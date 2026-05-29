"""Geometry utilities for 3D scene reconstruction pipeline.

Provides functions for:
- Quaternion / rotation matrix conversions
- Mesh scaling and transformation
- Alignment of generated meshes to 3D bounding boxes
- Pose refinement using correspondence points
- Mesh rendering for MARCO refinement
- Image cropping and keypoint sampling
- Depth-based 3D point extraction
"""

from __future__ import annotations

import logging
import numpy as np
import cv2
import trimesh
from typing import Optional, Tuple, List
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


# ─── Quaternion / Rotation Conversions ───────────────────────────

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (qw, qx, qy, qz) to 3x3 rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)  # normalize
    qw, qx, qy, qz = q

    R = np.array([
        [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qw*qz),      2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),      1 - 2*(qx**2 + qz**2),   2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),      2*(qy*qz + qw*qx),       1 - 2*(qx**2 + qy**2)]
    ])
    return R


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (qw, qx, qy, qz)."""
    rot = Rotation.from_matrix(R)
    q = rot.as_quat()  # returns (qx, qy, qz, qw)
    return np.array([q[3], q[0], q[1], q[2]])  # (qw, qx, qy, qz)


# ─── Mesh Operations ────────────────────────────────────────────

def scale_mesh(mesh: trimesh.Trimesh, scale: float) -> trimesh.Trimesh:
    """Scale a mesh uniformly by a factor."""
    scaled = mesh.copy()
    scaled.apply_scale(scale)
    return scaled


def transform_mesh(
    mesh: trimesh.Trimesh,
    rotation: np.ndarray = None,
    translation: np.ndarray = None,
    scale: float = 1.0,
) -> trimesh.Trimesh:
    """Apply scale, rotation, and translation to a mesh.

    The canonical mesh is first scaled, then rotated, then translated.
    Returns a new mesh (copy).
    """
    transformed = mesh.copy()

    # Scale
    if scale != 1.0:
        transformed.apply_scale(scale)

    # Build 4x4 homogeneous transform
    T = np.eye(4)
    if rotation is not None:
        T[:3, :3] = rotation
    if translation is not None:
        T[:3, 3] = translation

    transformed.apply_transform(T)
    return transformed


def _compute_alignment_2d_iou(
    mesh: trimesh.Trimesh,
    bbox_2d: np.ndarray,
    camera_intrinsics: np.ndarray,
) -> float:
    """Compute 2D IoU between a mesh's projection and a 2D bounding box.

    Projects the mesh's axis-aligned bounding box corners to 2D and
    computes IoU with the detected 2D bbox.

    Args:
        mesh: Transformed mesh (already in camera space)
        bbox_2d: (4,) xyxy detected 2D bbox
        camera_intrinsics: (3,3) camera K matrix

    Returns:
        IoU value (0.0 if no overlap)
    """
    try:
        bounds = mesh.bounds  # (2, 3) min, max
        mins, maxs = bounds
        corners = np.array([
            [mins[0], mins[1], mins[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], maxs[1], mins[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]],
            [mins[0], maxs[1], maxs[2]],
        ])
        corners_2d = project_points_to_2d(corners, camera_intrinsics)
        px1 = corners_2d[:, 0].min()
        py1 = corners_2d[:, 1].min()
        px2 = corners_2d[:, 0].max()
        py2 = corners_2d[:, 1].max()

        dx1, dy1, dx2, dy2 = bbox_2d.tolist()

        ix1 = max(px1, dx1)
        iy1 = max(py1, dy1)
        ix2 = min(px2, dx2)
        iy2 = min(py2, dy2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_proj = (px2 - px1) * (py2 - py1)
        area_det = (dx2 - dx1) * (dy2 - dy1)
        iou = inter / max(area_proj + area_det - inter, 1e-6)
        return float(iou)
    except Exception:
        return 0.0


def align_mesh_to_bbox(
    mesh: trimesh.Trimesh,
    bbox_center: np.ndarray,
    bbox_dims: np.ndarray,
    bbox_quat: np.ndarray,
    camera_intrinsics: Optional[np.ndarray] = None,
    bbox_2d: Optional[np.ndarray] = None,
) -> Tuple[trimesh.Trimesh, float, np.ndarray, np.ndarray]:
    """Align a canonically-posed mesh to a 3D bounding box.

    The generated mesh from Hunyuan3D is in a canonical pose centered at
    origin in Y-up convention (X-right, Y-up, Z-toward-viewer).  The mesh
    fits within a bounding box of approximately [-1, 1] on each axis.

    WildDet3D outputs bounding boxes in OpenCV camera coordinates (X-right,
    Y-down, Z-forward) with dimensions (W, L, H) where:
      - W (width)  maps to Z-axis in the bbox's local frame
      - L (length) maps to X-axis in the bbox's local frame
      - H (height) maps to Y-axis in the bbox's local frame

    Convention-based axis mapping (mesh → bbox local frame):
      mesh X → bbox local X (L) — both are left/right
      mesh Y → bbox local -Y (H) — up ↔ down flip via R_axis
      mesh Z → bbox local -Z (W) — toward-viewer ↔ forward flip

    The quaternion R_quat then rotates from the bbox local frame to camera
    coordinates.  When camera intrinsics and the 2D detection box are
    available, multiple axis permutations are tested and the one with
    the highest 2D projection IoU is selected.

    Args:
        mesh: The canonical trimesh.Trimesh object
        bbox_center: (3,) 3D center of the bounding box in meters
        bbox_dims: (3,) (W, L, H) dimensions in meters from WildDet3D
        bbox_quat: (4,) (qw, qx, qy, qz) quaternion for rotation
        camera_intrinsics: Optional (3,3) camera K matrix for 2D validation
        bbox_2d: Optional (4,) xyxy 2D detection bbox for validation

    Returns:
        Tuple of (aligned_mesh, scale_factor, rotation_matrix, translation)
    """
    # Get mesh bounding box
    mesh_extents = mesh.extents  # (3,) size along each axis in Y-up frame
    mesh_center = mesh.bounds.mean(axis=0)
    mesh_extents_safe = np.maximum(mesh_extents, 1e-6)

    # ── Remap bbox dimensions to match mesh axes ──────────────────
    #
    # WildDet3D bbox_dims = (W, L, H) where:
    #   W → bbox local Z ↔ mesh -Z (depth, with direction flip)
    #   L → bbox local X ↔ mesh X (width)
    #   H → bbox local Y ↔ mesh -Y (height, with up/down flip)
    #
    # Direct correspondence (before axis conversion):
    #   bbox local (X, Y, Z) = (L, H, W)
    #   mesh (X, Y, Z) in the canonical Y-up frame maps to:
    #     mesh X → bbox X (L), mesh Y → bbox -Y (H), mesh Z → bbox -Z (W)
    #
    # After R_axis (Y-up→Y-down, Z-toward→Z-forward), the extents match
    # directly because R_axis only flips signs (preserving magnitude).
    W, L, H = float(bbox_dims[0]), float(bbox_dims[1]), float(bbox_dims[2])
    bbox_dims_mapped = np.array([L, H, W], dtype=np.float64)

    # ── Rotation from quaternion ──────────────────────────────────
    R_quat = quaternion_to_rotation_matrix(bbox_quat)

    # ── Axis convention conversion ────────────────────────────────
    # Hunyuan3D mesh: X-right, Y-up, Z-toward-viewer
    # OpenCV camera:  X-right, Y-down, Z-forward
    # Conversion: (x, y, z)_mesh → (x, -y, -z)_opencv
    # diag(1, -1, -1) is a proper rotation (det = +1).
    R_axis = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1],
    ], dtype=np.float64)

    # ── Candidate permutations ────────────────────────────────────
    # The generated mesh may not have a predictable alignment of its
    # principal axes with the bbox axes.  We test four 90° rotations
    # around the Y axis (which preserve the Y-up / Y-down relationship)
    # and pick the one with the best 2D projection IoU.
    #
    # All four are proper rotations (det = +1):
    #   0°: identity
    #   90°: rotates X→-Z, Z→X
    #  180°: rotates X→-X, Z→-Z
    #  270°: rotates X→Z, Z→-X
    R_perms = [
        ("0deg", np.eye(3, dtype=np.float64)),
        ("90deg", np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64)),
        ("180deg", np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)),
        ("270deg", np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)),
    ]

    # Compute per-candidate scale factors and test 2D IoU.
    #
    # For each permutation R_perm, the effective dimension mapping is:
    #   p_bbox = R_perm @ p_mesh  (after centering & axis conversion)
    # So bbox_dims_mapped must be compared against the permuted mesh extents.
    #
    # The permuted mesh extents are: extents_perm[i] = sum_j |R_perm[i,j]| * mesh_extents[j]
    # (because R_perm permutes/reflects axes, and extents are always positive)
    candidates = []
    for name, R_perm in R_perms:
        # Compute the effective extent that each bbox axis "sees"
        # after the permutation is applied to the mesh.
        extents_perm = np.abs(R_perm) @ mesh_extents_safe
        ratios = bbox_dims_mapped / np.maximum(extents_perm, 1e-6)
        sf = float(np.median(ratios))
        candidates.append((name, R_perm, sf, ratios))

    # ── Select the best candidate ─────────────────────────────────
    if camera_intrinsics is not None and bbox_2d is not None:
        # Test each candidate by computing 2D projection IoU
        best_iou = -1.0
        best_idx = 0

        for idx, (name, R_perm, sf, ratios) in enumerate(candidates):
            R = R_quat @ R_axis @ R_perm
            test_mesh = mesh.copy()
            test_mesh.apply_translation(-mesh_center)
            test_mesh.apply_scale(sf)
            T_rot = np.eye(4)
            T_rot[:3, :3] = R
            test_mesh.apply_transform(T_rot)
            test_mesh.apply_translation(bbox_center)
            iou = _compute_alignment_2d_iou(test_mesh, bbox_2d, camera_intrinsics)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        best_name, R_perm, scale_factor, scale_ratios = candidates[best_idx]
        logger.info(
            f"  2D validation: tested 4 Y-rotations, "
            f"chose {best_name}, best_IoU={best_iou:.3f}"
        )
    else:
        # Without 2D validation, use the most balanced scale ratios
        # (lowest coefficient of variation = std/mean)
        best_cv = float("inf")
        best_idx = 0
        for idx, (name, R_perm, sf, ratios) in enumerate(candidates):
            cv = np.std(ratios) / max(np.mean(ratios), 1e-6)
            if cv < best_cv:
                best_cv = cv
                best_idx = idx
        best_name, R_perm, scale_factor, scale_ratios = candidates[best_idx]

    R = R_quat @ R_axis @ R_perm

    # Log detailed alignment info
    logger.info(
        f"  Scale computation: mesh_extents={mesh_extents}, "
        f"bbox_dims=(W={W:.3f}, L={L:.3f}, H={H:.3f}), "
        f"mapped=(L={L:.3f}, H={H:.3f}, W={W:.3f}), "
        f"scale_ratios={scale_ratios}, "
        f"scale={scale_factor:.4f}, "
        f"perm={best_name}"
    )

    # ── Build the full transformation ─────────────────────────────
    # 1. Center the mesh at origin
    # 2. Scale it
    # 3. Rotate it (includes permutation + axis conversion + quaternion)
    # 4. Translate to bbox center

    centered_mesh = mesh.copy()
    centered_mesh.apply_translation(-mesh_center)
    centered_mesh.apply_scale(scale_factor)

    T_rot = np.eye(4)
    T_rot[:3, :3] = R
    centered_mesh.apply_transform(T_rot)

    centered_mesh.apply_translation(bbox_center)

    return centered_mesh, scale_factor, R, bbox_center


def bbox3d_to_corners_opencv(
    bbox_center: np.ndarray,
    bbox_dims: np.ndarray,
    bbox_quat: np.ndarray,
) -> np.ndarray:
    """Compute 8 corners of a 3D bounding box in camera coordinates.

    WildDet3D convention (OpenCV): bbox_dims = (W, L, H) where
      W → Z-axis, L → X-axis, H → Y-axis.

    Args:
        bbox_center: (3,) 3D center in camera coords
        bbox_dims: (3,) (W, L, H) dimensions
        bbox_quat: (4,) (qw, qx, qy, qz) quaternion

    Returns:
        (8, 3) array of corner positions in camera coordinates
    """
    W, L, H = bbox_dims[0], bbox_dims[1], bbox_dims[2]

    # Corners in the box's local frame (before rotation)
    # X: ±L/2, Y: ±H/2, Z: ±W/2  (WildDet3D convention)
    x_corners = np.array([L/2, L/2, -L/2, -L/2, L/2, L/2, -L/2, -L/2])
    y_corners = np.array([H/2, H/2, H/2, H/2, -H/2, -H/2, -H/2, -H/2])
    z_corners = np.array([W/2, -W/2, -W/2, W/2, W/2, -W/2, -W/2, W/2])

    corners = np.stack([x_corners, y_corners, z_corners], axis=1)  # (8, 3)

    # Apply rotation
    R = quaternion_to_rotation_matrix(bbox_quat)
    corners = (R @ corners.T).T

    # Apply translation
    corners += bbox_center

    return corners


def project_points_to_2d(
    points_3d: np.ndarray,
    camera_intrinsics: np.ndarray,
) -> np.ndarray:
    """Project 3D points to 2D image coordinates.

    Args:
        points_3d: (N, 3) 3D points in camera coordinates
        camera_intrinsics: (3, 3) camera intrinsic matrix

    Returns:
        (N, 2) 2D pixel coordinates (u, v)
    """
    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]

    # Project: u = fx * X/Z + cx, v = fy * Y/Z + cy
    z = points_3d[:, 2]
    z_safe = np.where(np.abs(z) > 1e-6, z, 1e-6)

    u = fx * points_3d[:, 0] / z_safe + cx
    v = fy * points_3d[:, 1] / z_safe + cy

    return np.stack([u, v], axis=1)


def draw_bbox3d_on_image(
    image: np.ndarray,
    bbox_center: np.ndarray,
    bbox_dims: np.ndarray,
    bbox_quat: np.ndarray,
    camera_intrinsics: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw a 3D bounding box projected onto a 2D image.

    Args:
        image: (H, W, 3) uint8 RGB image
        bbox_center: (3,) 3D center in camera coords
        bbox_dims: (3,) (W, L, H) dimensions
        bbox_quat: (4,) (qw, qx, qy, qz) quaternion
        camera_intrinsics: (3, 3) camera intrinsic matrix
        color: RGB color for the box edges
        thickness: Line thickness

    Returns:
        Image with 3D bounding box drawn
    """
    corners_3d = bbox3d_to_corners_opencv(bbox_center, bbox_dims, bbox_quat)
    corners_2d = project_points_to_2d(corners_3d, camera_intrinsics)

    # Draw edges connecting the 8 corners
    # Bottom face: 0-1-2-3-0, Top face: 4-5-6-7-4, Verticals: 0-4, 1-5, 2-6, 3-7
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    ]

    vis = image.copy()
    h, w = vis.shape[:2]
    for i, j in edges:
        pt1 = corners_2d[i].astype(int)
        pt2 = corners_2d[j].astype(int)
        # Only draw if both points are reasonably within image bounds
        if (0 <= pt1[0] < w and 0 <= pt1[1] < h and
            0 <= pt2[0] < w and 0 <= pt2[1] < h):
            cv2.line(vis, tuple(pt1), tuple(pt2), color, thickness)

    return vis


# ─── Pose Refinement with Correspondences ────────────────────────

def refine_pose_with_correspondences(
    object_points_3d: np.ndarray,
    image_points_2d: np.ndarray,
    camera_intrinsics: np.ndarray,
    current_rotation: np.ndarray,
    current_translation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Refine 6-DoF object pose using 2D-3D correspondences.

    Uses PnP (Perspective-n-Point) to find the pose that best explains
    the observed 2D image points given the corresponding 3D object points.

    Args:
        object_points_3d: (N,3) 3D points in the object's local coordinate
                          frame (already scaled by scale_factor)
        image_points_2d: (N,2) 2D observed points in the camera image
                         (in the coordinate system matching camera_intrinsics)
        camera_intrinsics: (3,3) camera intrinsic matrix matching the
                           coordinate system of image_points_2d
        current_rotation: (3,3) current rotation matrix (fallback if PnP fails)
        current_translation: (3,) current translation vector (fallback)

    Returns:
        Tuple of (refined_rotation, refined_translation)
    """
    if len(object_points_3d) < 4:
        return current_rotation, current_translation

    dist_coeffs = np.zeros((4, 1))  # No distortion

    success, rvec, tvec = cv2.solvePnP(
        object_points_3d.astype(np.float64),
        image_points_2d.astype(np.float64),
        camera_intrinsics.astype(np.float64),
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return current_rotation, current_translation

    # Convert Rodrigues vector to rotation matrix
    refined_R, _ = cv2.Rodrigues(rvec)
    refined_t = tvec.flatten()

    return refined_R, refined_t


def refine_pose_icp(
    mesh: trimesh.Trimesh,
    current_transform: np.ndarray,
    correspondence_src_3d: np.ndarray,
    correspondence_tgt_3d: np.ndarray,
) -> np.ndarray:
    """Refine pose using ICP-like alignment with 3D-3D correspondences.

    Uses SVD-based rigid transformation estimation from matched 3D point pairs.

    Args:
        mesh: The mesh object
        current_transform: (4,4) current homogeneous transform
        correspondence_src_3d: (N,3) 3D points from current mesh pose
        correspondence_tgt_3d: (N,3) 3D points from target (ground truth)

    Returns:
        (4,4) refined homogeneous transform
    """
    if len(correspondence_src_3d) < 3:
        return current_transform

    # SVD-based rigid transform estimation
    src_centered = correspondence_src_3d - correspondence_src_3d.mean(axis=0)
    tgt_centered = correspondence_tgt_3d - correspondence_tgt_3d.mean(axis=0)

    H = src_centered.T @ tgt_centered
    U, S, Vt = np.linalg.svd(H)

    R_refine = Vt.T @ U.T

    # Ensure proper rotation (det = +1)
    if np.linalg.det(R_refine) < 0:
        Vt[-1, :] *= -1
        R_refine = Vt.T @ U.T

    t_refine = correspondence_tgt_3d.mean(axis=0) - R_refine @ correspondence_src_3d.mean(axis=0)

    # Build refinement transform
    T_refine = np.eye(4)
    T_refine[:3, :3] = R_refine
    T_refine[:3, 3] = t_refine

    # Compose with current transform
    return T_refine @ current_transform


# ─── Mesh Rendering ──────────────────────────────────────────────

def render_mesh(
    mesh: trimesh.Trimesh,
    resolution: int = 512,
    camera_intrinsics: np.ndarray = None,
    camera_pose: np.ndarray = None,
    original_image_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render a mesh to an RGB image and depth map.

    Uses pyrender or trimesh's offscreen rendering.

    When using pyrender, the mesh is assumed to be in OpenCV camera coordinates
    (Y-down, Z-forward). pyrender uses OpenGL convention (Y-up, Z-backward),
    so we apply the OpenCV→OpenGL flip matrix to the camera pose to compensate.

    When camera_intrinsics are provided, they must be for the original image
    resolution. They are automatically scaled to the render resolution.

    Args:
        mesh: trimesh.Trimesh to render
        resolution: Output image resolution (square)
        camera_intrinsics: (3,3) optional camera K matrix (at original image resolution)
        camera_pose: (4,4) optional camera extrinsic matrix (in OpenCV convention)
        original_image_shape: (H, W) of the original image, needed for intrinsics scaling

    Returns:
        Tuple of (rendered_image (H,W,3), depth_map (H,W))
    """
    try:
        import pyrender
    except ImportError:
        # Fallback: use trimesh's simple rendering
        return _render_mesh_trimesh(mesh, resolution, camera_intrinsics, camera_pose)

    scene = pyrender.Scene()
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh_pyrender)

    if camera_intrinsics is not None:
        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[1, 1]
        cx = camera_intrinsics[0, 2]
        cy = camera_intrinsics[1, 2]

        # Scale intrinsics from original image resolution to render resolution
        if original_image_shape is not None:
            orig_h, orig_w = original_image_shape
            scale_x = resolution / orig_w
            scale_y = resolution / orig_h
            fx *= scale_x
            fy *= scale_y
            cx *= scale_x
            cy *= scale_y
    else:
        # Default: fit camera to see the whole mesh
        fx = fy = resolution
        cx = cy = resolution / 2.0

    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=100.0)

    # OpenCV → OpenGL conversion
    cv2gl = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1],
    ], dtype=np.float64)

    if camera_pose is not None:
        gl_camera_pose = camera_pose @ cv2gl
    else:
        gl_camera_pose = cv2gl

    scene.add(camera, pose=gl_camera_pose)

    # Add lighting (in the same frame as the camera)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=gl_camera_pose)

    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    color, depth = renderer.render(scene)
    renderer.delete()

    return color, depth


def _render_mesh_trimesh(
    mesh: trimesh.Trimesh,
    resolution: int = 512,
    camera_intrinsics: np.ndarray = None,
    camera_pose: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback rendering using trimesh's built-in scene rendering."""
    scene = trimesh.Scene()
    scene.add_geometry(mesh)

    # Try to render using trimesh
    try:
        png = scene.save_image(resolution=[resolution, resolution])
        if png is not None:
            import io
            from PIL import Image
            img = Image.open(io.BytesIO(png))
            return np.array(img)[:, :, :3], np.zeros((resolution, resolution))
    except Exception:
        pass

    # Final fallback: create a simple depth-based silhouette
    vertices = mesh.vertices
    if camera_pose is not None:
        R = camera_pose[:3, :3]
        t = camera_pose[:3, 3]
        vertices = (R @ vertices.T).T + t

    if camera_intrinsics is not None:
        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[1, 1]
        cx = camera_intrinsics[0, 2]
        cy = camera_intrinsics[1, 2]
    else:
        fx = fy = resolution / 2.0
        cx = cy = resolution / 2.0

    img = np.ones((resolution, resolution, 3), dtype=np.uint8) * 255
    depth = np.zeros((resolution, resolution), dtype=np.float32)

    for v in vertices:
        if v[2] > 0:  # In front of camera
            u = int(fx * v[0] / v[2] + cx)
            vv = int(fy * v[1] / v[2] + cy)
            if 0 <= u < resolution and 0 <= vv < resolution:
                img[vv, u] = [128, 128, 128]
                depth[vv, u] = v[2]

    return img, depth


def render_mesh_for_marco(
    mesh: trimesh.Trimesh,
    transform: np.ndarray,
    camera_intrinsics: np.ndarray,
    resolution: int = 512,
    original_image_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Render a mesh with its transform for MARCO comparison.

    Renders the mesh from the same viewpoint as the original camera,
    so the rendered image can be compared with the cropped object image.

    Args:
        mesh: The canonical mesh
        transform: (4,4) current pose of the mesh
        camera_intrinsics: (3,3) camera K matrix (at original image resolution)
        resolution: Render resolution
        original_image_shape: (H, W) of the original image, for intrinsics scaling

    Returns:
        Tuple of (rendered RGB image (H, W, 3), depth_map (H, W))
    """
    # Apply transform to mesh
    rendered_mesh = mesh.copy()
    rendered_mesh.apply_transform(transform)

    # Camera at origin in OpenCV convention
    camera_pose = np.eye(4)

    rgb, depth = render_mesh(
        rendered_mesh,
        resolution=resolution,
        camera_intrinsics=camera_intrinsics,
        camera_pose=camera_pose,
        original_image_shape=original_image_shape,
    )
    return rgb, depth


def unproject_depth_to_3d(
    keypoints_2d: np.ndarray,
    depth_map: np.ndarray,
    camera_intrinsics: np.ndarray,
) -> np.ndarray:
    """Unproject 2D keypoints with depth values to 3D camera-space points.

    For each 2D keypoint, looks up the depth value and unprojects to
    get the 3D position in camera coordinates.

    Args:
        keypoints_2d: (N, 2) 2D pixel coordinates (u, v) in the depth map
        depth_map: (H, W) depth map from rendering (depth values in camera space)
        camera_intrinsics: (3,3) camera K matrix matching the depth map resolution

    Returns:
        (N, 3) 3D points in camera coordinates
    """
    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]

    h, w = depth_map.shape
    points_3d = []

    for kp in keypoints_2d:
        u, v = int(round(kp[0])), int(round(kp[1]))

        # Clamp to valid range
        u = max(0, min(u, w - 1))
        v = max(0, min(v, h - 1))

        z = float(depth_map[v, u])

        if z <= 0 or not np.isfinite(z):
            # Invalid depth — try a small neighborhood
            found = False
            for du in range(-2, 3):
                for dv in range(-2, 3):
                    nu, nv = u + du, v + dv
                    if 0 <= nu < w and 0 <= nv < h:
                        nz = float(depth_map[nv, nu])
                        if nz > 0 and np.isfinite(nz):
                            z = nz
                            u, v = nu, nv
                            found = True
                            break
                if found:
                    break

        if z <= 0 or not np.isfinite(z):
            points_3d.append(np.array([0.0, 0.0, 0.0]))
            continue

        # Unproject: X = Z * (u - cx) / fx, Y = Z * (v - cy) / fy
        x = z * (u - cx) / fx
        y = z * (v - cy) / fy
        points_3d.append(np.array([x, y, z]))

    return np.array(points_3d, dtype=np.float64)


# ─── Image Operations ────────────────────────────────────────────

def crop_image_with_mask(
    image: np.ndarray,
    bbox: np.ndarray,
    mask: np.ndarray,
    padding: int = 20,
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """Crop an image using bounding box and apply mask.

    Args:
        image: (H, W, 3) uint8 image
        bbox: (4,) xyxy bounding box in pixels
        mask: (H, W) boolean mask
        padding: Padding around the bounding box
        background_color: RGB color for masked-out regions

    Returns:
        Tuple of (cropped_image (H', W', 3), crop_mask (H', W'), crop_offset (x1, y1))
        The crop_offset is the top-left corner of the crop in the original image,
        useful for adjusting camera intrinsics when projecting from crop coordinates.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)

    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    # Crop
    crop = image[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2].copy()

    # Apply mask: set background to specified color
    bg_mask = ~crop_mask
    for c in range(3):
        crop[:, :, c][bg_mask] = background_color[c]

    return crop, crop_mask, (x1, y1)


def sample_keypoints_from_mask(
    mask: np.ndarray,
    num_keypoints: int = 20,
    method: str = "uniform",
) -> np.ndarray:
    """Sample 2D keypoints from a binary mask region.

    Args:
        mask: (H, W) boolean mask
        num_keypoints: Number of keypoints to sample
        method: "uniform" (grid) or "random" or "contour"

    Returns:
        (N, 2) array of (x, y) keypoint coordinates in image pixels
    """
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    if method == "uniform":
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        n_side = int(np.ceil(np.sqrt(num_keypoints)))
        x_grid = np.linspace(x_min, x_max, n_side)
        y_grid = np.linspace(y_min, y_max, n_side)

        points = []
        for x in x_grid:
            for y in y_grid:
                ix, iy = int(round(x)), int(round(y))
                if 0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1]:
                    if mask[iy, ix]:
                        points.append([x, y])

        points = np.array(points, dtype=np.float32)
        if len(points) > num_keypoints:
            indices = np.random.choice(len(points), num_keypoints, replace=False)
            points = points[indices]

        return points

    elif method == "contour":
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return np.zeros((0, 2), dtype=np.float32)

        contour = max(contours, key=cv2.contourArea)
        contour_pts = contour.squeeze(1)  # (N, 2) xy

        if len(contour_pts) <= num_keypoints:
            return contour_pts.astype(np.float32)

        indices = np.linspace(0, len(contour_pts) - 1, num_keypoints, dtype=int)
        return contour_pts[indices].astype(np.float32)

    else:  # random
        indices = np.random.choice(len(xs), min(num_keypoints, len(xs)), replace=False)
        return np.stack([xs[indices], ys[indices]], axis=1).astype(np.float32)


def _ray_mesh_intersect_moller_trumbore(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    eps: float = 1e-8,
) -> Optional[np.ndarray]:
    """Möller–Trumbore ray-triangle intersection (single ray vs all triangles).

    Vectorised pure-numpy implementation that tests one ray against every
    triangle and returns the closest hit.  No external acceleration
    structure (rtree / embree) is required, making this portable across
    all platforms and numpy versions.

    Args:
        ray_origin: (3,) ray origin
        ray_dir: (3,) normalized ray direction
        vertices: (V, 3) mesh vertices
        faces: (F, 3) triangle face indices
        eps: epsilon to avoid self-intersection

    Returns:
        (3,) intersection point or None if no hit
    """
    # Ensure correct dtypes for robust indexing
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    if len(faces) == 0:
        return None

    # Get triangle vertices for all faces at once
    v0 = vertices[faces[:, 0]]  # (F, 3)
    v1 = vertices[faces[:, 1]]  # (F, 3)
    v2 = vertices[faces[:, 2]]  # (F, 3)

    edge1 = v1 - v0  # (F, 3)
    edge2 = v2 - v0  # (F, 3)

    h = np.cross(ray_dir, edge2)  # (F, 3)
    a = (edge1 * h).sum(axis=1)  # (F,)

    # Check if ray is parallel to triangle
    valid = np.abs(a) > eps

    f = np.zeros(len(a), dtype=np.float64)
    f[valid] = 1.0 / a[valid]

    s = ray_origin - v0  # (F, 3)
    u = f * (s * h).sum(axis=1)  # (F,)

    valid &= (u >= 0.0) & (u <= 1.0)

    q = np.cross(s, edge1)  # (F, 3)
    v = f * (q * ray_dir).sum(axis=1)  # (F,)

    valid &= (v >= 0.0) & (u + v <= 1.0)

    t = f * (edge2 * q).sum(axis=1)  # (F,)

    valid &= (t > eps)

    if not np.any(valid):
        return None

    # Find closest valid intersection
    t_valid = np.where(valid, t, np.inf)
    closest_idx = np.argmin(t_valid)
    t_closest = t[closest_idx]

    return ray_origin + t_closest * ray_dir


def get_mesh_3d_points_for_2d_keypoints(
    mesh: trimesh.Trimesh,
    keypoints_2d: np.ndarray,
    camera_intrinsics: np.ndarray,
    mesh_transform: np.ndarray = None,
) -> np.ndarray:
    """Get 3D points on the mesh surface corresponding to 2D keypoints.

    Projects rays from the camera through 2D keypoints and finds
    intersections with the mesh surface. The mesh MUST be in camera
    coordinates for the ray casting to produce meaningful results.

    If mesh_transform is provided, the mesh is first transformed to
    camera space before ray casting. The returned 3D points are in
    camera space.

    Args:
        mesh: The trimesh object (canonical or already in camera space)
        keypoints_2d: (N, 2) 2D keypoint coordinates (u, v) in pixels
        camera_intrinsics: (3,3) camera K matrix matching the keypoint coordinates
        mesh_transform: (4,4) optional transform to apply to the mesh
                        (e.g., current pose to bring it to camera space)

    Returns:
        (N, 3) 3D points on the mesh surface (in camera space if
        mesh_transform was applied, otherwise in mesh local space)
    """
    if mesh_transform is not None:
        mesh = mesh.copy()
        mesh.apply_transform(mesh_transform)

    # Ensure mesh has valid geometry for ray casting
    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        logger.warning("Mesh has no faces/vertices; returning zero 3D points")
        return np.zeros((len(keypoints_2d), 3), dtype=np.float32)

    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]

    # Use trimesh's accelerated ray casting when available (rtree/embree),
    # otherwise fall back to the vectorised Möller–Trumbore implementation.
    # Both produce identical results; trimesh's version is faster for large
    # meshes when an acceleration structure is installed.
    use_trimesh_ray = False
    try:
        import rtree  # noqa: F401
        use_trimesh_ray = True
    except ImportError:
        pass
    if not use_trimesh_ray:
        try:
            import embree  # noqa: F401
            use_trimesh_ray = True
        except ImportError:
            pass

    points_3d = []
    for kp in keypoints_2d:
        u, v = float(kp[0]), float(kp[1])
        # Create ray from pixel through camera origin
        ray_dir = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        ray_origin = np.array([0.0, 0.0, 0.0])

        found = False

        if use_trimesh_ray:
            try:
                locations, index_ray, index_tri = mesh.ray.intersects_location(
                    ray_origins=[ray_origin],
                    ray_directions=[ray_dir],
                )
                if len(locations) > 0:
                    # Pick the closest intersection (smallest t along the ray)
                    t_values = (locations - ray_origin) @ ray_dir
                    closest_idx = np.argmin(t_values)
                    points_3d.append(locations[closest_idx])
                    found = True
            except Exception:
                pass

        if not found:
            # Vectorised Möller–Trumbore ray-triangle intersection
            try:
                hit = _ray_mesh_intersect_moller_trumbore(
                    ray_origin, ray_dir,
                    mesh.vertices, mesh.faces,
                )
                if hit is not None:
                    points_3d.append(hit)
                else:
                    points_3d.append(np.array([0.0, 0.0, 0.0]))
            except Exception:
                points_3d.append(np.array([0.0, 0.0, 0.0]))

    return np.array(points_3d, dtype=np.float32)
