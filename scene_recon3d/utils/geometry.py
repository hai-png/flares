"""Geometry utilities for 3D scene reconstruction pipeline.

Provides functions for:
- Quaternion / rotation matrix conversions
- Mesh scaling and transformation
- Alignment of generated meshes to 3D bounding boxes
- Pose refinement using correspondence points
- Mesh rendering for MARCO refinement
- Image cropping and keypoint sampling
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


def align_mesh_to_bbox(
    mesh: trimesh.Trimesh,
    bbox_center: np.ndarray,
    bbox_dims: np.ndarray,
    bbox_quat: np.ndarray,
    bbox_up_axis: str = "y",
) -> Tuple[trimesh.Trimesh, float, np.ndarray, np.ndarray]:
    """Align a canonically-posed mesh to a 3D bounding box.

    The generated mesh from Hunyuan3D is in a canonical pose centered at origin
    in Y-up convention (X-right, Y-up, Z-toward-viewer).

    WildDet3D outputs bounding boxes in OpenCV camera coordinates (X-right,
    Y-down, Z-forward) with dimensions (W, L, H) where:
      - W (width)  → Z-axis (forward/backward)
      - L (length) → X-axis (left/right)
      - H (height) → Y-axis (down/up, vertical)

    This is NOT a simple (X, Y, Z) mapping!  The scale computation must
    remap bbox dimensions to match the mesh's local axes before dividing.

    Steps:
    1. Remap bbox dims: (W, L, H) → (L, H, W) to match (mesh_X, mesh_Y, mesh_Z)
    2. Compute uniform scale factor from remapped ratios
    3. Convert quaternion to rotation matrix
    4. Apply axis conversion (OpenCV Y-down → mesh Y-up)
    5. Center, scale, rotate, and translate the mesh

    Args:
        mesh: The canonical trimesh.Trimesh object
        bbox_center: (3,) 3D center of the bounding box in meters
        bbox_dims: (3,) (W, L, H) dimensions in meters from WildDet3D
        bbox_quat: (4,) (qw, qx, qy, qz) quaternion for rotation
        bbox_up_axis: Up axis convention of the 3D bbox ("y" for WildDet3D/OpenCV)

    Returns:
        Tuple of (aligned_mesh, scale_factor, rotation_matrix, translation)
    """
    # Get mesh bounding box
    mesh_bounds = mesh.bounds  # (2, 3) min and max
    mesh_extents = mesh.extents  # (3,) size along each axis in Y-up frame
    mesh_center = mesh.bounds.mean(axis=0)

    # ── Remap bbox dimensions to match mesh axes ──────────────────
    #
    # WildDet3D bbox_dims = (W, L, H) where:
    #   W → camera Z (forward)  ↔  mesh Z (depth/front)
    #   L → camera X (right)    ↔  mesh X (width)
    #   H → camera Y (down)     ↔  mesh -Y (up)  [sign doesn't affect scale]
    #
    # So to compare with mesh_extents = (X, Y, Z):
    #   mesh X ↔ L = bbox_dims[1]
    #   mesh Y ↔ H = bbox_dims[2]
    #   mesh Z ↔ W = bbox_dims[0]
    W, L, H = bbox_dims[0], bbox_dims[1], bbox_dims[2]
    bbox_dims_remapped = np.array([L, H, W], dtype=np.float64)

    # Compute uniform scale: map mesh extents to bbox dims (remapped)
    mesh_extents_safe = np.maximum(mesh_extents, 1e-6)
    scale_ratios = bbox_dims_remapped / mesh_extents_safe
    # Use minimum ratio so the mesh fits inside the bbox on all axes
    scale_factor = float(np.min(scale_ratios))

    logger.info(
        f"  Scale computation: mesh_extents={mesh_extents}, "
        f"bbox_dims=(W={W:.3f}, L={L:.3f}, H={H:.3f}), "
        f"remapped=(L={L:.3f}, H={H:.3f}, W={W:.3f}), "
        f"ratios={scale_ratios}, scale={scale_factor:.4f}"
    )

    # ── Rotation from quaternion ──────────────────────────────────
    R = quaternion_to_rotation_matrix(bbox_quat)

    # ── Axis convention conversion ────────────────────────────────
    # WildDet3D uses OpenCV (Y-down, Z-forward).
    # Hunyuan3D generates meshes in Y-up convention.
    # The quaternion rotates the box in camera coordinates.
    # We need to convert the mesh from Y-up to camera Y-down BEFORE
    # applying the quaternion rotation.
    #
    # The conversion: (x, y, z)_mesh → (x, -y, -z)_camera
    # This is a 180° rotation around X: diag(1, -1, -1)
    #
    # Combined rotation: R_final = R_quat @ axis_conversion
    # Effect: first convert mesh to camera frame, then rotate by quaternion
    if bbox_up_axis == "y":
        axis_conversion = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1],
        ], dtype=np.float64)
        R = R @ axis_conversion

    # ── Build the full transformation ─────────────────────────────
    # 1. Center the mesh at origin
    # 2. Scale it
    # 3. Rotate it
    # 4. Translate to bbox center

    # Center the mesh at origin first
    centered_mesh = mesh.copy()
    centered_mesh.apply_translation(-mesh_center)

    # Apply scale
    centered_mesh.apply_scale(scale_factor)

    # Apply rotation
    T_rot = np.eye(4)
    T_rot[:3, :3] = R
    centered_mesh.apply_transform(T_rot)

    # Apply translation to bbox center
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
    current_rotation: np.ndarray,
    current_translation: np.ndarray,
    src_points_2d: np.ndarray,
    tgt_points_2d: np.ndarray,
    src_points_3d: np.ndarray,
    camera_intrinsics: np.ndarray,
    scale_factor: float,
    mesh: trimesh.Trimesh,
) -> Tuple[np.ndarray, np.ndarray]:
    """Refine 6-DoF object pose using 2D-3D correspondences from MARCO.

    Uses PnP (Perspective-n-Point) to refine the pose given semantic
    correspondence points between the cropped object image and rendered mesh.

    Args:
        current_rotation: (3,3) current rotation matrix
        current_translation: (3,) current translation vector
        src_points_2d: (N,2) source 2D points (from cropped object image)
        tgt_points_2d: (N,2) target 2D points (from rendered mesh image)
        src_points_3d: (N,3) corresponding 3D points on the mesh
        camera_intrinsics: (3,3) camera intrinsic matrix
        scale_factor: Scale factor applied to the mesh
        mesh: The mesh object

    Returns:
        Tuple of (refined_rotation, refined_translation)
    """
    if len(src_points_2d) < 4:
        # Not enough correspondences for PnP; return current pose
        return current_rotation, current_translation

    # Scale 3D points by the same scale factor used in alignment.
    # These points are in the object's local coordinate frame (centered at origin).
    src_points_3d_scaled = src_points_3d * scale_factor

    # solvePnP expects 3D points in the OBJECT's local coordinate frame and
    # 2D points in the image. It returns (rvec, tvec) such that:
    #   p_camera = R @ p_object + t
    # Previously the code mistakenly transformed 3D points to world/camera
    # frame before passing them, which makes solvePnP solve for an identity
    # transform — rendering the refinement useless.
    dist_coeffs = np.zeros((4, 1))  # No distortion

    success, rvec, tvec = cv2.solvePnP(
        src_points_3d_scaled.astype(np.float64),
        tgt_points_2d.astype(np.float64),
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
    """Render a mesh to an RGB image and extract 2D keypoints.

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
            # Without original dimensions, assume intrinsics are already
            # for the render resolution (backward compatibility)
            pass
    else:
        # Default: fit camera to see the whole mesh
        fx = fy = resolution
        cx = cy = resolution / 2.0

    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=100.0)

    # OpenCV convention: Y-down, Z-forward
    # OpenGL convention: Y-up, Z-backward (camera looks along -Z)
    # We apply the conversion matrix so the mesh (in OpenCV coords) is
    # rendered correctly by the OpenGL camera.
    #   OpenGL_pose = OpenCV_pose @ flip_matrix
    # where flip_matrix flips Y and Z axes.
    cv2gl = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1],
    ], dtype=np.float64)

    if camera_pose is not None:
        gl_camera_pose = camera_pose @ cv2gl
    else:
        # Camera at origin in OpenCV convention → in OpenGL, the camera
        # needs to be placed so it looks along -Z at the scene.
        gl_camera_pose = cv2gl

    camera_node = scene.add(camera, pose=gl_camera_pose)

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
        # trimesh uses pyglet/PIL for offscreen rendering
        png = scene.save_image(resolution=[resolution, resolution])
        if png is not None:
            import io
            from PIL import Image
            img = Image.open(io.BytesIO(png))
            return np.array(img)[:, :, :3], np.zeros((resolution, resolution))
    except Exception:
        pass

    # Final fallback: create a simple depth-based silhouette
    # Project mesh vertices to image
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

    # Simple projection
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
) -> np.ndarray:
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
        Rendered RGB image (H, W, 3)
    """
    # Apply transform to mesh
    rendered_mesh = mesh.copy()
    rendered_mesh.apply_transform(transform)

    # Set camera pose (looking at the object from front)
    # In camera coordinate system, camera is at origin looking along +Z
    camera_pose = np.eye(4)

    rgb, _ = render_mesh(
        rendered_mesh,
        resolution=resolution,
        camera_intrinsics=camera_intrinsics,
        camera_pose=camera_pose,
        original_image_shape=original_image_shape,
    )
    return rgb


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
        # Sample keypoints on a uniform grid within the mask
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
        # Sample keypoints along the contour of the mask
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return np.zeros((0, 2), dtype=np.float32)

        # Use the largest contour
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

    Pure-numpy fallback that does NOT require rtree or pyembree.
    Tests the ray against every triangle and returns the closest hit.

    Args:
        ray_origin: (3,) ray origin
        ray_dir: (3,) normalized ray direction
        vertices: (V, 3) mesh vertices
        faces: (F, 3) triangle face indices
        eps: epsilon to avoid self-intersection

    Returns:
        (3,) intersection point or None if no hit
    """
    # Get triangle vertices for all faces at once
    v0 = vertices[faces[:, 0]]  # (F, 3)
    v1 = vertices[faces[:, 1]]  # (F, 3)
    v2 = vertices[faces[:, 2]]  # (F, 3)

    edge1 = v1 - v0  # (F, 3)
    edge2 = v2 - v0  # (F, 3)

    # h = cross(ray_dir, edge2)
    h = np.cross(ray_dir, edge2)  # (F, 3)
    a = np.einsum('j,ij->i', edge1, h)  # dot product per row: (F,)

    # Check if ray is parallel to triangle
    valid = np.abs(a) > eps

    f = np.zeros(len(a))
    f[valid] = 1.0 / a[valid]

    # s = ray_origin - v0
    s = ray_origin - v0  # (F, 3)

    # u = f * dot(s, h)
    u = f * np.einsum('ij,ij->i', s, h)

    # Check u bounds
    valid &= (u >= 0.0) & (u <= 1.0)

    # q = cross(s, edge1)
    q = np.cross(s, edge1)  # (F, 3)

    # v = f * dot(ray_dir, q)
    v = f * np.einsum('j,ij->i', ray_dir, q)

    # Check v bounds
    valid &= (v >= 0.0) & (u + v <= 1.0)

    # t = f * dot(edge2, q)
    t = f * np.einsum('ij,ij->i', edge2, q)

    # Check t > eps (intersection in front of ray)
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

    Projects rays from 2D keypoints and finds intersections with the mesh.
    First tries trimesh's built-in ray intersection (which uses rtree/pyembree
    if available for speed). If that fails (e.g. rtree not installed), falls
    back to a pure-numpy Möller–Trumbore implementation.

    Args:
        mesh: The trimesh object
        keypoints_2d: (N, 2) 2D keypoint coordinates (x, y)
        camera_intrinsics: (3,3) camera K matrix
        mesh_transform: (4,4) optional transform applied to the mesh

    Returns:
        (N, 3) 3D points on the mesh surface
    """
    if mesh_transform is not None:
        mesh = mesh.copy()
        mesh.apply_transform(mesh_transform)

    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]

    # Try trimesh's ray intersection first (fast with rtree/pyembree)
    use_trimesh_ray = True
    try:
        import rtree  # noqa: F401
    except ImportError:
        use_trimesh_ray = False

    if not use_trimesh_ray:
        # Check if pyembree is available as an alternative accelerator
        try:
            import embree  # noqa: F401
            use_trimesh_ray = True
        except ImportError:
            logger.info(
                "rtree/embree not installed — using pure-numpy Möller–Trumbore "
                "ray-triangle intersection (slower but no extra dependencies). "
                "Install rtree for faster ray queries: pip install rtree"
            )

    points_3d = []
    for kp in keypoints_2d:
        u, v = kp[0], kp[1]
        # Create ray from pixel
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
                    distances = np.linalg.norm(locations, axis=1)
                    closest_idx = np.argmin(distances)
                    points_3d.append(locations[closest_idx])
                    found = True
            except Exception:
                # Fall through to Möller–Trumbore
                pass

        if not found:
            # Pure-numpy fallback: Möller–Trumbore ray-triangle intersection
            hit = _ray_mesh_intersect_moller_trumbore(
                ray_origin, ray_dir,
                mesh.vertices, mesh.faces,
            )
            if hit is not None:
                points_3d.append(hit)
            else:
                points_3d.append(np.array([0.0, 0.0, 0.0]))  # fallback

    return np.array(points_3d, dtype=np.float32)
