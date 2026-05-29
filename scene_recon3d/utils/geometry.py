"""Geometry utilities for 3D scene reconstruction pipeline.

Provides functions for:
- Quaternion / rotation matrix conversions
- Mesh scaling and transformation
- Alignment of generated meshes to 3D bounding boxes
- Pose refinement using correspondence points
- Mesh rendering for MARCO refinement
- Image cropping and keypoint sampling
- Depth-based 3D point extraction

Coordinate System Conventions
=============================
This module bridges three coordinate systems:

1. **Hunyuan3D mesh** (Y-up, glTF): X-right, Y-up, -Z-forward
   The generated mesh is centered at origin, approximately in [-1,1]^3.
   The "front" of the object faces -Z.

2. **WildDet3D 3D bounding box** (OpenCV camera): X-right, Y-down, Z-forward
   bbox format: [cx, cy, cz, W, L, H, qw, qx, qy, qz]
   Local frame convention: X=Length, Y=Height, Z=Width
   W → Z-extent, L → X-extent, H → Y-extent
   The quaternion (scalar-first) rotates from local frame → camera frame.

3. **Camera frame** (OpenCV): X-right, Y-down, Z-forward
   The camera looks along +Z.

Alignment Strategy
==================
The mesh-to-bbox alignment follows this logic:

  p_camera = R_quat @ R_convention @ R_yaw @ (scale * (p_mesh - mesh_center)) + bbox_center

Where:
  - R_convention: Fixed Y-up→OpenCV flip (diag(1, -1, -1))
    Maps mesh X→local X, mesh Y→local -Y, mesh Z→local -Z
  - R_yaw: Optional 90°/180°/270° rotation around the Y-axis (mesh up axis)
    Accounts for ambiguity in the generated mesh's yaw orientation
  - R_quat: The WildDet3D quaternion (local frame → camera frame)
  - scale: Uniform scale to fit the mesh inside the 3D bounding box
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
    """Convert quaternion (qw, qx, qy, qz) to 3x3 rotation matrix.

    Uses the scalar-first convention matching WildDet3D's output format.
    The quaternion is normalized before conversion.

    Args:
        q: (4,) quaternion in [qw, qx, qy, qz] order

    Returns:
        (3, 3) rotation matrix
    """
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    qw, qx, qy, qz = q

    R = np.array([
        [1 - 2*(qy**2 + qz**2),  2*(qx*qy - qw*qz),      2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz),      1 - 2*(qx**2 + qz**2),   2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy),      2*(qy*qz + qw*qx),       1 - 2*(qx**2 + qy**2)]
    ])
    return R


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (qw, qx, qy, qz).

    Args:
        R: (3, 3) rotation matrix

    Returns:
        (4,) quaternion in [qw, qx, qy, qz] order
    """
    rot = Rotation.from_matrix(R)
    q = rot.as_quat()  # scipy returns (qx, qy, qz, qw)
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

    if scale != 1.0:
        transformed.apply_scale(scale)

    T = np.eye(4)
    if rotation is not None:
        T[:3, :3] = rotation
    if translation is not None:
        T[:3, 3] = translation

    transformed.apply_transform(T)
    return transformed


# ─── 2D Projection Utilities ─────────────────────────────────────

def _compute_alignment_2d_iou(
    mesh: trimesh.Trimesh,
    bbox_2d: np.ndarray,
    camera_intrinsics: np.ndarray,
) -> float:
    """Compute 2D IoU between a mesh's projection and a 2D bounding box.

    Projects the mesh's oriented bounding box (OBB) corners to 2D and
    computes IoU with the detected 2D bbox.  Uses the OBB instead of the
    AABB because the aligned mesh is typically rotated, making the AABB
    much larger than the actual mesh silhouette and producing misleading
    IoU values.

    Only in-front-of-camera corners are used for the 2D bbox computation.
    If fewer than 3 corners are in front of the camera, returns 0.0.

    Args:
        mesh: Transformed mesh (already in camera space)
        bbox_2d: (4,) xyxy detected 2D bbox
        camera_intrinsics: (3,3) camera K matrix

    Returns:
        IoU value (0.0 if no overlap or too few visible corners)
    """
    try:
        # Use oriented bounding box for tighter fit on rotated meshes
        try:
            obb = mesh.bounding_box_oriented
            corners_3d = obb.vertices  # (8, 3) OBB corners
        except Exception:
            # Fallback to axis-aligned bounding box
            bounds = mesh.bounds  # (2, 3) min, max
            mins, maxs = bounds
            corners_3d = np.array([
                [mins[0], mins[1], mins[2]],
                [maxs[0], mins[1], mins[2]],
                [maxs[0], maxs[1], mins[2]],
                [mins[0], maxs[1], mins[2]],
                [mins[0], mins[1], maxs[2]],
                [maxs[0], mins[1], maxs[2]],
                [maxs[0], maxs[1], maxs[2]],
                [mins[0], maxs[1], maxs[2]],
            ])

        corners_2d, in_front = project_points_to_2d(
            corners_3d, camera_intrinsics, return_validity=True,
        )

        # Only use corners in front of the camera
        valid_corners = corners_2d[in_front]
        if len(valid_corners) < 3:
            return 0.0

        # Use np.nanmin/nanmax as safety net for any remaining NaN
        px1 = float(np.nanmin(valid_corners[:, 0]))
        py1 = float(np.nanmin(valid_corners[:, 1]))
        px2 = float(np.nanmax(valid_corners[:, 0]))
        py2 = float(np.nanmax(valid_corners[:, 1]))

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


# ─── Alignment ───────────────────────────────────────────────────

def _generate_yaw_rotations() -> List[Tuple[str, np.ndarray]]:
    """Generate the 4 rotations around the Y-axis (mesh up-axis).

    The Hunyuan3D mesh is Y-up with front facing -Z.  The yaw
    ambiguity (which direction the object faces relative to the bbox
    length axis) is captured by these 4 rotations:

      0°:   Identity — front faces -Z
      90°:  Front faces -X (rotated 90° around Y)
      180°: Front faces +Z (facing backward)
      270°: Front faces +X (rotated -90° around Y)

    These are the only physically plausible rotations because they
    preserve the up direction (Y-axis stays Y-axis).  Testing all 24
    cube rotations, as the previous implementation did, allows
    physically impossible orientations like the object lying on its
    side or upside-down.

    Returns:
        List of (name, 3×3 rotation matrix) tuples
    """
    rotations = []
    for angle_deg in [0, 90, 180, 270]:
        angle_rad = np.deg2rad(angle_deg)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        # Rotation around Y-axis:
        #   X' = c*X + s*Z
        #   Y' = Y
        #   Z' = -s*X + c*Z
        R = np.array([
            [ c, 0, s],
            [ 0, 1, 0],
            [-s, 0, c],
        ], dtype=np.float64)
        rotations.append((f"yaw{angle_deg}", R))
    return rotations


def _compute_scale_factor(
    mesh_extents: np.ndarray,
    bbox_dims_local: np.ndarray,
    R_yaw: np.ndarray,
    degenerate_threshold: float = 0.05,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Compute the optimal uniform scale factor to fit a mesh inside a 3D bbox.

    After applying R_yaw to the mesh, its extents along each local axis
    change.  The scale factor is chosen so that the mesh fits inside the
    bbox as tightly as possible on all axes.

    Strategy: Use the *maximum* scale ratio among non-degenerate axes.
    This ensures the mesh fills the bbox on its largest dimension, and
    may be smaller on other dimensions (which is fine — the mesh shape
    may not exactly match the bbox shape, but it shouldn't overflow).

    The previous implementation used "min" which made the mesh too small,
    and "median" which was a compromise.  "Max" is the correct choice
    because:
    - The bbox dimensions from WildDet3D define the object's size
    - We want the mesh to fill the bbox on its principal axis
    - Being smaller on secondary axes is better than overflowing

    Args:
        mesh_extents: (3,) mesh extents in Y-up frame [X, Y, Z]
        bbox_dims_local: (3,) bbox dimensions in local frame [L, H, W]
        R_yaw: (3, 3) yaw rotation matrix to apply to mesh first
        degenerate_threshold: Relative threshold for detecting flat axes

    Returns:
        (scale_factor, scale_ratios, degenerate_mask)
    """
    # Compute permuted extents: after R_yaw, each mesh axis may contribute
    # to multiple local axes.  For a rotation around Y:
    #   local_X = R_yaw[0,:] @ mesh_axes → affected by mesh X and Z
    #   local_Y = R_yaw[1,:] @ mesh_axes → only mesh Y
    #   local_Z = R_yaw[2,:] @ mesh_axes → affected by mesh X and Z
    extents_perm = np.abs(R_yaw) @ mesh_extents

    # Avoid division by near-zero
    extents_safe = np.maximum(extents_perm, 1e-6)
    ratios = bbox_dims_local / extents_safe

    # Identify degenerate (nearly flat) mesh dimensions
    max_extent = float(np.max(extents_perm))
    degenerate_mask = extents_perm < (max_extent * degenerate_threshold)

    # Select scale from non-degenerate axes
    valid_ratios = ratios[~degenerate_mask]
    if len(valid_ratios) == 0:
        # All axes degenerate — use the largest finite ratio as best guess
        finite_ratios = ratios[np.isfinite(ratios)]
        if len(finite_ratios) > 0:
            return float(np.max(finite_ratios)), ratios, degenerate_mask
        return 1.0, ratios, degenerate_mask

    # Use the maximum ratio so the mesh fills the bbox on its principal axis.
    # This prevents the mesh from being undersized relative to the bbox.
    scale_factor = float(np.max(valid_ratios))

    return scale_factor, ratios, degenerate_mask


def align_mesh_to_bbox(
    mesh: trimesh.Trimesh,
    bbox_center: np.ndarray,
    bbox_dims: np.ndarray,
    bbox_quat: np.ndarray,
    camera_intrinsics: Optional[np.ndarray] = None,
    bbox_2d: Optional[np.ndarray] = None,
) -> Tuple[trimesh.Trimesh, float, np.ndarray, np.ndarray]:
    """Align a canonically-posed mesh to a 3D bounding box.

    Coordinate System Mapping
    -------------------------
    Hunyuan3D mesh convention (Y-up, glTF):
      X = right, Y = up, -Z = forward (front faces -Z)

    WildDet3D bbox convention (OpenCV camera):
      bbox_dims = (W, L, H) where:
        W (width)  = Z-extent in local frame (depth direction)
        L (length) = X-extent in local frame (horizontal)
        H (height) = Y-extent in local frame (vertical, +Y = down)

    Local frame axes: X = Length, Y = Height, Z = Width

    The alignment applies the following transform chain:
      p_camera = R_quat @ R_convention @ R_yaw @ (scale * (p_mesh - center)) + bbox_center

    Where:
      - R_convention = diag(1, -1, -1) — maps Y-up→Y-down, -Z-forward→Z-forward
      - R_yaw — one of 4 rotations around the Y-axis (0°/90°/180°/270°)
      - R_quat — the WildDet3D quaternion (local frame → camera frame)
      - scale — uniform scale to fit mesh inside bbox

    When camera intrinsics and a 2D detection bbox are available, all 4
    yaw candidates are tested and the one with the highest 2D projection
    IoU is selected.  Otherwise, the candidate with the most uniform
    scale ratios (lowest coefficient of variation) is chosen.

    Args:
        mesh: The canonical trimesh.Trimesh object (Y-up, -Z forward)
        bbox_center: (3,) 3D center of the bounding box in meters
        bbox_dims: (3,) (W, L, H) dimensions in meters from WildDet3D
        bbox_quat: (4,) (qw, qx, qy, qz) quaternion for rotation
        camera_intrinsics: Optional (3,3) camera K matrix for 2D validation
        bbox_2d: Optional (4,) xyxy 2D detection bbox for validation

    Returns:
        Tuple of (aligned_mesh, scale_factor, rotation_matrix, translation)
    """
    # ── 1. Get mesh properties ────────────────────────────────────
    mesh_extents = mesh.extents  # (3,) size along each axis in Y-up frame
    mesh_center = mesh.bounds.mean(axis=0)  # center of mesh bounding box

    # ── 2. Map bbox dimensions to local frame axes ────────────────
    # WildDet3D: dims = (W, L, H)
    # Local frame: X = L, Y = H, Z = W
    W, L, H = float(bbox_dims[0]), float(bbox_dims[1]), float(bbox_dims[2])
    bbox_dims_local = np.array([L, H, W], dtype=np.float64)

    # ── 3. Build the fixed convention rotation ────────────────────
    # R_convention maps from mesh Y-up convention to bbox local frame
    # (which is in OpenCV convention: Y-down, Z-forward).
    #
    # mesh X (right)  → local X (right):  +1
    # mesh Y (up)     → local -Y (down):  -1  (Y-up to Y-down flip)
    # mesh Z (backward) → local -Z (backward): -1
    #   Equivalently: mesh -Z (forward) → local Z (forward)
    #
    # This is the OpenCV↔glTF conversion matrix.
    R_convention = np.diag([1.0, -1.0, -1.0])

    # ── 4. Decode the WildDet3D quaternion ────────────────────────
    R_quat = quaternion_to_rotation_matrix(bbox_quat)

    # ── 5. Build yaw rotation candidates ──────────────────────────
    yaw_candidates = _generate_yaw_rotations()

    # ── 6. Evaluate each candidate ────────────────────────────────
    candidates = []
    for name, R_yaw in yaw_candidates:
        scale_factor, ratios, deg_mask = _compute_scale_factor(
            mesh_extents, bbox_dims_local, R_yaw,
        )
        candidates.append((name, R_yaw, scale_factor, ratios, deg_mask))

    # ── 7. Select the best candidate ──────────────────────────────
    if camera_intrinsics is not None and bbox_2d is not None:
        # Use 2D projection IoU to select the best yaw
        best_iou = -1.0
        best_idx = 0
        for idx, (name, R_yaw, sf, ratios, deg_mask) in enumerate(candidates):
            R_full = R_quat @ R_convention @ R_yaw
            test_mesh = mesh.copy()
            test_mesh.apply_translation(-mesh_center)
            test_mesh.apply_scale(sf)
            T_rot = np.eye(4)
            T_rot[:3, :3] = R_full
            test_mesh.apply_transform(T_rot)
            test_mesh.apply_translation(bbox_center)
            iou = _compute_alignment_2d_iou(test_mesh, bbox_2d, camera_intrinsics)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        best_name, R_yaw, scale_factor, scale_ratios, best_deg = candidates[best_idx]
        logger.info(
            f"  2D validation: tested {len(candidates)} yaw candidates, "
            f"chose {best_name}, best_IoU={best_iou:.3f}"
        )
    else:
        # Without camera intrinsics, select the candidate with the most
        # uniform scale ratios (lowest coefficient of variation).  This
        # prefers orientations where the mesh fills the bbox evenly.
        best_cv = float("inf")
        best_idx = 0
        for idx, (name, R_yaw, sf, ratios, deg_mask) in enumerate(candidates):
            valid_ratios = ratios[~deg_mask]
            if len(valid_ratios) == 0:
                valid_ratios = ratios
            mean_r = np.mean(valid_ratios)
            if mean_r < 1e-6:
                continue
            cv = float(np.std(valid_ratios) / mean_r)
            if cv < best_cv:
                best_cv = cv
                best_idx = idx

        best_name, R_yaw, scale_factor, scale_ratios, best_deg = candidates[best_idx]
        logger.info(
            f"  No 2D validation: selected {best_name} by CV criterion"
        )

    # ── 8. Build the full rotation matrix ─────────────────────────
    R_full = R_quat @ R_convention @ R_yaw

    logger.info(
        f"  Scale computation: mesh_extents={mesh_extents}, "
        f"bbox_dims=(W={W:.3f}, L={L:.3f}, H={H:.3f}), "
        f"local=(L={L:.3f}, H={H:.3f}, W={W:.3f}), "
        f"scale_ratios={scale_ratios}, "
        f"scale={scale_factor:.4f}, "
        f"yaw={best_name}"
    )

    # ── 9. Apply the full transformation ──────────────────────────
    # Transform chain: center → scale → rotate → translate
    aligned_mesh = mesh.copy()
    aligned_mesh.apply_translation(-mesh_center)
    aligned_mesh.apply_scale(scale_factor)

    T_rot = np.eye(4)
    T_rot[:3, :3] = R_full
    aligned_mesh.apply_transform(T_rot)

    aligned_mesh.apply_translation(bbox_center)

    return aligned_mesh, scale_factor, R_full, bbox_center


# ─── 3D Bounding Box Corner Computation ──────────────────────────

def bbox3d_to_corners_opencv(
    bbox_center: np.ndarray,
    bbox_dims: np.ndarray,
    bbox_quat: np.ndarray,
) -> np.ndarray:
    """Compute 8 corners of a 3D bounding box in camera coordinates.

    Follows the WildDet3D/vis4d convention for AxisMode.OPENCV.
    The local frame uses: X = Length, Y = Height, Z = Width.

    Corner layout (before rotation, in local frame):
      In OpenCV (Y-down), +Y is down and +Z is forward:

        Corner 0: [ L/2,  H/2, -W/2]   right-bottom-back
        Corner 1: [ L/2,  H/2,  W/2]   right-bottom-front
        Corner 2: [-L/2,  H/2, -W/2]   left-bottom-back
        Corner 3: [-L/2,  H/2,  W/2]   left-bottom-front
        Corner 4: [ L/2, -H/2, -W/2]   right-top-back
        Corner 5: [ L/2, -H/2,  W/2]   right-top-front
        Corner 6: [-L/2, -H/2, -W/2]   left-top-back
        Corner 7: [-L/2, -H/2,  W/2]   left-top-front

    Edges:
      Bottom face (Y=+H/2, i.e. Y-down bottom): 0-1, 1-3, 3-2, 2-0
      Top face (Y=-H/2, i.e. Y-down top):       4-5, 5-7, 7-6, 6-4
      Verticals: 0-4, 1-5, 2-6, 3-7

    Args:
        bbox_center: (3,) 3D center in camera coords
        bbox_dims: (3,) (W, L, H) dimensions
        bbox_quat: (4,) (qw, qx, qy, qz) quaternion

    Returns:
        (8, 3) array of corner positions in camera coordinates
    """
    W, L, H = float(bbox_dims[0]), float(bbox_dims[1]), float(bbox_dims[2])

    # Half-dimensions in local frame (X=L, Y=H, Z=W)
    hl, hh, hw = L / 2.0, H / 2.0, W / 2.0

    # Corners in the box's local canonical frame (vis4d AxisMode.OPENCV)
    local_corners = np.array([
        [ hl,  hh, -hw],  # 0: right-bottom-back
        [ hl,  hh,  hw],  # 1: right-bottom-front
        [-hl,  hh, -hw],  # 2: left-bottom-back
        [-hl,  hh,  hw],  # 3: left-bottom-front
        [ hl, -hh, -hw],  # 4: right-top-back
        [ hl, -hh,  hw],  # 5: right-top-front
        [-hl, -hh, -hw],  # 6: left-top-back
        [-hl, -hh,  hw],  # 7: left-top-front
    ], dtype=np.float64)

    # Apply rotation (quaternion rotates from local frame → camera frame)
    R = quaternion_to_rotation_matrix(bbox_quat)
    corners = (R @ local_corners.T).T

    # Apply translation
    corners += np.asarray(bbox_center, dtype=np.float64)

    return corners


# ─── 2D Projection ───────────────────────────────────────────────

def project_points_to_2d(
    points_3d: np.ndarray,
    camera_intrinsics: np.ndarray,
    return_validity: bool = False,
) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    """Project 3D points to 2D image coordinates.

    Args:
        points_3d: (N, 3) 3D points in camera coordinates (OpenCV: Y-down, Z-forward)
        camera_intrinsics: (3, 3) camera intrinsic matrix
        return_validity: If True, also return a boolean mask indicating
                         which points are in front of the camera.

    Returns:
        (N, 2) 2D pixel coordinates (u, v).
        If return_validity=True, also returns (N,) boolean mask where True
        means the point is in front of the camera (z > 0).
    """
    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]

    # Project: u = fx * X/Z + cx, v = fy * Y/Z + cy
    z = points_3d[:, 2]
    in_front = z > 0
    z_safe = np.where(in_front, z, 1.0)

    u = fx * points_3d[:, 0] / z_safe + cx
    v = fy * points_3d[:, 1] / z_safe + cy

    # Invalidate behind-camera projections with NaN
    u[~in_front] = np.nan
    v[~in_front] = np.nan

    result = np.stack([u, v], axis=1)

    if return_validity:
        return result, in_front
    return result


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

    Uses the WildDet3D/vis4d corner ordering from bbox3d_to_corners_opencv.

    Edge connectivity (matching vis4d convention):
      Bottom face (Y=+H/2): 0-1, 1-3, 3-2, 2-0
      Top face (Y=-H/2):    4-5, 5-7, 7-6, 6-4
      Verticals:            0-4, 1-5, 2-6, 3-7

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
    corners_2d, in_front = project_points_to_2d(
        corners_3d, camera_intrinsics, return_validity=True,
    )

    # Edges matching the vis4d corner ordering
    # Bottom face (Y=+H/2 in Y-down = bottom of box): 0-1-3-2-0
    # Top face (Y=-H/2 in Y-down = top of box): 4-5-7-6-4
    # Verticals: 0-4, 1-5, 2-6, 3-7
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # bottom face
        (4, 5), (5, 7), (7, 6), (6, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    ]

    vis = image.copy()
    h, w = vis.shape[:2]
    for i, j in edges:
        # Skip edges where either endpoint is behind the camera
        if not in_front[i] or not in_front[j]:
            continue
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
    n_pts = len(object_points_3d)
    if n_pts < 4:
        return current_rotation, current_translation

    dist_coeffs = np.zeros((4, 1))

    obj_pts = object_points_3d.astype(np.float64)
    img_pts = image_points_2d.astype(np.float64)
    K = camera_intrinsics.astype(np.float64)

    # Choose PnP algorithm based on correspondences count
    if n_pts >= 6:
        flags = cv2.SOLVEPNP_ITERATIVE
    elif n_pts >= 4:
        flags = cv2.SOLVEPNP_EPNP
    else:
        return current_rotation, current_translation

    try:
        success, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K, dist_coeffs, flags=flags,
        )
    except cv2.error:
        return current_rotation, current_translation

    if not success:
        return current_rotation, current_translation

    # Convert Rodrigues vector to rotation matrix
    refined_R, _ = cv2.Rodrigues(rvec)
    refined_t = tvec.flatten()

    # Validate PnP result by computing reprojection error.
    projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist_coeffs)
    projected = projected.squeeze(1)
    reproj_err = np.sqrt(np.sum((projected - img_pts) ** 2, axis=1))
    mean_reproj_err = float(np.mean(reproj_err))

    # Threshold: 5% of the image diagonal
    image_diag = np.sqrt(K[0, 2] ** 2 + K[1, 2] ** 2) * 2.8
    max_reproj_err = 0.05 * image_diag

    if mean_reproj_err > max_reproj_err:
        logger.debug(
            f"PnP reprojection error too large ({mean_reproj_err:.1f}px > "
            f"{max_reproj_err:.1f}px); keeping current pose"
        )
        return current_rotation, current_translation

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
        return _render_mesh_trimesh(mesh, resolution, camera_intrinsics, camera_pose)

    scene = pyrender.Scene()
    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh_pyrender)

    if camera_intrinsics is not None:
        fx = camera_intrinsics[0, 0]
        fy = camera_intrinsics[1, 1]
        cx = camera_intrinsics[0, 2]
        cy = camera_intrinsics[1, 2]

        if original_image_shape is not None:
            orig_h, orig_w = original_image_shape
            uniform_scale = resolution / max(orig_h, orig_w)
            fx *= uniform_scale
            fy *= uniform_scale
            cx *= uniform_scale
            cy *= uniform_scale
    else:
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
        if v[2] > 0:
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
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    crop = image[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2].copy()

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
        contour_pts = contour.squeeze(1)

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
    triangle and returns the closest hit.

    Args:
        ray_origin: (3,) ray origin
        ray_dir: (3,) normalized ray direction
        vertices: (V, 3) mesh vertices
        faces: (F, 3) triangle face indices
        eps: epsilon to avoid self-intersection

    Returns:
        (3,) intersection point or None if no hit
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    if len(faces) == 0:
        return None

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    edge1 = v1 - v0
    edge2 = v2 - v0

    h = np.cross(ray_dir, edge2)
    a = (edge1 * h).sum(axis=1)

    valid = np.abs(a) > eps

    f = np.zeros(len(a), dtype=np.float64)
    f[valid] = 1.0 / a[valid]

    s = ray_origin - v0
    u = f * (s * h).sum(axis=1)

    valid &= (u >= 0.0) & (u <= 1.0)

    q = np.cross(s, edge1)
    v = f * (q * ray_dir).sum(axis=1)

    valid &= (v >= 0.0) & (u + v <= 1.0)

    t = f * (edge2 * q).sum(axis=1)

    valid &= (t > eps)

    if not np.any(valid):
        return None

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

    Returns:
        (N, 3) 3D points on the mesh surface
    """
    if mesh_transform is not None:
        mesh = mesh.copy()
        mesh.apply_transform(mesh_transform)

    if len(mesh.faces) == 0 or len(mesh.vertices) == 0:
        logger.warning("Mesh has no faces/vertices; returning zero 3D points")
        return np.zeros((len(keypoints_2d), 3), dtype=np.float32)

    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]

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
                    t_values = (locations - ray_origin) @ ray_dir
                    closest_idx = np.argmin(t_values)
                    points_3d.append(locations[closest_idx])
                    found = True
            except Exception:
                pass

        if not found:
            try:
                hit = _ray_mesh_intersect_moller_trumbore(
                    ray_origin, ray_dir,
                    mesh.vertices, mesh.faces,
                )
                if hit is not None:
                    points_3d.append(hit)
                    found = True
            except Exception:
                pass

        if not found:
            points_3d.append(np.array([0.0, 0.0, 0.0]))

    return np.array(points_3d, dtype=np.float64)
