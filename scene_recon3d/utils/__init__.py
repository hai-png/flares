"""Utility modules for 3D scene reconstruction pipeline."""

# Auto-setup repo paths on import (ensures WildDet3D, Hunyuan3D imports work)
from . import setup_paths  # noqa: F401 — side-effect import

from .geometry import (
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
    scale_mesh,
    transform_mesh,
    align_mesh_to_bbox,
    refine_pose_with_correspondences,
    render_mesh,
    crop_image_with_mask,
    sample_keypoints_from_mask,
)
from .data_types import (
    DetectedObject,
    SceneReconstructionResult,
    ObjectReconstructionResult,
)
