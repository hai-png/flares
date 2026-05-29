"""Data types for the 3D scene reconstruction pipeline."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import trimesh


@dataclass
class DetectedObject:
    """Represents a single detected object in the scene."""
    object_id: int
    class_name: str
    confidence: float

    # 2D detection results (from RF-DETR)
    bbox_2d: np.ndarray              # (4,) xyxy in pixels
    mask_2d: np.ndarray              # (H, W) boolean mask
    crop_image: Optional[np.ndarray] = None  # (H, W, 3) uint8 cropped+masked image
    crop_mask: Optional[np.ndarray] = None   # (H, W) boolean mask for the crop
    crop_offset: Optional[Tuple[int, int]] = None  # (x1, y1) top-left of crop in original image

    # 3D bounding box (from WildDet3D)
    bbox_3d: Optional[np.ndarray] = None     # (10,) cx,cy,cz,w,l,h,qw,qx,qy,qz
    bbox_3d_center: Optional[np.ndarray] = None  # (3,) 3D center in camera coords (meters)
    bbox_3d_dims: Optional[np.ndarray] = None    # (3,) w,l,h in meters
    bbox_3d_quat: Optional[np.ndarray] = None    # (4,) qw,qx,qy,qz quaternion
    score_3d: Optional[float] = None

    # 3D model (from Hunyuan3D)
    mesh: Optional[trimesh.Trimesh] = None
    mesh_path: Optional[str] = None

    # Alignment results
    aligned_mesh: Optional[trimesh.Trimesh] = None
    scale_factor: Optional[float] = None
    initial_rotation: Optional[np.ndarray] = None  # (3,3) rotation matrix
    initial_translation: Optional[np.ndarray] = None  # (3,) translation

    # Refinement results (from MARCO)
    refined_rotation: Optional[np.ndarray] = None   # (3,3) rotation matrix
    refined_translation: Optional[np.ndarray] = None  # (3,) translation
    correspondence_points_src: Optional[np.ndarray] = None  # (N,2) source keypoints
    correspondence_points_tgt: Optional[np.ndarray] = None  # (N,2) target keypoints


@dataclass
class ObjectReconstructionResult:
    """Result for a single reconstructed object."""
    object_id: int
    class_name: str
    mesh: trimesh.Trimesh
    transform: np.ndarray             # (4,4) homogeneous transform
    bbox_3d: Optional[np.ndarray] = None  # (10,) 3D bounding box params, or None
    confidence: float = 0.0


@dataclass
class SceneReconstructionResult:
    """Complete result of the 3D scene reconstruction pipeline."""
    objects: List[ObjectReconstructionResult] = field(default_factory=list)
    camera_intrinsics: Optional[np.ndarray] = None  # (3,3)
    image_shape: Optional[Tuple[int, int]] = None   # (H, W)

    def get_scene_mesh(self) -> trimesh.Scene:
        """Combine all objects into a single trimesh.Scene."""
        scene = trimesh.Scene()
        for obj in self.objects:
            mesh = obj.mesh.copy()
            mesh.apply_transform(obj.transform)
            mesh.metadata["class_name"] = obj.class_name
            mesh.metadata["object_id"] = obj.object_id
            scene.add_geometry(mesh, node_name=f"{obj.class_name}_{obj.object_id}")
        return scene

    def export_scene(self, path: str):
        """Export the combined scene to file."""
        scene = self.get_scene_mesh()
        scene.export(path)
