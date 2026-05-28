"""MARCO Module: Semantic Correspondence for Pose Refinement.

This module wraps MARCO (Navigating the Unseen Space of Semantic Correspondence)
to establish dense/sparse point correspondences between the cropped object image
and the rendered image of the generated 3D model. These correspondences are
then used to refine the 6-DoF pose of each object in the scene.

MARCO uses a DINOv2 backbone with AdaptFormer adapters and an upsampling head
to predict where semantic keypoints from a source image appear in a target image.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from ..utils.data_types import DetectedObject
from ..utils.geometry import (
    quaternion_to_rotation_matrix,
    refine_pose_with_correspondences,
    render_mesh_for_marco,
    sample_keypoints_from_mask,
    get_mesh_3d_points_for_2d_keypoints,
)

logger = logging.getLogger(__name__)


class MARCORefiner:
    """MARCO-based pose refinement using semantic correspondence.

    Given a cropped object image and a rendered image of its 3D model,
    uses MARCO to find semantic correspondences between them, then
    uses these correspondences to refine the 6-DoF pose.

    Example:
        >>> refiner = MARCORefiner()
        >>> refiner.load_model()
        >>> refined_objects = refiner.refine_poses(objects, camera_intrinsics)
    """

    def __init__(
        self,
        checkpoint: str = "marco_release.pth",
        use_torch_hub: bool = True,
        inference_res: int = 840,
        num_keypoints_per_object: int = 20,
        keypoint_sampling_method: str = "uniform",
        refinement_iterations: int = 3,
        device: str = "cuda",
    ):
        """Initialize MARCO refiner.

        Args:
            checkpoint: Path to MARCO checkpoint file
            use_torch_hub: Use torch.hub to auto-download the model
            inference_res: Longest-side resolution for MARCO inference
            num_keypoints_per_object: Number of keypoints to sample per object
            keypoint_sampling_method: How to sample keypoints ('uniform', 'contour', 'random')
            refinement_iterations: Number of MARCO-based refinement iterations
            device: Device to run on
        """
        self.checkpoint = checkpoint
        self.use_torch_hub = use_torch_hub
        self.inference_res = inference_res
        self.num_keypoints = num_keypoints_per_object
        self.sampling_method = keypoint_sampling_method
        self.refinement_iterations = refinement_iterations
        self.device = device
        self.model = None

    def load_model(self):
        """Load the MARCO model and weights.

        Uses torch.hub for auto-download by default, or loads from
        a local checkpoint file.
        """
        # Add MARCO repo to path if needed
        repo_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "repos", "MARCO"
        )
        if os.path.exists(repo_path) and repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        logger.info("Loading MARCO model...")

        if self.use_torch_hub:
            try:
                self.model = torch.hub.load(
                    "visinf/MARCO",
                    "marco",
                    pretrained=True,
                    trust_repo=True,
                    device=self.device,
                )
                self.model.eval()
                logger.info("MARCO model loaded via torch.hub")
                return
            except Exception as e:
                logger.warning(
                    f"torch.hub loading failed: {e}. "
                    "Trying manual loading..."
                )

        # Manual loading
        try:
            from models import build_marco

            self.model = build_marco()
            ckpt = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ckpt["model"], strict=False)
            self.model = self.model.to(self.device).eval()
            logger.info("MARCO model loaded from local checkpoint")

        except ImportError:
            raise ImportError(
                "MARCO is not installed. Please install it:\n"
                "  cd repos/MARCO && pip install -r requirements.txt\n"
                "  Or use use_torch_hub=True to auto-download"
            )

    def find_correspondences(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        source_keypoints: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find semantic correspondences between two images using MARCO.

        Given a source image with labeled keypoints and a target image,
        predicts where those same semantic keypoints appear in the target.

        Args:
            source_image: (H, W, 3) uint8 source image (cropped object)
            target_image: (H, W, 3) uint8 target image (rendered mesh)
            source_keypoints: (N, 2) source keypoint coordinates (x, y)
                              in original image pixel space

        Returns:
            Tuple of (source_keypoints, predicted_target_keypoints)
            Both in original image pixel coordinates (x, y)
        """
        if self.model is None:
            self.load_model()

        # Save images temporarily for MARCO's preprocess_data
        import tempfile
        from PIL import Image as PILImage

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1, \
             tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2:
            src_path = f1.name
            tgt_path = f2.name
            PILImage.fromarray(source_image).save(src_path)
            PILImage.fromarray(target_image).save(tgt_path)

        try:
            # Convert keypoints to list format for MARCO
            src_kps_list = source_keypoints.tolist()

            # Prepare input using MARCO's preprocessing
            inputs = self.model.preprocess_data(
                src_path, tgt_path, src_kps_list, device=self.device
            )

            # Predict target keypoint positions
            with torch.no_grad():
                pred_kps = self.model(**inputs)

            # pred_kps is in resized+padded space; convert back to original
            pred_kps_np = pred_kps.cpu().numpy()[0]  # (N, 2)

            # We need to convert from MARCO's resized+padded space back to
            # original target image space.
            # MARCO's preprocess_data pads images to square with the longest side
            # at inference_res. We need to reverse this transform.
            tgt_h, tgt_w = target_image.shape[:2]
            scale = self.inference_res / max(tgt_h, tgt_w)
            pad_w = self.inference_res - int(tgt_w * scale)
            pad_h = self.inference_res - int(tgt_h * scale)

            # Convert from padded space to original space
            pred_original = np.zeros_like(pred_kps_np)
            pred_original[:, 0] = (pred_kps_np[:, 0] - pad_w / 2) / scale
            pred_original[:, 1] = (pred_kps_np[:, 1] - pad_h / 2) / scale

            return source_keypoints, pred_original

        finally:
            os.unlink(src_path)
            os.unlink(tgt_path)

    def refine_single_object(
        self,
        obj: DetectedObject,
        camera_intrinsics: np.ndarray,
        render_resolution: int = 512,
    ) -> DetectedObject:
        """Refine the pose of a single object using MARCO correspondences.

        Steps:
        1. Sample keypoints on the cropped object image (source)
        2. Render the aligned 3D model from the same viewpoint (target)
        3. Use MARCO to find where source keypoints appear in the render
        4. Use 2D-3D correspondences to refine the pose via PnP

        Args:
            obj: DetectedObject with aligned mesh and initial pose
            camera_intrinsics: (3,3) camera intrinsic matrix
            render_resolution: Resolution for rendering the mesh

        Returns:
            Updated DetectedObject with refined pose
        """
        if self.model is None:
            self.load_model()

        if obj.crop_image is None or obj.aligned_mesh is None:
            logger.warning(
                f"Object {obj.object_id} missing crop_image or aligned_mesh; "
                "skipping refinement"
            )
            return obj

        # 1. Sample keypoints from the object mask
        crop_mask = obj.crop_mask if obj.crop_mask is not None else np.ones(
            obj.crop_image.shape[:2], dtype=bool
        )
        src_keypoints = sample_keypoints_from_mask(
            crop_mask,
            num_keypoints=self.num_keypoints,
            method=self.sampling_method,
        )

        if len(src_keypoints) < 4:
            logger.warning(
                f"Object {obj.object_id}: too few keypoints ({len(src_keypoints)}); "
                "skipping refinement"
            )
            return obj

        # Build current transform
        current_R = obj.initial_rotation if obj.initial_rotation is not None else np.eye(3)
        current_t = obj.initial_translation if obj.initial_translation is not None else np.zeros(3)
        scale_factor = obj.scale_factor if obj.scale_factor is not None else 1.0

        # Iterative refinement
        for iteration in range(self.refinement_iterations):
            logger.info(
                f"Object {obj.object_id} refinement iteration {iteration + 1}/"
                f"{self.refinement_iterations}"
            )

            # 2. Render the mesh with current pose
            current_transform = np.eye(4)
            current_transform[:3, :3] = current_R
            current_transform[:3, 3] = current_t

            try:
                rendered_image = render_mesh_for_marco(
                    obj.mesh,
                    current_transform,
                    camera_intrinsics,
                    resolution=render_resolution,
                )
            except Exception as e:
                logger.warning(f"Rendering failed for object {obj.object_id}: {e}")
                break

            # Resize source image to match render resolution for MARCO
            src_image = cv2.resize(
                obj.crop_image, (render_resolution, render_resolution)
            )

            # Scale source keypoints to render resolution
            if obj.crop_image is not None:
                h_ratio = render_resolution / obj.crop_image.shape[0]
                w_ratio = render_resolution / obj.crop_image.shape[1]
                src_kps_scaled = src_keypoints.copy()
                src_kps_scaled[:, 0] *= w_ratio
                src_kps_scaled[:, 1] *= h_ratio
            else:
                src_kps_scaled = src_keypoints

            # 3. Find correspondences using MARCO
            try:
                src_kps_out, tgt_kps = self.find_correspondences(
                    src_image, rendered_image, src_kps_scaled
                )
            except Exception as e:
                logger.warning(
                    f"MARCO correspondence failed for object {obj.object_id}: {e}"
                )
                break

            # Filter valid correspondences (within image bounds)
            valid = (
                (tgt_kps[:, 0] >= 0) & (tgt_kps[:, 0] < render_resolution) &
                (tgt_kps[:, 1] >= 0) & (tgt_kps[:, 1] < render_resolution)
            )
            src_kps_valid = src_kps_out[valid]
            tgt_kps_valid = tgt_kps[valid]

            if len(src_kps_valid) < 4:
                logger.warning(
                    f"Too few valid correspondences ({len(src_kps_valid)}) "
                    f"for object {obj.object_id}"
                )
                break

            # Store correspondence points
            obj.correspondence_points_src = src_kps_valid
            obj.correspondence_points_tgt = tgt_kps_valid

            # 4. Get 3D points on the mesh for these keypoints
            # Map source keypoints back to mesh surface
            src_kps_original = src_kps_valid.copy()
            if obj.crop_image is not None:
                src_kps_original[:, 0] /= w_ratio
                src_kps_original[:, 1] /= h_ratio

            try:
                # Get 3D points on the canonical mesh
                # We use the crop bbox to establish a mapping from
                # crop image coords to mesh coords
                points_3d = get_mesh_3d_points_for_2d_keypoints(
                    obj.mesh,
                    src_kps_original,
                    camera_intrinsics,
                    mesh_transform=None,  # canonical mesh
                )
            except Exception as e:
                logger.warning(
                    f"3D point extraction failed for object {obj.object_id}: {e}"
                )
                break

            # 5. Refine pose using PnP with correspondences
            refined_R, refined_t = refine_pose_with_correspondences(
                current_rotation=current_R,
                current_translation=current_t,
                src_points_2d=src_kps_valid,
                tgt_points_2d=tgt_kps_valid,
                src_points_3d=points_3d,
                camera_intrinsics=camera_intrinsics,
                scale_factor=scale_factor,
                mesh=obj.mesh,
            )

            current_R = refined_R
            current_t = refined_t

        # Update the object with refined pose
        obj.refined_rotation = current_R
        obj.refined_translation = current_t

        # Apply refined transform to get the final aligned mesh
        obj.aligned_mesh = obj.mesh.copy()
        # First center at origin
        mesh_center = obj.aligned_mesh.bounds.mean(axis=0)
        obj.aligned_mesh.apply_translation(-mesh_center)
        # Scale
        obj.aligned_mesh.apply_scale(scale_factor)
        # Apply refined rotation
        T_rot = np.eye(4)
        T_rot[:3, :3] = current_R
        obj.aligned_mesh.apply_transform(T_rot)
        # Apply refined translation
        obj.aligned_mesh.apply_translation(current_t)

        logger.info(
            f"Object {obj.object_id} pose refined. "
            f"Rotation diff: {np.linalg.norm(current_R - (obj.initial_rotation or np.eye(3))):.4f}, "
            f"Translation diff: {np.linalg.norm(current_t - (obj.initial_translation or np.zeros(3))):.4f}"
        )

        return obj

    def refine_poses(
        self,
        objects: List[DetectedObject],
        camera_intrinsics: np.ndarray,
        render_resolution: int = 512,
    ) -> List[DetectedObject]:
        """Refine poses for all objects using MARCO correspondences.

        Args:
            objects: List of DetectedObject with aligned meshes
            camera_intrinsics: (3,3) camera intrinsic matrix
            render_resolution: Resolution for rendering meshes

        Returns:
            Updated list of DetectedObject with refined poses
        """
        if self.model is None:
            self.load_model()

        refined_objects = []
        for obj in objects:
            if obj.aligned_mesh is not None:
                obj = self.refine_single_object(
                    obj, camera_intrinsics, render_resolution
                )
            refined_objects.append(obj)

        n_refined = sum(1 for o in refined_objects if o.refined_rotation is not None)
        logger.info(
            f"MARCO pose refinement complete: {n_refined}/{len(objects)} objects refined"
        )

        return refined_objects
