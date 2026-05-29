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
    unproject_depth_to_3d,
)

logger = logging.getLogger(__name__)


class MARCORefiner:
    """MARCO-based pose refinement using semantic correspondence.

    Given a cropped object image and a rendered image of its 3D model,
    uses MARCO to find semantic correspondences between them, then
    uses these correspondences to refine the 6-DoF pose.

    The refinement flow for each object:
    1. Sample 2D keypoints from the object's crop mask
    2. Render the aligned 3D mesh from the camera viewpoint
    3. Use MARCO to find where crop keypoints appear in the render
    4. Use the depth buffer to get 3D mesh points for the target keypoints
    5. Convert 3D points from camera space to object frame
    6. Use solvePnP with 3D object points + 2D source observations to refine pose

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
        repo_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "repos", "MARCO"
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

        MARCO's preprocess_data resizes the longest side to inference_res
        and bottom-right pads to a square.  Source keypoints are scaled from
        original pixels into the resized (top-left aligned) space by
        preprocess_data.  The model's predict_from_logits outputs target
        keypoints in the *padded square* pixel space.  Because the image
        content is top-left aligned (bottom-right padding), the padded-space
        coordinates are the same as resized-space coordinates, so converting
        back only requires dividing by the resize scale factor.

        Args:
            source_image: (H, W, 3) uint8 source image (cropped object)
            target_image: (H, W, 3) uint8 target image (rendered mesh)
            source_keypoints: (N, 2) source keypoint coordinates (x, y)
                              in source image pixel space

        Returns:
            Tuple of (source_keypoints, predicted_target_keypoints)
            Both in original image pixel coordinates (x, y)
        """
        if self.model is None:
            self.load_model()

        import tempfile
        from PIL import Image as PILImage

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1, \
             tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2:
            src_path = f1.name
            tgt_path = f2.name
            PILImage.fromarray(source_image).save(src_path)
            PILImage.fromarray(target_image).save(tgt_path)

        try:
            src_kps_list = source_keypoints.tolist()

            inputs = self.model.preprocess_data(
                src_path, tgt_path, src_kps_list, device=self.device
            )

            with torch.no_grad():
                pred_kps = self.model(**inputs)

            pred_kps_np = pred_kps.cpu().numpy()[0]  # (N, 2) in padded-square space

            # MARCO pads bottom-right (top-left aligned), so padded-space
            # coordinates equal resized-space coordinates.  The scale factor
            # from original → resized is the same for both X and Y because
            # the resize preserves aspect ratio:
            #   scale = inference_res / max(orig_w, orig_h)
            # To go back: coord_original = coord_padded / scale
            tgt_h, tgt_w = target_image.shape[:2]
            scale = self.inference_res / max(tgt_h, tgt_w)

            pred_original = np.zeros_like(pred_kps_np)
            pred_original[:, 0] = pred_kps_np[:, 0] / scale
            pred_original[:, 1] = pred_kps_np[:, 1] / scale

            return source_keypoints, pred_original

        finally:
            os.unlink(src_path)
            os.unlink(tgt_path)

    def refine_single_object(
        self,
        obj: DetectedObject,
        camera_intrinsics: np.ndarray,
        render_resolution: int = 512,
        original_image_shape: Optional[Tuple[int, int]] = None,
    ) -> DetectedObject:
        """Refine the pose of a single object using MARCO correspondences.

        Steps:
        1. Sample keypoints on the cropped object image (source)
        2. Render the aligned 3D model from the camera viewpoint (target)
        3. Use MARCO to find where source keypoints appear in the render
        4. Get 3D mesh points corresponding to the target keypoints
           (using depth buffer or ray casting)
        5. Convert 3D points from camera space to object frame
        6. Use solvePnP with 3D object points and 2D source observations
           to refine the pose

        Args:
            obj: DetectedObject with aligned mesh and initial pose
            camera_intrinsics: (3,3) camera intrinsic matrix (for the full image)
            render_resolution: Resolution for rendering the mesh
            original_image_shape: (H, W) of the original image

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

        # Simplify mesh for ray casting if very dense
        MARCO_MAX_FACES = 50000
        work_mesh = obj.mesh
        if work_mesh is not None and len(work_mesh.faces) > MARCO_MAX_FACES:
            try:
                work_mesh = work_mesh.simplify_quadric_decimation(face_count=MARCO_MAX_FACES)
                logger.info(
                    f"Object {obj.object_id}: decimated mesh from "
                    f"{len(obj.mesh.faces)} to {len(work_mesh.faces)} faces for MARCO"
                )
            except Exception as e:
                logger.debug(f"Mesh decimation failed ({e}); using original mesh")
                work_mesh = obj.mesh

        # Iterative refinement
        for iteration in range(self.refinement_iterations):
            logger.info(
                f"Object {obj.object_id} refinement iteration {iteration + 1}/"
                f"{self.refinement_iterations}"
            )

            # 2. Render the mesh with current pose
            T_scale = np.eye(4)
            T_scale[0, 0] = T_scale[1, 1] = T_scale[2, 2] = scale_factor
            T_rot = np.eye(4)
            T_rot[:3, :3] = current_R
            T_trans = np.eye(4)
            T_trans[:3, 3] = current_t
            current_transform = T_trans @ T_rot @ T_scale

            try:
                rendered_image, depth_map = render_mesh_for_marco(
                    work_mesh,
                    current_transform,
                    camera_intrinsics,
                    resolution=render_resolution,
                    original_image_shape=original_image_shape,
                )
            except Exception as e:
                logger.warning(f"Rendering failed for object {obj.object_id}: {e}")
                continue

            # Check if rendering produced a visible result
            has_valid_depth = depth_map is not None and np.any(depth_map > 0)
            if not has_valid_depth:
                logger.info(
                    f"Object {obj.object_id}: depth buffer empty at iteration "
                    f"{iteration + 1} - skipping"
                )
                continue

            # Resize source image to match render resolution for MARCO
            src_image = cv2.resize(
                obj.crop_image, (render_resolution, render_resolution)
            )

            # Scale source keypoints to render resolution
            crop_h, crop_w = obj.crop_image.shape[:2]
            h_ratio = render_resolution / crop_h
            w_ratio = render_resolution / crop_w
            src_kps_scaled = src_keypoints.copy()
            src_kps_scaled[:, 0] *= w_ratio
            src_kps_scaled[:, 1] *= h_ratio

            # 3. Find correspondences using MARCO
            try:
                src_kps_out, tgt_kps = self.find_correspondences(
                    src_image, rendered_image, src_kps_scaled
                )
            except Exception as e:
                logger.warning(
                    f"MARCO correspondence failed for object {obj.object_id}: {e}"
                )
                continue

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
                continue

            # Store correspondence points
            obj.correspondence_points_src = src_kps_valid
            obj.correspondence_points_tgt = tgt_kps_valid

            # 4. Get 3D points on the mesh corresponding to target keypoints
            #
            # The target keypoints are where MARCO says the source features
            # appear in the rendered image. We need to find the 3D mesh points
            # at those 2D locations.
            #
            # Approach: Use the depth buffer from rendering to unproject
            # the 2D target keypoints to 3D camera-space points.
            #
            # Build the scaled camera intrinsics for the render resolution
            K_scaled = camera_intrinsics.copy().astype(np.float64)
            if original_image_shape is not None:
                orig_h, orig_w = original_image_shape
                K_scaled[0, 0] *= render_resolution / orig_w
                K_scaled[0, 2] *= render_resolution / orig_w
                K_scaled[1, 1] *= render_resolution / orig_h
                K_scaled[1, 2] *= render_resolution / orig_h

            try:
                # Use depth buffer to get 3D points in camera space
                points_3d_cam = unproject_depth_to_3d(
                    tgt_kps_valid, depth_map, K_scaled,
                )

                # Filter out zero-valued points (failed extractions)
                valid_3d = np.any(points_3d_cam != 0, axis=1)

                # If depth buffer gave too few points, try ray casting
                if valid_3d.sum() < 6:
                    logger.info(
                        f"Object {obj.object_id}: depth unprojection yielded "
                        f"only {valid_3d.sum()} valid points, trying ray casting"
                    )
                    points_3d_cam = get_mesh_3d_points_for_2d_keypoints(
                        work_mesh,
                        tgt_kps_valid,
                        K_scaled,
                        mesh_transform=current_transform,
                    )
                    valid_3d = np.any(points_3d_cam != 0, axis=1)

                if valid_3d.sum() < 4:
                    logger.warning(
                        f"Object {obj.object_id}: too few valid 3D points "
                        f"({valid_3d.sum()}); skipping this iteration"
                    )
                    continue

                # Keep only valid points
                tgt_kps_valid = tgt_kps_valid[valid_3d]
                src_kps_valid = src_kps_valid[valid_3d]
                points_3d_cam = points_3d_cam[valid_3d]

            except Exception as e:
                logger.warning(
                    f"3D point extraction failed for object {obj.object_id}: {e}"
                )
                continue

            # 5. Convert 3D points from camera space to object frame
            #
            # The 3D points are in camera space. We need them in the object's
            # local coordinate frame for solvePnP.
            #
            # Current transform: p_cam = R @ (scale * p_obj) + t
            # Inverse: p_obj = R^T @ (p_cam - t) / scale
            points_3d_obj = (current_R.T @ (points_3d_cam - current_t).T).T / scale_factor

            # 6. Refine pose using solvePnP
            #
            # solvePnP finds R, t such that:
            #   p_image = K @ (R @ p_object + t)
            #
            # We use:
            # - object_points: p_obj * scale (scaled object-frame points)
            # - image_points: src_kps in full-image coordinates
            #   (the actual 2D observations of the object in the real image)
            # - camera_intrinsics: K for the full original image
            #
            # Convert source keypoints from crop coordinates to full-image
            # coordinates so they match the full-image camera intrinsics.
            src_kps_full = src_kps_valid.copy()
            # Undo the render_resolution scaling first
            src_kps_full[:, 0] /= w_ratio
            src_kps_full[:, 1] /= h_ratio
            # Now keypoints are in crop_image pixel coordinates.
            # If the crop was resized (crop_scale != 1.0), undo that too
            # to get back to the original (un-resized) crop coordinates
            # which match the crop_offset.
            crop_scale = obj.crop_scale if obj.crop_scale is not None else 1.0
            if crop_scale != 1.0:
                src_kps_full[:, 0] /= crop_scale
                src_kps_full[:, 1] /= crop_scale
            # Add crop offset to get full-image coordinates
            if obj.crop_offset is not None:
                src_kps_full[:, 0] += obj.crop_offset[0]
                src_kps_full[:, 1] += obj.crop_offset[1]
            else:
                # Fallback: estimate offset from 2D bbox
                x1, y1 = obj.bbox_2d[:2].astype(float)
                src_kps_full[:, 0] += x1
                src_kps_full[:, 1] += y1

            refined_R, refined_t = refine_pose_with_correspondences(
                object_points_3d=points_3d_obj * scale_factor,
                image_points_2d=src_kps_full,
                camera_intrinsics=camera_intrinsics.astype(np.float64),
                current_rotation=current_R,
                current_translation=current_t,
            )

            # Validate refined pose - reject implausibly large changes
            rot_diff = np.linalg.norm(refined_R - current_R)
            trans_diff = np.linalg.norm(refined_t - current_t)
            if rot_diff > 1.5 or trans_diff > 3.0:
                logger.warning(
                    f"Object {obj.object_id}: rejecting large pose change "
                    f"(rot_diff={rot_diff:.3f}, trans_diff={trans_diff:.3f}). "
                    f"Keeping previous pose."
                )
            else:
                current_R = refined_R
                current_t = refined_t

        # Update the object with refined pose
        obj.refined_rotation = current_R
        obj.refined_translation = current_t

        # Apply refined transform to get the final aligned mesh
        obj.aligned_mesh = obj.mesh.copy()
        mesh_center = obj.aligned_mesh.bounds.mean(axis=0)
        obj.aligned_mesh.apply_translation(-mesh_center)
        obj.aligned_mesh.apply_scale(scale_factor)
        T_rot = np.eye(4)
        T_rot[:3, :3] = current_R
        obj.aligned_mesh.apply_transform(T_rot)
        obj.aligned_mesh.apply_translation(current_t)

        init_R = obj.initial_rotation if obj.initial_rotation is not None else np.eye(3)
        init_t = obj.initial_translation if obj.initial_translation is not None else np.zeros(3)
        logger.info(
            f"Object {obj.object_id} pose refined. "
            f"Rotation diff: {np.linalg.norm(current_R - init_R):.4f}, "
            f"Translation diff: {np.linalg.norm(current_t - init_t):.4f}"
        )

        return obj

    def refine_poses(
        self,
        objects: List[DetectedObject],
        camera_intrinsics: np.ndarray,
        render_resolution: int = 512,
        original_image_shape: Optional[Tuple[int, int]] = None,
    ) -> List[DetectedObject]:
        """Refine poses for all objects using MARCO correspondences.

        Args:
            objects: List of DetectedObject with aligned meshes
            camera_intrinsics: (3,3) camera intrinsic matrix
            render_resolution: Resolution for rendering meshes
            original_image_shape: (H, W) of the original image

        Returns:
            Updated list of DetectedObject with refined poses
        """
        if self.model is None:
            self.load_model()

        refined_objects = []
        for obj in objects:
            if obj.aligned_mesh is not None:
                obj = self.refine_single_object(
                    obj, camera_intrinsics, render_resolution,
                    original_image_shape=original_image_shape,
                )
            refined_objects.append(obj)

        n_refined = sum(1 for o in refined_objects if o.refined_rotation is not None)
        logger.info(
            f"MARCO pose refinement complete: {n_refined}/{len(objects)} objects refined"
        )

        return refined_objects
