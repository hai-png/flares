"""WildDet3D Module: Monocular 3D Bounding Box Estimation.

This module wraps WildDet3D to lift 2D bounding box detections to
3D bounding boxes in camera coordinate space. It uses SAM3 + LingBot-Depth
for depth estimation and a 3D regression head for full 3D box prediction.

The 3D boxes are output in OpenCV camera coordinates:
  X-right, Y-down, Z-forward
  Format: (cx, cy, cz, w, l, h, qw, qx, qy, qz) in meters
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..utils.data_types import DetectedObject

logger = logging.getLogger(__name__)


class WildDet3DEstimator:
    """WildDet3D based 3D bounding box estimator.

    Takes 2D bounding box detections and estimates 3D bounding boxes
    in camera coordinate space using the geometric prompt mode.

    Example:
        >>> estimator = WildDet3DEstimator(checkpoint="ckpt/wilddet3d_alldata_all_prompt_v1.0.pt")
        >>> estimator.load_model()
        >>> objects_3d = estimator.estimate_3d(image, objects_2d, intrinsics)
    """

    def __init__(
        self,
        checkpoint: str = "ckpt/wilddet3d_alldata_all_prompt_v1.0.pt",
        score_threshold: float = 0.3,
        score_3d_threshold: float = 0.1,
        use_predicted_intrinsics: bool = True,
        canonical_rotation: bool = True,
        device: str = "cuda",
    ):
        """Initialize WildDet3D estimator.

        Args:
            checkpoint: Path to the WildDet3D checkpoint file
            score_threshold: Combined 2D×3D score threshold
            score_3d_threshold: Standalone 3D confidence threshold
            use_predicted_intrinsics: Use predicted camera intrinsics for
                                      in-the-wild images without known K
            canonical_rotation: Normalize dimensions W≤L, yaw∈[0,π)
            device: Device to run on
        """
        self.checkpoint = checkpoint
        self.score_threshold = score_threshold
        self.score_3d_threshold = score_3d_threshold
        self.use_predicted_intrinsics = use_predicted_intrinsics
        self.canonical_rotation = canonical_rotation
        self.device = device
        self.model = None

    def load_model(self):
        """Load the WildDet3D model and weights.

        Requires the WildDet3D package to be installed and the
        checkpoint file to be available. Downloads from HuggingFace
        if not present.
        """
        # Add WildDet3D repo to path if needed
        repo_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "repos", "WildDet3D"
        )
        if os.path.exists(repo_path) and repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        # Check for checkpoint
        if not os.path.exists(self.checkpoint):
            logger.info(
                f"Checkpoint not found at {self.checkpoint}. "
                "Downloading from HuggingFace..."
            )
            os.makedirs(os.path.dirname(self.checkpoint) or ".", exist_ok=True)
            os.system(
                f"huggingface-cli download allenai/WildDet3D "
                f"wilddet3d_alldata_all_prompt_v1.0.pt "
                f"--local-dir {os.path.dirname(self.checkpoint) or '.'}"
            )

        try:
            from wilddet3d import build_model
        except ImportError:
            raise ImportError(
                "WildDet3D is not installed. Please install it:\n"
                "  cd repos/WildDet3D && pip install -e .\n"
                "  pip install vis4d==1.0.0\n"
                "  pip install git+https://github.com/SysCV/vis4d_cuda_ops.git --no-build-isolation"
            )

        logger.info("Loading WildDet3D model...")
        self.model = build_model(
            checkpoint=self.checkpoint,
            score_threshold=self.score_threshold,
            score_3d_threshold=self.score_3d_threshold,
            skip_pretrained=True,
            canonical_rotation=self.canonical_rotation,
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info("WildDet3D model loaded successfully")

    def estimate_3d(
        self,
        image: np.ndarray,
        objects: List[DetectedObject],
        intrinsics: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
    ) -> List[DetectedObject]:
        """Estimate 3D bounding boxes for detected objects.

        Uses the geometric prompt mode (one-to-one): each 2D box is
        lifted to a 3D bounding box in camera coordinates.

        Args:
            image: (H, W, 3) uint8 RGB image
            objects: List of DetectedObject with 2D bboxes populated
            intrinsics: Optional (3,3) camera intrinsic matrix.
                        If None, uses predicted intrinsics.
            class_names: Optional class name list for prompt_text

        Returns:
            Updated list of DetectedObject with 3D bbox fields populated
        """
        if self.model is None:
            self.load_model()

        if not objects:
            logger.warning("No objects to estimate 3D boxes for")
            return objects

        from wilddet3d import preprocess

        # Preprocess the image
        image_float = image.astype(np.float32)
        data = preprocess(
            image_float,
            intrinsics if intrinsics is not None else None,
        )

        # Process each object with geometric prompt
        for obj in objects:
            # Use the 2D bounding box as geometric prompt
            x1, y1, x2, y2 = obj.bbox_2d.tolist()

            # Build prompt text
            prompt_text = "geometric"
            if class_names and obj.class_name in class_names:
                prompt_text = f"geometric: {obj.class_name}"

            with torch.no_grad():
                try:
                    results = self.model(
                        images=data["images"].to(self.device),
                        intrinsics=data["intrinsics"].to(self.device)[None],
                        input_hw=[data["input_hw"]],
                        original_hw=[data["original_hw"]],
                        padding=[data["padding"]],
                        input_boxes=[[x1, y1, x2, y2]],
                        prompt_text=prompt_text,
                        use_predicted_intrinsics=self.use_predicted_intrinsics,
                    )

                    if self.use_predicted_intrinsics:
                        boxes, boxes3d, scores, scores_2d, scores_3d, class_ids, depth_maps = results[:7]
                    else:
                        boxes, boxes3d, scores, scores_2d, scores_3d, class_ids, depth_maps = results[:7]

                    # Extract 3D box for this object
                    if len(boxes3d) > 0 and len(boxes3d[0]) > 0:
                        bbox_3d = boxes3d[0][0].cpu().numpy()  # (10,)
                        obj.bbox_3d = bbox_3d
                        obj.bbox_3d_center = bbox_3d[:3]    # cx, cy, cz
                        obj.bbox_3d_dims = bbox_3d[3:6]     # w, l, h
                        obj.bbox_3d_quat = bbox_3d[6:10]    # qw, qx, qy, qz

                        if len(scores_3d) > 0 and len(scores_3d[0]) > 0:
                            obj.score_3d = float(scores_3d[0][0].cpu())

                        logger.info(
                            f"Object {obj.object_id} ({obj.class_name}): "
                            f"3D center={obj.bbox_3d_center}, "
                            f"dims={obj.bbox_3d_dims}, "
                            f"3D score={obj.score_3d:.3f}"
                        )
                    else:
                        logger.warning(
                            f"No 3D box returned for object {obj.object_id} ({obj.class_name})"
                        )

                except Exception as e:
                    logger.error(
                        f"WildDet3D inference failed for object {obj.object_id}: {e}"
                    )

        # Also store the camera intrinsics used
        if intrinsics is not None:
            self._last_intrinsics = intrinsics
        elif "original_intrinsics" in data:
            self._last_intrinsics = data["original_intrinsics"].numpy()
        else:
            self._last_intrinsics = data["intrinsics"].numpy()

        return objects

    def get_intrinsics(self) -> Optional[np.ndarray]:
        """Get the camera intrinsics used in the last estimation.

        Returns:
            (3,3) camera intrinsic matrix or None if not available
        """
        return getattr(self, "_last_intrinsics", None)
