"""RF-DETR Module: 2D Object Detection + Instance Segmentation.

This module wraps RF-DETR to detect objects in a scene image and produce
2D bounding boxes and instance segmentation masks.

RF-DETR (Roboflow Detection Transformer) uses a DINOv2 ViT backbone
with a transformer decoder for real-time detection and segmentation.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..utils.data_types import DetectedObject

logger = logging.getLogger(__name__)

# Mapping from config variant name to RF-DETR class name
VARIANT_MAP = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large": "RFDETRLarge",
    "seg_nano": "RFDETRSegNano",
    "seg_small": "RFDETRSegSmall",
    "seg_medium": "RFDETRSegMedium",
    "seg_large": "RFDETRSegLarge",
    "seg_xlarge": "RFDETRSegXLarge",
    "seg_2xlarge": "RFDETRSeg2XLarge",
}


class RFDETRDetector:
    """RF-DETR based object detector with instance segmentation.

    Detects objects in images and returns 2D bounding boxes and
    instance segmentation masks. Must use a 'seg_*' variant for
    mask output.

    Example:
        >>> detector = RFDETRDetector(variant="seg_medium")
        >>> objects = detector.detect(image_array)
        >>> for obj in objects:
        ...     print(obj.class_name, obj.bbox_2d, obj.mask_2d.shape)
    """

    def __init__(
        self,
        variant: str = "seg_medium",
        confidence_threshold: float = 0.5,
        device: str = "cuda",
        include_source_image: bool = True,
    ):
        """Initialize RF-DETR detector.

        Args:
            variant: Model variant name. Must start with 'seg_' for mask output.
                     Options: seg_nano, seg_small, seg_medium, seg_large,
                              seg_xlarge, seg_2xlarge
                     Or detection-only: nano, small, medium, large
            confidence_threshold: Minimum confidence for detections
            device: Device to run on ('cuda' or 'cpu')
            include_source_image: Whether to include source image in results
        """
        self.variant = variant
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.include_source_image = include_source_image
        self.model = None
        self.class_names = None

        # Warn if using detection-only variant (no masks)
        if not variant.startswith("seg_"):
            logger.warning(
                f"Variant '{variant}' does not produce segmentation masks. "
                "Use a 'seg_*' variant for the 3D reconstruction pipeline. "
                "Falling back to 'seg_medium'."
            )
            self.variant = "seg_medium"

    def load_model(self):
        """Load the RF-DETR model and weights.

        Downloads pretrained COCO weights automatically on first use.
        Weights are cached at ~/.roboflow/models/.
        """
        try:
            from rfdetr import (
                RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium,
                RFDETRSegLarge, RFDETRSegXLarge, RFDETRSeg2XLarge,
            )
        except ImportError as e:
            raise ImportError(
                "Failed to import rfdetr. Please install it:\n"
                "  cd repos/rf-detr && pip install -e .\n"
                f"Original error: {e}"
            )

        class_map = {
            "seg_nano": RFDETRSegNano,
            "seg_small": RFDETRSegSmall,
            "seg_medium": RFDETRSegMedium,
            "seg_large": RFDETRSegLarge,
            "seg_xlarge": RFDETRSegXLarge,
            "seg_2xlarge": RFDETRSeg2XLarge,
        }

        if self.variant not in class_map:
            raise ValueError(
                f"Unknown variant '{self.variant}'. "
                f"Must be one of: {list(class_map.keys())}"
            )

        logger.info(f"Loading RF-DETR model variant: {self.variant}")
        self.model = class_map[self.variant]()

        # Load COCO class names
        try:
            from rfdetr.assets.coco_classes import COCO_CLASSES
            self.class_names = COCO_CLASSES
        except ImportError:
            # Try alternative import path
            try:
                import rfdetr
                asset_dir = None
                for path in rfdetr.__path__:
                    candidate = os.path.join(path, "assets", "coco_classes.py")
                    if os.path.exists(candidate):
                        asset_dir = path
                        break
                if asset_dir:
                    import importlib.util
                    spec = importlib.util.spec_from_file_location(
                        "coco_classes",
                        os.path.join(asset_dir, "assets", "coco_classes.py")
                    )
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    self.class_names = mod.COCO_CLASSES
                else:
                    raise ImportError("coco_classes.py not found in rfdetr package")
            except Exception:
                logger.warning("Could not load COCO class names from rfdetr")
                self.class_names = [f"class_{i}" for i in range(91)]

        logger.info(f"RF-DETR model loaded successfully")

    def detect(
        self,
        image: np.ndarray,
        threshold: float = None,
        min_area: int = 100,
        max_detections: int = 50,
    ) -> List[DetectedObject]:
        """Detect objects in an image.

        Args:
            image: (H, W, 3) uint8 RGB image
            threshold: Confidence threshold (overrides init value)
            min_area: Minimum mask area in pixels to keep an object
            max_detections: Maximum number of objects to return

        Returns:
            List of DetectedObject instances with 2D boxes and masks
        """
        if self.model is None:
            self.load_model()

        threshold = threshold or self.confidence_threshold

        # Run inference
        detections = self.model.predict(
            image,
            threshold=threshold,
            include_source_image=self.include_source_image,
        )

        objects = []

        # Handle different RF-DETR API versions
        # Some versions return a Detections object, others a dict or tuple
        if hasattr(detections, 'xyxy'):
            # supervision-style Detections object
            n_dets = len(detections.xyxy)
            for i in range(min(n_dets, max_detections)):
                bbox = detections.xyxy[i]  # (4,) xyxy
                confidence = float(detections.confidence[i])
                class_id = int(detections.class_id[i])

                # Get class name
                if self.class_names and class_id < len(self.class_names):
                    class_name = self.class_names[class_id]
                else:
                    class_name = f"class_{class_id}"

                # Get mask
                mask = detections.mask[i] if detections.mask is not None else None

                if mask is None:
                    # Create a rectangular mask from bbox if no segmentation
                    mask = np.zeros(image.shape[:2], dtype=bool)
                    x1, y1, x2, y2 = bbox.astype(int)
                    mask[max(0,y1):min(image.shape[0],y2), max(0,x1):min(image.shape[1],x2)] = True

                # Filter by minimum area
                if mask.sum() < min_area:
                    continue

                obj = DetectedObject(
                    object_id=i,
                    class_name=class_name,
                    confidence=confidence,
                    bbox_2d=bbox.astype(np.float64),
                    mask_2d=mask,
                )
                objects.append(obj)
        else:
            logger.warning(
                f"Unexpected detection format: {type(detections)}. "
                "RF-DETR API may have changed."
            )

        logger.info(
            f"RF-DETR detected {len(objects)} objects "
            f"(threshold={threshold:.2f})"
        )
        return objects

    def detect_from_path(
        self,
        image_path: str,
        threshold: float = None,
        min_area: int = 100,
        max_detections: int = 50,
    ) -> Tuple[List[DetectedObject], np.ndarray]:
        """Detect objects from an image file path.

        Args:
            image_path: Path to the image file
            threshold: Confidence threshold
            min_area: Minimum mask area in pixels
            max_detections: Maximum number of objects

        Returns:
            Tuple of (list of DetectedObject, image array)
        """
        image = np.array(Image.open(image_path).convert("RGB"))
        objects = self.detect(image, threshold, min_area, max_detections)
        return objects, image

    def visualize(
        self,
        image: np.ndarray,
        objects: List[DetectedObject],
    ) -> np.ndarray:
        """Visualize detections on the image.

        Args:
            image: (H, W, 3) uint8 RGB image
            objects: List of DetectedObject

        Returns:
            Annotated image (H, W, 3) uint8
        """
        try:
            import supervision as sv
        except ImportError:
            logger.warning("supervision not installed; skipping visualization")
            return image

        # Convert to supervision Detections format
        if not objects:
            return image

        xyxy = np.stack([obj.bbox_2d for obj in objects])
        confidence = np.array([obj.confidence for obj in objects])
        class_id = np.array([obj.object_id for obj in objects])
        masks = np.stack([obj.mask_2d for obj in objects])

        dets = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
            mask=masks,
        )

        labels = [f"{obj.class_name} {obj.confidence:.2f}" for obj in objects]

        annotated = sv.BoxAnnotator().annotate(image.copy(), dets)
        annotated = sv.MaskAnnotator().annotate(annotated, dets)
        annotated = sv.LabelAnnotator().annotate(annotated, dets, labels)

        return annotated
