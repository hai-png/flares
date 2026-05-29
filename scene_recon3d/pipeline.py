"""Main Pipeline Orchestrator: 3D Scene Reconstruction.

This module ties together all the pipeline stages:
  1. RF-DETR    → 2D object detection + instance segmentation
  2. WildDet3D  → 3D bounding box estimation from 2D boxes
  3. Hunyuan3D  → Per-object 3D mesh generation
  4. Alignment  → Scale and pose alignment using 3D bounding boxes
  5. MARCO      → Semantic correspondence for pose refinement

The pipeline processes a single RGB image (with optional camera intrinsics)
and produces a complete 3D scene reconstruction with textured meshes
placed in a unified 3D coordinate space.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
from omegaconf import OmegaConf

from .modules.rfdetr_detector import RFDETRDetector
from .modules.wilddet3d_estimator import WildDet3DEstimator
from .modules.hunyuan3d_generator import Hunyuan3DGenerator
from .modules.marco_refiner import MARCORefiner
from .utils.data_types import DetectedObject, ObjectReconstructionResult, SceneReconstructionResult
from .utils.geometry import align_mesh_to_bbox, crop_image_with_mask, draw_bbox3d_on_image

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Timing and statistics for each pipeline stage."""
    total_time: float = 0.0
    stage_times: Dict[str, float] = field(default_factory=dict)
    num_objects_detected: int = 0
    num_objects_3d: int = 0
    num_objects_meshed: int = 0
    num_objects_refined: int = 0


class SceneReconstructionPipeline:
    """Complete 3D scene reconstruction pipeline.

    Combines RF-DETR, WildDet3D, Hunyuan3D-2.1+FlashVDM, and MARCO
    to reconstruct a complete 3D scene from a single RGB image.

    Pipeline flow:
        ┌─────────────┐     ┌─────────────┐
        │   Input      │────▶│   RF-DETR   │
        │   Image      │     │ (2D boxes + │
        │              │     │   masks)    │
        └─────────────┘     └──────┬──────┘
                                    │
                          ┌─────────▼──────────┐
                          │    WildDet3D        │
                          │  (3D bounding boxes │
                          │   in camera space)  │
                          └─────────┬──────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
              ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
              │ Hunyuan3D  │  │ Hunyuan3D  │  │ Hunyuan3D  │
              │ (mesh for  │  │ (mesh for  │  │ (mesh for  │
              │  obj 1)    │  │  obj 2)    │  │  obj N)    │
              └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
                    │               │               │
                    └───────┬───────┘───────────────┘
                            │
                  ┌─────────▼──────────┐
                  │  Scale + Pose      │
                  │  Alignment         │
                  │  (using 3D bboxes) │
                  └─────────┬──────────┘
                            │
                  ┌─────────▼──────────┐
                  │     MARCO          │
                  │  (pose refinement  │
                  │   via semantic     │
                  │   correspondence)  │
                  └─────────┬──────────┘
                            │
                  ┌─────────▼──────────┐
                  │  Final 3D Scene    │
                  │  Reconstruction    │
                  └────────────────────┘

    Example:
        >>> pipeline = SceneReconstructionPipeline.from_config("configs/pipeline_config.yaml")
        >>> result = pipeline.reconstruct("scene_image.jpg")
        >>> result.export_scene("output/scene.glb")
    """

    def __init__(
        self,
        # RF-DETR params
        rfdetr_variant: str = "seg_medium",
        rfdetr_confidence: float = 0.5,
        # WildDet3D params
        wilddet3d_checkpoint: str = "ckpt/wilddet3d_alldata_all_prompt_v1.0.pt",
        wilddet3d_score_threshold: float = 0.3,
        wilddet3d_use_predicted_intrinsics: bool = True,
        # Hunyuan3D params
        hunyuan3d_model_path: str = "tencent/Hunyuan3D-2.1",
        hunyuan3d_subfolder: str = "hunyuan3d-dit-v2-1",
        hunyuan3d_enable_flashvdm: bool = True,
        hunyuan3d_num_steps: int = 50,
        hunyuan3d_guidance_scale: float = 5.0,
        hunyuan3d_octree_resolution: int = 384,
        hunyuan3d_generate_texture: bool = True,
        # MARCO params
        marco_use_torch_hub: bool = True,
        marco_inference_res: int = 840,
        marco_num_keypoints: int = 20,
        marco_refinement_iterations: int = 3,
        # General params
        device: str = "cuda",
        output_dir: str = "output/scene",
        save_intermediate: bool = True,
        render_resolution: int = 512,
        min_object_area: int = 100,
        low_vram_mode: bool = False,
    ):
        """Initialize the pipeline with all module configurations.

        Args:
            rfdetr_variant: RF-DETR model variant (seg_* for masks)
            rfdetr_confidence: Detection confidence threshold
            wilddet3d_checkpoint: Path to WildDet3D checkpoint
            wilddet3d_score_threshold: 3D detection score threshold
            wilddet3d_use_predicted_intrinsics: Predict intrinsics for in-the-wild images
            hunyuan3d_model_path: Hunyuan3D model path (HF or local)
            hunyuan3d_subfolder: DiT model subfolder
            hunyuan3d_enable_flashvdm: Enable FlashVDM acceleration
            hunyuan3d_num_steps: Denoising steps
            hunyuan3d_guidance_scale: CFG guidance scale
            hunyuan3d_octree_resolution: Mesh resolution
            hunyuan3d_generate_texture: Generate PBR textures
            marco_use_torch_hub: Auto-download MARCO via torch.hub
            marco_inference_res: MARCO inference resolution
            marco_num_keypoints: Keypoints per object for refinement
            marco_refinement_iterations: Number of refinement iterations
            device: Compute device
            output_dir: Directory for output files
            save_intermediate: Save intermediate results
            render_resolution: Resolution for mesh rendering
            min_object_area: Minimum mask area to process an object
            low_vram_mode: Load/unload models stage-by-stage to reduce peak GPU memory
        """
        self.device = device
        self.output_dir = output_dir
        self.save_intermediate = save_intermediate
        self.render_resolution = render_resolution
        self.min_object_area = min_object_area
        self.low_vram_mode = low_vram_mode

        # Initialize all modules (lazy loading - models loaded on first use)
        self.detector = RFDETRDetector(
            variant=rfdetr_variant,
            confidence_threshold=rfdetr_confidence,
            device=device,
        )

        self.estimator_3d = WildDet3DEstimator(
            checkpoint=wilddet3d_checkpoint,
            score_threshold=wilddet3d_score_threshold,
            use_predicted_intrinsics=wilddet3d_use_predicted_intrinsics,
            device=device,
        )

        self.generator = Hunyuan3DGenerator(
            model_path=hunyuan3d_model_path,
            subfolder=hunyuan3d_subfolder,
            enable_flashvdm=hunyuan3d_enable_flashvdm,
            num_inference_steps=hunyuan3d_num_steps,
            guidance_scale=hunyuan3d_guidance_scale,
            octree_resolution=hunyuan3d_octree_resolution,
            generate_texture=hunyuan3d_generate_texture,
            device=device,
            low_vram_mode=low_vram_mode,
        )

        self.refiner = MARCORefiner(
            use_torch_hub=marco_use_torch_hub,
            inference_res=marco_inference_res,
            num_keypoints_per_object=marco_num_keypoints,
            refinement_iterations=marco_refinement_iterations,
            device=device,
        )

        self._stats = PipelineStats()

    @classmethod
    def from_config(cls, config_path: str) -> "SceneReconstructionPipeline":
        """Create a pipeline from a YAML configuration file.

        Args:
            config_path: Path to the YAML config file

        Returns:
            Configured SceneReconstructionPipeline instance
        """
        cfg = OmegaConf.load(config_path)

        rfdetr = cfg.get("rfdetr", {})
        wilddet3d = cfg.get("wilddet3d", {})
        hunyuan3d = cfg.get("hunyuan3d", {})
        marco = cfg.get("marco", {})
        pipeline = cfg.get("pipeline", {})

        return cls(
            # RF-DETR
            rfdetr_variant=rfdetr.get("model_variant", "seg_medium"),
            rfdetr_confidence=rfdetr.get("confidence_threshold", 0.5),
            # WildDet3D
            wilddet3d_checkpoint=wilddet3d.get(
                "checkpoint", "ckpt/wilddet3d_alldata_all_prompt_v1.0.pt"
            ),
            wilddet3d_score_threshold=wilddet3d.get("score_threshold", 0.3),
            wilddet3d_use_predicted_intrinsics=wilddet3d.get(
                "use_predicted_intrinsics", True
            ),
            # Hunyuan3D
            hunyuan3d_model_path=hunyuan3d.get("model_path", "tencent/Hunyuan3D-2.1"),
            hunyuan3d_subfolder=hunyuan3d.get("subfolder", "hunyuan3d-dit-v2-1"),
            hunyuan3d_enable_flashvdm=hunyuan3d.get("enable_flashvdm", True),
            hunyuan3d_num_steps=hunyuan3d.get("num_inference_steps", 50),
            hunyuan3d_guidance_scale=hunyuan3d.get("guidance_scale", 5.0),
            hunyuan3d_octree_resolution=hunyuan3d.get("octree_resolution", 384),
            hunyuan3d_generate_texture=hunyuan3d.get("generate_texture", True),
            # MARCO
            marco_use_torch_hub=marco.get("use_torch_hub", True),
            marco_inference_res=marco.get("inference_res", 840),
            marco_num_keypoints=marco.get("num_keypoints_per_object", 20),
            marco_refinement_iterations=marco.get("refinement_iterations", 3),
            # Pipeline
            device=pipeline.get("device", "cuda"),
            output_dir=pipeline.get("output_dir", "output/scene"),
            save_intermediate=pipeline.get("save_intermediate", True),
            render_resolution=pipeline.get("render_resolution", 512),
            min_object_area=pipeline.get("min_object_area", 100),
            low_vram_mode=pipeline.get("low_vram_mode", False),
        )

    @staticmethod
    def _gpu_mem_info() -> str:
        """Return a short GPU + system memory summary string."""
        parts = []
        try:
            import torch
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                total = torch.cuda.get_device_properties(0).total_mem / 1024**3
                parts.append(f"GPU: {alloc:.1f}/{total:.1f}GB alloc, {reserved:.1f}GB reserved")
        except Exception:
            pass
        try:
            import psutil
            vm = psutil.virtual_memory()
            used_gb = vm.used / 1024**3
            total_gb = vm.total / 1024**3
            parts.append(f"RAM: {used_gb:.1f}/{total_gb:.1f}GB")
        except ImportError:
            pass
        return " | ".join(parts) if parts else ""

    def _free_memory(self):
        """Aggressively free GPU and CPU memory before loading a new model.

        Runs Python garbage collection and clears the CUDA cache.
        Call this BEFORE loading a model to ensure maximum free memory.
        """
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def _unload_model(self, module_name: str):
        """Unload a model from GPU memory to free VRAM.

        Moves the model to CPU first, then deletes it and runs garbage
        collection. The model will be re-loaded on next use (lazy loading).
        """
        unloaded = False

        if module_name == "detector" and self.detector.model is not None:
            logger.info(f"Unloading RF-DETR from GPU... {self._gpu_mem_info()}")
            try:
                self.detector.model = self.detector.model.cpu()
            except Exception:
                pass
            del self.detector.model
            self.detector.model = None
            unloaded = True
        elif module_name == "estimator_3d" and self.estimator_3d.model is not None:
            logger.info(f"Unloading WildDet3D from GPU... {self._gpu_mem_info()}")
            try:
                self.estimator_3d.model = self.estimator_3d.model.cpu()
            except Exception:
                pass
            del self.estimator_3d.model
            self.estimator_3d.model = None
            unloaded = True
        elif module_name == "generator" and self.generator.shape_pipeline is not None:
            logger.info(f"Unloading Hunyuan3D from GPU... {self._gpu_mem_info()}")
            # Use the dedicated unload method for thorough cleanup
            self.generator.unload_model()
            unloaded = True
        elif module_name == "refiner" and self.refiner.model is not None:
            logger.info(f"Unloading MARCO from GPU... {self._gpu_mem_info()}")
            try:
                self.refiner.model = self.refiner.model.cpu()
            except Exception:
                pass
            del self.refiner.model
            self.refiner.model = None
            unloaded = True

        if not unloaded:
            return  # Nothing to unload

        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.info(f"  After unload: {self._gpu_mem_info()}")

    def load_all_models(self):
        """Pre-load all models before running the pipeline.

        In low_vram_mode (default True), models are loaded stage-by-stage
        during pipeline execution and unloaded after each stage completes.
        This keeps peak GPU memory significantly lower than loading all
        models at once.

        When low_vram_mode=False, all models are loaded at once (requires
        sufficient GPU memory for all models simultaneously).
        """
        if self.low_vram_mode:
            logger.info(
                "Low VRAM mode enabled — models will be loaded/unloaded "
                "stage-by-stage during pipeline execution. "
                "This keeps peak GPU usage low for broader hardware compatibility."
            )
            # Do a quick import check to verify all modules are importable,
            # but don't actually load the heavy model weights.
            logger.info("Verifying all modules are importable (without loading weights)...")
            try:
                import rfdetr  # noqa: F401
                logger.info("  ✓ RF-DETR importable")
            except ImportError as e:
                logger.error(f"  ✗ RF-DETR not importable: {e}")
            try:
                from scene_recon3d.utils.setup_paths import setup_repo_paths
                setup_repo_paths()
                from wilddet3d import build_model  # noqa: F401
                logger.info("  ✓ WildDet3D importable")
            except ImportError as e:
                logger.error(f"  ✗ WildDet3D not importable: {e}")
            try:
                from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
                logger.info("  ✓ Hunyuan3D importable")
            except ImportError as e:
                logger.error(f"  ✗ Hunyuan3D not importable: {e}")
            logger.info("Low VRAM mode ready — models will load on demand")
        else:
            logger.info("Pre-loading all pipeline models (requires enough VRAM for all models)...")
            logger.info("[1/4] Loading RF-DETR...")
            self.detector.load_model()
            logger.info("[2/4] Loading WildDet3D...")
            self.estimator_3d.load_model()
            logger.info("[3/4] Loading Hunyuan3D-2.1 + FlashVDM...")
            self.generator.load_model()
            logger.info("[4/4] Loading MARCO...")
            self.refiner.load_model()
            logger.info("All models loaded successfully!")

    def reconstruct(
        self,
        image: np.ndarray | str,
        intrinsics: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ) -> SceneReconstructionResult:
        """Run the full 3D scene reconstruction pipeline.

        Args:
            image: Input image as numpy array (H,W,3) or file path string
            intrinsics: Optional (3,3) camera intrinsic matrix
            class_names: Optional list of class names for 3D prompting
            output_dir: Override output directory

        Returns:
            SceneReconstructionResult with all reconstructed objects
        """
        start_time = time.time()
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load image if path provided
        if isinstance(image, str):
            from PIL import Image as PILImage
            image = np.array(PILImage.open(image).convert("RGB"))

        h, w = image.shape[:2]
        logger.info(f"Starting 3D scene reconstruction on {w}×{h} image")

        # ═══════════════════════════════════════════════════════════
        # Stage 1: RF-DETR — 2D Object Detection + Instance Segmentation
        # ═══════════════════════════════════════════════════════════
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("Stage 1: RF-DETR — 2D Detection + Segmentation")
        logger.info("=" * 60)

        # Free any stale GPU/CPU memory before starting
        self._free_memory()
        logger.info(f"Memory before Stage 1: {self._gpu_mem_info()}")

        objects = self.detector.detect(
            image,
            min_area=self.min_object_area,
        )
        self._stats.stage_times["1_rfdetr"] = time.time() - t0
        self._stats.num_objects_detected = len(objects)

        if not objects:
            logger.warning("No objects detected! Returning empty result.")
            return SceneReconstructionResult(image_shape=(h, w))

        # Save detection visualization
        if self.save_intermediate:
            vis = self.detector.visualize(image, objects)
            import cv2
            cv2.imwrite(os.path.join(output_dir, "1_detections.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

            # Save per-object crop images
            crops_dir = os.path.join(output_dir, "crops")
            os.makedirs(crops_dir, exist_ok=True)
            for obj in objects:
                if obj.crop_image is not None:
                    crop_path = os.path.join(crops_dir, f"{obj.class_name}_{obj.object_id}_crop.png")
                    cv2.imwrite(crop_path, cv2.cvtColor(obj.crop_image, cv2.COLOR_RGB2BGR))
                if obj.crop_mask is not None:
                    mask_path = os.path.join(crops_dir, f"{obj.class_name}_{obj.object_id}_mask.png")
                    cv2.imwrite(mask_path, (obj.crop_mask * 255).astype(np.uint8))

        # Free RF-DETR from GPU before loading WildDet3D (low VRAM mode)
        if self.low_vram_mode:
            self._unload_model("detector")

        # ═══════════════════════════════════════════════════════════
        # Stage 2: WildDet3D — 3D Bounding Box Estimation
        # ═══════════════════════════════════════════════════════════
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("Stage 2: WildDet3D — 3D Bounding Box Estimation")
        logger.info("=" * 60)

        # Ensure maximum free memory before loading WildDet3D
        self._free_memory()
        logger.info(f"Memory before Stage 2: {self._gpu_mem_info()}")

        objects = self.estimator_3d.estimate_3d(
            image, objects, intrinsics=intrinsics, class_names=class_names
        )
        camera_intrinsics = self.estimator_3d.get_intrinsics()
        self._stats.stage_times["2_wilddet3d"] = time.time() - t0
        self._stats.num_objects_3d = sum(1 for o in objects if o.bbox_3d is not None)

        # Free WildDet3D from GPU before loading Hunyuan3D (low VRAM mode)
        # Hunyuan3D is the heaviest model (~7-10GB VRAM) and needs the space
        if self.low_vram_mode:
            self._unload_model("estimator_3d")

        # Save 3D bbox visualization and data
        if self.save_intermediate:
            import cv2
            import json

            # Draw 3D bounding boxes projected to 2D
            if camera_intrinsics is not None:
                vis_3d = image.copy()
                colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
                for i, obj in enumerate(objects):
                    if obj.bbox_3d is not None:
                        color = colors[i % len(colors)]
                        vis_3d = draw_bbox3d_on_image(
                            vis_3d,
                            obj.bbox_3d_center,
                            obj.bbox_3d_dims,
                            obj.bbox_3d_quat,
                            camera_intrinsics,
                            color=color,
                            thickness=2,
                        )
                        # Add label
                        corners_3d = obj.bbox_3d_center.reshape(1, 3)
                        from .utils.geometry import project_points_to_2d
                        center_2d, center_in_front = project_points_to_2d(
                            corners_3d, camera_intrinsics, return_validity=True,
                        )
                        if center_in_front[0]:
                            center_2d = center_2d[0]
                            label = f"{obj.class_name} (3D: {obj.score_3d:.2f})"
                            cv2.putText(vis_3d, label,
                                        (int(center_2d[0]) - 40, int(center_2d[1]) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.imwrite(os.path.join(output_dir, "2_bbox3d_projection.png"),
                            cv2.cvtColor(vis_3d, cv2.COLOR_RGB2BGR))

            # Save 3D bbox data as JSON
            bbox_data = []
            for obj in objects:
                if obj.bbox_3d is not None:
                    bbox_data.append({
                        "object_id": obj.object_id,
                        "class_name": obj.class_name,
                        "bbox_3d_center": obj.bbox_3d_center.tolist(),
                        "bbox_3d_dims_WLH": obj.bbox_3d_dims.tolist(),
                        "bbox_3d_quat_wxyz": obj.bbox_3d_quat.tolist(),
                        "score_3d": float(obj.score_3d) if obj.score_3d is not None else None,
                        "bbox_2d_xyxy": obj.bbox_2d.tolist(),
                    })
            with open(os.path.join(output_dir, "2_bbox3d_data.json"), "w") as f:
                json.dump(bbox_data, f, indent=2)

            # Save intrinsics
            if camera_intrinsics is not None:
                np.save(os.path.join(output_dir, "camera_intrinsics.npy"), camera_intrinsics)

        # ═══════════════════════════════════════════════════════════
        # Stage 3: Hunyuan3D — Per-Object 3D Mesh Generation
        # ═══════════════════════════════════════════════════════════
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("Stage 3: Hunyuan3D-2.1 + FlashVDM — 3D Mesh Generation")
        logger.info("=" * 60)

        # CRITICAL: Free every last byte before loading Hunyuan3D — it's
        # the largest model and both GPU *and* system RAM are very tight.
        self._free_memory()
        logger.info(f"Memory before Stage 3: {self._gpu_mem_info()}")

        mesh_dir = os.path.join(output_dir, "meshes")
        objects = self.generator.generate_meshes_for_objects(
            image, objects, output_dir=mesh_dir
        )
        self._stats.stage_times["3_hunyuan3d"] = time.time() - t0
        self._stats.num_objects_meshed = sum(1 for o in objects if o.mesh is not None)

        # Save Stage 3 debug info (mesh dimensions for alignment debugging)
        if self.save_intermediate:
            import json
            mesh_debug = []
            for obj in objects:
                if obj.mesh is not None:
                    mesh_debug.append({
                        "object_id": obj.object_id,
                        "class_name": obj.class_name,
                        "mesh_extents": obj.mesh.extents.tolist(),
                        "mesh_bounds_min": obj.mesh.bounds[0].tolist(),
                        "mesh_bounds_max": obj.mesh.bounds[1].tolist(),
                        "num_vertices": len(obj.mesh.vertices),
                        "num_faces": len(obj.mesh.faces),
                        "mesh_path": obj.mesh_path,
                    })
            with open(os.path.join(output_dir, "3_mesh_debug.json"), "w") as f:
                json.dump(mesh_debug, f, indent=2)

        # Free Hunyuan3D from GPU — always unload before MARCO, even in
        # non-low-VRAM mode, because MARCO needs significant GPU memory
        # (DINOv2-giant + feature extraction) and Hunyuan3D is not needed
        # for Stages 4 or 5.  On a T4 (14.5 GB), keeping Hunyuan3D loaded
        # leaves only ~300 MB free, causing MARCO to OOM.
        if self.generator.shape_pipeline is not None:
            self._unload_model("generator")

        # ═══════════════════════════════════════════════════════════
        # Stage 4: Scale & Pose Alignment using 3D Bounding Boxes
        # ═══════════════════════════════════════════════════════════
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("Stage 4: Scale + Pose Alignment (3D Bounding Boxes)")
        logger.info("=" * 60)

        self._free_memory()
        logger.info(f"Memory before Stage 4: {self._gpu_mem_info()}")

        for obj in objects:
            if obj.mesh is None or obj.bbox_3d is None:
                logger.warning(
                    f"Object {obj.object_id}: skipping alignment "
                    f"(mesh={'✓' if obj.mesh is not None else '✗'}, "
                    f"3D bbox={'✓' if obj.bbox_3d is not None else '✗'})"
                )
                continue

            try:
                aligned_mesh, scale_factor, rotation, translation = align_mesh_to_bbox(
                    mesh=obj.mesh,
                    bbox_center=obj.bbox_3d_center,
                    bbox_dims=obj.bbox_3d_dims,
                    bbox_quat=obj.bbox_3d_quat,
                    camera_intrinsics=camera_intrinsics,
                    bbox_2d=obj.bbox_2d,
                )

                obj.aligned_mesh = aligned_mesh
                obj.scale_factor = scale_factor
                obj.initial_rotation = rotation
                obj.initial_translation = translation

                logger.info(
                    f"Object {obj.object_id} ({obj.class_name}): "
                    f"scale={scale_factor:.4f}, "
                    f"center={obj.bbox_3d_center}, "
                    f"dims(W,L,H)={obj.bbox_3d_dims}, "
                    f"quat(w,x,y,z)={obj.bbox_3d_quat}"
                )
            except Exception as e:
                logger.error(
                    f"Object {obj.object_id} alignment failed: {e}. Skipping."
                )
                continue

        self._stats.stage_times["4_alignment"] = time.time() - t0

        # Save aligned meshes and alignment debug data
        if self.save_intermediate:
            import cv2
            import json

            aligned_dir = os.path.join(output_dir, "aligned")
            os.makedirs(aligned_dir, exist_ok=True)
            alignment_debug = []
            for obj in objects:
                if obj.aligned_mesh is not None:
                    # Save aligned mesh
                    path = os.path.join(aligned_dir, f"{obj.class_name}_{obj.object_id}.glb")
                    obj.aligned_mesh.export(path)

                    # Collect debug info
                    debug_info = {
                        "object_id": obj.object_id,
                        "class_name": obj.class_name,
                        "scale_factor": obj.scale_factor,
                        "initial_rotation": obj.initial_rotation.tolist() if obj.initial_rotation is not None else None,
                        "initial_translation": obj.initial_translation.tolist() if obj.initial_translation is not None else None,
                        "bbox_3d_center": obj.bbox_3d_center.tolist() if obj.bbox_3d_center is not None else None,
                        "bbox_3d_dims_WLH": obj.bbox_3d_dims.tolist() if obj.bbox_3d_dims is not None else None,
                        "bbox_3d_quat_wxyz": obj.bbox_3d_quat.tolist() if obj.bbox_3d_quat is not None else None,
                        "aligned_mesh_extents": obj.aligned_mesh.extents.tolist(),
                        "canonical_mesh_extents": obj.mesh.extents.tolist() if obj.mesh is not None else None,
                    }
                    alignment_debug.append(debug_info)

                    # Validation: project aligned mesh bounding box back to 2D
                    # and compare with original 2D detection
                    if camera_intrinsics is not None and obj.bbox_2d is not None:
                        from .utils.geometry import _compute_alignment_2d_iou
                        iou = _compute_alignment_2d_iou(
                            obj.aligned_mesh, obj.bbox_2d, camera_intrinsics
                        )

                        # Also compute projected 2D bbox for debug logging
                        from .utils.geometry import project_points_to_2d
                        try:
                            obb = obj.aligned_mesh.bounding_box_oriented
                            corners_3d = obb.vertices
                        except Exception:
                            aligned_bounds = obj.aligned_mesh.bounds
                            mins, maxs = aligned_bounds
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
                        valid_2d = corners_2d[in_front]
                        if len(valid_2d) >= 3:
                            proj_x1 = float(np.nanmin(valid_2d[:, 0]))
                            proj_y1 = float(np.nanmin(valid_2d[:, 1]))
                            proj_x2 = float(np.nanmax(valid_2d[:, 0]))
                            proj_y2 = float(np.nanmax(valid_2d[:, 1]))
                        else:
                            proj_x1 = proj_y1 = proj_x2 = proj_y2 = 0.0

                        det_x1, det_y1, det_x2, det_y2 = obj.bbox_2d.tolist()

                        debug_info["reprojection_2d"] = {
                            "projected_bbox_xyxy": [proj_x1, proj_y1, proj_x2, proj_y2],
                            "detected_bbox_xyxy": [det_x1, det_y1, det_x2, det_y2],
                            "iou": float(iou),
                        }

                        if iou < 0.3:
                            logger.warning(
                                f"Object {obj.object_id} alignment has low 2D IoU={iou:.3f}. "
                                f"Projected [{proj_x1:.0f},{proj_y1:.0f},{proj_x2:.0f},{proj_y2:.0f}] "
                                f"vs detected [{det_x1:.0f},{det_y1:.0f},{det_x2:.0f},{det_y2:.0f}]. "
                                f"Alignment may be inaccurate."
                            )

            # Save alignment debug data
            with open(os.path.join(output_dir, "4_alignment_debug.json"), "w") as f:
                json.dump(alignment_debug, f, indent=2)

            # Draw alignment validation visualization
            if camera_intrinsics is not None:
                vis_align = image.copy()
                for i, obj in enumerate(objects):
                    if obj.aligned_mesh is not None and obj.bbox_2d is not None:
                        from .utils.geometry import project_points_to_2d, _compute_alignment_2d_iou
                        # Draw detected 2D bbox (green)
                        x1, y1, x2, y2 = obj.bbox_2d.astype(int)
                        cv2.rectangle(vis_align, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Draw projected aligned mesh bbox (red)
                        # Use OBB for tighter fit on rotated meshes
                        try:
                            obb = obj.aligned_mesh.bounding_box_oriented
                            corners_3d = obb.vertices
                        except Exception:
                            aligned_bounds = obj.aligned_mesh.bounds
                            mins, maxs = aligned_bounds
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
                        valid_2d = corners_2d[in_front]
                        if len(valid_2d) >= 3:
                            px1 = int(np.nanmin(valid_2d[:, 0]))
                            py1 = int(np.nanmin(valid_2d[:, 1]))
                            px2 = int(np.nanmax(valid_2d[:, 0]))
                            py2 = int(np.nanmax(valid_2d[:, 1]))
                            cv2.rectangle(vis_align, (px1, py1), (px2, py2), (255, 0, 0), 2)

                        # Compute IoU between projected aligned mesh and 2D detection
                        iou_val = _compute_alignment_2d_iou(
                            obj.aligned_mesh, obj.bbox_2d, camera_intrinsics
                        )
                        cv2.putText(vis_align, f"{obj.class_name} IoU={iou_val:.2f}",
                                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                cv2.imwrite(os.path.join(output_dir, "4_alignment_validation.png"),
                            cv2.cvtColor(vis_align, cv2.COLOR_RGB2BGR))

        # ═══════════════════════════════════════════════════════════
        # Stage 5: MARCO — Pose Refinement via Semantic Correspondence
        # ═══════════════════════════════════════════════════════════
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("Stage 5: MARCO — Pose Refinement")
        logger.info("=" * 60)

        self._free_memory()
        logger.info(f"Memory before Stage 5: {self._gpu_mem_info()}")

        if camera_intrinsics is not None:
            objects = self.refiner.refine_poses(
                objects, camera_intrinsics, render_resolution=self.render_resolution,
                original_image_shape=(h, w),
            )
        else:
            logger.warning(
                "No camera intrinsics available; skipping MARCO refinement"
            )

        self._stats.stage_times["5_marco"] = time.time() - t0
        self._stats.num_objects_refined = sum(
            1 for o in objects if o.refined_rotation is not None
        )

        # Save refined meshes and MARCO debug data
        if self.save_intermediate:
            import json

            refined_dir = os.path.join(output_dir, "refined")
            os.makedirs(refined_dir, exist_ok=True)
            marco_debug = []
            for obj in objects:
                if obj.aligned_mesh is not None:
                    # Determine the final rotation and translation
                    if obj.refined_rotation is not None:
                        R = obj.refined_rotation
                        t = obj.refined_translation
                    else:
                        R = obj.initial_rotation
                        t = obj.initial_translation
                    scale_factor = obj.scale_factor if obj.scale_factor is not None else 1.0

                    # Save refined/final mesh (apply transform to canonical mesh)
                    refined_mesh = obj.mesh.copy()
                    mesh_center = refined_mesh.bounds.mean(axis=0)
                    refined_mesh.apply_translation(-mesh_center)
                    refined_mesh.apply_scale(scale_factor)
                    T_rot = np.eye(4)
                    T_rot[:3, :3] = R
                    refined_mesh.apply_transform(T_rot)
                    refined_mesh.apply_translation(t)
                    suffix = "_refined" if obj.refined_rotation is not None else "_aligned"
                    path = os.path.join(refined_dir, f"{obj.class_name}_{obj.object_id}{suffix}.glb")
                    refined_mesh.export(path)

                    init_R = obj.initial_rotation if obj.initial_rotation is not None else np.eye(3)
                    init_t = obj.initial_translation if obj.initial_translation is not None else np.zeros(3)
                    debug_info = {
                        "object_id": obj.object_id,
                        "class_name": obj.class_name,
                        "was_refined": obj.refined_rotation is not None,
                        "scale_factor": float(scale_factor),
                        "refined_rotation": obj.refined_rotation.tolist() if obj.refined_rotation is not None else None,
                        "refined_translation": obj.refined_translation.tolist() if obj.refined_translation is not None else None,
                        "initial_rotation": obj.initial_rotation.tolist() if obj.initial_rotation is not None else None,
                        "initial_translation": obj.initial_translation.tolist() if obj.initial_translation is not None else None,
                        "rotation_diff_norm": float(np.linalg.norm(R - init_R)),
                        "translation_diff_norm": float(np.linalg.norm(t - init_t)),
                        "num_correspondence_src": len(obj.correspondence_points_src) if obj.correspondence_points_src is not None else 0,
                        "num_correspondence_tgt": len(obj.correspondence_points_tgt) if obj.correspondence_points_tgt is not None else 0,
                    }
                    marco_debug.append(debug_info)

            with open(os.path.join(output_dir, "5_marco_debug.json"), "w") as f:
                json.dump(marco_debug, f, indent=2)

        # ═══════════════════════════════════════════════════════════
        # Build Final Result
        # ═══════════════════════════════════════════════════════════
        result_objects = []
        for obj in objects:
            if obj.aligned_mesh is None:
                continue

            # Build the final transform from the canonical mesh to world/camera coords.
            # The canonical mesh is centered at origin, so we need:
            #   T = Translate(bbox_center) @ Rotate(R) @ Scale(scale_factor)
            # The aligned_mesh already has this transform baked in.
            # To avoid a double-transform when get_scene_mesh() applies
            # obj.transform, we store the canonical mesh + the full transform.
            if obj.refined_rotation is not None:
                R = obj.refined_rotation
                t = obj.refined_translation
            else:
                R = obj.initial_rotation
                t = obj.initial_translation

            scale_factor = obj.scale_factor if obj.scale_factor is not None else 1.0

            # Build the full homogeneous transform that maps the canonical
            # (origin-centered) mesh to its final world position:
            #   T = Translate(t) @ Rotate(R) @ Scale(s)
            T_scale = np.eye(4)
            T_scale[0, 0] = scale_factor
            T_scale[1, 1] = scale_factor
            T_scale[2, 2] = scale_factor

            T_rot = np.eye(4)
            T_rot[:3, :3] = R

            T_trans = np.eye(4)
            T_trans[:3, 3] = t

            T = T_trans @ T_rot @ T_scale

            # Use the canonical mesh (NOT aligned_mesh) so that
            # get_scene_mesh() can apply the transform exactly once.
            canonical_mesh = obj.mesh
            if canonical_mesh is None:
                continue

            # Center the canonical mesh at origin (it should already be, but
            # ensure consistency with align_mesh_to_bbox which does this)
            centered = canonical_mesh.copy()
            mesh_center = centered.bounds.mean(axis=0)
            centered.apply_translation(-mesh_center)

            result_obj = ObjectReconstructionResult(
                object_id=obj.object_id,
                class_name=obj.class_name,
                mesh=centered,
                transform=T,
                bbox_3d=obj.bbox_3d if obj.bbox_3d is not None else None,
                confidence=obj.confidence,
            )
            result_objects.append(result_obj)

        result = SceneReconstructionResult(
            objects=result_objects,
            camera_intrinsics=camera_intrinsics,
            image_shape=(h, w),
        )

        # Export final scene
        scene_path = os.path.join(output_dir, "scene.glb")
        result.export_scene(scene_path)
        logger.info(f"Final scene exported to {scene_path}")

        # Also export individual object meshes (with transform baked in)
        if self.save_intermediate:
            final_dir = os.path.join(output_dir, "final_objects")
            os.makedirs(final_dir, exist_ok=True)
            for result_obj in result_objects:
                mesh = result_obj.mesh.copy()
                mesh.apply_transform(result_obj.transform)
                obj_path = os.path.join(final_dir, f"{result_obj.class_name}_{result_obj.object_id}.glb")
                mesh.export(obj_path)

        # Save pipeline summary
        if self.save_intermediate:
            import json
            summary = {
                "image_shape": [h, w],
                "num_objects_detected": self._stats.num_objects_detected,
                "num_objects_3d": self._stats.num_objects_3d,
                "num_objects_meshed": self._stats.num_objects_meshed,
                "num_objects_refined": self._stats.num_objects_refined,
                "total_time_s": self._stats.total_time if self._stats.total_time > 0 else time.time() - start_time,
                "stage_times_s": self._stats.stage_times,
                "objects": [
                    {
                        "object_id": o.object_id,
                        "class_name": o.class_name,
                        "confidence": o.confidence,
                    }
                    for o in result_objects
                ],
            }
            with open(os.path.join(output_dir, "pipeline_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)

        self._stats.total_time = time.time() - start_time
        self._log_stats()

        return result

    def _log_stats(self):
        """Log pipeline execution statistics."""
        logger.info("=" * 60)
        logger.info("Pipeline Statistics")
        logger.info("=" * 60)
        logger.info(f"Total time: {self._stats.total_time:.2f}s")
        for stage, t in self._stats.stage_times.items():
            logger.info(f"  {stage}: {t:.2f}s")
        logger.info(f"Objects detected: {self._stats.num_objects_detected}")
        logger.info(f"Objects with 3D bbox: {self._stats.num_objects_3d}")
        logger.info(f"Objects with mesh: {self._stats.num_objects_meshed}")
        logger.info(f"Objects refined: {self._stats.num_objects_refined}")

    def get_stats(self) -> PipelineStats:
        """Get the execution statistics from the last pipeline run."""
        return self._stats
