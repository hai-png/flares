"""Hunyuan3D-2.1 + FlashVDM Module: Image-to-3D Mesh Generation.

This module wraps Hunyuan3D-2.1 with FlashVDM acceleration to generate
high-fidelity textured 3D meshes from masked object images.

Hunyuan3D-2.1 is a two-stage pipeline:
  Stage 1: Shape Generation (3.3B DiT + VAE) → untextured mesh
  Stage 2: Texture Generation (2B Paint pipeline) → PBR-textured mesh

FlashVDM accelerates Stage 1 by:
  - Using sparse top-K cross-attention in the VAE decoder
  - Hierarchical octree-based volume decoding
  - Optionally using distilled turbo DiT (5 steps instead of 50)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh
from PIL import Image

from ..utils.data_types import DetectedObject
from ..utils.geometry import crop_image_with_mask

logger = logging.getLogger(__name__)


class Hunyuan3DGenerator:
    """Hunyuan3D-2.1 + FlashVDM based 3D mesh generator.

    Takes masked object images and generates high-fidelity 3D meshes
    with optional PBR textures.

    When low_vram_mode=True (recommended for T4 15GB):
      - Shape pipeline loaded with low_cpu_mem_usage=True to minimise system RAM
      - Texture (paint) pipeline is NOT loaded by default; enable explicitly
      - octree_resolution reduced to 256 (from 384) to cut peak VRAM
      - num_inference_steps reduced to 30 (from 50) to reduce intermediate tensors

    Example:
        >>> generator = Hunyuan3DGenerator(enable_flashvdm=True, low_vram_mode=True)
        >>> generator.load_model()
        >>> mesh = generator.generate_mesh(cropped_image, output_path="obj.glb")
    """

    def __init__(
        self,
        model_path: str = "tencent/Hunyuan3D-2.1",
        subfolder: str = "hunyuan3d-dit-v2-1",
        enable_flashvdm: bool = True,
        flashvdm_adaptive_kv: bool = True,
        flashvdm_topk_mode: str = "mean",
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        octree_resolution: int = 384,
        mc_algo: str = "dmc",
        generate_texture: bool = True,
        texture_max_num_view: int = 6,
        texture_resolution: int = 512,
        texture_size: int = 4096,
        realesrgan_ckpt: str = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth",
        device: str = "cuda",
        dtype: str = "float16",
        low_vram_mode: bool = True,
    ):
        """Initialize Hunyuan3D generator.

        Args:
            model_path: HuggingFace model path or local path
            subfolder: Subfolder for the DiT model weights
            enable_flashvdm: Enable FlashVDM for fast volume decoding
            flashvdm_adaptive_kv: Use adaptive KV selection in cross-attention
            flashvdm_topk_mode: Top-K mode: 'mean' or 'merge'
            num_inference_steps: Flow-matching denoising steps (50 standard, 5 turbo)
            guidance_scale: Classifier-free guidance scale
            octree_resolution: Mesh resolution (higher = finer)
            mc_algo: Marching cubes algorithm ('mc' or 'dmc')
            generate_texture: Whether to run texture generation stage
            texture_max_num_view: Number of views for texture generation
            texture_resolution: Resolution per view for texturing
            texture_size: Output texture atlas size
            realesrgan_ckpt: Path to RealESRGAN checkpoint for super-resolution
            device: Device to run on
            dtype: Data type ('float16', 'bfloat16', 'float32')
        """
        self.model_path = model_path
        self.subfolder = subfolder
        self.enable_flashvdm = enable_flashvdm
        self.flashvdm_adaptive_kv = flashvdm_adaptive_kv
        self.flashvdm_topk_mode = flashvdm_topk_mode
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.octree_resolution = octree_resolution
        self.mc_algo = mc_algo
        self.generate_texture = generate_texture
        self.texture_max_num_view = texture_max_num_view
        self.texture_resolution = texture_resolution
        self.texture_size = texture_size
        self.realesrgan_ckpt = realesrgan_ckpt
        self.device = device
        self.dtype = dtype
        self.low_vram_mode = low_vram_mode

        # In low VRAM mode, apply conservative defaults that reduce peak memory
        if self.low_vram_mode:
            if self.octree_resolution > 256:
                logger.info(
                    f"Low VRAM: reducing octree_resolution "
                    f"{self.octree_resolution}→256 to save GPU memory"
                )
                self.octree_resolution = 256
            if self.num_inference_steps > 30:
                logger.info(
                    f"Low VRAM: reducing num_inference_steps "
                    f"{self.num_inference_steps}→30 to save GPU memory"
                )
                self.num_inference_steps = 30
            if self.generate_texture:
                logger.info(
                    "Low VRAM: disabling texture generation by default "
                    "(re-enable with --texture or low_vram_mode=False)"
                )
                self.generate_texture = False

        self.shape_pipeline = None
        self.paint_pipeline = None
        self.rembg = None

    def _setup_paths(self):
        """Add Hunyuan3D-2.1 directories to sys.path.

        Hunyuan3D-2.1 is NOT pip-installable. It uses sys.path manipulation:
        - ./hy3dshape must be on sys.path for `from hy3dshape.pipelines import ...`
        - ./hy3dpaint must be on sys.path for `from textureGenPipeline import ...`
        The hy3dshape dir contains a nested package: hy3dshape/hy3dshape/
        """
        # Module is at flares/scene_recon3d/modules/ → 2 levels up to flares/
        repo_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "repos", "Hunyuan3D-2.1"
        )
        repo_path = os.path.abspath(repo_path)

        # Check common locations: ./repos/Hunyuan3D-2.1 and ./repos/Hunyuan3D-2.1
        # Also support FLARES_REPO_DIR env var for custom locations
        env_repo = os.environ.get("FLARES_REPO_DIR", "")
        search_paths = [repo_path]
        if env_repo:
            search_paths.insert(0, os.path.join(env_repo, "Hunyuan3D-2.1"))

        for candidate in search_paths:
            hy3dshape_dir = os.path.join(candidate, "hy3dshape")
            hy3dpaint_dir = os.path.join(candidate, "hy3dpaint")
            if os.path.isdir(os.path.join(hy3dshape_dir, "hy3dshape")):
                # Found the nested package structure: hy3dshape/hy3dshape/
                if hy3dshape_dir not in sys.path:
                    sys.path.insert(0, hy3dshape_dir)
                    logger.info(f"Added to sys.path: {hy3dshape_dir}")
                if hy3dpaint_dir not in sys.path:
                    sys.path.insert(0, hy3dpaint_dir)
                    logger.info(f"Added to sys.path: {hy3dpaint_dir}")
                return True

        return False

    def _ensure_model_downloaded(self):
        """Ensure Hunyuan3D model weights are fully downloaded.

        Hunyuan3D's smart_load_model() only checks if the model *directory* exists,
        not whether the actual checkpoint file is present inside it. If a partial
        download left an empty/incomplete directory, the download is skipped and
        from_single_file() raises FileNotFoundError.

        This method pre-downloads the model using huggingface_hub.snapshot_download
        to guarantee all required files are present before from_pretrained() is called.
        """
        from huggingface_hub import snapshot_download

        base_dir = os.environ.get("HY3DGEN_MODELS", os.path.expanduser("~/.cache/hy3dgen"))
        model_dir = os.path.expanduser(os.path.join(base_dir, self.model_path, self.subfolder))

        # Check if the checkpoint file actually exists (not just the directory)
        ckpt_path = os.path.join(model_dir, "model.fp16.ckpt")
        config_path = os.path.join(model_dir, "config.yaml")

        if os.path.isfile(ckpt_path) and os.path.isfile(config_path):
            logger.info(f"Hunyuan3D model already cached at {model_dir}")
            return

        logger.info(f"Downloading Hunyuan3D model weights from HuggingFace ({self.model_path})...")
        logger.info("This may take several minutes on first run (~5GB for shape model).")

        try:
            local_dir = os.path.expanduser(os.path.join(base_dir, self.model_path))
            snapshot_download(
                repo_id=self.model_path,
                allow_patterns=[f"{self.subfolder}/*"],
                local_dir=local_dir,
            )
            logger.info("Hunyuan3D model weights downloaded successfully")
        except Exception as e:
            logger.warning(
                f"Hunyuan3D model auto-download failed: {e}. "
                "Will try from_pretrained() which may also attempt the download."
            )

    def load_model(self):
        """Load the Hunyuan3D-2.1 shape and texture pipelines.

        Downloads pretrained weights from HuggingFace on first use.
        Weights are cached at ~/.cache/hy3dgen/.
        """
        # Setup sys.path for Hunyuan3D imports
        found = self._setup_paths()
        if not found:
            logger.warning(
                "Hunyuan3D-2.1 repo not found in expected locations. "
                "Set FLARES_REPO_DIR environment variable to the repos directory."
            )

        # Set environment variable for model cache
        os.environ.setdefault("HY3DGEN_MODELS", os.path.expanduser("~/.cache/hy3dgen"))

        # Pre-download model weights to avoid smart_load_model partial-cache bug
        try:
            self._ensure_model_downloaded()
        except Exception as e:
            logger.warning(f"Model pre-download check failed: {e}")

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.float16)

        # Load shape pipeline
        logger.info("Loading Hunyuan3D-2.1 shape pipeline...")
        try:
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        except ImportError as e:
            raise ImportError(
                "Failed to import Hunyuan3D-2.1. The repo uses sys.path, not pip install.\n"
                "Make sure the repos are cloned:\n"
                "  git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git repos/Hunyuan3D-2.1\n"
                "The pipeline auto-adds repos/Hunyuan3D-2.1/hy3dshape and hy3dpaint to sys.path.\n"
                f"Original error: {e}"
            )

        # Load shape pipeline — low_cpu_mem_usage=True loads weights in
        # shards directly to the target device, keeping system RAM ~1-2 GB
        # instead of peaking at 2× model size (~10 GB for fp32 staging).
        logger.info(
            f"Loading shape pipeline (low_cpu_mem_usage={self.low_vram_mode})..."
        )
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            self.model_path,
            subfolder=self.subfolder,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=self.low_vram_mode,
        )

        if pipeline is None:
            raise RuntimeError(
                "Hunyuan3DDiTFlowMatchingPipeline.from_pretrained() returned "
                "None. Check that the model weights are correctly downloaded "
                f"to {self.model_path}/{self.subfolder}."
            )

        # Move pipeline to the target device.  Some diffusers-style
        # pipelines return self from .to(), but others may return None
        # (e.g. when sub-modules are already on the device via
        # low_cpu_mem_usage).  Guard against None to avoid losing the
        # reference.
        try:
            moved = pipeline.to(self.device)
            if moved is not None:
                pipeline = moved
            else:
                logger.warning(
                    "pipeline.to(device) returned None; keeping original "
                    "pipeline reference (model may already be on device)"
                )
        except Exception as e:
            logger.warning(f"pipeline.to(device) failed: {e}. "
                           "Model may already be on the target device.")

        self.shape_pipeline = pipeline

        # Enable FlashVDM — only if the method exists on this pipeline
        # version.  Older or minimal installs may not ship FlashVDM.
        if self.enable_flashvdm:
            if hasattr(self.shape_pipeline, "enable_flashvdm"):
                logger.info("Enabling FlashVDM acceleration...")
                self.shape_pipeline.enable_flashvdm(
                    enabled=True,
                    adaptive_kv_selection=self.flashvdm_adaptive_kv,
                    topk_mode=self.flashvdm_topk_mode,
                    mc_algo=self.mc_algo,
                    replace_vae=True,
                )
            else:
                logger.warning(
                    "FlashVDM requested but shape_pipeline has no "
                    "enable_flashvdm() method. Continuing without "
                    "FlashVDM acceleration (mesh generation will be slower)."
                )

        # Load texture pipeline (optional — skipped in low_vram_mode unless
        # explicitly requested)
        if self.generate_texture:
            logger.info("Loading Hunyuan3D-2.1 texture pipeline...")
            try:
                from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

                # Find config and checkpoint paths
                repo_base = os.path.join(
                    os.path.dirname(__file__), "..", "..", "repos", "Hunyuan3D-2.1"
                )
                realesrgan_path = self.realesrgan_ckpt
                if not os.path.isabs(realesrgan_path):
                    realesrgan_path = os.path.join(repo_base, realesrgan_path)

                paint_cfg_path = os.path.join(repo_base, "hy3dpaint", "cfgs", "hunyuan-paint-pbr.yaml")
                custom_pipeline_path = os.path.join(repo_base, "hy3dpaint", "hunyuanpaintpbr")

                conf = Hunyuan3DPaintConfig(
                    max_num_view=self.texture_max_num_view,
                    resolution=self.texture_resolution,
                )
                conf.realesrgan_ckpt_path = realesrgan_path
                conf.multiview_cfg_path = paint_cfg_path
                conf.custom_pipeline = custom_pipeline_path

                self.paint_pipeline = Hunyuan3DPaintPipeline(conf)
                logger.info("Texture pipeline loaded successfully")

            except Exception as e:
                logger.warning(
                    f"Failed to load texture pipeline: {e}. "
                    "Will generate untextured meshes only."
                )
                self.paint_pipeline = None

        # Load background remover (optional — the pipeline already produces
        # RGBA images with transparent backgrounds from segmentation masks,
        # so rembg is only needed for images that lack an alpha channel.)
        #
        # IMPORTANT: rembg depends on onnxruntime.  When onnxruntime is not
        # installed, rembg's __init__.py prints a warning and calls
        # sys.exit(1), which kills the entire process.  We must check for
        # onnxruntime *before* importing anything that depends on rembg.
        self.rembg = None
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            logger.info(
                "onnxruntime not installed — BackgroundRemover disabled. "
                "Pipeline will use segmentation masks for background removal "
                "instead. To enable rembg: pip install onnxruntime"
            )
        else:
            try:
                from hy3dshape.rembg import BackgroundRemover
                self.rembg = BackgroundRemover()
            except (ImportError, OSError, RuntimeError) as e:
                logger.info(
                    f"BackgroundRemover not available ({type(e).__name__}: {e}). "
                    "Pipeline will use segmentation masks instead."
                )
                self.rembg = None

        # After loading, reclaim any temporary CPU memory from deserialization
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Hunyuan3D-2.1 model loaded successfully")

    def prepare_object_image(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        mask: np.ndarray,
        padding: int = 20,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        min_size: int = 64,
    ) -> Tuple[Image.Image, np.ndarray, Tuple[int, int]]:
        """Prepare an object crop image for Hunyuan3D.

        Crops the object from the scene using its bbox and mask,
        applies the mask to remove background, and converts to
        RGBA PIL Image (transparent background).

        Args:
            image: (H, W, 3) uint8 RGB scene image
            bbox: (4,) xyxy bounding box
            mask: (H, W) boolean segmentation mask
            padding: Padding around bounding box
            background_color: Color for masked-out regions (used if no alpha)
            min_size: Minimum crop dimension

        Returns:
            Tuple of (PIL RGBA image, crop_mask (H', W'), crop_offset (x1, y1))
        """
        # Crop with mask
        crop, crop_mask, crop_offset = crop_image_with_mask(
            image, bbox, mask, padding=padding, background_color=background_color
        )

        # Create RGBA image with transparent background
        h, w = crop.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = crop
        rgba[:, :, 3] = (crop_mask * 255).astype(np.uint8)

        # Ensure minimum size
        if min(h, w) < min_size:
            scale = min_size / min(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            rgba = np.array(
                Image.fromarray(rgba).resize((new_w, new_h), Image.LANCZOS)
            )

        pil_image = Image.fromarray(rgba, mode="RGBA")
        return pil_image, crop_mask, crop_offset

    def unload_model(self):
        """Unload all Hunyuan3D pipelines from GPU / CPU memory.

        Frees both the shape pipeline and the (optional) paint pipeline,
        then forces Python garbage-collection and CUDA cache clear.
        """
        import gc

        if self.shape_pipeline is not None:
            logger.info("Unloading Hunyuan3D shape pipeline...")
            # Move to CPU first to free GPU allocations before deleting
            try:
                self.shape_pipeline = self.shape_pipeline.to("cpu")
            except Exception:
                pass
            del self.shape_pipeline
            self.shape_pipeline = None

        if self.paint_pipeline is not None:
            logger.info("Unloading Hunyuan3D paint pipeline...")
            try:
                del self.paint_pipeline
            except Exception:
                pass
            self.paint_pipeline = None

        if self.rembg is not None:
            try:
                del self.rembg
            except Exception:
                pass
            self.rembg = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        logger.info("Hunyuan3D models unloaded, GPU cache cleared")

    def generate_mesh(
        self,
        image: Image.Image,
        output_path: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        octree_resolution: Optional[int] = None,
    ) -> trimesh.Trimesh:
        """Generate a 3D mesh from a masked object image.

        Args:
            image: PIL RGBA image with transparent background
            output_path: Optional path to save the mesh
            num_inference_steps: Override number of denoising steps
            guidance_scale: Override guidance scale
            octree_resolution: Override mesh resolution

        Returns:
            trimesh.Trimesh object with the generated 3D mesh
        """
        if self.shape_pipeline is None:
            self.load_model()

        steps = num_inference_steps or self.num_inference_steps
        gs = guidance_scale or self.guidance_scale
        res = octree_resolution or self.octree_resolution

        # Ensure RGBA with transparent background
        if image.mode != "RGBA":
            if self.rembg is not None:
                try:
                    image = self.rembg(image)
                except Exception as e:
                    logger.warning(
                        f"rembg background removal failed ({e}); "
                        "falling back to simple RGBA conversion"
                    )
                    image = image.convert("RGBA")
            else:
                image = image.convert("RGBA")

        logger.info(f"Generating 3D mesh (steps={steps}, guidance={gs}, resolution={res})...")

        with torch.no_grad():
            mesh = self.shape_pipeline(
                image=image,
                num_inference_steps=steps,
                guidance_scale=gs,
                octree_resolution=res,
                num_chunks=8000,
                box_v=1.01,
                output_type="trimesh",
            )[0]

        # Post-process: remove floaters and degenerate faces
        try:
            from hy3dshape import FaceReducer, FloaterRemover, DegenerateFaceRemover
            mesh = FloaterRemover()(mesh)
            mesh = DegenerateFaceRemover()(mesh)
            mesh = FaceReducer()(mesh, max_facenum=10000)
        except ImportError:
            logger.debug("Post-processing tools not available; skipping")

        # Apply texture if paint pipeline is available
        if self.paint_pipeline is not None and output_path:
            logger.info("Generating PBR textures...")
            try:
                # Save untextured mesh temporarily
                untextured_path = output_path.replace(".glb", "_untextured.glb")
                mesh.export(untextured_path)

                # Generate textured mesh
                textured_path = self.paint_pipeline(
                    mesh_path=untextured_path,
                    image_path=None,  # Will use the same image
                    output_mesh_path=output_path,
                    save_glb=True,
                )

                # Reload the textured mesh
                loaded = trimesh.load(output_path)
                # trimesh.load may return a Scene for multi-geometry GLBs;
                # convert to a single Trimesh for consistent downstream use.
                if isinstance(loaded, trimesh.Scene):
                    mesh = loaded.to_mesh()
                else:
                    mesh = loaded

                # Clean up temp file
                if os.path.exists(untextured_path):
                    os.remove(untextured_path)

            except Exception as e:
                logger.warning(f"Texture generation failed: {e}. Using untextured mesh.")
                # Save untextured mesh as fallback (the paint pipeline's
                # output path was never written, so we must save here)
                mesh.export(output_path)

        # Save if path provided
        if output_path and not (self.paint_pipeline is not None):
            mesh.export(output_path)

        n_verts = len(mesh.vertices)
        n_faces = len(mesh.faces)
        logger.info(f"Generated mesh: {n_verts} vertices, {n_faces} faces")

        return mesh

    def generate_meshes_for_objects(
        self,
        scene_image: np.ndarray,
        objects: List[DetectedObject],
        output_dir: str = "output/meshes",
        padding: int = 20,
    ) -> List[DetectedObject]:
        """Generate 3D meshes for all detected objects.

        For each object, crops the image using the mask, generates a 3D mesh,
        and stores it in the DetectedObject.

        Args:
            scene_image: (H, W, 3) uint8 RGB scene image
            objects: List of DetectedObject with 2D bboxes and masks
            output_dir: Directory to save mesh files
            padding: Padding around object crops

        Returns:
            Updated list of DetectedObject with mesh fields populated
        """
        os.makedirs(output_dir, exist_ok=True)

        for obj in objects:
            logger.info(
                f"Generating mesh for object {obj.object_id} ({obj.class_name})..."
            )

            # Prepare the object image
            pil_image, crop_mask, crop_offset = self.prepare_object_image(
                scene_image,
                obj.bbox_2d,
                obj.mask_2d,
                padding=padding,
            )

            # Store the crop info
            obj.crop_image = np.array(pil_image)[:, :, :3]
            obj.crop_mask = crop_mask
            obj.crop_offset = crop_offset

            # Generate mesh
            output_path = os.path.join(
                output_dir, f"{obj.class_name}_{obj.object_id}.glb"
            )

            try:
                mesh = self.generate_mesh(
                    image=pil_image,
                    output_path=output_path,
                )
                obj.mesh = mesh
                obj.mesh_path = output_path

                logger.info(
                    f"Object {obj.object_id} mesh generated: "
                    f"{len(mesh.vertices)} verts, {len(mesh.faces)} faces"
                )
            except Exception as e:
                logger.error(
                    f"Mesh generation failed for object {obj.object_id}: {e}"
                )

        return objects
