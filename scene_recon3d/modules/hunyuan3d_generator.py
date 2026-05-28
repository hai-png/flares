"""Hunyuan3D-2.1 + FlashVDM Module: Image-to-3D Mesh Generation.

This module wraps Hunyuan3D-2.1 with FlashVDM acceleration to generate
high-fidelity textured 3D meshes from masked object images.

Hunyuan3D-2.1 is a two-stage pipeline:
  Stage 1: Shape Generation (3.3B DiT + VAE) -> untextured mesh
  Stage 2: Texture Generation (2B Paint pipeline) -> PBR-textured mesh

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

    Example:
        >>> generator = Hunyuan3DGenerator(enable_flashvdm=True)
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
        # Determine repos directory from multiple possible locations
        this_dir = os.path.dirname(os.path.abspath(__file__))
        package_dir = os.path.dirname(this_dir)      # scene_recon3d/
        project_dir = os.path.dirname(package_dir)    # flares/

        # Support FLARES_REPO_DIR env var for custom locations
        env_repo = os.environ.get("FLARES_REPO_DIR", "")
        search_bases = [project_dir]
        if env_repo:
            search_bases.insert(0, env_repo)

        for base in search_bases:
            candidate = os.path.join(base, "repos", "Hunyuan3D-2.1")
            hy3dshape_dir = os.path.join(candidate, "hy3dshape")
            hy3dpaint_dir = os.path.join(candidate, "hy3dpaint")

            # Check for nested package structure: hy3dshape/hy3dshape/
            if os.path.isdir(os.path.join(hy3dshape_dir, "hy3dshape")):
                if hy3dshape_dir not in sys.path:
                    sys.path.insert(0, hy3dshape_dir)
                    logger.info(f"Added to sys.path: {hy3dshape_dir}")
                if hy3dpaint_dir not in sys.path:
                    sys.path.insert(0, hy3dpaint_dir)
                    logger.info(f"Added to sys.path: {hy3dpaint_dir}")
                return True

            # Also check if the repo itself is the base (e.g. running inside the repo)
            if os.path.isdir(os.path.join(base, "hy3dshape", "hy3dshape")):
                hy3dshape_inner = os.path.join(base, "hy3dshape")
                hy3dpaint_inner = os.path.join(base, "hy3dpaint")
                if hy3dshape_inner not in sys.path:
                    sys.path.insert(0, hy3dshape_inner)
                    logger.info(f"Added to sys.path: {hy3dshape_inner}")
                if hy3dpaint_inner not in sys.path:
                    sys.path.insert(0, hy3dpaint_inner)
                    logger.info(f"Added to sys.path: {hy3dpaint_inner}")
                return True

        # Try to find it in the current working directory structure
        for candidate_dir in ["repos/Hunyuan3D-2.1", "Hunyuan3D-2.1"]:
            hy3dshape_dir = os.path.join(candidate_dir, "hy3dshape")
            hy3dpaint_dir = os.path.join(candidate_dir, "hy3dpaint")
            if os.path.isdir(os.path.join(hy3dshape_dir, "hy3dshape")):
                abs_hy3dshape = os.path.abspath(hy3dshape_dir)
                abs_hy3dpaint = os.path.abspath(hy3dpaint_dir)
                if abs_hy3dshape not in sys.path:
                    sys.path.insert(0, abs_hy3dshape)
                    logger.info(f"Added to sys.path: {abs_hy3dshape}")
                if abs_hy3dpaint not in sys.path:
                    sys.path.insert(0, abs_hy3dpaint)
                    logger.info(f"Added to sys.path: {abs_hy3dpaint}")
                return True

        return False

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
                f"Current sys.path entries with 'hy3d': {[p for p in sys.path if 'hy3d' in p.lower()]}\n"
                f"Original error: {e}"
            )

        self.shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            self.model_path,
            subfolder=self.subfolder,
            torch_dtype=torch_dtype,
        )

        # Enable FlashVDM
        if self.enable_flashvdm:
            logger.info("Enabling FlashVDM acceleration...")
            try:
                self.shape_pipeline.enable_flashvdm(
                    enabled=True,
                    adaptive_kv_selection=self.flashvdm_adaptive_kv,
                    topk_mode=self.flashvdm_topk_mode,
                    mc_algo=self.mc_algo,
                    replace_vae=True,
                )
            except Exception as e:
                logger.warning(
                    f"FlashVDM enable failed: {e}. Continuing without FlashVDM."
                )
                self.enable_flashvdm = False

        # Load texture pipeline (optional)
        if self.generate_texture:
            logger.info("Loading Hunyuan3D-2.1 texture pipeline...")
            try:
                from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

                # Find config and checkpoint paths
                # Search for the repo base directory
                repo_base = None
                for p in sys.path:
                    if "hy3dpaint" in p and os.path.isdir(p):
                        parent = os.path.dirname(p)
                        if os.path.isdir(os.path.join(parent, "hy3dshape")):
                            repo_base = parent
                            break

                if repo_base is None:
                    # Fallback: use relative path
                    repo_base = os.path.join(
                        os.path.dirname(__file__), "..", "..", "repos", "Hunyuan3D-2.1"
                    )
                    repo_base = os.path.abspath(repo_base)

                realesrgan_path = self.realesrgan_ckpt
                if not os.path.isabs(realesrgan_path):
                    realesrgan_path = os.path.join(repo_base, realesrgan_ckpt)

                paint_cfg_path = os.path.join(repo_base, "hy3dpaint", "cfgs", "hunyuan-paint-pbr.yaml")
                custom_pipeline_path = os.path.join(repo_base, "hy3dpaint", "hunyuanpaintpbr")

                if not os.path.exists(paint_cfg_path):
                    logger.warning(f"Paint config not found at {paint_cfg_path}, skipping texture pipeline")
                    self.paint_pipeline = None
                else:
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

        # Load background remover
        try:
            from hy3dshape.rembg import BackgroundRemover
            self.rembg = BackgroundRemover()
        except ImportError:
            logger.warning("BackgroundRemover not available; input images must have transparent backgrounds")
            self.rembg = None

        logger.info("Hunyuan3D-2.1 model loaded successfully")

    def prepare_object_image(
        self,
        image: np.ndarray,
        bbox: np.ndarray,
        mask: np.ndarray,
        padding: int = 20,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        min_size: int = 64,
    ) -> Tuple[Image.Image, np.ndarray]:
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
            Tuple of (PIL RGBA image, crop_mask (H', W'))
        """
        # Crop with mask
        crop, crop_mask = crop_image_with_mask(
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
        return pil_image, crop_mask

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
                image = self.rembg(image)
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
                mesh = trimesh.load(output_path)

                # Clean up temp file
                if os.path.exists(untextured_path):
                    os.remove(untextured_path)

            except Exception as e:
                logger.warning(f"Texture generation failed: {e}. Using untextured mesh.")

        # Save if path provided (and texture pipeline didn't already save)
        if output_path and self.paint_pipeline is None:
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
            pil_image, crop_mask = self.prepare_object_image(
                scene_image,
                obj.bbox_2d,
                obj.mask_2d,
                padding=padding,
            )

            # Store the crop info
            obj.crop_image = np.array(pil_image)[:, :, :3]
            obj.crop_mask = crop_mask

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
