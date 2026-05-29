#!/usr/bin/env python3
"""Main entry point for the 3D Scene Reconstruction Pipeline.

Usage:
    python run_pipeline.py --image scene.jpg [--config configs/pipeline_config.yaml]
    python run_pipeline.py --image scene.jpg --intrinsics K.npy --output output/scene
    python run_pipeline.py --check-env
"""

import argparse
import logging
import os
import sys

# Ensure the repo root (which contains the scene_recon3d package) is on sys.path.
# When running `python run_pipeline.py` from the repo root, Python adds the
# script's directory to sys.path[0], which is the repo root itself.
# This means `from scene_recon3d.pipeline import ...` should work directly.
# But as a safety net, we also add the parent directory:
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

import numpy as np


def _check_numpy_compat():
    """Warn early if numpy>=2.0 is installed, which breaks the environment.

    Installing rembg[gpu] or onnxruntime-gpu can pull in numpy>=2.3,
    which triggers a library conflict: pymeshlab's bundled libcrypto.so.3
    needs OPENSSL_3.3.0, and the updated numpy/onnxruntime stack changes
    the shared-library loading order so that huggingface_hub (and thus
    RF-DETR, Hunyuan3D) fail to import.

    Fix: pip install "numpy>=1.24,<2.0"
    """
    major = int(np.__version__.split(".")[0])
    if major >= 2:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"NumPy {np.__version__} detected — this is likely incompatible. "
            f"Several pipeline components (RF-DETR, Hunyuan3D) depend on "
            f"pymeshlab which conflicts with numpy>=2.0 via OpenSSL linkage. "
            f"Run: pip install \"numpy>=1.24,<2.0\" to fix."
        )


_check_numpy_compat()


def _suppress_pymeshlab_warnings():
    """Import pymeshlab while suppressing noisy plugin-loading warnings.

    When libOpenGL.so.0 is missing, pymeshlab prints ~40 lines of warnings
    about plugins that cannot load. These plugins are optional and not used
    by our pipeline. Since pymeshlab prints via print() in its __init__.py,
    we redirect stdout during import.
    """
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        import pymeshlab  # noqa: F401
    except ImportError:
        pass
    finally:
        sys.stdout = old_stdout


_suppress_pymeshlab_warnings()


def setup_logging(level: str = "INFO"):
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def check_environment():
    """Check that all required dependencies are available."""
    import torch
    logger = logging.getLogger(__name__)

    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    import numpy as np
    logger.info(f"NumPy: {np.__version__}")

    # Check each pipeline component
    checks = {}

    try:
        import rfdetr
        checks["RF-DETR"] = True
    except ImportError as e:
        checks["RF-DETR"] = False
        logger.warning(f"RF-DETR not available: {e}")

    try:
        # Ensure paths are set up for repo-based imports
        from scene_recon3d.utils.setup_paths import setup_repo_paths
        setup_repo_paths()
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        checks["Hunyuan3D-Shape"] = True
    except ImportError as e:
        checks["Hunyuan3D-Shape"] = False
        logger.warning(f"Hunyuan3D-Shape not available: {e}")

    try:
        # Mock bpy if not installed — the paint pipeline imports it at the
        # top level but we don't call bpy-dependent functions (save_glb=False)
        if "bpy" not in sys.modules:
            import types
            sys.modules["bpy"] = types.ModuleType("bpy")
        from textureGenPipeline import Hunyuan3DPaintPipeline
        checks["Hunyuan3D-Paint"] = True
    except ImportError:
        checks["Hunyuan3D-Paint"] = False
        logger.info("Hunyuan3D-Paint not available (optional, texture generation will be skipped)")

    try:
        from wilddet3d import build_model
        checks["WildDet3D"] = True
    except ImportError as e:
        checks["WildDet3D"] = False
        logger.warning(f"WildDet3D not available: {e}")

    # Print summary
    logger.info("Component availability:")
    for name, available in checks.items():
        status = "OK" if available else "MISSING"
        logger.info(f"  {name}: {status}")

    return checks


def main():
    parser = argparse.ArgumentParser(
        description="3D Scene Reconstruction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Stages:
  1. RF-DETR     -- 2D object detection + instance segmentation
  2. WildDet3D   -- 3D bounding box estimation from 2D boxes
  3. Hunyuan3D   -- Per-object 3D mesh generation
  4. Alignment   -- Scale and pose alignment using 3D bounding boxes
  5. MARCO       -- Pose refinement via semantic correspondence

Examples:
  # Basic usage with auto-downloaded models
  python run_pipeline.py --image scene.jpg

  # With camera intrinsics and custom config
  python run_pipeline.py --image scene.jpg --intrinsics K.npy --config configs/pipeline_config.yaml

  # Quick test with smaller models
  python run_pipeline.py --image scene.jpg --rfdetr-variant seg_nano --no-flashvdm --steps 5

  # Check environment only
  python run_pipeline.py --check-env
        """,
    )

    # Required arguments
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to the input scene image"
    )

    # Optional arguments
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--intrinsics", type=str, default=None,
        help="Path to camera intrinsics matrix (.npy file, 3x3)"
    )
    parser.add_argument(
        "--output", type=str, default="output/scene",
        help="Output directory (default: output/scene)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Compute device (default: cuda)"
    )

    # Override individual settings
    parser.add_argument("--rfdetr-variant", type=str, default=None, help="RF-DETR variant")
    parser.add_argument("--confidence", type=float, default=None, help="Detection confidence threshold")
    parser.add_argument("--no-flashvdm", action="store_true", help="Disable FlashVDM")
    parser.add_argument("--steps", type=int, default=None, help="Hunyuan3D denoising steps")
    parser.add_argument("--guidance", type=float, default=None, help="Hunyuan3D guidance scale")
    parser.add_argument("--no-texture", action="store_true", help="Skip texture generation")
    parser.add_argument("--texture", action="store_true", help="Force-enable texture generation (even in low VRAM mode)")
    parser.add_argument("--refinement-iters", type=int, default=None, help="MARCO refinement iterations")
    parser.add_argument("--preload", action="store_true", help="Pre-load all models before running (ignored in low VRAM mode)")
    parser.add_argument("--no-low-vram", action="store_true", help="Disable low VRAM mode (keep all models in GPU, needs ~25GB)")
    parser.add_argument("--low-vram", action="store_true", help="Force low VRAM mode (auto-detected by default)")
    parser.add_argument("--check-env", action="store_true", help="Check environment and exit")

    # Logging
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Check environment
    checks = check_environment()

    if args.check_env:
        print("\nEnvironment check complete. Exiting.")
        return

    # Require --image unless --check-env
    if args.image is None:
        parser.error("--image is required (use --check-env to verify environment only)")

    # Verify critical components
    missing = [name for name, ok in checks.items() if not ok and name in ["RF-DETR", "Hunyuan3D-Shape", "WildDet3D"]]
    if missing:
        logger.error(
            f"Critical components missing: {missing}. "
            "Please run setup.sh first or install missing dependencies."
        )
        logger.error(
            "Tip: Set FLARES_REPO_DIR if repos are in a non-default location. "
            "E.g.: export FLARES_REPO_DIR=/content/flares/repos"
        )
        sys.exit(1)

    # --- Build Pipeline ---
    from scene_recon3d.pipeline import SceneReconstructionPipeline

    # Determine low VRAM mode from CLI flags
    if args.no_low_vram:
        low_vram_flag = False
    elif args.low_vram:
        low_vram_flag = True
    else:
        low_vram_flag = False  # Auto-detected inside Hunyuan3DGenerator based on GPU VRAM

    if args.config:
        logger.info(f"Loading config from {args.config}")
        pipeline = SceneReconstructionPipeline.from_config(args.config)
        if args.no_low_vram:
            pipeline.low_vram_mode = False
            pipeline.generator.low_vram_mode = False
        elif args.low_vram:
            pipeline.low_vram_mode = True
            pipeline.generator.low_vram_mode = True
    else:
        pipeline = SceneReconstructionPipeline(
            device=args.device,
            output_dir=args.output,
            low_vram_mode=low_vram_flag,  # Auto-detected if neither flag is set
        )

    # Apply CLI overrides
    if args.rfdetr_variant:
        pipeline.detector.variant = args.rfdetr_variant
        pipeline.detector.model = None  # Force reload with new variant
    if args.confidence is not None:
        pipeline.detector.confidence_threshold = args.confidence
    if args.no_flashvdm:
        pipeline.generator.enable_flashvdm = False
    if args.steps is not None:
        pipeline.generator.num_inference_steps = args.steps
    if args.guidance is not None:
        pipeline.generator.guidance_scale = args.guidance
    if args.no_texture:
        pipeline.generator.generate_texture = False
    elif args.texture:
        pipeline.generator.generate_texture = True
        logger.info("Texture generation force-enabled (may use additional GPU memory)")
    if args.refinement_iters is not None:
        pipeline.refiner.refinement_iterations = args.refinement_iters

    # Pre-load models if requested
    if args.preload:
        pipeline.load_all_models()

    # --- Load Inputs ---
    from PIL import Image

    logger.info(f"Loading image: {args.image}")
    image = np.array(Image.open(args.image).convert("RGB"))

    intrinsics = None
    if args.intrinsics:
        logger.info(f"Loading intrinsics: {args.intrinsics}")
        intrinsics = np.load(args.intrinsics)

    # --- Run Pipeline ---
    logger.info("Starting 3D scene reconstruction pipeline...")
    result = pipeline.reconstruct(
        image=image,
        intrinsics=intrinsics,
        output_dir=args.output,
    )

    # --- Report Results ---
    stats = pipeline.get_stats()
    logger.info("=" * 60)
    logger.info("Reconstruction Complete!")
    logger.info("=" * 60)
    logger.info(f"  Objects reconstructed: {len(result.objects)}")
    logger.info(f"  Total time: {stats.total_time:.2f}s")
    for stage, t in stats.stage_times.items():
        logger.info(f"    {stage}: {t:.2f}s")
    logger.info(f"  Output directory: {args.output}")
    logger.info(f"  Scene file: {os.path.join(args.output, 'scene.glb')}")


if __name__ == "__main__":
    main()
