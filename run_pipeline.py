#!/usr/bin/env python3
"""Main entry point for the 3D Scene Reconstruction Pipeline.

Usage:
    python run_pipeline.py --image scene.jpg [--config configs/pipeline_config.yaml]
    python run_pipeline.py --image scene.jpg --intrinsics K.npy --output output/scene
"""

import argparse
import logging
import os
import sys

import numpy as np


def setup_logging(level: str = "INFO"):
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="3D Scene Reconstruction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Stages:
  1. RF-DETR     — 2D object detection + instance segmentation
  2. WildDet3D   — 3D bounding box estimation from 2D boxes
  3. Hunyuan3D   — Per-object 3D mesh generation
  4. Alignment   — Scale and pose alignment using 3D bounding boxes
  5. MARCO       — Pose refinement via semantic correspondence

Examples:
  # Basic usage with auto-downloaded models
  python run_pipeline.py --image scene.jpg

  # With camera intrinsics and custom config
  python run_pipeline.py --image scene.jpg --intrinsics K.npy --config configs/pipeline_config.yaml

  # Quick test with smaller models
  python run_pipeline.py --image scene.jpg --rfdetr-variant seg_nano --no-flashvdm --steps 5
        """,
    )

    # Required arguments
    parser.add_argument(
        "--image", type=str, required=True,
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
    parser.add_argument("--refinement-iters", type=int, default=None, help="MARCO refinement iterations")
    parser.add_argument("--preload", action="store_true", help="Pre-load all models before running")

    # Logging
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # ─── Build Pipeline ───────────────────────────────────────
    from scene_recon3d.pipeline import SceneReconstructionPipeline

    if args.config:
        logger.info(f"Loading config from {args.config}")
        pipeline = SceneReconstructionPipeline.from_config(args.config)
    else:
        pipeline = SceneReconstructionPipeline(
            device=args.device,
            output_dir=args.output,
        )

    # Apply CLI overrides
    if args.rfdetr_variant:
        pipeline.detector.variant = args.rfdetr_variant
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
    if args.refinement_iters is not None:
        pipeline.refiner.refinement_iterations = args.refinement_iters

    # Pre-load models if requested
    if args.preload:
        pipeline.load_all_models()

    # ─── Load Inputs ──────────────────────────────────────────
    from PIL import Image

    logger.info(f"Loading image: {args.image}")
    image = np.array(Image.open(args.image).convert("RGB"))

    intrinsics = None
    if args.intrinsics:
        logger.info(f"Loading intrinsics: {args.intrinsics}")
        intrinsics = np.load(args.intrinsics)

    # ─── Run Pipeline ─────────────────────────────────────────
    logger.info("Starting 3D scene reconstruction pipeline...")
    result = pipeline.reconstruct(
        image=image,
        intrinsics=intrinsics,
        output_dir=args.output,
    )

    # ─── Report Results ───────────────────────────────────────
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
