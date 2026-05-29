#!/usr/bin/env python3
"""Demo script for the 3D Scene Reconstruction Pipeline.

This script demonstrates the pipeline with a sample image,
or can be used to test individual stages.

Usage:
    # Full pipeline demo
    python scripts/demo.py --image scene.jpg

    # Test individual stages
    python scripts/demo.py --image scene.jpg --stage detect
    python scripts/demo.py --image scene.jpg --stage detect3d
    python scripts/demo.py --image scene.jpg --stage generate
    python scripts/demo.py --image scene.jpg --stage refine
"""

import argparse
import logging
import os
import sys

import numpy as np
from PIL import Image

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def demo_detection(image: np.ndarray, args):
    """Demo Stage 1: RF-DETR detection."""
    from scene_recon3d.modules.rfdetr_detector import RFDETRDetector

    detector = RFDETRDetector(
        variant=args.rfdetr_variant,
        confidence_threshold=args.confidence,
    )
    detector.load_model()

    objects = detector.detect(image, min_area=args.min_area)

    print(f"\nDetected {len(objects)} objects:")
    for obj in objects:
        area = obj.mask_2d.sum()
        print(f"  [{obj.object_id}] {obj.class_name} "
              f"(confidence={obj.confidence:.3f}, mask_area={area}px)")

    # Visualize
    vis = detector.visualize(image, objects)
    output_path = os.path.join(args.output, "demo_detections.png")
    import cv2
    cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"\nDetection visualization saved to: {output_path}")

    return objects


def demo_3d_estimation(image: np.ndarray, objects, args):
    """Demo Stage 2: WildDet3D 3D bounding box estimation."""
    from scene_recon3d.modules.wilddet3d_estimator import WildDet3DEstimator

    estimator = WildDet3DEstimator(
        checkpoint=args.wilddet3d_checkpoint,
        use_predicted_intrinsics=True,
    )
    estimator.load_model()

    objects = estimator.estimate_3d(image, objects)

    print(f"\n3D Bounding Boxes:")
    for obj in objects:
        if obj.bbox_3d is not None:
            print(f"  [{obj.object_id}] {obj.class_name}: "
                  f"center={obj.bbox_3d_center}, "
                  f"dims={obj.bbox_3d_dims}, "
                  f"score_3d={obj.score_3d:.3f}")
        else:
            print(f"  [{obj.object_id}] {obj.class_name}: No 3D box")

    intrinsics = estimator.get_intrinsics()
    return objects, intrinsics


def demo_mesh_generation(image: np.ndarray, objects, args):
    """Demo Stage 3: Hunyuan3D mesh generation."""
    from scene_recon3d.modules.hunyuan3d_generator import Hunyuan3DGenerator

    generator = Hunyuan3DGenerator(
        enable_flashvdm=not args.no_flashvdm,
        num_inference_steps=args.steps,
        octree_resolution=args.octree_resolution,
        generate_texture=not args.no_texture,
    )
    generator.load_model()

    mesh_dir = os.path.join(args.output, "demo_meshes")
    objects = generator.generate_meshes_for_objects(
        image, objects, output_dir=mesh_dir
    )

    print(f"\nGenerated meshes:")
    for obj in objects:
        if obj.mesh is not None:
            print(f"  [{obj.object_id}] {obj.class_name}: "
                  f"{len(obj.mesh.vertices)} vertices, {len(obj.mesh.faces)} faces")
        else:
            print(f"  [{obj.object_id}] {obj.class_name}: No mesh")

    return objects


def demo_pose_refinement(objects, intrinsics, args):
    """Demo Stage 5: MARCO pose refinement."""
    from scene_recon3d.modules.marco_refiner import MARCORefiner
    from scene_recon3d.utils.geometry import align_mesh_to_bbox

    # First align meshes to 3D bounding boxes
    print("\nAligning meshes to 3D bounding boxes...")
    for obj in objects:
        if obj.mesh is not None and obj.bbox_3d is not None:
            aligned_mesh, scale_factor, rotation, translation = align_mesh_to_bbox(
                obj.mesh,
                obj.bbox_3d_center,
                obj.bbox_3d_dims,
                obj.bbox_3d_quat,
            )
            obj.aligned_mesh = aligned_mesh
            obj.scale_factor = scale_factor
            obj.initial_rotation = rotation
            obj.initial_translation = translation
            print(f"  [{obj.object_id}] {obj.class_name}: scale={scale_factor:.4f}")

    # Then refine with MARCO
    refiner = MARCORefiner(
        num_keypoints_per_object=args.num_keypoints,
        refinement_iterations=args.refinement_iters,
    )
    refiner.load_model()

    objects = refiner.refine_poses(
        objects, intrinsics, render_resolution=args.render_resolution
    )

    print(f"\nRefined poses:")
    for obj in objects:
        if obj.refined_rotation is not None:
            rot_diff = np.linalg.norm(
                obj.refined_rotation - (obj.initial_rotation or np.eye(3))
            )
            trans_diff = np.linalg.norm(
                obj.refined_translation - (obj.initial_translation or np.zeros(3))
            )
            print(f"  [{obj.object_id}] {obj.class_name}: "
                  f"rot_diff={rot_diff:.4f}, trans_diff={trans_diff:.4f}")
        else:
            print(f"  [{obj.object_id}] {obj.class_name}: Not refined")

    return objects


def demo_full_pipeline(image: np.ndarray, args):
    """Demo the full pipeline."""
    from scene_recon3d.pipeline import SceneReconstructionPipeline

    pipeline = SceneReconstructionPipeline(
        rfdetr_variant=args.rfdetr_variant,
        rfdetr_confidence=args.confidence,
        wilddet3d_checkpoint=args.wilddet3d_checkpoint,
        hunyuan3d_enable_flashvdm=not args.no_flashvdm,
        hunyuan3d_num_steps=args.steps,
        hunyuan3d_octree_resolution=args.octree_resolution,
        hunyuan3d_generate_texture=not args.no_texture,
        marco_refinement_iterations=args.refinement_iters,
        output_dir=args.output,
        min_object_area=args.min_area,
        render_resolution=args.render_resolution,
    )

    if args.preload:
        pipeline.load_all_models()

    result = pipeline.reconstruct(image=image)

    print(f"\n{'=' * 60}")
    print(f"Full Pipeline Results")
    print(f"{'=' * 60}")
    print(f"Objects reconstructed: {len(result.objects)}")
    for obj in result.objects:
        print(f"  [{obj.object_id}] {obj.class_name}: "
              f"{len(obj.mesh.vertices)} vertices, "
              f"confidence={obj.confidence:.3f}")

    stats = pipeline.get_stats()
    print(f"\nTotal time: {stats.total_time:.2f}s")
    for stage, t in stats.stage_times.items():
        print(f"  {stage}: {t:.2f}s")

    print(f"\nScene exported to: {os.path.join(args.output, 'scene.glb')}")


def main():
    parser = argparse.ArgumentParser(description="3D Scene Reconstruction Demo")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, default="output/demo", help="Output directory")
    parser.add_argument("--stage", type=str, default="full",
                       choices=["full", "detect", "detect3d", "generate", "refine"],
                       help="Which pipeline stage to demo")

    # Model settings
    parser.add_argument("--rfdetr-variant", type=str, default="seg_medium")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--wilddet3d-checkpoint", type=str,
                       default="ckpt/wilddet3d_alldata_all_prompt_v1.0.pt")
    parser.add_argument("--no-flashvdm", action="store_true")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--octree-resolution", type=int, default=384)
    parser.add_argument("--no-texture", action="store_true")
    parser.add_argument("--num-keypoints", type=int, default=20)
    parser.add_argument("--refinement-iters", type=int, default=3)
    parser.add_argument("--render-resolution", type=int, default=512)
    parser.add_argument("--min-area", type=int, default=100)
    parser.add_argument("--preload", action="store_true")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Load image
    image = np.array(Image.open(args.image).convert("RGB"))
    print(f"Image loaded: {image.shape[1]}×{image.shape[0]}")

    # Run requested stage
    if args.stage == "full":
        demo_full_pipeline(image, args)
    elif args.stage == "detect":
        demo_detection(image, args)
    elif args.stage == "detect3d":
        objects = demo_detection(image, args)
        demo_3d_estimation(image, objects, args)
    elif args.stage == "generate":
        objects = demo_detection(image, args)
        demo_mesh_generation(image, objects, args)
    elif args.stage == "refine":
        objects = demo_detection(image, args)
        objects, intrinsics = demo_3d_estimation(image, objects, args)
        objects = demo_mesh_generation(image, objects, args)
        demo_pose_refinement(objects, intrinsics, args)


if __name__ == "__main__":
    main()
