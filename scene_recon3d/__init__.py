"""
3D Scene Reconstruction Pipeline
=================================
A complete pipeline that combines:
  1. RF-DETR    — 2D object detection + instance segmentation
  2. WildDet3D  — Monocular 3D bounding box estimation
  3. Hunyuan3D-2.1 + FlashVDM — Image-to-3D mesh generation
  4. MARCO      — Semantic correspondence for pose refinement

Pipeline flow:
  Image → RF-DETR (2D boxes + masks) → WildDet3D (3D boxes)
        → Hunyuan3D (per-object 3D meshes) → Scale/Pose alignment
        → MARCO (pose refinement) → Final 3D scene
"""

__version__ = "1.0.0"
