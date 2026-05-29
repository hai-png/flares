# 3D Scene Reconstruction Pipeline

A complete pipeline that reconstructs a 3D scene from a single RGB image by combining state-of-the-art models for detection, 3D estimation, mesh generation, and pose refinement.

## Architecture Overview

```
Input Image
     │
     ▼
┌──────────────┐     2D Boxes + Masks
│   RF-DETR    │─────────────────────────┐
│ (Detection + │                         │
│  Segmentation)│                        │
└──────────────┘                         │
     │                                   │
     │ 2D Boxes                          │ Masks
     ▼                                   ▼
┌──────────────┐               ┌──────────────────┐
│  WildDet3D   │               │  Hunyuan3D-2.1   │
│  (3D BBox    │               │  + FlashVDM      │
│  Estimation) │               │  (3D Mesh Gen)   │
└──────┬───────┘               └────────┬─────────┘
       │                                │
       │ 3D BBox (scale, pose)          │ Canonical Mesh
       │                                │
       └────────────┬───────────────────┘
                    │
                    ▼
          ┌──────────────────┐
          │  Scale + Pose    │
          │  Alignment       │
          │  (3D BBox → Mesh)│
          └────────┬─────────┘
                   │
                   │ Aligned Mesh + Crop Image
                   ▼
          ┌──────────────────┐
          │     MARCO        │
          │  (Semantic       │
          │  Correspondence  │
          │  Pose Refinement)│
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │  Final 3D Scene  │
          │  Reconstruction  │
          └──────────────────┘
```

## Pipeline Stages

### Stage 1: RF-DETR — 2D Object Detection + Instance Segmentation
- **Model**: RF-DETR-Seg (DINOv2 ViT backbone + Transformer decoder + Segmentation head)
- **Input**: Single RGB image
- **Output**: 2D bounding boxes (xyxy) + instance masks per object
- **Key feature**: Real-time detection with segmentation masks using windowed DINOv2 attention

### Stage 2: WildDet3D — 3D Bounding Box Estimation
- **Model**: WildDet3D (SAM3 + LingBot-Depth + 3D regression head)
- **Input**: Image + 2D bounding boxes (geometric prompt mode)
- **Output**: 3D bounding boxes in camera coordinates: (cx, cy, cz, w, l, h, qw, qx, qy, qz)
- **Key feature**: Predicts metric 3D boxes with camera intrinsics prediction for in-the-wild images

### Stage 3: Hunyuan3D-2.1 + FlashVDM — 3D Mesh Generation
- **Model**: Hunyuan3D-2.1 (3.3B DiT + ShapeVAE) with FlashVDM acceleration
- **Input**: Masked object crop image
- **Output**: High-fidelity textured 3D mesh (trimesh.Trimesh)
- **Key feature**: FlashVDM provides >45× faster volume decoding via sparse top-K cross-attention + hierarchical octree

### Stage 4: Scale & Pose Alignment
- **Method**: 3D bounding box guided alignment
- **Input**: Canonical mesh + 3D bounding box (center, dims, rotation)
- **Output**: Mesh transformed to fit the 3D bounding box in scene space
- **Key feature**: Scales and rotates canonically-posed meshes using 3D bbox dimensions and quaternion

### Stage 5: MARCO — Pose Refinement
- **Model**: MARCO (DINOv2 + AdaptFormer + Upsampling head)
- **Input**: Cropped object image + rendered mesh image
- **Output**: Semantic correspondence points → refined 6-DoF pose
- **Key feature**: 3× smaller and 10× faster than diffusion-based correspondence methods; generalizes to unseen categories

## Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.5+ with CUDA 12.x
- NVIDIA GPU with ≥16GB VRAM (≥24GB recommended for texture generation)

### Quick Setup
```bash
# Clone the pipeline repository
git clone <this-repo>
cd scene_recon3d

# Run the setup script (installs all dependencies + downloads weights)
bash scripts/setup.sh

# Or skip downloads if you have limited bandwidth
bash scripts/setup.sh --skip-download
```

### Manual Installation

```bash
# 1. Install core dependencies
pip install omegaconf opencv-python pillow scipy trimesh einops tqdm \
    huggingface_hub safetensors supervision pyrender

# 2. Install RF-DETR
cd repos/rf-detr && pip install -e .

# 3. Install Hunyuan3D-2.1
cd repos/Hunyuan3D-2.1/hy3dshape && pip install -e .
cd repos/Hunyuan3D-2.1/hy3dpaint && pip install -e .

# 4. Install WildDet3D
pip install vis4d==1.0.0
pip install git+https://github.com/SysCV/vis4d_cuda_ops.git --no-build-isolation

# 5. Install MARCO dependencies
pip install timm pandas mediapy h5py scikit-learn gdown

# 6. Download model weights
huggingface-cli download allenai/WildDet3D wilddet3d_alldata_all_prompt_v1.0.pt --local-dir ckpt/
# Hunyuan3D and RF-DETR weights auto-download on first use
# MARCO weights auto-download via torch.hub
```

## Usage

### Basic Usage
```bash
python run_pipeline.py --image scene.jpg --preload
```

### With Camera Intrinsics
```bash
python run_pipeline.py --image scene.jpg --intrinsics K.npy --output output/my_scene
```

### With Custom Config
```bash
python run_pipeline.py --image scene.jpg --config configs/pipeline_config.yaml
```

### Quick Test (Smaller Models)
```bash
python run_pipeline.py --image scene.jpg \
    --rfdetr-variant seg_nano \
    --no-flashvdm \
    --steps 5 \
    --no-texture
```

### Python API
```python
from scene_recon3d.pipeline import SceneReconstructionPipeline

# Create pipeline from config
pipeline = SceneReconstructionPipeline.from_config("configs/pipeline_config.yaml")

# Or create with explicit parameters
pipeline = SceneReconstructionPipeline(
    rfdetr_variant="seg_medium",
    hunyuan3d_enable_flashvdm=True,
    hunyuan3d_num_steps=50,
    marco_refinement_iterations=3,
)

# Pre-load all models
pipeline.load_all_models()

# Run reconstruction
import numpy as np
from PIL import Image

image = np.array(Image.open("scene.jpg").convert("RGB"))
result = pipeline.reconstruct(image=image)

# Access results
print(f"Reconstructed {len(result.objects)} objects")
for obj in result.objects:
    print(f"  {obj.class_name}: {len(obj.mesh.vertices)} vertices")

# Export scene
result.export_scene("output/scene.glb")
```

## Configuration

See `configs/pipeline_config.yaml` for all configurable options:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `rfdetr` | `model_variant` | `seg_medium` | RF-DETR variant (seg_nano to seg_2xlarge) |
| `rfdetr` | `confidence_threshold` | `0.5` | Detection confidence threshold |
| `wilddet3d` | `use_predicted_intrinsics` | `true` | Predict camera K for in-the-wild images |
| `wilddet3d` | `canonical_rotation` | `true` | Normalize 3D box rotation |
| `hunyuan3d` | `enable_flashvdm` | `true` | Enable FlashVDM acceleration |
| `hunyuan3d` | `num_inference_steps` | `50` | Denoising steps (5 for turbo) |
| `hunyuan3d` | `octree_resolution` | `384` | Mesh resolution |
| `hunyuan3d` | `generate_texture` | `true` | Generate PBR textures |
| `marco` | `num_keypoints_per_object` | `20` | Keypoints sampled per object |
| `marco` | `refinement_iterations` | `3` | Number of MARCO refinement iterations |
| `pipeline` | `min_object_area` | `100` | Min mask area to process an object |

## Output Structure

```
output/scene/
├── scene.glb              # Final combined 3D scene
├── 1_detections.png       # Stage 1: Detection visualization
├── meshes/                # Stage 3: Per-object meshes
│   ├── chair_0.glb
│   ├── table_1.glb
│   └── cup_2.glb
└── aligned/               # Stage 4: Aligned meshes (before refinement)
    ├── chair_0.glb
    ├── table_1.glb
    └── cup_2.glb
```

## Dependencies

| Component | Repository | License |
|-----------|-----------|---------|
| RF-DETR | [roboflow/rf-detr](https://github.com/roboflow/rf-detr) | Apache 2.0 |
| WildDet3D | [allenai/WildDet3D](https://github.com/allenai/WildDet3D) | Apache 2.0 |
| Hunyuan3D-2.1 | [Tencent-Hunyuan/Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1) | Tencent Hunyuan |
| FlashVDM | [Tencent-Hunyuan/FlashVDM](https://github.com/Tencent-Hunyuan/FlashVDM) | Tencent Hunyuan |
| MARCO | [visinf/MARCO](https://github.com/visinf/MARCO) | MIT |

## Troubleshooting

### CUDA Out of Memory
- Use `--no-texture` to skip the texture generation stage (saves ~21GB VRAM)
- Use `--rfdetr-variant seg_nano` for smaller detection model
- Use `--steps 5` with the turbo DiT model for faster generation
- Reduce `octree_resolution` to 256 or lower

### WildDet3D Installation Issues
- Requires `vis4d` and `vis4d_cuda_ops` which need CUDA build tools
- Use `--skip-cuda-ext` flag and install CUDA extensions separately

### MARCO torch.hub Download Fails
- Manually download from: https://github.com/visinf/MARCO/releases/download/v1.0/marco_release.pth
- Place in `ckpt/marco_release.pth` and set `marco.use_torch_hub: false` in config

## Citation

If you use this pipeline, please cite the original papers:

```bibtex
@article{rfdetr2024,
  title={RF-DETR: Neural Architecture Search for Real-Time Detection Transformers},
  journal={arXiv:2511.09554},
  year={2024}
}

@article{wilddet3d2025,
  title={WildDet3D: Promptable Monocular 3D Object Detection in the Wild},
  author={Allen AI},
  year={2025}
}

@article{hunyuan3d2025,
  title={Hunyuan3D 2.1: High-Fidelity 3D Generation with PBR},
  author={Tencent Hunyuan},
  year={2025}
}

@article{flashvdm2025,
  title={FlashVDM: Unleashing Vecset Diffusion Model for Fast Shape Generation},
  journal={ICCV 2025},
  year={2025}
}

@article{marco2025,
  title={MARCO: Navigating the Unseen Space of Semantic Correspondence},
  journal={CVPR 2026},
  year={2025}
}
```
