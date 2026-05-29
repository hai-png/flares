#!/bin/bash
# =============================================================
# Setup Script for FLARES - 3D Scene Reconstruction Pipeline
# =============================================================
# This script installs all dependencies and prepares the environment
# for Google Colab or similar CUDA-equipped Linux environments.
#
# IMPORTANT: This script NEVER uses the repos' requirements.txt
# files because they contain conflicting version pins (e.g.,
# torch==2.5.1, numpy==1.24.4) that break other components.
# Instead, we install known-compatible versions directly.
#
# Project structure:
#   flares/                        <- repo root
#     scene_recon3d/               <- Python package
#       __init__.py
#       pipeline.py
#       modules/
#       utils/
#     run_pipeline.py              <- CLI entry point
#     setup.py                     <- pip install -e .
#     scripts/setup.sh             <- this script
#     repos/                       <- cloned repos (git-ignored)
#
# Usage:
#   bash scripts/setup.sh [--skip-download] [--skip-cuda-ext]
# =============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPOS_DIR="$PROJECT_DIR/repos"

echo "============================================================"
echo " FLARES - 3D Scene Reconstruction Pipeline - Setup"
echo "============================================================"
echo "Project dir: $PROJECT_DIR"
echo "Repos dir:   $REPOS_DIR"
echo ""

# Parse arguments
SKIP_DOWNLOAD=false
SKIP_CUDA_EXT=false
PYTHON_CMD="${PYTHON_CMD:-python3}"

for arg in "$@"; do
    case $arg in
        --skip-download)  SKIP_DOWNLOAD=true ;;
        --skip-cuda-ext)  SKIP_CUDA_EXT=true ;;
        --python=*)       PYTHON_CMD="${arg#*=}" ;;
        -h|--help)
            echo "Usage: bash setup.sh [--skip-download] [--skip-cuda-ext] [--python=python3]"
            exit 0
            ;;
    esac
done

# ─── 1. Check Prerequisites ────────────────────────────────────
echo "[1/8] Checking prerequisites..."

if ! command -v git &> /dev/null; then
    echo "ERROR: git is not installed"
    exit 1
fi

if ! $PYTHON_CMD -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch is not installed. Please install it first:"
    echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    exit 1
fi

CUDA_VERSION=$($PYTHON_CMD -c "import torch; print(torch.version.cuda or 'cpu')" 2>/dev/null)
TORCH_VERSION=$($PYTHON_CMD -c "import torch; print(torch.__version__)" 2>/dev/null)
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
echo "  PyTorch: $TORCH_VERSION"
echo "  PyTorch CUDA: $CUDA_VERSION"
echo "  Python: $PYTHON_VERSION"

# Record original torch version for later restore
SAVED_TORCH_VERSION="$TORCH_VERSION"
SAVED_TORCHVISION_VERSION=$($PYTHON_CMD -c "import torchvision; print(torchvision.__version__)" 2>/dev/null || echo "unknown")
echo "  Saved torch==$SAVED_TORCH_VERSION, torchvision==$SAVED_TORCHVISION_VERSION"

# ─── 2. Clone Repositories ─────────────────────────────────────
echo ""
echo "[2/8] Checking repositories..."

REPOS=(
    "https://github.com/visinf/MARCO.git"
    "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git"
    "https://github.com/Tencent-Hunyuan/FlashVDM.git"
    "https://github.com/roboflow/rf-detr.git"
    "https://github.com/allenai/WildDet3D.git"
)

mkdir -p "$REPOS_DIR"
for repo_url in "${REPOS[@]}"; do
    repo_name=$(basename "$repo_url" .git)
    if [ ! -d "$REPOS_DIR/$repo_name" ]; then
        echo "  Cloning $repo_name..."
        git clone --recurse-submodules "$repo_url" "$REPOS_DIR/$repo_name"
    else
        echo "  ✓ $repo_name already exists"
        if [ "$repo_name" = "WildDet3D" ]; then
            cd "$REPOS_DIR/WildDet3D"
            git submodule update --init --recursive 2>/dev/null || true
        fi
    fi
done

# ─── 3. Install Core Dependencies ─────────────────────────────
echo ""
echo "[3/8] Installing core Python dependencies..."

$PYTHON_CMD -m pip install --upgrade pip "setuptools<82" wheel

# Install core deps without strict version pins.
# These are shared across multiple pipeline components.
# Pin numpy<2.0 to avoid conflicts with pandas, scikit-learn, vis4d,
# and pymeshlab's OpenSSL linkage.  rembg>=2.0.50 requires numpy>=2.3,
# so we install rembg<2.0.50 which works with numpy 1.x.
# NEVER install rembg[gpu] — it pulls numpy>=2.3 and onnxruntime-gpu
# which break the pymeshlab → OpenSSL → huggingface_hub import chain.
$PYTHON_CMD -m pip install --quiet \
    "numpy>=1.24,<2.0" \
    omegaconf \
    pyyaml \
    opencv-python-headless \
    pillow \
    scipy \
    scikit-image \
    trimesh \
    einops \
    tqdm \
    huggingface_hub \
    safetensors \
    supervision \
    "rembg>=2.0,<2.0.50" \
    pygltflib

# ─── 4. Install the scene_recon3d package ──────────────────────
echo ""
echo "[4/8] Installing scene_recon3d package..."

cd "$PROJECT_DIR"
$PYTHON_CMD -m pip install -e . --no-deps --quiet 2>/dev/null || \
    echo "  ⚠ pip install -e . had issues (non-critical if running from repo root)"

if $PYTHON_CMD -c "from scene_recon3d.pipeline import SceneReconstructionPipeline; print('OK')" 2>/dev/null; then
    echo "  ✓ scene_recon3d package importable"
else
    echo "  ⚠ scene_recon3d package not importable — make sure you run from the repo root"
fi

# ─── 5. Install RF-DETR ────────────────────────────────────────
echo ""
echo "[5/8] Installing RF-DETR..."

cd "$REPOS_DIR/rf-detr"
# Install RF-DETR with --no-deps first to avoid its dependencies pulling
# incompatible torch/numpy versions, then install its actual needed deps.
$PYTHON_CMD -m pip install -e . --no-deps --quiet 2>/dev/null || \
    echo "  ⚠ RF-DETR no-deps install had issues"

# Install RF-DETR's actual runtime dependencies (without strict pins)
$PYTHON_CMD -m pip install --quiet \
    "transformers>=5.1.0,<6.0.0" \
    diffusers \
    accelerate \
    timm \
    supervision 2>/dev/null || true

if $PYTHON_CMD -c "import rfdetr" 2>/dev/null; then
    echo "  ✓ RF-DETR installed"
else
    echo "  ⚠ RF-DETR import failed (will retry after torch restore)"
fi

# ─── 6. Install Hunyuan3D-2.1 dependencies ─────────────────────
echo ""
echo "[6/8] Setting up Hunyuan3D-2.1 + FlashVDM..."

# Verify repo structure
HUNYUAN_DIR="$REPOS_DIR/Hunyuan3D-2.1"
if [ -d "$HUNYUAN_DIR/hy3dshape/hy3dshape" ]; then
    echo "  ✓ Hunyuan3D-2.1 repo structure verified"
else
    echo "  ⚠ Expected Hunyuan3D-2.1 structure not found at $HUNYUAN_DIR"
fi

# Install ALL Hunyuan3D dependencies WITHOUT using their requirements.txt.
# Their requirements.txt pins numpy==1.24.4 (broken on Python 3.12) and
# torch==2.5.1 which downgrades Colab's torch.
#
# The hy3dshape package requires these imports:
#   pipelines.py:      torch, numpy, PIL, trimesh, diffusers, transformers,
#                       omegaconf, yaml, tqdm, huggingface_hub, safetensors, accelerate
#   postprocessors.py:  pymeshlab, trimesh, numpy
#   preprocessors.py:   opencv-python, einops, numpy, PIL
#   conditioner.py:     transformers (CLIP, DINOv2), torchvision, numpy
#   denoisers/:         torch, einops
#   rembg.py:           rembg (optional)
echo "  Installing Hunyuan3D-compatible dependencies (no strict pins)..."

# Core shape pipeline dependencies
$PYTHON_CMD -m pip install --quiet \
    pymeshlab \
    pyyaml \
    accelerate \
    xatlas \
    psutil 2>/dev/null || {
    echo "  ⚠ Some Hunyuan3D deps failed. Trying individually..."
    $PYTHON_CMD -m pip install --quiet pymeshlab 2>/dev/null || echo "  ⚠ pymeshlab install failed"
    $PYTHON_CMD -m pip install --quiet pyyaml 2>/dev/null || echo "  ⚠ pyyaml install failed"
    $PYTHON_CMD -m pip install --quiet accelerate 2>/dev/null || echo "  ⚠ accelerate install failed"
    $PYTHON_CMD -m pip install --quiet xatlas 2>/dev/null || echo "  ⚠ xatlas install failed"
}

# Install OpenGL library to suppress pymeshlab plugin warnings
# (pymeshlab plugins like libfilter_ao.so need libOpenGL.so.0)
if ! ldconfig -p 2>/dev/null | grep -q "libOpenGL.so.0"; then
    $PYTHON_CMD -c "
import subprocess, sys
try:
    # Try apt (Ubuntu/Debian)
    subprocess.run(['apt-get', 'install', '-y', 'libopengl0'], check=True, capture_output=True)
except Exception:
    try:
        # Try conda
        subprocess.run(['conda', 'install', '-y', 'libopengl0'], check=True, capture_output=True)
    except Exception:
        print('  ⚠ Could not install libopengl0 (pymeshlab will show warnings)')
" 2>/dev/null || true
fi

# Optional: open3d (for point cloud / mesh I/O)
$PYTHON_CMD -m pip install --quiet open3d 2>/dev/null || \
    echo "  ⚠ open3d install failed (optional)"

# Quick import test for hy3dshape core
echo "  Testing hy3dshape import..."
$PYTHON_CMD -c "
import sys, os
hy3dshape_dir = os.path.join('$HUNYUAN_DIR', 'hy3dshape')
if os.path.isdir(os.path.join(hy3dshape_dir, 'hy3dshape')):
    sys.path.insert(0, hy3dshape_dir)
    try:
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        print('  ✓ hy3dshape imports successfully')
    except Exception as e:
        print(f'  ✗ hy3dshape import failed: {e}')
        # Try to identify which module is missing
        import importlib
        missing = []
        for mod in ['pymeshlab', 'yaml', 'trimesh', 'diffusers', 'transformers',
                     'einops', 'omegaconf', 'accelerate', 'huggingface_hub',
                     'safetensors', 'cv2', 'PIL', 'tqdm']:
            try:
                importlib.import_module(mod)
            except ImportError:
                missing.append(mod)
        if missing:
            print(f'    Missing modules: {missing}')
else:
    print('  ✗ hy3dshape directory not found')
" 2>&1

# Build CUDA extensions for texture pipeline (optional - requires CUDA toolkit + ninja)
if [ "$SKIP_CUDA_EXT" = false ]; then
    echo "  Building CUDA rasterizer extensions (optional, requires CUDA toolkit)..."
    if command -v nvcc &> /dev/null; then
        echo "    CUDA toolkit found: $(nvcc --version | grep release | head -1)"

        # Install ninja for faster CUDA builds
        $PYTHON_CMD -m pip install --quiet ninja 2>/dev/null || true

        cd "$HUNYUAN_DIR/hy3dpaint/custom_rasterizer" 2>/dev/null
        if [ -f "setup.py" ] || [ -f "pyproject.toml" ]; then
            $PYTHON_CMD -m pip install -e . 2>/dev/null || \
                echo "  ⚠ custom_rasterizer build failed (non-critical, texture pipeline will be unavailable)"
        else
            echo "  ⚠ custom_rasterizer setup.py/pyproject.toml not found, skipping"
        fi

        cd "$HUNYUAN_DIR/hy3dpaint/DifferentiableRenderer" 2>/dev/null
        if [ -f "compile_mesh_painter.sh" ]; then
            bash compile_mesh_painter.sh 2>/dev/null || \
                echo "  ⚠ DifferentiableRenderer build failed (non-critical)"
        else
            echo "  ⚠ compile_mesh_painter.sh not found, skipping"
        fi
    else
        echo "  ⚠ nvcc not found, skipping CUDA extension builds. Install CUDA toolkit for texture pipeline."
    fi
else
    echo "  Skipping CUDA extension builds (--skip-cuda-ext)"
fi

# Download RealESRGAN weights
if [ "$SKIP_DOWNLOAD" = false ]; then
    ESRGAN_CKPT="$HUNYUAN_DIR/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    if [ ! -f "$ESRGAN_CKPT" ]; then
        mkdir -p "$(dirname "$ESRGAN_CKPT")"
        echo "  Downloading RealESRGAN weights..."
        wget -q "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
            -O "$ESRGAN_CKPT" 2>/dev/null || \
            echo "  ⚠ RealESRGAN download failed"
    fi
fi

echo "  ✓ Hunyuan3D-2.1 setup complete"

# ─── 7. Install WildDet3D + MARCO dependencies ────────────────
echo ""
echo "[7/8] Setting up WildDet3D + MARCO..."

WILDDET_DIR="$REPOS_DIR/WildDet3D"
if [ -d "$WILDDET_DIR/wilddet3d" ]; then
    echo "  ✓ WildDet3D package directory verified"
else
    echo "  ⚠ Expected wilddet3d package not found"
fi

# Verify submodules
for submodule in sam3 lingbot_depth; do
    if [ -d "$WILDDET_DIR/third_party/$submodule" ] && \
       [ "$(ls -A "$WILDDET_DIR/third_party/$submodule" 2>/dev/null)" ]; then
        echo "  ✓ WildDet3D third_party/$submodule populated"
    else
        echo "  Initializing WildDet3D submodules..."
        cd "$WILDDET_DIR"
        git submodule update --init --recursive 2>/dev/null || true
    fi
done

# Install WildDet3D dependencies WITHOUT using their requirements.txt
echo "  Installing WildDet3D-compatible dependencies (no strict pins)..."
$PYTHON_CMD -m pip install --quiet \
    pyquaternion \
    ftfy \
    regex \
    iopath \
    pyarrow \
    ml_collections \
    terminaltables \
    timm \
    pycocotools \
    scalabel \
    cloudpickle 2>/dev/null || true

# Install utils3d (needed by WildDet3D)
$PYTHON_CMD -m pip install --quiet \
    "utils3d @ git+https://github.com/EasternJournalist/utils3d.git" 2>/dev/null || \
    echo "  ⚠ utils3d install failed"

# Install vis4d framework
# NOTE: vis4d requires pydantic<2.0 but rfdetr requires pydantic>=2.0.
# We install vis4d with --no-deps to avoid the pydantic conflict,
# then install vis4d's other deps manually.
if $PYTHON_CMD -c "import vis4d" 2>/dev/null; then
    echo "  ✓ vis4d already installed"
else
    echo "  Installing vis4d (with --no-deps to avoid pydantic conflict)..."
    $PYTHON_CMD -m pip install --quiet vis4d==1.0.0 --no-deps 2>/dev/null || {
        echo "  ⚠ vis4d install failed, trying without version pin..."
        $PYTHON_CMD -m pip install --quiet vis4d --no-deps 2>/dev/null || \
            echo "  ⚠ vis4d install failed completely"
    }
fi

# Install vis4d CUDA ops (optional)
if [ "$SKIP_CUDA_EXT" = false ]; then
    if $PYTHON_CMD -c "import vis4d_cuda_ops" 2>/dev/null; then
        echo "  ✓ vis4d_cuda_ops already installed"
    else
        echo "  Installing vis4d_cuda_ops..."
        $PYTHON_CMD -m pip install --quiet "git+https://github.com/SysCV/vis4d_cuda_ops.git" 2>/dev/null || \
            echo "  ⚠ vis4d_cuda_ops build failed (non-critical)"
    fi
fi

# Install MARCO dependencies
$PYTHON_CMD -m pip install --quiet \
    pandas \
    mediapy \
    h5py \
    scikit-learn \
    torch-kmeans \
    gdown 2>/dev/null || true

echo "  ✓ WildDet3D + MARCO setup complete"

# ─── 8. Restore torch + Reinstall RF-DETR ─────────────────────
echo ""
echo "[8/8] Restoring torch and reinstalling RF-DETR..."

CURRENT_TORCH=$($PYTHON_CMD -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")

if [ "$CURRENT_TORCH" != "$SAVED_TORCH_VERSION" ]; then
    echo "  ⚠ torch was changed from $SAVED_TORCH_VERSION to $CURRENT_TORCH"
    echo "  Restoring torch==$SAVED_TORCH_VERSION..."

    # Determine the correct PyTorch index URL based on CUDA version
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    if echo "$CUDA_VERSION" | grep -q "^12\.1"; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    elif echo "$CUDA_VERSION" | grep -q "^11\.8"; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    fi
    echo "  Using PyTorch index: $TORCH_INDEX"

    $PYTHON_CMD -m pip install --quiet \
        torch=="$SAVED_TORCH_VERSION" \
        torchvision=="$SAVED_TORCHVISION_VERSION" \
        --index-url "$TORCH_INDEX" 2>/dev/null || {
        echo "  Exact version restore failed, installing latest compatible..."
        $PYTHON_CMD -m pip install --quiet torch torchvision \
            --index-url "$TORCH_INDEX"
    }

    echo "  ✓ torch restored to $($PYTHON_CMD -c "import torch; print(torch.__version__)")"
else
    echo "  ✓ torch version unchanged ($CURRENT_TORCH)"
fi

# Reinstall RF-DETR with restored torch
echo "  Reinstalling RF-DETR with current torch..."
cd "$REPOS_DIR/rf-detr"
$PYTHON_CMD -m pip install -e . --no-deps --quiet 2>/dev/null || \
    echo "  ⚠ RF-DETR reinstall had issues"

# Ensure pydantic >= 2.0 for rfdetr
$PYTHON_CMD -m pip install --quiet "pydantic>=2.0" 2>/dev/null || \
    echo "  ⚠ pydantic upgrade failed"

# Ensure transformers >= 5.1.0 for rfdetr (BackboneConfigMixin is v5+ only)
CURRENT_TRANSFORMERS=$($PYTHON_CMD -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "0.0.0")
echo "  Current transformers: $CURRENT_TRANSFORMERS"
$PYTHON_CMD -m pip install --quiet "transformers>=5.1.0,<6.0.0" 2>/dev/null || \
    echo "  ⚠ transformers upgrade failed"
NEW_TRANSFORMERS=$($PYTHON_CMD -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "unknown")
echo "  Upgraded transformers: $NEW_TRANSFORMERS"

if $PYTHON_CMD -c "import rfdetr" 2>/dev/null; then
    echo "  ✓ RF-DETR installed successfully"
else
    echo "  ⚠ RF-DETR import still failing — check transformers version"
    $PYTHON_CMD -c "import transformers; print(f'  transformers version: {transformers.__version__}')" 2>/dev/null
fi

# ─── Download Model Weights ────────────────────────────────────
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo ""
    echo "Downloading model weights..."

    CKPT_DIR="$PROJECT_DIR/ckpt"
    mkdir -p "$CKPT_DIR"

    if [ ! -f "$CKPT_DIR/wilddet3d_alldata_all_prompt_v1.0.pt" ]; then
        echo "  Downloading WildDet3D checkpoint (~2GB)..."
        $PYTHON_CMD -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    'allenai/WildDet3D',
    'wilddet3d_alldata_all_prompt_v1.0.pt',
    local_dir='$CKPT_DIR'
)
" 2>/dev/null || echo "  ⚠ WildDet3D download failed"
    else
        echo "  ✓ WildDet3D checkpoint already exists"
    fi

    echo "  MARCO checkpoint will be auto-downloaded via torch.hub on first use"
    echo "  RF-DETR models will be auto-downloaded on first use"

    # Pre-download Hunyuan3D-2.1 model weights (optional but recommended)
    # Hunyuan3D's smart_load_model has a bug: if the cache directory exists but
    # the actual checkpoint file is missing (partial download), it skips re-downloading.
    # We pre-download here to avoid that issue at runtime.
    echo "  Pre-downloading Hunyuan3D-2.1 model weights (~5GB, may take several minutes)..."
    $PYTHON_CMD -c "
import os, sys
from huggingface_hub import snapshot_download

base_dir = os.environ.get('HY3DGEN_MODELS', os.path.expanduser('~/.cache/hy3dgen'))
model_path = 'tencent/Hunyuan3D-2.1'
subfolder = 'hunyuan3d-dit-v2-1'
model_dir = os.path.expanduser(os.path.join(base_dir, model_path, subfolder))
ckpt_path = os.path.join(model_dir, 'model.fp16.ckpt')
config_path = os.path.join(model_dir, 'config.yaml')

if os.path.isfile(ckpt_path) and os.path.isfile(config_path):
    print(f'  ✓ Hunyuan3D-2.1 shape model already cached')
else:
    print(f'  Downloading {model_path} (subfolder: {subfolder})...')
    local_dir = os.path.expanduser(os.path.join(base_dir, model_path))
    snapshot_download(
        repo_id=model_path,
        allow_patterns=[f'{subfolder}/*'],
        local_dir=local_dir,
    )
    print(f'  ✓ Hunyuan3D-2.1 shape model downloaded')
" 2>/dev/null || echo "  ⚠ Hunyuan3D-2.1 model pre-download failed (will retry at runtime)"
fi

# ─── Final Verification ────────────────────────────────────────
echo ""
echo "============================================================"
echo " Setup Complete!"
echo "============================================================"
echo ""
echo "Verification:"

$PYTHON_CMD -c "
import sys, os

# Add project root to path
project_dir = '$PROJECT_DIR'
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

repos_dir = '$REPOS_DIR'

# WildDet3D
wilddet_dir = os.path.join(repos_dir, 'WildDet3D')
if os.path.isdir(os.path.join(wilddet_dir, 'wilddet3d')):
    sys.path.insert(0, wilddet_dir)
    for sub in ['sam3', 'lingbot_depth', 'moge']:
        p = os.path.join(wilddet_dir, 'third_party', sub)
        if os.path.isdir(p):
            sys.path.insert(0, p)

# Hunyuan3D-2.1
hunyuan_dir = os.path.join(repos_dir, 'Hunyuan3D-2.1')
hy3dshape_dir = os.path.join(hunyuan_dir, 'hy3dshape')
hy3dpaint_dir = os.path.join(hunyuan_dir, 'hy3dpaint')
if os.path.isdir(os.path.join(hy3dshape_dir, 'hy3dshape')):
    sys.path.insert(0, hy3dshape_dir)
if os.path.isdir(hy3dpaint_dir):
    sys.path.insert(0, hy3dpaint_dir)

# MARCO
marco_dir = os.path.join(repos_dir, 'MARCO')
if os.path.isdir(marco_dir):
    sys.path.insert(0, marco_dir)

# Test imports
try:
    from scene_recon3d.pipeline import SceneReconstructionPipeline
    print('  ✓ scene_recon3d package')
except Exception as e:
    print(f'  ✗ scene_recon3d package: {e}')

try:
    import rfdetr
    print('  ✓ RF-DETR')
except Exception as e:
    print(f'  ✗ RF-DETR: {e}')
    import transformers
    print(f'    (transformers version: {transformers.__version__}, need >=5.1.0 for BackboneConfigMixin)')

try:
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    print('  ✓ Hunyuan3D-Shape')
except Exception as e:
    print(f'  ✗ Hunyuan3D-Shape: {e}')

try:
    from textureGenPipeline import Hunyuan3DPaintPipeline
    print('  ✓ Hunyuan3D-Paint')
except Exception as e:
    print(f'  ✗ Hunyuan3D-Paint (optional): {e}')

try:
    from wilddet3d import build_model
    print('  ✓ WildDet3D')
except Exception as e:
    print(f'  ✗ WildDet3D: {e}')

import torch
import numpy as np
print(f'  ✓ torch {torch.__version__} + numpy {np.__version__}')
if torch.cuda.is_available():
    print(f'  ✓ CUDA: {torch.cuda.get_device_name(0)}')
else:
    print('  ⚠ CUDA not available')
" 2>&1

echo ""
echo "To run the pipeline (low VRAM mode, recommended for T4):"
echo "  cd $PROJECT_DIR"
echo "  python run_pipeline.py --image <path_to_image>"
echo ""
echo "To run with textures enabled (needs more GPU memory):"
echo "  python run_pipeline.py --image <path_to_image> --texture"
echo ""
echo "To check environment only:"
echo "  python run_pipeline.py --check-env"
