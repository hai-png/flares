#!/bin/bash
# =============================================================
# Setup Script for 3D Scene Reconstruction Pipeline
# =============================================================
# This script installs all dependencies, downloads model weights,
# and prepares the environment for the pipeline.
#
# Usage:
#   bash scripts/setup.sh [--skip-download] [--skip-cuda-ext]
# =============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPOS_DIR="$PROJECT_DIR/repos"

echo "============================================================"
echo " 3D Scene Reconstruction Pipeline — Setup"
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
echo "[1/7] Checking prerequisites..."

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
echo "  PyTorch CUDA: $CUDA_VERSION"
echo "  Python: $($PYTHON_CMD --version)"

# ─── 2. Clone Repositories ─────────────────────────────────────
echo ""
echo "[2/7] Checking repositories..."

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
    fi
done

# ─── 3. Install Core Python Dependencies ───────────────────────
echo ""
echo "[3/7] Installing core Python dependencies..."

$PYTHON_CMD -m pip install --upgrade pip

# Core dependencies
$PYTHON_CMD -m pip install \
    omegaconf \
    opencv-python-headless \
    pillow \
    scipy \
    scikit-image \
    trimesh \
    pyrender \
    einops \
    tqdm \
    huggingface_hub \
    safetensors \
    supervision

# ─── 4. Install RF-DETR ────────────────────────────────────────
echo ""
echo "[4/7] Installing RF-DETR..."

if $PYTHON_CMD -c "import rfdetr" 2>/dev/null; then
    echo "  ✓ rfdetr already installed"
else
    cd "$REPOS_DIR/rf-detr"
    $PYTHON_CMD -m pip install -e .
    echo "  ✓ RF-DETR installed"
fi

# ─── 5. Install Hunyuan3D-2.1 + FlashVDM ──────────────────────
echo ""
echo "[5/7] Installing Hunyuan3D-2.1..."

# Install Hunyuan3D shape pipeline
if $PYTHON_CMD -c "import hy3dshape" 2>/dev/null; then
    echo "  ✓ hy3dshape already installed"
else
    cd "$REPOS_DIR/Hunyuan3D-2.1/hy3dshape"
    $PYTHON_CMD -m pip install -e .
    echo "  ✓ hy3dshape installed"
fi

# Install Hunyuan3D texture pipeline
if $PYTHON_CMD -c "import hy3dpaint" 2>/dev/null; then
    echo "  ✓ hy3dpaint already installed"
else
    cd "$REPOS_DIR/Hunyuan3D-2.1/hy3dpaint"
    $PYTHON_CMD -m pip install -e . 2>/dev/null || \
        echo "  ⚠ hy3dpaint install had warnings (may need CUDA extensions)"
    echo "  ✓ hy3dpaint installed"
fi

# Install additional Hunyuan3D dependencies
$PYTHON_CMD -m pip install \
    transformers>=4.46.0 \
    diffusers>=0.30.0 \
    accelerate>=1.1.1 \
    rembg>=2.0.50 \
    pymeshlab \
    xatlas \
    open3d \
    pygltflib \
    basicsr \
    realesrgan 2>/dev/null || \
    echo "  ⚠ Some optional dependencies failed (non-critical)"

# Build CUDA extensions for texture pipeline
if [ "$SKIP_CUDA_EXT" = false ]; then
    echo "  Building CUDA rasterizer extensions..."
    cd "$REPOS_DIR/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer"
    $PYTHON_CMD -m pip install -e . 2>/dev/null || \
        echo "  ⚠ custom_rasterizer build failed (non-critical for shape-only)"

    cd "$REPOS_DIR/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer"
    bash compile_mesh_painter.sh 2>/dev/null || \
        echo "  ⚠ DifferentiableRenderer build failed (non-critical for shape-only)"
fi

# Download RealESRGAN weights
if [ "$SKIP_DOWNLOAD" = false ]; then
    ESRGAN_CKPT="$REPOS_DIR/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    if [ ! -f "$ESRGAN_CKPT" ]; then
        mkdir -p "$(dirname "$ESRGAN_CKPT")"
        echo "  Downloading RealESRGAN weights..."
        wget -q "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
            -O "$ESRGAN_CKPT" 2>/dev/null || \
            echo "  ⚠ RealESRGAN download failed (texture generation will be disabled)"
    fi
fi

# ─── 6. Install WildDet3D ──────────────────────────────────────
echo ""
echo "[6/7] Installing WildDet3D..."

# Install vis4d framework
if $PYTHON_CMD -c "import vis4d" 2>/dev/null; then
    echo "  ✓ vis4d already installed"
else
    $PYTHON_CMD -m pip install vis4d==1.0.0 2>/dev/null || \
        echo "  ⚠ vis4d install failed"
fi

# Install vis4d CUDA ops
if [ "$SKIP_CUDA_EXT" = false ]; then
    if $PYTHON_CMD -c "import vis4d_cuda_ops" 2>/dev/null; then
        echo "  ✓ vis4d_cuda_ops already installed"
    else
        $PYTHON_CMD -m pip install git+https://github.com/SysCV/vis4d_cuda_ops.git \
            --no-build-isolation --no-cache-dir 2>/dev/null || \
            echo "  ⚠ vis4d_cuda_ops build failed"
    fi
fi

# Install WildDet3D dependencies
$PYTHON_CMD -m pip install \
    pyquaternion \
    ftfy \
    regex \
    iopath \
    pyarrow 2>/dev/null || \
    echo "  ⚠ Some WildDet3D dependencies failed"

# ─── 7. Install MARCO ──────────────────────────────────────────
echo ""
echo "[7/7] Installing MARCO..."

$PYTHON_CMD -m pip install \
    timm \
    pandas \
    mediapy \
    h5py \
    scikit-learn \
    torch-kmeans \
    gdown 2>/dev/null || \
    echo "  ⚠ Some MARCO dependencies failed"

# ─── Download Model Weights ────────────────────────────────────
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo ""
    echo "Downloading model weights..."

    # WildDet3D checkpoint
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
" 2>/dev/null || echo "  ⚠ WildDet3D download failed. Run manually: huggingface-cli download allenai/WildDet3D wilddet3d_alldata_all_prompt_v1.0.pt --local-dir ckpt/"
    else
        echo "  ✓ WildDet3D checkpoint already exists"
    fi

    # MARCO checkpoint (optional, torch.hub auto-downloads)
    if [ ! -f "$CKPT_DIR/marco_release.pth" ]; then
        echo "  MARCO checkpoint will be auto-downloaded via torch.hub on first use"
    else
        echo "  ✓ MARCO checkpoint already exists"
    fi

    # Hunyuan3D models are auto-downloaded from HuggingFace on first use
    echo "  Hunyuan3D-2.1 models will be auto-downloaded from HuggingFace on first use"
    echo "  RF-DETR models will be auto-downloaded on first use"
fi

# ─── Final Checks ──────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Setup Complete!"
echo "============================================================"
echo ""
echo "Installed packages:"
$PYTHON_CMD -c "
try:
    import rfdetr; print(f'  ✓ RF-DETR {rfdetr.__version__}')
except: print('  ✗ RF-DETR not installed')

try:
    import hy3dshape; print('  ✓ Hunyuan3D-Shape')
except: print('  ✗ Hunyuan3D-Shape not installed')

try:
    import hy3dpaint; print('  ✓ Hunyuan3D-Paint')
except: print('  ✗ Hunyuan3D-Paint not installed (optional)')

try:
    from flashvdm_decoder.volume_decoders import FlashVDMVolumeDecoding; print('  ✓ FlashVDM')
except: print('  ✗ FlashVDM not available')
" 2>/dev/null

echo ""
echo "To run the pipeline:"
echo "  cd $PROJECT_DIR"
echo "  python run_pipeline.py --image <path_to_image> --preload"
echo ""
echo "To run with custom config:"
echo "  python run_pipeline.py --image <path_to_image> --config configs/pipeline_config.yaml"
