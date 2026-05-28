#!/bin/bash
# =============================================================
# Setup Script for 3D Scene Reconstruction Pipeline
# =============================================================
# This script installs all dependencies, downloads model weights,
# and prepares the environment for the pipeline.
#
# Usage:
#   bash scripts/setup.sh [--skip-download] [--skip-cuda-ext]
#   bash scripts/setup.sh --python=python3
#
# IMPORTANT: This script carefully manages dependency versions to
# avoid conflicts between repos that pin different torch/numpy versions.
# We filter out strict version pins and use the system torch instead.
# =============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REPOS_DIR="$PROJECT_DIR/repos"

echo "============================================================"
echo " 3D Scene Reconstruction Pipeline - Setup"
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

# ─── Helper: Install requirements filtering out problematic pins ──────
# Some repos pin specific torch/numpy/torchvision versions that conflict
# with each other and with the system PyTorch. We filter these out and
# install the rest.

install_requirements_filtered() {
    local req_file="$1"
    local label="$2"
    # Packages whose versions we want to filter out (keep the package, drop the version pin)
    local FILTER_PKGS="^torch==|^numpy==|^torchvision==|^triton==|^nvidia-|^sympy=="

    if [ ! -f "$req_file" ]; then
        echo "  ⚠ $label requirements file not found: $req_file"
        return 1
    fi

    # Create a filtered temp requirements file
    local TMP_REQ
    TMP_REQ=$(mktemp /tmp/filtered_requirements.XXXXXX.txt)

    # Filter: keep lines that don't match the problematic version pins
    # For lines that DO match, keep the package name but drop the version
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip empty lines and comments
        [[ -z "$line" || "$line" =~ ^# ]] && continue

        # Check if this line pins a problematic version
        if echo "$line" | grep -qE "$FILTER_PKGS"; then
            # Extract package name and install latest compatible version
            pkg_name=$(echo "$line" | sed 's/[<>=!].*//' | sed 's/\[.*//')
            echo "# FILTERED (original: $line)" >> "$TMP_REQ"
        else
            echo "$line" >> "$TMP_REQ"
        fi
    done < "$req_file"

    echo "  Installing $label dependencies (filtered)..."
    $PYTHON_CMD -m pip install -r "$TMP_REQ" 2>&1 | tail -5 || true

    rm -f "$TMP_REQ"
}

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
TORCH_VERSION=$($PYTHON_CMD -c "import torch; print(torch.__version__)" 2>/dev/null)
echo "  PyTorch: $TORCH_VERSION"
echo "  PyTorch CUDA: $CUDA_VERSION"
echo "  Python: $($PYTHON_CMD --version)"

# Record the original torch version so we can restore it later if needed
SAVED_TORCH_VERSION="$TORCH_VERSION"
SAVED_TORCHVISION_VERSION=$($PYTHON_CMD -c "import torchvision; print(torchvision.__version__)" 2>/dev/null || echo "unknown")
echo "  Saved torch=$SAVED_TORCH_VERSION torchvision=$SAVED_TORCHVISION_VERSION"

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
        # Ensure submodules are initialized for WildDet3D
        if [ "$repo_name" = "WildDet3D" ]; then
            cd "$REPOS_DIR/WildDet3D"
            git submodule update --init --recursive 2>/dev/null || true
        fi
    fi
done

# ─── 3. Install Core Python Dependencies ───────────────────────
echo ""
echo "[3/7] Installing core Python dependencies..."

$PYTHON_CMD -m pip install --upgrade pip setuptools wheel

# Core dependencies that don't conflict with torch
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
    supervision \
    rembg \
    pymeshlab \
    xatlas \
    pygltflib \
    basicsr \
    realesrgan 2>/dev/null || true

# ─── 4. Install RF-DETR ────────────────────────────────────────
echo ""
echo "[4/7] Installing RF-DETR..."

# RF-DETR needs to be installed first, but we'll reinstall it at the end
# if torch got downgraded
cd "$REPOS_DIR/rf-detr"
$PYTHON_CMD -m pip install -e . 2>/dev/null || \
    echo "  ⚠ RF-DETR install had issues, will retry after torch fix"

if $PYTHON_CMD -c "import rfdetr" 2>/dev/null; then
    echo "  ✓ RF-DETR installed"
else
    echo "  ⚠ RF-DETR import failed (will retry after torch restore)"
fi

# ─── 5. Install Hunyuan3D-2.1 + FlashVDM ──────────────────────
echo ""
echo "[5/7] Setting up Hunyuan3D-2.1 + FlashVDM..."

# Hunyuan3D-2.1 is NOT pip-installable. It uses sys.path manipulation.
# The repo uses nested packages: hy3dshape/hy3dshape/ and hy3dpaint/
# We verify the directory structure and install dependencies instead.

HUNYUAN_DIR="$REPOS_DIR/Hunyuan3D-2.1"
if [ -d "$HUNYUAN_DIR/hy3dshape/hy3dshape" ]; then
    echo "  ✓ Hunyuan3D-2.1 repo structure verified (nested hy3dshape package)"
else
    echo "  ⚠ Expected Hunyuan3D-2.1 structure not found at $HUNYUAN_DIR"
    echo "  Checking alternative structure..."
    # List what's actually in hy3dshape/
    ls -la "$HUNYUAN_DIR/hy3dshape/" 2>/dev/null || echo "  hy3dshape/ not found"
fi

if [ -d "$HUNYUAN_DIR/hy3dpaint" ]; then
    echo "  ✓ Hunyuan3D-2.1 hy3dpaint directory verified"
else
    echo "  ⚠ Expected hy3dpaint directory not found at $HUNYUAN_DIR/hy3dpaint"
fi

# Install Hunyuan3D Python dependencies with version filtering
# (Hunyuan3D pins numpy==1.24.4 which is incompatible with Python 3.12)
if [ -f "$HUNYUAN_DIR/requirements.txt" ]; then
    install_requirements_filtered "$HUNYUAN_DIR/requirements.txt" "Hunyuan3D-2.1"
else
    echo "  Installing known Hunyuan3D dependencies..."
    $PYTHON_CMD -m pip install \
        transformers \
        diffusers \
        accelerate \
        rembg \
        pymeshlab \
        xatlas \
        pygltflib \
        basicsr \
        realesrgan 2>/dev/null || true
fi

# Build CUDA extensions for texture pipeline (optional)
if [ "$SKIP_CUDA_EXT" = false ]; then
    echo "  Building CUDA rasterizer extensions (optional)..."
    cd "$HUNYUAN_DIR/hy3dpaint/custom_rasterizer"
    $PYTHON_CMD -m pip install -e . 2>/dev/null || \
        echo "  ⚠ custom_rasterizer build failed (non-critical for shape-only)"

    cd "$HUNYUAN_DIR/hy3dpaint/DifferentiableRenderer"
    bash compile_mesh_painter.sh 2>/dev/null || \
        echo "  ⚠ DifferentiableRenderer build failed (non-critical for shape-only)"
else
    echo "  Skipping CUDA extension builds (--skip-cuda-ext)"
fi

# Download RealESRGAN weights (needed for texture pipeline)
if [ "$SKIP_DOWNLOAD" = false ]; then
    ESRGAN_CKPT="$HUNYUAN_DIR/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    if [ ! -f "$ESRGAN_CKPT" ]; then
        mkdir -p "$(dirname "$ESRGAN_CKPT")"
        echo "  Downloading RealESRGAN weights..."
        wget -q "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
            -O "$ESRGAN_CKPT" 2>/dev/null || \
            echo "  ⚠ RealESRGAN download failed (texture generation will be disabled)"
    fi
fi

echo "  ✓ Hunyuan3D-2.1 setup complete (uses sys.path at runtime)"

# ─── 6. Install WildDet3D ──────────────────────────────────────
echo ""
echo "[6/7] Setting up WildDet3D..."

# WildDet3D is NOT pip-installable. It uses sys.path manipulation.
# The repo has a wilddet3d/ package directory with __init__.py
# that auto-adds third_party submodules to sys.path.
# We verify structure and install dependencies.

WILDDET_DIR="$REPOS_DIR/WildDet3D"
if [ -d "$WILDDET_DIR/wilddet3d" ]; then
    echo "  ✓ WildDet3D package directory verified"
else
    echo "  ⚠ Expected wilddet3d package not found at $WILDDET_DIR/wilddet3d"
fi

# Verify third_party submodules
for submodule in sam3 lingbot_depth; do
    if [ -d "$WILDDET_DIR/third_party/$submodule" ] && \
       [ "$(ls -A "$WILDDET_DIR/third_party/$submodule" 2>/dev/null)" ]; then
        echo "  ✓ WildDet3D third_party/$submodule populated"
    else
        echo "  ⚠ WildDet3D third_party/$submodule is empty - initializing submodules..."
        cd "$WILDDET_DIR"
        git submodule update --init --recursive 2>/dev/null || \
            echo "  ⚠ Failed to init submodule $submodule"
    fi
done

# Install WildDet3D Python dependencies with version filtering
# (WildDet3D pins torch==2.5.1 and numpy<2.0.0 which break other packages)
if [ -f "$WILDDET_DIR/requirements.txt" ]; then
    install_requirements_filtered "$WILDDET_DIR/requirements.txt" "WildDet3D"
else
    echo "  Installing known WildDet3D dependencies..."
    $PYTHON_CMD -m pip install \
        pyquaternion \
        ftfy \
        regex \
        iopath \
        pyarrow \
        einops \
        timm \
        transformers \
        ml_collections \
        terminaltables 2>/dev/null || true
fi

# Install vis4d framework (required by WildDet3D)
if $PYTHON_CMD -c "import vis4d" 2>/dev/null; then
    echo "  ✓ vis4d already installed"
else
    echo "  Installing vis4d..."
    # vis4d pulls in its own numpy pin - install with --no-deps then add missing deps
    $PYTHON_CMD -m pip install vis4d==1.0.0 2>/dev/null || {
        echo "  ⚠ vis4d install failed, trying with --no-deps..."
        $PYTHON_CMD -m pip install vis4d==1.0.0 --no-deps 2>/dev/null || \
            echo "  ⚠ vis4d install failed completely"
    }
fi

# Install vis4d CUDA ops
if [ "$SKIP_CUDA_EXT" = false ]; then
    if $PYTHON_CMD -c "import vis4d_cuda_ops" 2>/dev/null; then
        echo "  ✓ vis4d_cuda_ops already installed"
    else
        echo "  Installing vis4d_cuda_ops..."
        $PYTHON_CMD -m pip install git+https://github.com/SysCV/vis4d_cuda_ops.git 2>/dev/null || \
            echo "  ⚠ vis4d_cuda_ops build failed"
    fi
else
    echo "  Skipping vis4d_cuda_ops build (--skip-cuda-ext)"
fi

echo "  ✓ WildDet3D setup complete (uses sys.path at runtime)"

# ─── 7. Install MARCO ──────────────────────────────────────────
echo ""
echo "[7/7] Installing MARCO dependencies..."

# MARCO is loaded via torch.hub or sys.path. Install its dependencies.
$PYTHON_CMD -m pip install \
    timm \
    pandas \
    mediapy \
    h5py \
    scikit-learn \
    torch-kmeans \
    gdown 2>/dev/null || \
    echo "  ⚠ Some MARCO dependencies failed"

# ─── Restore torch version if it was downgraded ────────────────
echo ""
echo "Checking if torch needs to be restored..."

CURRENT_TORCH=$($PYTHON_CMD -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
echo "  Current torch: $CURRENT_TORCH"
echo "  Original torch: $SAVED_TORCH_VERSION"

if [ "$CURRENT_TORCH" != "$SAVED_TORCH_VERSION" ]; then
    echo "  ⚠ torch was downgraded! Restoring to original version..."
    # Determine the install source based on CUDA version
    if echo "$SAVED_TORCH_VERSION" | grep -q "+cu"; then
        # Get CUDA version from the version string (e.g., 2.11.0+cu128 → cu128)
        CUDA_SUFFIX=$(echo "$SAVED_TORCH_VERSION" | grep -oP '\+cu\d+')
        CUDA_VER=${CUDA_SUFFIX#+cu}  # e.g., 128
        # Map to PyTorch index URL format (128 → cu121, cu124, etc.)
        # PyTorch uses cu121, cu124, cu126 etc.
        # For cu128, try cu124 as it's the closest available
        INDEX_CUDA="cu124"
        echo "  Reinstalling torch==$SAVED_TORCH_VERSION from CUDA $INDEX_CUDA index..."
        $PYTHON_CMD -m pip install torch=="$SAVED_TORCH_VERSION" torchvision=="$SAVED_TORCHVISION_VERSION" \
            --index-url "https://download.pytorch.org/whl/$INDEX_CUDA" 2>/dev/null || {
            echo "  Exact version restore failed, trying compatible version..."
            $PYTHON_CMD -m pip install torch torchvision \
                --index-url "https://download.pytorch.org/whl/$INDEX_CUDA"
        }
    else
        echo "  Reinstalling torch==$SAVED_TORCH_VERSION..."
        $PYTHON_CMD -m pip install torch=="$SAVED_TORCH_VERSION" torchvision=="$SAVED_TORCHVISION_VERSION" 2>/dev/null || {
            $PYTHON_CMD -m pip install torch torchvision
        }
    fi

    # Reinstall RF-DETR since torch was changed
    echo "  Reinstalling RF-DETR (was broken by torch change)..."
    cd "$REPOS_DIR/rf-detr"
    $PYTHON_CMD -m pip install -e . --no-deps 2>/dev/null || \
        echo "  ⚠ RF-DETR reinstall failed"

    echo "  ✓ torch restored to $SAVED_TORCH_VERSION"
else
    echo "  ✓ torch version unchanged ($CURRENT_TORCH)"
fi

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
    echo "  MARCO checkpoint will be auto-downloaded via torch.hub on first use"

    # Hunyuan3D models are auto-downloaded from HuggingFace on first use
    echo "  Hunyuan3D-2.1 models will be auto-downloaded from HuggingFace on first use"

    # RF-DETR models are auto-downloaded on first use
    echo "  RF-DETR models will be auto-downloaded on first use"
fi

# ─── Final Verification ────────────────────────────────────────
echo ""
echo "============================================================"
echo " Setup Complete!"
echo "============================================================"
echo ""
echo "Verification:"

# Set up sys.path the same way the pipeline does
$PYTHON_CMD -c "
import sys, os

# Add repo paths the same way setup_paths.py does
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

# Test imports
try:
    import rfdetr
    print('  ✓ RF-DETR')
except Exception as e:
    print(f'  ✗ RF-DETR: {e}')

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

try:
    import torch
    import numpy as np
    print(f'  ✓ torch {torch.__version__} + numpy {np.__version__}')
    if torch.cuda.is_available():
        print(f'  ✓ CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        print('  ⚠ CUDA not available')
except Exception as e:
    print(f'  ✗ torch/numpy check failed: {e}')
" 2>&1 | head -20

echo ""
echo "To run the pipeline:"
echo "  cd $PROJECT_DIR"
echo "  python run_pipeline.py --image <path_to_image> --preload"
echo ""
echo "To run with custom config:"
echo "  python run_pipeline.py --image <path_to_image> --config configs/pipeline_config.yaml"
echo ""
echo "If imports fail, try restoring torch manually:"
echo "  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
