#!/usr/bin/env python3
"""Script to download all pretrained model weights for the pipeline.

Usage:
    python scripts/download_weights.py [--all] [--wilddet3d] [--marco] [--hunyuan3d] [--rfdetr]
"""

import argparse
import os
import sys


def download_wilddet3d(output_dir: str = "ckpt"):
    """Download WildDet3D checkpoint from HuggingFace."""
    print("Downloading WildDet3D checkpoint (~2GB)...")
    os.makedirs(output_dir, exist_ok=True)

    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        "allenai/WildDet3D",
        "wilddet3d_alldata_all_prompt_v1.0.pt",
        local_dir=output_dir,
    )
    print(f"  ✓ WildDet3D saved to: {path}")


def download_marco(output_dir: str = "ckpt"):
    """Download MARCO checkpoint from GitHub Releases or Google Drive."""
    print("Downloading MARCO checkpoint...")
    os.makedirs(output_dir, exist_ok=True)

    # Try torch.hub first
    try:
        import torch
        model = torch.hub.load("visinf/MARCO", "marco", pretrained=True, trust_repo=True)
        print("  ✓ MARCO downloaded via torch.hub")
        return
    except Exception as e:
        print(f"  torch.hub download failed: {e}")

    # Fallback: try gdown
    try:
        import gdown
        url = "https://drive.google.com/uc?id=1_of8iQjenTttF5Jld69LNf9M0vnM2Xbx"
        output = os.path.join(output_dir, "marco_spair.pth")
        gdown.download(url, output, quiet=False)
        print(f"  ✓ MARCO saved to: {output}")
    except Exception as e:
        print(f"  gdown download failed: {e}")
        print("  Please download manually from:")
        print("    https://github.com/visinf/MARCO/releases/download/v1.0/marco_release.pth")


def download_hunyuan3d():
    """Download Hunyuan3D-2.1 weights from HuggingFace.

    Note: These are auto-downloaded on first use via the pipeline,
    but can be pre-downloaded here.
    """
    print("Downloading Hunyuan3D-2.1 weights from HuggingFace...")
    print("  This will be cached at ~/.cache/hy3dgen/")

    from huggingface_hub import snapshot_download
    snapshot_download("tencent/Hunyuan3D-2.1", cache_dir=os.path.expanduser("~/.cache/hy3dgen"))
    print("  ✓ Hunyuan3D-2.1 weights downloaded")


def download_rfdetr():
    """Download RF-DETR weights.

    Note: These are auto-downloaded on first use.
    """
    print("Downloading RF-DETR weights...")
    print("  These are auto-downloaded on first use via the rfdetr package.")
    print("  Triggering download now...")

    from rfdetr import RFDETRSegMedium
    model = RFDETRSegMedium()
    # Just loading the model triggers the download
    print("  ✓ RF-DETR weights downloaded")


def download_realesrgan(output_dir: str = None):
    """Download RealESRGAN weights for texture super-resolution."""
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(__file__), "..", "repos", "Hunyuan3D-2.1", "hy3dpaint", "ckpt"
        )
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "RealESRGAN_x4plus.pth")
    if os.path.exists(output_path):
        print(f"  ✓ RealESRGAN already exists at: {output_path}")
        return

    print("Downloading RealESRGAN weights...")
    import urllib.request
    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    urllib.request.urlretrieve(url, output_path)
    print(f"  ✓ RealESRGAN saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument("--all", action="store_true", help="Download all weights")
    parser.add_argument("--wilddet3d", action="store_true", help="Download WildDet3D only")
    parser.add_argument("--marco", action="store_true", help="Download MARCO only")
    parser.add_argument("--hunyuan3d", action="store_true", help="Download Hunyuan3D only")
    parser.add_argument("--rfdetr", action="store_true", help="Download RF-DETR only")
    parser.add_argument("--realesrgan", action="store_true", help="Download RealESRGAN only")
    parser.add_argument("--output-dir", type=str, default="ckpt", help="Output directory")
    args = parser.parse_args()

    if not any([args.all, args.wilddet3d, args.marco, args.hunyuan3d, args.rfdetr, args.realesrgan]):
        args.all = True

    if args.all or args.wilddet3d:
        download_wilddet3d(args.output_dir)

    if args.all or args.marco:
        download_marco(args.output_dir)

    if args.all or args.hunyuan3d:
        download_hunyuan3d()

    if args.all or args.rfdetr:
        download_rfdetr()

    if args.all or args.realesrgan:
        download_realesrgan()

    print("\nAll requested weights downloaded successfully!")


if __name__ == "__main__":
    main()
