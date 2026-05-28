"""Setup configuration for the scene_recon3d package."""

from setuptools import setup, find_packages

setup(
    name="scene_recon3d",
    version="1.0.0",
    description="3D Scene Reconstruction Pipeline combining RF-DETR, WildDet3D, Hunyuan3D-2.1+FlashVDM, and MARCO",
    author="Scene Recon3D",
    python_requires=">=3.10",
    packages=find_packages(exclude=["scripts", "configs"]),
    install_requires=[
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "numpy>=1.24",
        "opencv-python-headless>=4.8",
        "Pillow>=10.0",
        "scipy>=1.11",
        "scikit-image>=0.21",
        "trimesh>=4.0",
        "einops>=0.7",
        "omegaconf>=2.3",
        "pyyaml>=6.0",
        "tqdm>=4.65",
        "huggingface_hub>=0.19",
        "safetensors>=0.4",
        "supervision>=0.18",
    ],
    extras_require={
        "full": [
            "rfdetr",
            "transformers>=4.46",
            "diffusers>=0.30",
            "accelerate>=1.1",
            "rembg>=2.0",
            "timm>=1.0",
            "pyrender",
            "open3d>=0.18",
            "pymeshlab",
        ],
        "dev": [
            "pytest",
            "black",
            "ruff",
        ],
    },
    # run_pipeline.py is at the repo root, not inside the package,
    # so we don't use entry_points for it. Run it directly:
    #   python run_pipeline.py --image scene.jpg
)
