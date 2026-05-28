"""Path setup utility for the 3D scene reconstruction pipeline.

This module ensures that all repository paths are correctly added to sys.path
before any imports are attempted. It supports both the default layout
(./repos/ next to this package) and a custom location via the
FLARES_REPO_DIR environment variable.

Usage:
    import scene_recon3d.utils.setup_paths  # noqa: F401 — side-effect import
    # Now all pipeline imports will work
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)


def setup_repo_paths():
    """Add all required repository paths to sys.path.

    Searches for repos in:
    1. FLARES_REPO_DIR environment variable (if set)
    2. ./repos/ directory relative to this package

    Adds:
    - WildDet3D repo root (for `from wilddet3d import ...`)
    - WildDet3D/third_party/{sam3,lingbot_depth,moge}
    - Hunyuan3D-2.1/hy3dshape (for `from hy3dshape.pipelines import ...`)
    - Hunyuan3D-2.1/hy3dpaint (for `from textureGenPipeline import ...`)
    """
    # Determine repos directory
    this_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(this_dir)  # scene_recon3d/
    project_dir = os.path.dirname(package_dir)  # flares/

    env_repo = os.environ.get("FLARES_REPO_DIR", "")
    repos_dir = os.path.join(project_dir, "repos")

    if env_repo:
        repos_dir = env_repo

    if not os.path.isdir(repos_dir):
        logger.debug(f"Repos directory not found: {repos_dir}")
        return

    # ─── WildDet3D ─────────────────────────────────────────────
    wilddet_dir = os.path.join(repos_dir, "WildDet3D")
    if os.path.isdir(os.path.join(wilddet_dir, "wilddet3d")):
        if wilddet_dir not in sys.path:
            sys.path.insert(0, wilddet_dir)
        for submodule in ["sam3", "lingbot_depth", "moge"]:
            sub_path = os.path.join(wilddet_dir, "third_party", submodule)
            if os.path.isdir(sub_path) and sub_path not in sys.path:
                sys.path.insert(0, sub_path)

    # ─── Hunyuan3D-2.1 ────────────────────────────────────────
    hunyuan_dir = os.path.join(repos_dir, "Hunyuan3D-2.1")
    hy3dshape_dir = os.path.join(hunyuan_dir, "hy3dshape")
    hy3dpaint_dir = os.path.join(hunyuan_dir, "hy3dpaint")
    if os.path.isdir(os.path.join(hy3dshape_dir, "hy3dshape")):
        if hy3dshape_dir not in sys.path:
            sys.path.insert(0, hy3dshape_dir)
        if hy3dpaint_dir not in sys.path:
            sys.path.insert(0, hy3dpaint_dir)


# Auto-setup on import
setup_repo_paths()
