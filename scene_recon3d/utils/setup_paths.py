"""Path setup utility for the 3D scene reconstruction pipeline.

This module ensures that all repository paths are correctly added to sys.path
before any imports are attempted. It supports both the default layout
(./repos/ next to this package) and a custom location via the
FLARES_REPO_DIR environment variable.

Usage:
    import scene_recon3d.utils.setup_paths  # noqa: F401 -- side-effect import
    # Now all pipeline imports will work
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)


def _find_repos_dir():
    """Find the repos directory from multiple possible locations.

    Search order:
    1. FLARES_REPO_DIR environment variable (if set)
    2. ./repos/ relative to this package's parent
    3. ./repos/ relative to current working directory
    4. /content/flares/repos/ (Google Colab default)
    """
    # 1. Environment variable
    env_repo = os.environ.get("FLARES_REPO_DIR", "")
    if env_repo and os.path.isdir(env_repo):
        return env_repo

    # 2. Relative to this package
    this_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(this_dir)      # scene_recon3d/
    project_dir = os.path.dirname(package_dir)    # flares/
    repos_dir = os.path.join(project_dir, "repos")
    if os.path.isdir(repos_dir):
        return repos_dir

    # 3. Relative to cwd
    repos_dir_cwd = os.path.join(os.getcwd(), "repos")
    if os.path.isdir(repos_dir_cwd):
        return repos_dir_cwd

    # 4. Google Colab default
    colab_path = "/content/flares/repos"
    if os.path.isdir(colab_path):
        return colab_path

    return None


def setup_repo_paths():
    """Add all required repository paths to sys.path.

    Adds:
    - WildDet3D repo root (for `from wilddet3d import ...`)
    - WildDet3D/third_party/{sam3,lingbot_depth,moge}
    - Hunyuan3D-2.1/hy3dshape (for `from hy3dshape.pipelines import ...`)
    - Hunyuan3D-2.1/hy3dpaint (for `from textureGenPipeline import ...`)
    - MARCO repo root (for `from models import ...`)
    """
    repos_dir = _find_repos_dir()

    if repos_dir is None:
        logger.debug("Repos directory not found in any expected location")
        return

    # --- WildDet3D ---
    wilddet_dir = os.path.join(repos_dir, "WildDet3D")
    if os.path.isdir(os.path.join(wilddet_dir, "wilddet3d")):
        if wilddet_dir not in sys.path:
            sys.path.insert(0, wilddet_dir)
            logger.debug(f"Added to sys.path: {wilddet_dir}")
        for submodule in ["sam3", "lingbot_depth", "moge"]:
            sub_path = os.path.join(wilddet_dir, "third_party", submodule)
            if os.path.isdir(sub_path) and sub_path not in sys.path:
                sys.path.insert(0, sub_path)
                logger.debug(f"Added to sys.path: {sub_path}")

    # --- Hunyuan3D-2.1 ---
    hunyuan_dir = os.path.join(repos_dir, "Hunyuan3D-2.1")
    hy3dshape_dir = os.path.join(hunyuan_dir, "hy3dshape")
    hy3dpaint_dir = os.path.join(hunyuan_dir, "hy3dpaint")
    if os.path.isdir(os.path.join(hy3dshape_dir, "hy3dshape")):
        if hy3dshape_dir not in sys.path:
            sys.path.insert(0, hy3dshape_dir)
            logger.debug(f"Added to sys.path: {hy3dshape_dir}")
        if hy3dpaint_dir not in sys.path:
            sys.path.insert(0, hy3dpaint_dir)
            logger.debug(f"Added to sys.path: {hy3dpaint_dir}")

    # --- MARCO ---
    marco_dir = os.path.join(repos_dir, "MARCO")
    if os.path.isdir(marco_dir) and marco_dir not in sys.path:
        sys.path.insert(0, marco_dir)
        logger.debug(f"Added to sys.path: {marco_dir}")


# Auto-setup on import
setup_repo_paths()
