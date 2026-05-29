"""Allow running the pipeline as: python -m scene_recon3d"""

import os
import sys

# Ensure the repo root is on sys.path so run_pipeline can be found
_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from run_pipeline import main

if __name__ == "__main__":
    main()
