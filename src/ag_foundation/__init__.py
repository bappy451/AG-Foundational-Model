"""Agricultural foundation-model package."""

import os
from pathlib import Path

# Setup local models caching directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MODELS_DIR = _PROJECT_ROOT / "models"
_MODELS_DIR.mkdir(exist_ok=True)

if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(_MODELS_DIR)
if "TORCH_HOME" not in os.environ:
    os.environ["TORCH_HOME"] = str(_MODELS_DIR)

# Disable the annoying symlink warning on Windows when running without Developer Mode
if "HF_HUB_DISABLE_SYMLINKS_WARNING" not in os.environ:
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from .cli import main

__all__ = ["main"]
