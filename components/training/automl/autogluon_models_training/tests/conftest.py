"""Test fixtures for autogluon_models_training (embedded shared helpers)."""

import sys
from pathlib import Path

_shared_dir = str(Path(__file__).resolve().parents[2] / "shared")
if _shared_dir not in sys.path:
    sys.path.insert(0, _shared_dir)
