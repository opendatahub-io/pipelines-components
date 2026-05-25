"""Expose shared modules the same way KFP embedded tasks do (flat imports)."""

import sys
from pathlib import Path

_shared_dir = str(Path(__file__).resolve().parents[1])
if _shared_dir not in sys.path:
    sys.path.insert(0, _shared_dir)
