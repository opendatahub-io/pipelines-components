"""Skip optional generator tests without Python 3.12 and ai4rag."""

import importlib.util
import sys

if sys.version_info < (3, 12) or importlib.util.find_spec("ai4rag") is None:
    collect_ignore_glob = ["test_*.py"]
