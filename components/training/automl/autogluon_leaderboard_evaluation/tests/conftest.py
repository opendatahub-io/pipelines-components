"""Test fixtures for leaderboard_evaluation (embedded shared run_status helpers)."""

import sys
from pathlib import Path
from unittest import mock

import pytest

_shared_dir = str(Path(__file__).resolve().parents[2] / "shared")
if _shared_dir not in sys.path:
    sys.path.insert(0, _shared_dir)


@pytest.fixture(autouse=True)
def inject_workspace_path(monkeypatch, tmp_path):
    """Inject workspace_path when tests omit it."""
    from ..component import leaderboard_evaluation

    original = leaderboard_evaluation.python_func

    def wrapper(*args, **kwargs):
        kwargs.setdefault("workspace_path", str(tmp_path))
        run_status = mock.MagicMock()
        run_status.path = str(tmp_path / "run_status_out")
        run_status.metadata = {}
        kwargs.setdefault("run_status_artifact", run_status)
        return original(*args, **kwargs)

    monkeypatch.setattr(leaderboard_evaluation, "python_func", wrapper)
