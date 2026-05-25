"""Test fixtures for autogluon_models_training (embedded shared helpers)."""

import sys
from pathlib import Path
from unittest import mock

import pytest

_shared_dir = str(Path(__file__).resolve().parents[2] / "shared")
if _shared_dir not in sys.path:
    sys.path.insert(0, _shared_dir)


@pytest.fixture(autouse=True)
def inject_run_status_artifact(monkeypatch, tmp_path):
    """Inject run_status_artifact when tests omit it."""
    from ..component import autogluon_models_training

    original = autogluon_models_training.python_func

    def wrapper(*args, **kwargs):
        if "run_status_artifact" not in kwargs:
            art = mock.MagicMock()
            art.path = str(tmp_path / "run_status_out")
            art.metadata = {}
            kwargs["run_status_artifact"] = art
        return original(*args, **kwargs)

    monkeypatch.setattr(autogluon_models_training, "python_func", wrapper)
