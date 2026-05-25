"""Test fixtures for tabular_data_loader (embedded shared run_status helpers)."""

import sys
from pathlib import Path
from unittest import mock

import pytest

_shared_dir = str(Path(__file__).resolve().parents[4] / "training" / "automl" / "shared")
if _shared_dir not in sys.path:
    sys.path.insert(0, _shared_dir)

from run_status import PIPELINE_TABULAR_TRAINING  # noqa: E402


def _make_run_status_artifact():
    art = mock.MagicMock()
    art.path = "/tmp/run_status_artifact"
    art.metadata = {}
    return art


@pytest.fixture(autouse=True)
def inject_run_status_defaults(monkeypatch):
    """Inject KFP placeholder args when tests omit pipeline_name and run_id."""
    from ..component import automl_data_loader

    original = automl_data_loader.python_func

    def wrapper(*args, **kwargs):
        kwargs.setdefault("pipeline_name", "test-pipeline")
        kwargs.setdefault("run_id", "test-run-id")
        kwargs.setdefault("run_status_pipeline_id", PIPELINE_TABULAR_TRAINING)
        kwargs.setdefault("run_status_artifact", _make_run_status_artifact())
        return original(*args, **kwargs)

    monkeypatch.setattr(automl_data_loader, "python_func", wrapper)
