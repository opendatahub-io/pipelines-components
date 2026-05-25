"""Test fixtures for tabular_data_loader (embedded MLflow shared helpers)."""

import sys
from pathlib import Path
from unittest import mock

import pytest

_shared_dir = str(Path(__file__).resolve().parents[4] / "training" / "automl" / "shared")
if _shared_dir not in sys.path:
    sys.path.insert(0, _shared_dir)


def _make_mlflow_tracking_artifact():
    art = mock.MagicMock()
    art.path = "/tmp/mlflow_tracking_test"
    return art


@pytest.fixture(autouse=True)
def inject_mlflow_tracking_defaults(monkeypatch):
    """Inject MLflow tracking artifact and KFP placeholders when tests omit them."""
    from ..component import automl_data_loader

    original = automl_data_loader.python_func

    def wrapper(*args, **kwargs):
        kwargs.setdefault("mlflow_tracking_artifact", _make_mlflow_tracking_artifact())
        kwargs.setdefault("pipeline_name", "test-pipeline")
        kwargs.setdefault("run_id", "test-run-id")
        return original(*args, **kwargs)

    monkeypatch.setattr(automl_data_loader, "python_func", wrapper)
