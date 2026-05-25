"""Test fixtures for tabular_data_loader (embedded shared run_status helpers)."""

import sys
from pathlib import Path
from unittest import mock

import pytest

_SHARED_DIR = str(Path(__file__).resolve().parents[4] / "training" / "automl" / "shared")
_RUN_STATUS_PIPELINE_ID = "autogluon-tabular-training-pipeline"


def _make_run_status_artifact():
    art = mock.MagicMock()
    art.path = "/tmp/run_status_artifact"
    art.metadata = {}
    return art


def _make_embedded_artifact():
    art = mock.MagicMock()
    art.path = _SHARED_DIR
    return art


@pytest.fixture(autouse=True)
def inject_run_status_defaults(monkeypatch):
    """Inject KFP placeholder args when tests omit pipeline_name and run_id."""
    if _SHARED_DIR not in sys.path:
        sys.path.insert(0, _SHARED_DIR)

    from ..component import automl_data_loader

    original = automl_data_loader.python_func

    def wrapper(*args, **kwargs):
        kwargs.setdefault("pipeline_name", "test-pipeline")
        kwargs.setdefault("run_id", "test-run-id")
        kwargs.setdefault("run_status_pipeline_id", _RUN_STATUS_PIPELINE_ID)
        kwargs.setdefault("run_status_artifact", _make_run_status_artifact())
        kwargs.setdefault("embedded_artifact", _make_embedded_artifact())
        return original(*args, **kwargs)

    monkeypatch.setattr(automl_data_loader, "python_func", wrapper)
