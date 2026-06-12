"""Fixtures for test_data_loader unit tests."""

import pytest
from kfp_components.components.data_processing.autorag.pytest_support import wrap_component_python_func

from ..component import test_data_loader


@pytest.fixture(autouse=True)
def inject_autorag_embedded_artifact(monkeypatch, tmp_path):
    """Inject the embedded shared artifact when unit tests omit KFP runtime parameters."""
    wrap_component_python_func(test_data_loader, monkeypatch, tmp_path)
