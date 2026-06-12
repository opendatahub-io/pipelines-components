"""Fixtures for text_extraction unit tests."""

import pytest
from kfp_components.components.data_processing.autorag.pytest_support import wrap_component_python_func

from ..component import text_extraction


@pytest.fixture(autouse=True)
def inject_autorag_embedded_artifact(monkeypatch, tmp_path):
    """Inject the embedded shared artifact when unit tests omit KFP runtime parameters."""
    wrap_component_python_func(text_extraction, monkeypatch, tmp_path)
