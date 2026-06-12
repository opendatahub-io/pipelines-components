"""Fixtures for search_space_preparation unit tests."""

import pytest
from kfp_components.components.training.autorag.pytest_support import wrap_component_python_func

from ..component import search_space_preparation


@pytest.fixture(autouse=True)
def inject_autorag_embedded_artifact(monkeypatch, tmp_path):
    """Inject the embedded shared artifact when unit tests omit KFP runtime parameters."""
    wrap_component_python_func(search_space_preparation, monkeypatch, tmp_path)
