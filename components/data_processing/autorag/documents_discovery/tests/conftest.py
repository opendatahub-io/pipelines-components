"""Fixtures for documents_discovery unit tests."""

import pytest
from kfp_components.components.data_processing.autorag.pytest_support import wrap_component_python_func

from ..component import documents_discovery


@pytest.fixture(autouse=True)
def inject_autorag_component_outputs(monkeypatch, tmp_path):
    """Inject KFP output-artifact mocks omitted by unit tests."""
    wrap_component_python_func(documents_discovery, monkeypatch, tmp_path)
