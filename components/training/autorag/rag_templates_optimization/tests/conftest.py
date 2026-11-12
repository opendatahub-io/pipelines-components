"""Fixtures for rag_templates_optimization unit tests."""

import pytest
from kfp_components.components.training.autorag.pytest_support import wrap_component_python_func

from ..component import rag_templates_optimization


@pytest.fixture(autouse=True)
def inject_autorag_component_outputs(monkeypatch, tmp_path):
    """Inject KFP output-artifact mocks omitted by unit tests."""
    wrap_component_python_func(rag_templates_optimization, monkeypatch, tmp_path)
