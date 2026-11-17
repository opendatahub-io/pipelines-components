"""Fixtures for models_pre_selector unit tests."""

import pytest
from kfp_components.components.training.autorag.pytest_support import wrap_component_python_func

from ..component import models_pre_selector


@pytest.fixture(autouse=True)
def inject_autorag_component_outputs(monkeypatch, tmp_path):
    """Inject KFP output-artifact mocks omitted by unit tests."""
    wrap_component_python_func(models_pre_selector, monkeypatch, tmp_path)
