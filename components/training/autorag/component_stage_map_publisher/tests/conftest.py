"""Fixtures for component_stage_map_publisher unit tests."""

import pytest
from kfp_components.components.training.autorag.pytest_support import autorag_shared_dir, wrap_component_python_func

from ..component import publish_component_stage_map


@pytest.fixture(autouse=True)
def inject_autorag_embedded_artifact(monkeypatch, tmp_path):
    """Inject the embedded shared artifact when unit tests omit KFP runtime parameters."""
    wrap_component_python_func(
        publish_component_stage_map,
        monkeypatch,
        tmp_path,
        embedded_path=str(autorag_shared_dir() / "run_status_templates"),
    )
