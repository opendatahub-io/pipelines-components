"""Fixtures for leaderboard_evaluation unit tests."""

import pytest
from kfp_components.components.training.autorag.pytest_support import wrap_component_python_func

from ..component import leaderboard_evaluation


@pytest.fixture(autouse=True)
def inject_autorag_embedded_artifact(monkeypatch, tmp_path):
    """Inject the embedded shared artifact when unit tests omit KFP runtime parameters."""
    wrap_component_python_func(leaderboard_evaluation, monkeypatch, tmp_path)
