"""Tests for ManagedPipelineEntry and collect_managed_pipelines."""

from pathlib import Path

import pytest

from ..generate_managed_pipelines import (
    ManagedPipelineEntry,
    ManagedPipelineMetadataError,
    STABILITY_VALUES,
)


def test_from_managed_pipeline_dir_success(tmp_path: Path):
    """Build entry when name, stability, and path are valid."""
    repo = tmp_path
    pipe_dir = repo / "pipelines" / "training" / "p"
    pipe_dir.mkdir(parents=True)
    (pipe_dir / "pipeline.py").write_text("from kfp import dsl\n\n@dsl.pipeline\ndef p():\n    pass\n")

    metadata = {
        "name": "my_pipeline",
        "stability": "alpha",
        "managed": True,
    }
    entry = ManagedPipelineEntry.from_managed_pipeline_dir(
        dir_path=pipe_dir,
        repo_root=repo,
        metadata=metadata,
    )
    assert entry.name == "my_pipeline"
    assert entry.stability == "alpha"
    assert entry.path == "pipelines/training/p/pipeline.py"
    assert isinstance(entry.description, str)


@pytest.mark.parametrize(
    ("metadata", "reason"),
    [
        ({}, "empty metadata"),
        ({"name": "x"}, "missing stability"),
        ({"stability": "alpha"}, "missing name"),
        ({"name": "", "stability": "alpha"}, "empty name"),
        ({"name": "x", "stability": ""}, "empty stability"),
        ({"name": "x", "stability": "omega"}, "invalid stability"),
        ({"name": 1, "stability": "alpha"}, "non-str name"),
    ],
)
def test_from_managed_pipeline_dir_invalid_metadata_raises(tmp_path: Path, metadata: dict, reason: str):
    """Invalid metadata raises ManagedPipelineMetadataError."""
    _ = reason
    repo = tmp_path
    pipe_dir = repo / "pipelines" / "training" / "p"
    pipe_dir.mkdir(parents=True)
    (pipe_dir / "pipeline.py").touch()

    with pytest.raises(ManagedPipelineMetadataError):
        ManagedPipelineEntry.from_managed_pipeline_dir(
            dir_path=pipe_dir,
            repo_root=repo,
            metadata=metadata,
        )


def test_stability_values_matches_known_set():
    """Keep in sync with metadata schema (experimental, alpha, beta, stable)."""
    assert STABILITY_VALUES == frozenset({"experimental", "alpha", "beta", "stable"})
