"""Tests for ManagedPipelineEntry, collect_managed_pipelines, and compile_managed_pipeline."""

import json
import textwrap
from pathlib import Path

import pytest
import yaml

from ..generate_managed_pipelines import (
    METADATA_STABILITY_VALUES,
    STABILITY_TO_MANAGED_DISPLAY,
    ManagedPipelineCompilationError,
    ManagedPipelineMetadataError,
    collect_managed_pipelines,
    compile_managed_pipeline,
    main,
    managed_pipeline_entry_from_dir,
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
    entry = managed_pipeline_entry_from_dir(
        dir_path=pipe_dir,
        repo_root=repo,
        metadata=metadata,
    )
    assert entry.name == "my_pipeline"
    assert entry.stability == "Development Preview"
    assert entry.path == "pipelines/training/p/pipeline.py"
    assert isinstance(entry.description, str)
    assert entry.description == ""


def test_from_managed_pipeline_dir_success_with_pipeline_description(tmp_path: Path):
    """Build entry with description extracted from @dsl.pipeline(description=...)."""
    repo = tmp_path
    pipe_dir = repo / "pipelines" / "training" / "p"
    pipe_dir.mkdir(parents=True)
    (pipe_dir / "pipeline.py").write_text(
        'from kfp import dsl\n\n@dsl.pipeline(description="keyword description")\ndef p():\n    pass\n'
    )

    metadata = {
        "name": "my_pipeline",
        "stability": "alpha",
        "managed": True,
    }
    entry = managed_pipeline_entry_from_dir(
        dir_path=pipe_dir,
        repo_root=repo,
        metadata=metadata,
    )
    assert entry.description == "keyword description"


@pytest.mark.parametrize(
    ("metadata", "reason"),
    [
        ({}, "empty metadata"),
        ({"name": "x"}, "missing stability"),
        ({"stability": "alpha"}, "missing name"),
        ({"name": "", "stability": "alpha"}, "empty name"),
        ({"name": "x", "stability": ""}, "empty stability"),
        ({"name": "x", "stability": "omega"}, "invalid stability"),
        ({"name": "x", "stability": "experimental"}, "experimental disallowed"),
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
        managed_pipeline_entry_from_dir(
            dir_path=pipe_dir,
            repo_root=repo,
            metadata=metadata,
        )


@pytest.mark.parametrize(
    ("metadata_stability", "expected_display"),
    [
        ("alpha", "Development Preview"),
        ("beta", "Technology Preview"),
        ("stable", "General Availability"),
    ],
)
def test_stability_mapped_to_display_labels(tmp_path: Path, metadata_stability: str, expected_display: str):
    """JSON stability uses product labels, not metadata keywords."""
    repo = tmp_path
    pipe_dir = repo / "pipelines" / "x" / "p"
    pipe_dir.mkdir(parents=True)
    (pipe_dir / "pipeline.py").touch()

    entry = managed_pipeline_entry_from_dir(
        dir_path=pipe_dir,
        repo_root=repo,
        metadata={"name": "n", "stability": metadata_stability, "managed": True},
    )
    assert entry.stability == expected_display


def test_metadata_stability_values_match_validate_metadata():
    """Keep in sync with scripts/validate_metadata STABILITY_OPTIONS."""
    assert METADATA_STABILITY_VALUES == frozenset({"experimental", "alpha", "beta", "stable"})
    assert set(STABILITY_TO_MANAGED_DISPLAY) == {"alpha", "beta", "stable"}


def test_collect_managed_pipelines_missing_pipelines_root_raises(tmp_path: Path):
    """Missing pipelines/ must not yield an empty list silently."""
    repo = tmp_path
    with pytest.raises(FileNotFoundError, match="pipelines directory not found"):
        collect_managed_pipelines(repo)


def test_collect_managed_pipelines_skips_non_mapping_metadata(tmp_path: Path):
    """Non-mapping YAML content should be ignored safely."""
    repo = tmp_path
    pipe_dir = repo / "pipelines" / "training" / "p"
    pipe_dir.mkdir(parents=True)
    (pipe_dir / "pipeline.py").touch()
    (pipe_dir / "metadata.yaml").write_text("- not-a-mapping\n", encoding="utf-8")

    assert collect_managed_pipelines(repo) == []


# ---------------------------------------------------------------------------
# compile_managed_pipeline
# ---------------------------------------------------------------------------

_VALID_PIPELINE_SRC = textwrap.dedent("""\
    from kfp import dsl

    @dsl.component
    def helper(x: int) -> int:
        return x + 1

    @dsl.pipeline(name="test-pipeline")
    def test_pipeline():
        helper(x=1)
""")


class TestCompileManagedPipeline:
    """Tests for compile_managed_pipeline."""

    def test_compiles_valid_pipeline_to_yaml(self, tmp_path: Path):
        """Valid pipeline.py produces a pipeline.yaml containing a KFP pipeline spec."""
        pipeline_py = tmp_path / "pipeline.py"
        pipeline_py.write_text(_VALID_PIPELINE_SRC)
        output_yaml = tmp_path / "pipeline.yaml"

        compile_managed_pipeline(
            pipeline_py=pipeline_py,
            output_path=output_yaml,
            repo_root=tmp_path,
        )

        assert output_yaml.is_file()
        with open(output_yaml) as f:
            docs = list(yaml.safe_load_all(f))
        assert len(docs) >= 1
        spec = docs[0]
        assert isinstance(spec, dict)
        assert "deploymentSpec" in spec or "root" in spec

    def test_overwrites_existing_output(self, tmp_path: Path):
        """Pre-existing pipeline.yaml is overwritten by a fresh compilation."""
        pipeline_py = tmp_path / "pipeline.py"
        pipeline_py.write_text(_VALID_PIPELINE_SRC)
        output_yaml = tmp_path / "pipeline.yaml"
        output_yaml.write_text("stale content\n")

        compile_managed_pipeline(
            pipeline_py=pipeline_py,
            output_path=output_yaml,
            repo_root=tmp_path,
        )

        content = output_yaml.read_text()
        assert "stale content" not in content
        with open(output_yaml) as f:
            docs = list(yaml.safe_load_all(f))
        assert len(docs) >= 1

    def test_raises_when_no_pipeline_function(self, tmp_path: Path):
        """Module without @dsl.pipeline raises ManagedPipelineCompilationError."""
        pipeline_py = tmp_path / "pipeline.py"
        pipeline_py.write_text("x = 1\n")
        output_yaml = tmp_path / "pipeline.yaml"

        with pytest.raises(ManagedPipelineCompilationError, match="No @dsl.pipeline"):
            compile_managed_pipeline(
                pipeline_py=pipeline_py,
                output_path=output_yaml,
                repo_root=tmp_path,
            )

    def test_raises_when_module_cannot_be_loaded(self, tmp_path: Path):
        """Module with import errors raises ManagedPipelineCompilationError."""
        pipeline_py = tmp_path / "pipeline.py"
        pipeline_py.write_text("import nonexistent_module_xyz_123\n")
        output_yaml = tmp_path / "pipeline.yaml"

        with pytest.raises(ManagedPipelineCompilationError):
            compile_managed_pipeline(
                pipeline_py=pipeline_py,
                output_path=output_yaml,
                repo_root=tmp_path,
            )

    def test_raises_when_compilation_fails(self, tmp_path: Path):
        """Compilation write failure is wrapped in ManagedPipelineCompilationError."""
        pipeline_py = tmp_path / "pipeline.py"
        pipeline_py.write_text(_VALID_PIPELINE_SRC)
        output_yaml = tmp_path / "no_such_dir" / "pipeline.yaml"

        with pytest.raises(ManagedPipelineCompilationError, match="Failed to compile"):
            compile_managed_pipeline(
                pipeline_py=pipeline_py,
                output_path=output_yaml,
                repo_root=tmp_path,
            )


# ---------------------------------------------------------------------------
# main() integration
# ---------------------------------------------------------------------------


def test_main_generates_json_and_compiles_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """main() writes managed-pipelines.json AND compiles pipeline.yaml for each managed pipeline."""
    repo = tmp_path
    pipe_dir = repo / "pipelines" / "training" / "p"
    pipe_dir.mkdir(parents=True)
    (pipe_dir / "pipeline.py").write_text(_VALID_PIPELINE_SRC)
    (pipe_dir / "metadata.yaml").write_text(
        yaml.dump({"name": "my_pipeline", "stability": "alpha", "managed": True}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "scripts.generate_managed_pipelines.generate_managed_pipelines.get_repo_root",
        lambda: repo,
    )
    monkeypatch.setattr("sys.argv", ["prog"])

    rc = main()

    assert rc == 0

    json_path = repo / "managed-pipelines.json"
    assert json_path.is_file()
    entries = json.loads(json_path.read_text())
    assert len(entries) == 1
    assert entries[0]["name"] == "my_pipeline"

    compiled = pipe_dir / "pipeline.yaml"
    assert compiled.is_file()
    with open(compiled) as f:
        docs = list(yaml.safe_load_all(f))
    assert len(docs) >= 1
    assert isinstance(docs[0], dict)
