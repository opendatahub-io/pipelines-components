"""Unit tests for AutoRAG publish_component_stage_map."""

import json
from pathlib import Path
from unittest import mock

import pytest

from ..component import publish_component_stage_map

PIPELINE_RAG_OPTIMIZATION = "documents-rag-optimization-pipeline"


@pytest.fixture
def component_stage_map_artifact(tmp_path):
    """Mock KFP output artifact for component_stage_map."""
    artifact = mock.MagicMock()
    artifact.path = str(tmp_path / "component_stage_map")
    artifact.metadata = {}
    return artifact


class TestPublishComponentStageMap:
    """Unit tests for the AutoRAG component stage map publisher."""

    def test_component_is_callable(self):
        """Component is importable and exposes python_func."""
        assert callable(publish_component_stage_map)
        assert hasattr(publish_component_stage_map, "python_func")

    def test_publishes_stage_map_from_template(self, component_stage_map_artifact):
        """Publish AutoRAG template as component_stage_map.json with runtime fields."""
        publish_component_stage_map.python_func(
            pipeline_id=PIPELINE_RAG_OPTIMIZATION,
            run_id="run-abc",
            component_stage_map=component_stage_map_artifact,
        )
        output_file = Path(component_stage_map_artifact.path) / "component_stage_map.json"
        assert output_file.is_file()
        document = json.loads(output_file.read_text(encoding="utf-8"))
        assert document["pipeline_id"] == PIPELINE_RAG_OPTIMIZATION
        assert document["kfp_run_id"] == "run-abc"
        assert "published_at" in document
        assert len(document["components"]) >= 1
        assert component_stage_map_artifact.metadata["display_name"] == "Component Stage Map"

    def test_unknown_pipeline_id_raises(self, component_stage_map_artifact):
        """Raise when template has no components for pipeline_id."""
        with pytest.raises(FileNotFoundError, match="nonexistent-pipeline"):
            publish_component_stage_map.python_func(
                pipeline_id="nonexistent-pipeline",
                run_id="run-1",
                component_stage_map=component_stage_map_artifact,
            )
