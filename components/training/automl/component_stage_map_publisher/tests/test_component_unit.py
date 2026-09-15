"""Unit tests for publish_component_stage_map."""

import json
from pathlib import Path
from unittest import mock

import pytest

from ..component import publish_component_stage_map

PIPELINE_TABULAR = "autogluon-tabular-training-pipeline"


@pytest.fixture
def component_stage_map_artifact(tmp_path):
    """Mock KFP output artifact for component_stage_map."""
    artifact = mock.MagicMock()
    artifact.path = str(tmp_path / "component_stage_map")
    artifact.metadata = {}
    return artifact


class TestPublishComponentStageMap:
    """Unit tests for the component stage map publisher."""

    def test_component_is_callable(self):
        """Component is importable and exposes python_func."""
        assert callable(publish_component_stage_map)
        assert hasattr(publish_component_stage_map, "python_func")

    def test_publishes_stage_map_from_template(self, component_stage_map_artifact):
        """Publish tabular template as component_stage_map.json with runtime fields."""
        publish_component_stage_map.python_func(
            pipeline_id=PIPELINE_TABULAR,
            run_id="run-abc",
            component_stage_map=component_stage_map_artifact,
        )
        output_file = Path(component_stage_map_artifact.path) / "component_stage_map.json"
        assert output_file.is_file()
        document = json.loads(output_file.read_text(encoding="utf-8"))
        assert document["pipeline_id"] == PIPELINE_TABULAR
        assert document["kfp_run_id"] == "run-abc"
        assert "published_at" in document
        assert len(document["components"]) >= 1
        assert "initial_document" not in document
        assert component_stage_map_artifact.metadata["display_name"] == "Component Stage Map"
        assert component_stage_map_artifact.metadata["pipeline_id"] == PIPELINE_TABULAR

    def test_mlflow_block_disabled_when_config_unset(self, component_stage_map_artifact, monkeypatch):
        """Emit a disabled mlflow block when KFP_MLFLOW_CONFIG is unset."""
        monkeypatch.delenv("KFP_MLFLOW_CONFIG", raising=False)
        publish_component_stage_map.python_func(
            pipeline_id=PIPELINE_TABULAR,
            run_id="run-abc",
            component_stage_map=component_stage_map_artifact,
        )
        document = json.loads(
            (Path(component_stage_map_artifact.path) / "component_stage_map.json").read_text(encoding="utf-8")
        )
        assert document["mlflow"] == {"tracking_enabled": False}
        assert component_stage_map_artifact.metadata["mlflow_tracking_enabled"] == "False"

    def test_mlflow_block_enabled_from_env(self, component_stage_map_artifact, monkeypatch):
        """Populate the mlflow block from KFP_MLFLOW_CONFIG when tracking is enabled."""
        monkeypatch.setenv(
            "KFP_MLFLOW_CONFIG",
            json.dumps(
                {
                    "endpoint": "https://mlflow.example.com",
                    "experimentId": "7",
                    "parentRunId": "parent-run",
                }
            ),
        )
        publish_component_stage_map.python_func(
            pipeline_id=PIPELINE_TABULAR,
            run_id="run-abc",
            component_stage_map=component_stage_map_artifact,
        )
        document = json.loads(
            (Path(component_stage_map_artifact.path) / "component_stage_map.json").read_text(encoding="utf-8")
        )
        assert document["mlflow"]["tracking_enabled"] is True
        assert document["mlflow"]["tracking_uri"] == "https://mlflow.example.com"
        assert document["mlflow"]["run_url"] == "https://mlflow.example.com/#/experiments/7/runs/parent-run"
        assert component_stage_map_artifact.metadata["mlflow_tracking_enabled"] == "True"

    def test_rejects_empty_pipeline_id(self, component_stage_map_artifact):
        """Reject blank pipeline_id."""
        with pytest.raises(ValueError, match="pipeline_id"):
            publish_component_stage_map.python_func(
                pipeline_id="  ",
                run_id="run-1",
                component_stage_map=component_stage_map_artifact,
            )

    def test_rejects_empty_run_id(self, component_stage_map_artifact):
        """Reject blank run_id."""
        with pytest.raises(ValueError, match="run_id"):
            publish_component_stage_map.python_func(
                pipeline_id=PIPELINE_TABULAR,
                run_id="",
                component_stage_map=component_stage_map_artifact,
            )

    def test_unknown_pipeline_id_raises(self, component_stage_map_artifact):
        """Raise when template has no components for pipeline_id."""
        with pytest.raises(FileNotFoundError, match="nonexistent-pipeline"):
            publish_component_stage_map.python_func(
                pipeline_id="nonexistent-pipeline",
                run_id="run-1",
                component_stage_map=component_stage_map_artifact,
            )

    def test_strips_initial_document_from_legacy_template(self, component_stage_map_artifact, monkeypatch):
        """Omit legacy initial_document from published artifact."""
        legacy_manifest = {
            "pipeline_id": PIPELINE_TABULAR,
            "initial_document": {"components": []},
            "components": [{"id": "automl_data_loader", "stages": []}],
        }
        mock_load = mock.Mock(return_value=legacy_manifest.copy())
        monkeypatch.setattr(
            "kfp_components.components.training.automl.shared.run_status.load_pipeline_run_status_manifest",
            mock_load,
        )
        publish_component_stage_map.python_func(
            pipeline_id=PIPELINE_TABULAR,
            run_id="run-1",
            component_stage_map=component_stage_map_artifact,
        )
        mock_load.assert_called_once_with(PIPELINE_TABULAR)
        document = json.loads(
            (Path(component_stage_map_artifact.path) / "component_stage_map.json").read_text(encoding="utf-8")
        )
        assert "initial_document" not in document
