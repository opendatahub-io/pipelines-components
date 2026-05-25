"""Tests for shared MLflow tracking helpers."""

import json
import os
from pathlib import Path
from unittest import mock

import pytest

from ..mlflow_tracking import (
    TRACKING_ARTIFACT_FILENAME,
    build_mlflow_run_url,
    build_tracking_artifact_payload,
    mlflow_enabled,
    parse_model_display_name,
    write_tracking_artifact,
)


class TestMlflowEnabled:
    def test_disabled_when_uri_unset(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        assert mlflow_enabled() is False

    def test_enabled_when_uri_set(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
        assert mlflow_enabled() is True


class TestBuildMlflowRunUrl:
    def test_builds_url_with_workspace(self):
        url = build_mlflow_run_url(
            "https://mlflow.example.com/",
            "5",
            "abc123",
            "my-project",
        )
        assert url == "https://mlflow.example.com/#/experiments/5/runs/abc123?workspace=my-project"

    def test_returns_none_when_missing_ids(self):
        assert build_mlflow_run_url("https://mlflow.example.com", None, "abc", None) is None


class TestTrackingArtifactPayload:
    def test_disabled_payload(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        payload = build_tracking_artifact_payload(kfp_run_id="run-1", pipeline_name="my-pipeline")
        assert payload == {
            "tracking_enabled": False,
            "kfp_run_id": "run-1",
            "pipeline_name": "my-pipeline",
        }

    def test_enabled_payload(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")
        monkeypatch.setenv("MLFLOW_EXPERIMENT_ID", "5")
        monkeypatch.setenv("MLFLOW_RUN_ID", "parent-run")
        monkeypatch.setenv("MLFLOW_WORKSPACE", "ds-project")
        payload = build_tracking_artifact_payload(kfp_run_id="kfp-1", pipeline_name="tabular-pipeline")
        assert payload["tracking_enabled"] is True
        assert payload["mlflow_run_id"] == "parent-run"
        assert "mlflow_run_url" in payload
        assert payload["mlflow_run_url"].startswith("https://mlflow.example.com")


class TestWriteTrackingArtifact:
    def test_writes_json_file(self, tmp_path):
        payload = {"tracking_enabled": False, "kfp_run_id": "r1", "pipeline_name": "p1"}
        write_tracking_artifact(str(tmp_path), payload)
        out_file = tmp_path / TRACKING_ARTIFACT_FILENAME
        assert out_file.exists()
        assert json.loads(out_file.read_text()) == payload


class TestParseModelDisplayName:
    @pytest.mark.parametrize(
        ("name", "model_type", "stack_level"),
        [
            ("LightGBM_BAG_L1_FULL", "LightGBM", 1),
            ("WeightedEnsemble_L3_FULL", "WeightedEnsemble", 3),
            ("Naive", "Naive", 1),
        ],
    )
    def test_parses_autogluon_names(self, name, model_type, stack_level):
        assert parse_model_display_name(name) == (model_type, stack_level)


class TestLogTabularTrainingToMlflow:
    def test_skips_when_mlflow_disabled(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        from ..mlflow_tracking import log_tabular_training_to_mlflow

        log_tabular_training_to_mlflow(
            models_artifact_path="/models",
            models_artifact_uri="s3://models",
            model_names=["M1_FULL"],
            eval_results_by_model={"M1_FULL": {"accuracy": 0.9}},
            eval_metric="accuracy",
            task_type="binary",
            pipeline_name="p",
            run_id="r",
            model_config={"preset": "medium"},
            top_n=1,
            label_column="y",
        )
