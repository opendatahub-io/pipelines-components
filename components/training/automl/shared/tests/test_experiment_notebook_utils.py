"""Tests for experiment notebook generation helpers."""

# ruff: noqa: D102
# pylint: disable=missing-function-docstring

import ast
import json

from ..experiment_notebook_utils import (
    EXPERIMENT_NOTEBOOK_FILENAME,
    EXPERIMENT_NOTEBOOK_RELATIVE_PATH,
    TabularExperimentNotebookConfig,
    TimeseriesExperimentNotebookConfig,
    include_user_test_data_in_notebook,
    replace_placeholder_in_notebook,
    tabular_experiment_notebook_replacements,
    timeseries_experiment_notebook_replacements,
    write_experiment_notebook,
)


def _code_source(notebook: dict) -> str:
    return "".join(
        line for cell in notebook["cells"] if cell.get("cell_type") == "code" for line in cell.get("source", [])
    )


def _tabular_config(**overrides) -> TabularExperimentNotebookConfig:
    defaults = {
        "train_data_secret_name": "secret",
        "train_data_bucket_name": "bucket",
        "train_data_file_key": "datasets/train.csv",
        "test_data_bucket_name": "",
        "test_data_file_key": "",
        "label_column": "target",
        "task_type": "binary",
        "top_n": 3,
        "positive_class": "",
        "eval_metric": "accuracy",
        "preset": "speed",
    }
    defaults.update(overrides)
    return TabularExperimentNotebookConfig(**defaults)


def _timeseries_config(**overrides) -> TimeseriesExperimentNotebookConfig:
    defaults = {
        "train_data_secret_name": "secret",
        "train_data_bucket_name": "bucket",
        "train_data_file_key": "datasets/ts.csv",
        "test_data_bucket_name": "",
        "test_data_file_key": "",
        "target": "sales",
        "id_column": "item_id",
        "timestamp_column": "timestamp",
        "known_covariates_names": [],
        "prediction_length": 12,
        "top_n": 3,
        "eval_metric": "mean_absolute_scaled_error",
        "preset": "speed",
    }
    defaults.update(overrides)
    return TimeseriesExperimentNotebookConfig(**defaults)


class TestExperimentNotebookUtils:
    """Unit tests for experiment notebook helper functions."""

    def test_replace_placeholder_in_notebook_replaces_code_cells_only(self):
        notebook = {
            "cells": [
                {"cell_type": "markdown", "source": ["<REPLACE_TASK_TYPE>\n"]},
                {"cell_type": "code", "source": ["task_type = <REPLACE_TASK_TYPE>\n"]},
            ]
        }
        updated = replace_placeholder_in_notebook(notebook, {"<REPLACE_TASK_TYPE>": '"regression"'})
        assert updated["cells"][0]["source"] == ["<REPLACE_TASK_TYPE>\n"]
        assert updated["cells"][1]["source"] == ['task_type = "regression"\n']

    def test_include_user_test_data_in_notebook(self):
        assert not include_user_test_data_in_notebook("", "")
        assert not include_user_test_data_in_notebook("bucket", "")
        assert not include_user_test_data_in_notebook("", "datasets/test.csv")
        assert include_user_test_data_in_notebook("bucket", "datasets/test.csv")

    def test_tabular_experiment_notebook_replacements_maps_values(self):
        replacements = tabular_experiment_notebook_replacements(
            _tabular_config(
                train_data_secret_name="secret",
                train_data_file_key="datasets/train.csv",
                label_column="price",
                task_type="regression",
                eval_metric="r2",
            )
        )
        assert replacements["<REPLACE_S3_SECRET>"] == '"secret"'
        assert replacements["<REPLACE_TASK_TYPE>"] == '"regression"'
        assert replacements["<REPLACE_TOP_N>"] == "3"

    def test_tabular_replacements_escape_injection_payload(self, tmp_path):
        payload = '"; __import__("os").system("id"); #'
        replacements = tabular_experiment_notebook_replacements(
            _tabular_config(train_data_file_key=payload, eval_metric="")
        )
        destination = write_experiment_notebook(
            output_dir=tmp_path,
            kind="tabular",
            replacements=replacements,
        )
        notebook = json.loads(destination.read_text(encoding="utf-8"))
        config_source = ""
        for cell in notebook["cells"]:
            cell_source = "".join(cell.get("source", []))
            if cell.get("cell_type") == "code" and "train_data_file_key" in cell_source:
                config_source = cell_source
                break
        ast.parse(config_source)
        assert f"train_data_file_key = {json.dumps(payload)}" in config_source

    def test_timeseries_experiment_notebook_replacements_serializes_covariates(self):
        replacements = timeseries_experiment_notebook_replacements(
            _timeseries_config(
                test_data_bucket_name="test-bucket",
                test_data_file_key="datasets/test.csv",
                known_covariates_names=["promo"],
                prediction_length=24,
                top_n=2,
                preset="balanced",
            )
        )
        assert replacements["<REPLACE_KNOWN_COVARIATES_NAMES>"] == '["promo"]'
        assert replacements["<REPLACE_PREDICTION_LENGTH>"] == "24"

    def test_write_experiment_notebook_tabular(self, tmp_path):
        config = _tabular_config(
            train_data_secret_name="my-secret",
            train_data_bucket_name="my-bucket",
            positive_class="yes",
        )
        destination = write_experiment_notebook(
            output_dir=tmp_path,
            kind="tabular",
            replacements=tabular_experiment_notebook_replacements(config),
        )
        assert destination == tmp_path / EXPERIMENT_NOTEBOOK_FILENAME
        assert destination.exists()
        notebook = json.loads(destination.read_text(encoding="utf-8"))
        full_source = "".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        source = _code_source(notebook)
        assert 'pipeline_name = "autogluon-tabular-training-pipeline"' in source
        assert "kfp-connection" in full_source
        assert "run-defaults" in full_source
        assert "training-data" in full_source
        assert "pipeline-parameters" in full_source
        assert "<REPLACE_S3_SECRET>" not in source
        assert 'train_data_secret_name = "my-secret"' in source
        assert 'task_type = "binary"' in source
        assert 'positive_class = "yes"' in source
        assert '"positive_class": positive_class' in source
        assert "test_data_bucket_name" not in source
        assert "test_data_file_key" not in source
        training_data_source = next(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if "train_data_file_key =" in "".join(cell.get("source", []))
        )
        assert training_data_source.endswith('train_data_file_key = "datasets/train.csv"\n')
        assert "verify=False" not in source
        assert "KF_PIPELINES_ENDPOINT" in source
        assert "Set in this cell, then re-run:" in full_source
        assert "ELYRA_RUNTIME_CONFIG" in source
        assert "_resolve_kfp_ssl_ca_cert" in source
        assert "ssl_ca_cert" in source
        assert "KFP_TOKEN requires an HTTPS KFP host" in source
        assert "s3_verify" in source
        assert "Unsafe artifact key" in source
        assert "kfp_components" not in source
        assert "resolve_pipeline_template" in source
        assert "get_pipeline_and_versions" in source
        assert "list_pipeline_version_rows" not in source
        assert '"version_id": resolved_version_id' in source
        assert "get_pipeline_id" not in source
        assert "client.run_pipeline" in source
        assert "preflight-checks" in full_source
        assert "submit_run" not in source
        assert "head_bucket" in source
        assert "Open this run in Kubeflow Pipelines" in source
        assert "RHOAI_DASHBOARD_URL" in source
        assert "/develop-train/pipelines/runs/" in source

    def test_write_tabular_notebook_omits_empty_positive_class(self, tmp_path):
        destination = write_experiment_notebook(
            output_dir=tmp_path,
            kind="tabular",
            replacements=tabular_experiment_notebook_replacements(_tabular_config()),
        )
        source = _code_source(json.loads(destination.read_text(encoding="utf-8")))
        assert "positive_class" not in source

    def test_write_experiment_notebook_tabular_with_user_test_data(self, tmp_path):
        config = _tabular_config(
            train_data_secret_name="my-secret",
            train_data_bucket_name="my-bucket",
            test_data_bucket_name="test-bucket",
            test_data_file_key="datasets/test.csv",
            positive_class="yes",
        )
        destination = write_experiment_notebook(
            output_dir=tmp_path,
            kind="tabular",
            include_user_test_data=config.include_user_test_data,
            replacements=tabular_experiment_notebook_replacements(config),
        )
        source = _code_source(json.loads(destination.read_text(encoding="utf-8")))
        assert 'test_data_bucket_name = "test-bucket"' in source
        assert 'test_data_file_key = "datasets/test.csv"' in source
        assert '"test_data_bucket_name": test_data_bucket_name' in source
        assert 'positive_class = "yes"' in source
        assert '"positive_class": positive_class' in source

    def test_write_experiment_notebook_timeseries(self, tmp_path):
        destination = write_experiment_notebook(
            output_dir=tmp_path,
            kind="timeseries",
            replacements=timeseries_experiment_notebook_replacements(_timeseries_config()),
        )
        assert destination.name == EXPERIMENT_NOTEBOOK_FILENAME
        source = _code_source(json.loads(destination.read_text(encoding="utf-8")))
        assert 'pipeline_name = "autogluon-timeseries-training-pipeline"' in source
        assert 'target = "sales"' in source
        assert "test_data_bucket_name" not in source
        assert "test_data_file_key" not in source
        assert EXPERIMENT_NOTEBOOK_RELATIVE_PATH.endswith(EXPERIMENT_NOTEBOOK_FILENAME)
