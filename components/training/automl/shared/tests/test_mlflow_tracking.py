"""Unit tests for MLflow tracking helpers."""

import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest
from kfp_components.components.training.automl.shared import mlflow_tracking
from kfp_components.components.training.automl.shared.mlflow_tracking import (
    MLFLOW_WORKSPACE_HEADER,
    MlflowConfig,
    _metrics_for_task,
    _normalize_model_metrics,
    build_mlflow_run_url,
    build_mlflow_stage_map_block,
    configure_mlflow_client,
    display_model_run_name,
    experiment_run_logger,
    is_mlflow_enabled,
    parse_model_name,
    resolve_mlflow_config,
)


def _set_kfp_mlflow_config(
    monkeypatch,
    *,
    endpoint="https://mlflow.example.com",
    experiment_id="1",
    parent_run_id="parent-run",
    workspace="",
    workspaces_enabled=None,
    auth_type="",
    timeout="30s",
    extra=None,
):
    """Set KFP_MLFLOW_CONFIG to a JSON blob matching what the platform injects."""
    cfg = {
        "endpoint": endpoint,
        "experimentId": experiment_id,
        "parentRunId": parent_run_id,
        "authType": auth_type,
        "timeout": timeout,
    }
    if workspace:
        cfg["workspace"] = workspace
        cfg["workspacesEnabled"] = True if workspaces_enabled is None else workspaces_enabled
    elif workspaces_enabled is not None:
        cfg["workspacesEnabled"] = workspaces_enabled
    if extra:
        cfg.update(extra)
    monkeypatch.setenv("KFP_MLFLOW_CONFIG", json.dumps(cfg))
    return cfg


class TestMlflowTrackingHelpers:
    """Tests for MLflow env helpers and tracking artifact builders."""

    def test_is_mlflow_enabled_false_when_unset(self, monkeypatch):
        """Return False when KFP_MLFLOW_CONFIG is unset."""
        monkeypatch.delenv("KFP_MLFLOW_CONFIG", raising=False)
        assert is_mlflow_enabled() is False

    def test_is_mlflow_enabled_true_when_set(self, monkeypatch):
        """Return True when KFP_MLFLOW_CONFIG carries an endpoint."""
        _set_kfp_mlflow_config(monkeypatch)
        assert is_mlflow_enabled() is True

    def test_resolve_mlflow_config_parses_blob(self, monkeypatch):
        """Parse endpoint/parentRunId/experimentId/workspace/authType from the blob."""
        _set_kfp_mlflow_config(
            monkeypatch,
            endpoint="https://mlflow.example.com/mlflow",
            experiment_id="8",
            parent_run_id="6aae16e6",
            workspace="ns-automl-benchmarking",
            auth_type="kubernetes",
        )
        config = resolve_mlflow_config()
        assert config is not None
        assert config.mode == "kfp"
        assert config.tracking_uri == "https://mlflow.example.com/mlflow"
        assert config.experiment_id == "8"
        assert config.run_id == "6aae16e6"
        assert config.workspace == "ns-automl-benchmarking"
        assert config.auth_type == "kubernetes"

    def test_resolve_mlflow_config_none_when_absent(self, monkeypatch):
        """Return None when KFP_MLFLOW_CONFIG is not set."""
        monkeypatch.delenv("KFP_MLFLOW_CONFIG", raising=False)
        assert resolve_mlflow_config() is None

    def test_resolve_mlflow_config_none_when_invalid_json(self, monkeypatch):
        """Return None (not raise) when KFP_MLFLOW_CONFIG is malformed."""
        monkeypatch.setenv("KFP_MLFLOW_CONFIG", "{not json")
        assert resolve_mlflow_config() is None

    def test_resolve_mlflow_config_none_when_no_endpoint(self, monkeypatch):
        """Return None when the blob has no endpoint."""
        monkeypatch.setenv("KFP_MLFLOW_CONFIG", json.dumps({"parentRunId": "x"}))
        assert resolve_mlflow_config() is None

    def test_resolve_mlflow_config_ignores_workspace_when_disabled(self, monkeypatch):
        """Drop the workspace when workspacesEnabled is false."""
        _set_kfp_mlflow_config(monkeypatch, workspace="ns-automl-benchmarking", workspaces_enabled=False)
        config = resolve_mlflow_config()
        assert config is not None
        assert config.workspace == ""

    def test_resolve_mlflow_config_none_when_kubernetes_auth_over_http(self, monkeypatch):
        """Disable tracking for kubernetes auth over cleartext HTTP (bearer-token leak, CWE-319)."""
        _set_kfp_mlflow_config(monkeypatch, endpoint="http://mlflow.example.com", auth_type="kubernetes")
        assert resolve_mlflow_config() is None

    def test_resolve_mlflow_config_allows_non_kubernetes_auth_over_http(self, monkeypatch):
        """Non-kubernetes auth sends no bearer token, so an HTTP endpoint is left enabled."""
        _set_kfp_mlflow_config(monkeypatch, endpoint="http://mlflow.example.com", auth_type="")
        config = resolve_mlflow_config()
        assert config is not None
        assert config.tracking_uri == "http://mlflow.example.com"

    def test_build_mlflow_run_url(self):
        """Build a deep-link URL for the MLflow UI."""
        url = build_mlflow_run_url("https://mlflow.example.com/", "5", "abc123")
        assert url == "https://mlflow.example.com/#/experiments/5/runs/abc123"

    def test_build_mlflow_stage_map_block_disabled(self, monkeypatch):
        """Emit minimal mlflow block when tracking is disabled."""
        monkeypatch.delenv("KFP_MLFLOW_CONFIG", raising=False)
        block = build_mlflow_stage_map_block()
        assert block == {"tracking_enabled": False}

    def test_build_mlflow_stage_map_block_uri_only(self, monkeypatch):
        """Include tracking URI when only an endpoint (no ids/workspace) is available."""
        _set_kfp_mlflow_config(
            monkeypatch,
            experiment_id="",
            parent_run_id="",
        )
        block = build_mlflow_stage_map_block()
        assert block == {
            "tracking_enabled": True,
            "tracking_uri": "https://mlflow.example.com",
        }

    def test_build_mlflow_stage_map_block_full(self, monkeypatch):
        """Include MLflow IDs and workspace when the full blob is present."""
        _set_kfp_mlflow_config(
            monkeypatch,
            experiment_id="7",
            parent_run_id="parent-run",
            workspace="ds-project",
        )
        block = build_mlflow_stage_map_block()
        assert block == {
            "tracking_enabled": True,
            "tracking_uri": "https://mlflow.example.com",
            "experiment_id": "7",
            "run_id": "parent-run",
            "workspace": "ds-project",
            "run_url": "https://mlflow.example.com/#/experiments/7/runs/parent-run",
        }

    def test_configure_uses_native_set_workspace_when_available(self):
        """Prefer mlflow.set_workspace when the client exposes it."""
        mlflow = mock.Mock(spec=["set_tracking_uri", "set_workspace"])
        config = MlflowConfig(
            mode="kfp",
            tracking_uri="https://mlflow.example.com/mlflow",
            workspace="ns-automl-benchmarking",
        )
        configure_mlflow_client(mlflow, config)
        mlflow.set_tracking_uri.assert_called_once_with("https://mlflow.example.com/mlflow")
        mlflow.set_workspace.assert_called_once_with("ns-automl-benchmarking")

    def test_configure_injects_workspace_header_when_no_set_workspace(self, monkeypatch):
        """Fall back to the x-mlflow-workspace request header on plain MLflow."""
        import requests

        recorded: list[tuple[str, dict | None]] = []

        def fake_request(self, method, url, *args, **kwargs):
            recorded.append((url, kwargs.get("headers")))
            return "ok"

        monkeypatch.setattr(requests.Session, "request", fake_request)

        mlflow = mock.Mock(spec=["set_tracking_uri"])  # no set_workspace attribute
        config = MlflowConfig(
            mode="kfp",
            tracking_uri="https://mlflow.example.com/mlflow",
            workspace="ns-automl-benchmarking",
        )
        configure_mlflow_client(mlflow, config)

        session = requests.Session()
        session.request("GET", "https://mlflow.example.com/api/2.0/mlflow/experiments/search")
        session.request("GET", "https://s3.other.example.com/bucket/object")

        tracking_headers = recorded[0][1]
        assert tracking_headers is not None
        assert tracking_headers[MLFLOW_WORKSPACE_HEADER] == "ns-automl-benchmarking"

        other_host_headers = recorded[1][1] or {}
        assert MLFLOW_WORKSPACE_HEADER not in other_host_headers

    def test_configure_skips_workspace_when_empty(self):
        """No workspace handling when the workspace is empty."""
        mlflow = mock.Mock(spec=["set_tracking_uri", "set_workspace"])
        config = MlflowConfig(mode="kfp", tracking_uri="https://mlflow.example.com", workspace="")
        configure_mlflow_client(mlflow, config)
        mlflow.set_workspace.assert_not_called()

    def test_configure_kubernetes_auth_sets_token(self, monkeypatch, tmp_path):
        """Populate MLFLOW_TRACKING_TOKEN from the SA token file when authType=kubernetes."""
        token_file = tmp_path / "token"
        token_file.write_text("sa-token-value\n", encoding="utf-8")
        monkeypatch.setattr(mlflow_tracking, "SERVICE_ACCOUNT_TOKEN_PATH", str(token_file))
        monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)

        mlflow = mock.Mock(spec=["set_tracking_uri"])
        config = MlflowConfig(mode="kfp", tracking_uri="https://mlflow.example.com", auth_type="kubernetes")
        configure_mlflow_client(mlflow, config)

        assert os.environ["MLFLOW_TRACKING_TOKEN"] == "sa-token-value"

    def test_configure_kubernetes_auth_missing_token_does_not_raise(self, monkeypatch, tmp_path):
        """A missing SA token leaves auth unset without raising."""
        monkeypatch.setattr(mlflow_tracking, "SERVICE_ACCOUNT_TOKEN_PATH", str(tmp_path / "absent"))
        monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)

        mlflow = mock.Mock(spec=["set_tracking_uri"])
        config = MlflowConfig(mode="kfp", tracking_uri="https://mlflow.example.com", auth_type="kubernetes")
        configure_mlflow_client(mlflow, config)

        assert "MLFLOW_TRACKING_TOKEN" not in os.environ

    def test_configure_applies_request_timeout(self, monkeypatch):
        """Translate the platform ``timeout`` (e.g. "30s") into MLFLOW_HTTP_REQUEST_TIMEOUT seconds."""
        monkeypatch.delenv("MLFLOW_HTTP_REQUEST_TIMEOUT", raising=False)
        mlflow = mock.Mock(spec=["set_tracking_uri"])
        config = MlflowConfig(mode="kfp", tracking_uri="https://mlflow.example.com", timeout="30s")
        configure_mlflow_client(mlflow, config)
        assert os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] == "30"

    def test_configure_skips_timeout_when_absent_or_invalid(self, monkeypatch):
        """Leave MLFLOW_HTTP_REQUEST_TIMEOUT unset for empty or unparseable timeouts."""
        monkeypatch.delenv("MLFLOW_HTTP_REQUEST_TIMEOUT", raising=False)
        mlflow = mock.Mock(spec=["set_tracking_uri"])
        for bad in ("", "  ", "abc"):
            config = MlflowConfig(mode="kfp", tracking_uri="https://mlflow.example.com", timeout=bad)
            configure_mlflow_client(mlflow, config)
            assert "MLFLOW_HTTP_REQUEST_TIMEOUT" not in os.environ

    @pytest.mark.parametrize(
        ("model_name", "expected_type", "expected_level"),
        [
            ("WeightedEnsemble_L3_FULL", "WeightedEnsemble", 3),
            ("CatBoost_BAG_L1_FULL", "CatBoost", 1),
            ("Naive", "Naive", 1),
        ],
    )
    def test_parse_model_name(self, model_name, expected_type, expected_level):
        """Parse model family and stack level from AutoGluon model names."""
        model_type, stack_level = parse_model_name(model_name)
        assert model_type == expected_type
        assert stack_level == expected_level

    @pytest.mark.parametrize(
        ("model_name", "expected_display"),
        [
            ("WeightedEnsemble_L3_FULL", "WeightedEnsemble_L3"),
            ("CatBoost_BAG_L1_FULL", "CatBoost_BAG_L1"),
            ("Naive", "Naive"),
        ],
    )
    def test_display_model_run_name(self, model_name, expected_display):
        """Strip refit suffix from MLflow child run names."""
        assert display_model_run_name(model_name) == expected_display

    def test_normalize_model_metrics_flattens_test_data(self):
        """Flatten nested test_data metrics from artifact metadata."""
        payload = {"test_data": {"accuracy": 0.91, "f1": 0.88}}
        assert _normalize_model_metrics(payload) == {"accuracy": 0.91, "f1": 0.88}

    def test_metrics_for_task_binary(self):
        """Log every computed scalar metric (incl. log_loss); non-scalars are skipped."""
        metrics = _metrics_for_task(
            "binary",
            {"accuracy": 0.91, "f1": 0.88, "roc_auc": 0.95, "log_loss": 0.31, "note": "n/a"},
        )
        assert metrics == {"accuracy": 0.91, "f1": 0.88, "roc_auc": 0.95, "log_loss": 0.31}


def _mock_run_context(run_id: str, experiment_id: str = "1") -> mock.MagicMock:
    ctx = mock.MagicMock()
    ctx.info.run_id = run_id
    ctx.info.experiment_id = experiment_id
    ctx.data.tags = {}
    ctx.__enter__ = mock.Mock(return_value=ctx)
    ctx.__exit__ = mock.Mock(return_value=False)
    return ctx


def _write_model_metrics(base_path: Path, model_name: str, metrics: dict) -> Path:
    metrics_dir = base_path / model_name / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return metrics_dir


def _make_mock_mlflow(parent_ctx, child_ctxs) -> mock.MagicMock:
    """Build a mock ``mlflow`` module for the incremental logger lifecycle.

    ``start_run`` is called once for the parent run then once per model (nested child);
    ``active_run`` is read once in ``log_header`` (parent) then once per ``log_model`` (child).
    """
    mock_mlflow = mock.MagicMock()
    mock_mlflow.start_run.side_effect = [parent_ctx, *child_ctxs]
    mock_mlflow.active_run.side_effect = [parent_ctx, *child_ctxs]
    mock_mlflow.entities.RunTag = mock.Mock(side_effect=lambda key, value: (key, value))
    mock_mlflow.MlflowClient.return_value.get_run.return_value = parent_ctx
    return mock_mlflow


def _run_logger_lifecycle(
    mock_mlflow,
    *,
    tmp_path: Path,
    model_names: list[str],
    task_type: str = "binary",
    eval_metric: str = "accuracy",
    metrics_by_model: dict | None = None,
    total_fit_time_seconds: float | None = None,
):
    """Drive a full ``experiment_run_logger`` lifecycle and return ``result()``."""
    metrics_by_model = metrics_by_model or {name: {"accuracy": 0.9} for name in model_names}
    with mock.patch.dict(sys.modules, {"mlflow": mock_mlflow}):
        with experiment_run_logger(
            task_type=task_type,
            eval_metric=eval_metric,
        ) as run_logger:
            run_logger.log_header(
                pipeline_name="autogluon-tabular-training-pipeline",
                kfp_run_id="run-1",
                preset="speed",
                top_n=len(model_names),
            )
            for model_name in model_names:
                run_logger.log_model(
                    model_name=model_name,
                    model_uri=f"s3://bucket/models/{model_name}",
                    metrics={"test_data": metrics_by_model[model_name]},
                )
            run_logger.finalize(
                total_fit_time_seconds=total_fit_time_seconds,
            )
    return run_logger.result()


class TestMlflowExperimentLogger:
    """Tests for the incremental ``experiment_run_logger`` / ``MlflowExperimentLogger`` API."""

    def test_skips_when_mlflow_disabled(self, tmp_path, monkeypatch):
        """Yield a no-op logger (no MLflow import) when tracking is disabled."""
        monkeypatch.delenv("KFP_MLFLOW_CONFIG", raising=False)
        with experiment_run_logger(task_type="binary", eval_metric="accuracy") as run_logger:
            assert run_logger.enabled is False
            run_logger.log_header(pipeline_name="p", kfp_run_id="run-1")
            run_logger.log_model(
                model_name="M_FULL",
                model_uri="s3://bucket/models/M_FULL",
                metrics={"test_data": {"accuracy": 0.9}},
            )
            run_logger.finalize()
        logged, tracking_info = run_logger.result()
        assert logged is False
        assert tracking_info == {}

    def test_logs_parent_and_child_runs_kfp_mode(self, tmp_path, monkeypatch):
        """Resume the parent run and open one nested child run for a single model."""
        _set_kfp_mlflow_config(monkeypatch, parent_run_id="parent-run", experiment_id="1")

        model_name = "LightGBM_BAG_L1_FULL"
        mock_mlflow = _make_mock_mlflow(_mock_run_context("parent-run", "1"), [_mock_run_context("child-run-1", "1")])
        logged, tracking_info = _run_logger_lifecycle(
            mock_mlflow,
            tmp_path=tmp_path,
            model_names=[model_name],
            metrics_by_model={model_name: {"accuracy": 0.91, "f1": 0.88}},
        )

        assert logged is True
        assert tracking_info["tracking_mode"] == "kfp"
        assert tracking_info["mlflow_child_run_count"] == "1"
        assert tracking_info["mlflow_child_run_ids"] == "child-run-1"
        mock_mlflow.start_run.assert_any_call(run_id="parent-run")
        mock_mlflow.start_run.assert_any_call(run_name="LightGBM_BAG_L1", nested=True)
        mock_mlflow.set_tags.assert_called()
        mock_mlflow.log_params.assert_called()
        mock_mlflow.log_metrics.assert_called()
        mock_mlflow.log_artifact.assert_not_called()
        mock_mlflow.log_artifacts.assert_not_called()

    def test_reuses_live_child_run_created_by_callback(self, tmp_path, monkeypatch):
        """Refit reopens the live run the progress callback created for a model (no duplicate)."""
        _set_kfp_mlflow_config(monkeypatch, parent_run_id="parent-run", experiment_id="1")

        model_name = "LightGBM_BAG_L1_FULL"
        display_name = "LightGBM_BAG_L1"
        _write_model_metrics(tmp_path, model_name, {"accuracy": 0.91})
        html_path = tmp_path / "leaderboard.html"
        html_path.write_text("<html></html>", encoding="utf-8")

        mock_mlflow = _make_mock_mlflow(_mock_run_context("parent-run", "1"), [_mock_run_context("live-child-1", "1")])
        with mock.patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            with experiment_run_logger(task_type="binary", eval_metric="accuracy") as run_logger:
                run_logger.log_header(pipeline_name="p", kfp_run_id="run-1", top_n=1)
                # Simulate the progress callback having created a live run during fit().
                run_logger._live_child_runs[display_name] = "live-child-1"
                run_logger.log_model(
                    model_name=model_name,
                    model_uri=f"s3://bucket/models/{model_name}",
                    metrics={"test_data": {"accuracy": 0.91}},
                )
                run_logger.finalize()
        logged, tracking_info = run_logger.result()

        assert logged is True
        # Reopened the live run by id (nested, since the parent stays active) and did NOT
        # open a second nested run via run_name for it.
        mock_mlflow.start_run.assert_any_call(run_id="live-child-1", nested=True)
        new_child_calls = [c for c in mock_mlflow.start_run.call_args_list if c.kwargs.get("run_name")]
        assert new_child_calls == []
        assert tracking_info["mlflow_child_run_ids"] == "live-child-1"

    def test_prune_live_child_runs_deletes_non_top_n(self, tmp_path, monkeypatch):
        """Pruning deletes the live runs for candidates outside the top-N and keeps the rest."""
        _set_kfp_mlflow_config(monkeypatch, parent_run_id="parent-run", experiment_id="1")

        mock_mlflow = _make_mock_mlflow(_mock_run_context("parent-run", "1"), [])
        mock_client = mock.MagicMock()
        mock_mlflow.MlflowClient.return_value = mock_client
        with mock.patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            with experiment_run_logger(task_type="binary", eval_metric="accuracy") as run_logger:
                run_logger.log_header(pipeline_name="p", kfp_run_id="run-1", top_n=1)
                # The callback registered a live run for every candidate trained during fit().
                run_logger._live_child_runs.update(
                    {
                        "LightGBM_BAG_L1": "run-lgbm",
                        "CatBoost_BAG_L1": "run-cat",
                        "WeightedEnsemble_L2": "run-ens",
                    }
                )
                run_logger.prune_live_child_runs(["LightGBM_BAG_L1"])

        # Only the two non-top-N runs are deleted; the kept one survives.
        deleted = {c.args[0] for c in mock_client.delete_run.call_args_list}
        assert deleted == {"run-cat", "run-ens"}
        assert run_logger._live_child_runs == {"LightGBM_BAG_L1": "run-lgbm"}

    def test_prune_live_child_runs_swallows_delete_errors(self, tmp_path, monkeypatch):
        """A failing delete is logged, not raised, and does not block the others."""
        _set_kfp_mlflow_config(monkeypatch, parent_run_id="parent-run", experiment_id="1")

        mock_mlflow = _make_mock_mlflow(_mock_run_context("parent-run", "1"), [])
        mock_client = mock.MagicMock()
        mock_client.delete_run.side_effect = RuntimeError("boom")
        mock_mlflow.MlflowClient.return_value = mock_client
        with mock.patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            with experiment_run_logger(task_type="binary", eval_metric="accuracy") as run_logger:
                run_logger.log_header(pipeline_name="p", kfp_run_id="run-1", top_n=1)
                run_logger._live_child_runs["CatBoost_BAG_L1"] = "run-cat"
                # Must not raise even though delete_run blows up.
                run_logger.prune_live_child_runs(["LightGBM_BAG_L1"])

        mock_client.delete_run.assert_called_once_with("run-cat")

    def test_creates_experiment_and_parent_when_platform_gives_no_run_id(self, tmp_path, monkeypatch):
        """Tech-preview path: no parentRunId/experimentId -> create experiment + parent named after run."""
        _set_kfp_mlflow_config(monkeypatch, parent_run_id="", experiment_id="")

        model_name = "LightGBM_BAG_L1_FULL"
        _write_model_metrics(tmp_path, model_name, {"accuracy": 0.9})
        html_path = tmp_path / "leaderboard.html"
        html_path.write_text("<html></html>", encoding="utf-8")

        mock_mlflow = _make_mock_mlflow(_mock_run_context("new-parent", "42"), [_mock_run_context("child-1", "42")])
        with mock.patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            with experiment_run_logger(
                task_type="binary",
                eval_metric="accuracy",
                run_name="Mlflow-test",
            ) as run_logger:
                run_logger.log_header(pipeline_name="p", kfp_run_id="run-1", kfp_run_name="Mlflow-test", top_n=1)
                run_logger.log_model(
                    model_name=model_name,
                    model_uri=f"s3://bucket/models/{model_name}",
                    metrics={"test_data": {"accuracy": 0.9}},
                )
                run_logger.finalize()
        logged, tracking_info = run_logger.result()

        assert logged is True
        # An experiment named after the run was get-or-created, then a new parent run started.
        mock_mlflow.set_experiment.assert_called_once_with("Mlflow-test")
        mock_mlflow.start_run.assert_any_call(run_name="Mlflow-test")
        # Did NOT try to resume a (nonexistent) platform parent run.
        resume_calls = [c for c in mock_mlflow.start_run.call_args_list if "run_id" in c.kwargs]
        assert resume_calls == []
        # Real ids come from the created run, not the (empty) config.
        assert tracking_info["mlflow_run_id"] == "new-parent"
        assert tracking_info["mlflow_experiment_id"] == "42"

    def test_starts_parent_in_given_experiment_when_only_run_id_missing(self, tmp_path, monkeypatch):
        """With an experimentId but no parentRunId, start a new run in that experiment (no set_experiment)."""
        _set_kfp_mlflow_config(monkeypatch, parent_run_id="", experiment_id="7")

        model_name = "LightGBM_BAG_L1_FULL"
        _write_model_metrics(tmp_path, model_name, {"accuracy": 0.9})
        html_path = tmp_path / "leaderboard.html"
        html_path.write_text("<html></html>", encoding="utf-8")

        mock_mlflow = _make_mock_mlflow(_mock_run_context("new-parent", "7"), [_mock_run_context("child-1", "7")])
        with mock.patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            with experiment_run_logger(
                task_type="binary", eval_metric="accuracy", run_name="Mlflow-test"
            ) as run_logger:
                run_logger.log_header(pipeline_name="p", kfp_run_id="run-1", top_n=1)
                run_logger.log_model(
                    model_name=model_name,
                    model_uri=f"s3://bucket/models/{model_name}",
                    metrics={"test_data": {"accuracy": 0.9}},
                )
                run_logger.finalize()
        logged, _ = run_logger.result()

        assert logged is True
        mock_mlflow.set_experiment.assert_not_called()
        mock_mlflow.start_run.assert_any_call(experiment_id="7", run_name="Mlflow-test")

    def test_logs_multiple_child_runs(self, tmp_path, monkeypatch):
        """Create one child MLflow run per model under the parent run."""
        _set_kfp_mlflow_config(monkeypatch, parent_run_id="parent-run", experiment_id="99")

        model_names = ["WeightedEnsemble_L3_FULL", "CatBoost_BAG_L1_FULL"]
        metrics_by_model = {}
        for model_name in model_names:
            _write_model_metrics(tmp_path, model_name, {"accuracy": 0.9, "f1": 0.88, "roc_auc": 0.95})
            metrics_by_model[model_name] = {"accuracy": 0.9, "f1": 0.88, "roc_auc": 0.95}

        mock_mlflow = _make_mock_mlflow(
            _mock_run_context("parent-run", "99"),
            [_mock_run_context("child-1", "99"), _mock_run_context("child-2", "99")],
        )
        logged, tracking_info = _run_logger_lifecycle(
            mock_mlflow,
            tmp_path=tmp_path,
            model_names=model_names,
            metrics_by_model=metrics_by_model,
        )

        assert logged is True
        assert tracking_info["mlflow_child_run_count"] == "2"
        assert tracking_info["mlflow_experiment_id"] == "99"
        mock_mlflow.start_run.assert_any_call(run_id="parent-run")
        mock_mlflow.start_run.assert_any_call(run_name="WeightedEnsemble_L3", nested=True)
        mock_mlflow.start_run.assert_any_call(run_name="CatBoost_BAG_L1", nested=True)

    def test_logs_kfp_artifact_references_without_uploading_them(self, tmp_path, monkeypatch):
        """Keep pipeline artifacts in S3 and log only their references to MLflow."""
        _set_kfp_mlflow_config(monkeypatch, parent_run_id="parent-run", experiment_id="1")

        model_name = "LightGBM_BAG_L1_FULL"
        mock_mlflow = _make_mock_mlflow(_mock_run_context("parent-run", "1"), [_mock_run_context("child-run-1", "1")])
        logged, _ = _run_logger_lifecycle(
            mock_mlflow,
            tmp_path=tmp_path,
            model_names=[model_name],
            metrics_by_model={model_name: {"accuracy": 0.91}},
        )

        assert logged is True
        child_params = [
            call.args[0] for call in mock_mlflow.log_params.call_args_list if "predictor_path" in call.args[0]
        ]
        assert child_params == [
            {
                "model_name": model_name,
                "model_type": "LightGBM",
                "stack_level": "1",
                "metrics_path": f"s3://bucket/models/{model_name}/metrics",
                "predictor_path": f"s3://bucket/models/{model_name}/predictor",
                "notebook_path": f"s3://bucket/models/{model_name}/notebooks/automl_predictor_notebook.ipynb",
            }
        ]
        mock_mlflow.log_artifact.assert_not_called()
        mock_mlflow.log_artifacts.assert_not_called()

    def test_disables_when_parent_run_open_fails(self, tmp_path, monkeypatch):
        """Disable tracking (no raise) when the parent run cannot be opened."""
        _set_kfp_mlflow_config(monkeypatch, parent_run_id="parent-run", experiment_id="1")

        model_name = "LightGBM_BAG_L1_FULL"
        _write_model_metrics(tmp_path, model_name, {"accuracy": 0.91})

        mock_mlflow = mock.MagicMock()
        mock_mlflow.start_run.side_effect = RuntimeError(
            '{"status": "Failure", "reason": "NotAcceptable", "code": 406}'
        )

        logged, tracking_info = _run_logger_lifecycle(
            mock_mlflow,
            tmp_path=tmp_path,
            model_names=[model_name],
            metrics_by_model={model_name: {"accuracy": 0.91}},
        )

        assert logged is False
        # Configured but broken: the reason is surfaced instead of an empty (silently
        # successful-looking) result.
        assert "parent run could not be opened" in tracking_info["mlflow_tracking_error"]


class TestTrackingFailureReporting:
    """``result()`` reports failure when MLflow was configured but nothing was persisted."""

    def test_reports_not_logged_when_every_write_fails(self, tmp_path, monkeypatch):
        """All parent/child writes swallowed -> logged is False with a reason, not a false success."""
        _set_kfp_mlflow_config(monkeypatch, parent_run_id="parent-run", experiment_id="1")

        model_name = "LightGBM_BAG_L1_FULL"
        _write_model_metrics(tmp_path, model_name, {"accuracy": 0.91})

        # The parent run opens, but every write to the tracking server is rejected.
        mock_mlflow = _make_mock_mlflow(_mock_run_context("parent-run", "1"), [_mock_run_context("child-1", "1")])
        mock_mlflow.set_tags.side_effect = RuntimeError("tracking server unreachable")
        mock_mlflow.log_params.side_effect = RuntimeError("tracking server unreachable")
        mock_mlflow.log_metric.side_effect = RuntimeError("tracking server unreachable")
        mock_mlflow.log_metrics.side_effect = RuntimeError("tracking server unreachable")

        logged, tracking_info = _run_logger_lifecycle(
            mock_mlflow,
            tmp_path=tmp_path,
            model_names=[model_name],
            metrics_by_model={model_name: {"accuracy": 0.91}},
        )

        assert logged is False
        assert "tracking server unreachable" in tracking_info["mlflow_tracking_error"]

    def test_reports_logged_when_only_child_runs_fail(self, tmp_path, monkeypatch):
        """A healthy parent run still counts as logged even if a child run failed."""
        _set_kfp_mlflow_config(monkeypatch, parent_run_id="parent-run", experiment_id="1")

        model_name = "LightGBM_BAG_L1_FULL"
        _write_model_metrics(tmp_path, model_name, {"accuracy": 0.91})

        mock_mlflow = _make_mock_mlflow(_mock_run_context("parent-run", "1"), [])
        # Only the child run fails to open (both the nested call and the client fallback);
        # the parent header and finalize still write.
        mock_mlflow.start_run.side_effect = [_mock_run_context("parent-run", "1"), RuntimeError("nested run rejected")]
        mock_mlflow.MlflowClient.side_effect = RuntimeError("child run rejected")

        logged, tracking_info = _run_logger_lifecycle(
            mock_mlflow,
            tmp_path=tmp_path,
            model_names=[model_name],
            metrics_by_model={model_name: {"accuracy": 0.91}},
        )

        assert logged is True
        assert "mlflow_tracking_error" not in tracking_info
        assert model_name in tracking_info["mlflow_child_run_errors"]

    def test_configured_flag_separates_disabled_from_broken(self, monkeypatch):
        """``configured`` is False only when the platform injected no MLflow config."""
        monkeypatch.delenv("KFP_MLFLOW_CONFIG", raising=False)
        with experiment_run_logger(task_type="binary", eval_metric="accuracy") as run_logger:
            assert run_logger.configured is False

        _set_kfp_mlflow_config(monkeypatch, parent_run_id="parent-run", experiment_id="1")
        mock_mlflow = mock.MagicMock()
        mock_mlflow.start_run.side_effect = RuntimeError("boom")
        with mock.patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            with experiment_run_logger(task_type="binary", eval_metric="accuracy") as run_logger:
                assert run_logger.enabled is False
                assert run_logger.configured is True


class TestParentRunAdrFields:
    """The parent run records the ADR-required identity/metadata fields."""

    def _log_header(self, monkeypatch, *, dataset_uri="", test_dataset_uri="") -> mock.MagicMock:
        _set_kfp_mlflow_config(monkeypatch, parent_run_id="parent-run", experiment_id="1")
        parent_ctx = _mock_run_context("parent-run", "1")
        mock_mlflow = mock.MagicMock()
        mock_mlflow.start_run.return_value = parent_ctx
        mock_mlflow.active_run.return_value = parent_ctx
        mock_mlflow.MlflowClient.return_value.get_run.return_value = parent_ctx
        with mock.patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            with experiment_run_logger(task_type="binary", eval_metric="accuracy") as run_logger:
                run_logger.log_header(
                    pipeline_name="p",
                    kfp_run_id="run-1",
                    preset="speed",
                    top_n=3,
                    dataset_uri=dataset_uri,
                    test_dataset_uri=test_dataset_uri,
                )
        return mock_mlflow

    def test_task_type_is_a_param_not_a_tag(self, monkeypatch):
        """ADR: task_type is a run parameter, not a tag."""
        mock_mlflow = self._log_header(monkeypatch)
        tags = mock_mlflow.set_tags.call_args.args[0]
        params = mock_mlflow.log_params.call_args.args[0]
        assert "task_type" not in tags
        assert params["task_type"] == "binary"

    def test_versions_and_image_logged_as_tags_only(self, monkeypatch):
        """Record versions and image once as tags, not duplicate parameters."""
        mock_mlflow = self._log_header(monkeypatch)
        tags = mock_mlflow.set_tags.call_args.args[0]
        params = mock_mlflow.log_params.call_args.args[0]
        for key in ("kfp_version", "image", "autogluon_version"):
            assert key in tags
            assert key not in params

    def test_dataset_uri_logged_when_provided(self, monkeypatch):
        """ADR: the non-secret dataset URI is recorded when available."""
        mock_mlflow = self._log_header(monkeypatch, dataset_uri="s3://bucket/data.csv")
        params = mock_mlflow.log_params.call_args.args[0]
        assert params["dataset_uri"] == "s3://bucket/data.csv"

    def test_dataset_uri_omitted_when_empty(self, monkeypatch):
        """No dataset_uri param when the caller has no bucket/key."""
        mock_mlflow = self._log_header(monkeypatch, dataset_uri="")
        params = mock_mlflow.log_params.call_args.args[0]
        assert "dataset_uri" not in params

    def test_test_dataset_uri_logged_when_provided(self, monkeypatch):
        """Record both train and user-provided test dataset identities."""
        mock_mlflow = self._log_header(
            monkeypatch,
            dataset_uri="s3://bucket/train.csv",
            test_dataset_uri="s3://bucket/test.csv",
        )
        params = mock_mlflow.log_params.call_args.args[0]
        assert params["dataset_uri"] == "s3://bucket/train.csv"
        assert params["test_dataset_uri"] == "s3://bucket/test.csv"

    def test_test_dataset_uri_omitted_when_empty(self, monkeypatch):
        """Do not record an absent user-provided test dataset."""
        mock_mlflow = self._log_header(monkeypatch, dataset_uri="s3://bucket/train.csv")
        params = mock_mlflow.log_params.call_args.args[0]
        assert "test_dataset_uri" not in params

    def test_total_fit_time_logged_as_parent_metric(self, tmp_path, monkeypatch):
        """ADR: finalize logs total_fit_time_seconds as a parent metric."""
        _set_kfp_mlflow_config(monkeypatch, parent_run_id="parent-run", experiment_id="1")
        model_name = "LightGBM_BAG_L1_FULL"
        _write_model_metrics(tmp_path, model_name, {"accuracy": 0.9})
        mock_mlflow = _make_mock_mlflow(_mock_run_context("parent-run", "1"), [_mock_run_context("child-run-1", "1")])
        _run_logger_lifecycle(
            mock_mlflow,
            tmp_path=tmp_path,
            model_names=[model_name],
            metrics_by_model={model_name: {"accuracy": 0.9}},
            total_fit_time_seconds=12.5,
        )
        assert mock.call("total_fit_time_seconds", 12.5) in mock_mlflow.log_metric.call_args_list

    def test_total_fit_time_omitted_when_none(self, tmp_path, monkeypatch):
        """No total_fit_time_seconds metric when the caller did not measure it."""
        _set_kfp_mlflow_config(monkeypatch, parent_run_id="parent-run", experiment_id="1")
        model_name = "LightGBM_BAG_L1_FULL"
        _write_model_metrics(tmp_path, model_name, {"accuracy": 0.9})
        mock_mlflow = _make_mock_mlflow(_mock_run_context("parent-run", "1"), [_mock_run_context("child-run-1", "1")])
        _run_logger_lifecycle(
            mock_mlflow,
            tmp_path=tmp_path,
            model_names=[model_name],
            metrics_by_model={model_name: {"accuracy": 0.9}},
        )
        logged_metric_names = [c.args[0] for c in mock_mlflow.log_metric.call_args_list]
        assert "total_fit_time_seconds" not in logged_metric_names


class TestKfpTagDeduplication:
    """The parent run carries the KFP identity under one set of keys."""

    def _log_header_with_parent_tags(self, monkeypatch, parent_tags: dict) -> tuple[dict, mock.MagicMock]:
        _set_kfp_mlflow_config(monkeypatch, parent_run_id="parent-run", experiment_id="1")
        parent_ctx = _mock_run_context("parent-run", "1")
        parent_ctx.data.tags = parent_tags
        mock_mlflow = _make_mock_mlflow(parent_ctx, [])
        with mock.patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            with experiment_run_logger(task_type="binary", eval_metric="accuracy") as run_logger:
                run_logger.log_header(pipeline_name="p", kfp_run_id="run-1", kfp_run_name="Fraud detection")
        return mock_mlflow.set_tags.call_args.args[0], mock_mlflow

    def test_skips_automl_kfp_tags_when_platform_set_them(self, monkeypatch):
        """Platform KFP tags prevent duplicate AutoML run-identity tags."""
        tags, _ = self._log_header_with_parent_tags(
            monkeypatch,
            {"kfp.pipeline_run_id": "parent-run", "kfp.pipeline_id": "pipeline-123"},
        )
        assert "kfp_run_id" not in tags
        assert "kfp_run_name" not in tags
        assert tags["pipeline_name"] == "p"

    def test_sets_automl_kfp_tags_when_platform_absent(self, monkeypatch):
        """Retain AutoML fallback identity tags outside the platform integration."""
        tags, _ = self._log_header_with_parent_tags(monkeypatch, {})
        assert tags["kfp_run_id"] == "run-1"
        assert tags["kfp_run_name"] == "Fraud detection"

    def test_falls_back_to_automl_tags_when_parent_tags_unreadable(self, monkeypatch):
        """A failed parent-tag lookup must not drop the run identity."""
        _set_kfp_mlflow_config(monkeypatch, parent_run_id="parent-run", experiment_id="1")
        parent_ctx = _mock_run_context("parent-run", "1")
        mock_mlflow = _make_mock_mlflow(parent_ctx, [])
        mock_mlflow.MlflowClient.side_effect = RuntimeError("tracking server unreachable")

        with mock.patch.dict(sys.modules, {"mlflow": mock_mlflow}):
            with experiment_run_logger(task_type="binary", eval_metric="accuracy") as run_logger:
                run_logger.log_header(pipeline_name="p", kfp_run_id="run-1")

        assert mock_mlflow.set_tags.call_args.args[0]["kfp_run_id"] == "run-1"
