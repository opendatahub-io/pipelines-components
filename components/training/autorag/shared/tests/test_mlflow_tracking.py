"""Tests for AutoRAG MLflow tracking helpers."""

from __future__ import annotations

import json
from unittest import mock

import pytest
from kfp_components.components.training.autorag.pytest_support import (
    FakeMlflow,
    FakeRun,
)
from kfp_components.components.training.autorag.pytest_support import (
    mlflow_pattern_payload as _pattern_payload,
)
from kfp_components.components.training.autorag.shared.mlflow_tracking import (
    KFP_MLFLOW_CONFIG_ENV,
    MLFLOW_WORKSPACE_HEADER,
    MlflowConfig,
    MlflowPatternEventHandler,
    MlflowPatternLogger,
    build_mlflow_run_url,
    build_mlflow_stage_map_block,
    configure_mlflow_client,
    experiment_run_logger,
    is_mlflow_enabled,
    optimization_metric_key,
    parent_mlflow_run,
    pattern_metrics,
    pattern_params,
    pattern_score,
    resolve_mlflow_config,
)

TRACKING_URI = "https://mlflow.example.com"


def _kfp_config(**overrides) -> str:
    """Serialize a platform KFP_MLFLOW_CONFIG blob with the given overrides."""
    payload = {
        "endpoint": TRACKING_URI,
        "workspacesEnabled": True,
        "workspace": "ns-autorag",
        "parentRunId": "parent-run-1",
        "experimentId": "7",
        "authType": "kubernetes",
        "timeout": "30s",
    }
    payload.update(overrides)
    return json.dumps(payload)


class TestResolveMlflowConfig:
    """Parsing of the platform-injected KFP_MLFLOW_CONFIG blob."""

    def test_returns_none_when_env_absent(self, monkeypatch):
        """Tracking is off when the platform injects nothing."""
        monkeypatch.delenv(KFP_MLFLOW_CONFIG_ENV, raising=False)
        assert resolve_mlflow_config() is None
        assert is_mlflow_enabled() is False

    @pytest.mark.parametrize("raw", ["not json", "[1, 2]", "{}", '{"endpoint": "  "}'])
    def test_returns_none_for_unusable_blob(self, monkeypatch, raw):
        """Malformed or endpoint-less blobs disable tracking instead of raising."""
        monkeypatch.setenv(KFP_MLFLOW_CONFIG_ENV, raw)
        assert resolve_mlflow_config() is None

    def test_parses_full_blob(self, monkeypatch):
        """All platform fields are surfaced on the resolved config."""
        monkeypatch.setenv(KFP_MLFLOW_CONFIG_ENV, _kfp_config())
        config = resolve_mlflow_config()
        assert config == MlflowConfig(
            mode="kfp",
            tracking_uri=TRACKING_URI,
            experiment_id="7",
            run_id="parent-run-1",
            workspace="ns-autorag",
            auth_type="kubernetes",
            timeout="30s",
        )
        assert is_mlflow_enabled() is True

    def test_workspace_ignored_when_workspaces_disabled(self, monkeypatch):
        """A workspace is only sent when the server has workspaces enabled."""
        monkeypatch.setenv(KFP_MLFLOW_CONFIG_ENV, _kfp_config(workspacesEnabled=False))
        config = resolve_mlflow_config()
        assert config is not None
        assert config.workspace == ""

    def test_refuses_kubernetes_auth_over_plain_http(self, monkeypatch):
        """Never send a ServiceAccount bearer token over cleartext (CWE-319)."""
        monkeypatch.setenv(KFP_MLFLOW_CONFIG_ENV, _kfp_config(endpoint="http://mlflow.example.com"))
        assert resolve_mlflow_config() is None

    def test_allows_plain_http_without_kubernetes_auth(self, monkeypatch):
        """Plain HTTP is fine when no bearer token is attached."""
        monkeypatch.setenv(KFP_MLFLOW_CONFIG_ENV, _kfp_config(endpoint="http://mlflow.example.com", authType="none"))
        config = resolve_mlflow_config()
        assert config is not None
        assert config.tracking_uri == "http://mlflow.example.com"


class TestConfigureMlflowClient:
    """Auth, timeout, and workspace application."""

    def test_reads_service_account_token(self, monkeypatch, tmp_path):
        """The pod's SA token becomes MLFLOW_TRACKING_TOKEN for bearer auth."""
        token_file = tmp_path / "token"
        token_file.write_text("tok-123\n", encoding="utf-8")
        monkeypatch.setattr(
            "kfp_components.components.training.autorag.shared.mlflow_tracking.SERVICE_ACCOUNT_TOKEN_PATH",
            str(token_file),
        )
        # Register both variables before the implementation writes to os.environ so
        # monkeypatch restores their original values after this test.
        monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "")
        monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "")
        monkeypatch.delenv("MLFLOW_TRACKING_TOKEN")
        monkeypatch.delenv("MLFLOW_HTTP_REQUEST_TIMEOUT")
        fake = FakeMlflow()
        configure_mlflow_client(fake, MlflowConfig("kfp", TRACKING_URI, auth_type="kubernetes", timeout="30s"))
        import os

        assert os.environ["MLFLOW_TRACKING_TOKEN"] == "tok-123"
        assert os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] == "30"
        assert fake.tracking_uri == TRACKING_URI

    def test_missing_token_clears_inherited_token(self, monkeypatch, tmp_path):
        """An unreadable token clears inherited credentials rather than reusing them."""
        monkeypatch.setattr(
            "kfp_components.components.training.autorag.shared.mlflow_tracking.SERVICE_ACCOUNT_TOKEN_PATH",
            str(tmp_path / "absent"),
        )
        monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "inherited-token")
        configure_mlflow_client(FakeMlflow(), MlflowConfig("kfp", TRACKING_URI, auth_type="kubernetes"))
        import os

        assert "MLFLOW_TRACKING_TOKEN" not in os.environ

    def test_empty_token_clears_inherited_token(self, monkeypatch, tmp_path):
        """An empty mounted token also clears inherited credentials."""
        token_file = tmp_path / "token"
        token_file.write_text("\n", encoding="utf-8")
        monkeypatch.setattr(
            "kfp_components.components.training.autorag.shared.mlflow_tracking.SERVICE_ACCOUNT_TOKEN_PATH",
            str(token_file),
        )
        monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "inherited-token")
        configure_mlflow_client(FakeMlflow(), MlflowConfig("kfp", TRACKING_URI, auth_type="kubernetes"))
        import os

        assert "MLFLOW_TRACKING_TOKEN" not in os.environ

    def test_prefers_native_set_workspace(self):
        """When the client exposes set_workspace, use it instead of patching requests."""
        fake = FakeMlflow()
        fake.set_workspace = mock.Mock()
        configure_mlflow_client(fake, MlflowConfig("kfp", TRACKING_URI, workspace="ns-autorag"))
        fake.set_workspace.assert_called_once_with("ns-autorag")

    def test_installs_workspace_header_on_tracking_host_only(self, monkeypatch):
        """The workspace header is scoped to the tracking host, not artifact stores."""
        import requests

        seen: list[tuple[str, dict]] = []

        def fake_request(self, method, url, *args, **kwargs):
            seen.append((url, dict(kwargs.get("headers") or {})))
            return "ok"

        monkeypatch.setattr(requests.Session, "request", fake_request)
        configure_mlflow_client(FakeMlflow(), MlflowConfig("kfp", TRACKING_URI, workspace="ns-autorag"))

        session = requests.Session()
        session.request("GET", f"{TRACKING_URI}/api/2.0/mlflow/runs/get")
        session.request("GET", "https://s3.example.com/bucket/object")

        assert seen[0][1][MLFLOW_WORKSPACE_HEADER] == "ns-autorag"
        assert MLFLOW_WORKSPACE_HEADER not in seen[1][1]


class TestPatternMapping:
    """Mapping of an ai4rag pattern payload onto MLflow params and metrics."""

    def test_params_flatten_settings(self):
        """Settings become dotted params, including conditional retrieval keys."""
        params = pattern_params(_pattern_payload())
        assert params["chunking.method"] == "recursive"
        assert params["chunking.chunk_size"] == "512"
        assert params["embedding.model_id"] == "embed-a"
        assert params["generation.model_id"] == "gen-a"
        assert params["vector_store.provider_type"] == "milvus"
        assert params["retrieval.window_size"] == "2"
        assert params["iteration"] == "2"

    def test_params_exclude_prompt_text(self):
        """Prompt templates stay in KFP/S3 artifacts, not MLflow params."""
        params = pattern_params(_pattern_payload())
        assert not any("message_text" in key or "context_template" in key for key in params)

    def test_params_are_truncated(self):
        """Over-long values are truncated so MLflow does not reject the whole batch."""
        payload = _pattern_payload()
        payload["settings"]["embedding"]["embedding_params"] = {"x": "y" * 9000}
        assert len(pattern_params(payload)["embedding.embedding_params"]) == 5000

    def test_metrics_include_scores_and_bounds(self):
        """Each metric yields a mean plus its CI bounds when ai4rag computed them."""
        metrics = pattern_metrics(_pattern_payload())
        assert metrics["unitxt_faithfulness"] == 0.82
        assert metrics["unitxt_faithfulness_ci_low"] == 0.75
        assert metrics["unitxt_faithfulness_ci_high"] == 0.9
        assert metrics["custom_overall_score"] == 0.77
        assert "custom_overall_score_ci_low" not in metrics
        assert metrics["duration_seconds"] == 12.5

    def test_metrics_tolerate_missing_evaluation(self):
        """A payload with no evaluation block yields no metrics rather than raising."""
        assert pattern_metrics({"name": "p"}) == {}

    def test_optimization_metric_is_identified(self):
        """The flagged metric drives ranking and is exposed as a tag/score."""
        assert optimization_metric_key(_pattern_payload()) == "custom_overall_score"
        assert pattern_score(_pattern_payload()) == 0.77

    def test_metric_keys_are_sanitized(self):
        """Characters MLflow rejects in metric keys are replaced."""
        payload = _pattern_payload()
        payload["evaluation"]["metrics"] = [
            {"name": "ndcg@10", "evaluator": "ragas", "scores": {"mean": 0.5}},
        ]
        assert "ragas_ndcg_10" in pattern_metrics(payload)


class TestBuildMlflowStageMapBlock:
    """The `mlflow` block published for the odh-dashboard BFF."""

    def test_disabled_without_config(self, monkeypatch):
        """Tracking-off is reported explicitly so the dashboard can hide the link."""
        monkeypatch.delenv(KFP_MLFLOW_CONFIG_ENV, raising=False)
        assert build_mlflow_stage_map_block() == {"tracking_enabled": False}

    def test_includes_deep_link(self, monkeypatch):
        """A run URL is emitted when both experiment and run are known."""
        monkeypatch.setenv(KFP_MLFLOW_CONFIG_ENV, _kfp_config())
        block = build_mlflow_stage_map_block()
        assert block["tracking_enabled"] is True
        assert block["workspace"] == "ns-autorag"
        assert block["run_url"] == f"{TRACKING_URI}/#/experiments/7/runs/parent-run-1"

    def test_run_url_omitted_when_ids_missing(self):
        """No deep link without both ids."""
        assert build_mlflow_run_url(TRACKING_URI, "", "r") == ""


class TestParentMlflowRun:
    """Parent run resumption and fallback creation."""

    def test_resumes_platform_parent_run(self, monkeypatch):
        """The platform's parentRunId is resumed so results land in the user's experiment."""
        monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
        fake = FakeMlflow()
        config = MlflowConfig("kfp", TRACKING_URI, experiment_id="7", run_id="parent-run-1")
        with parent_mlflow_run(fake, config, fallback_name="job-x") as run:
            assert run.info.run_id == "parent-run-1"
        # Activate the platform experiment so nested child runs do not look up id=0.
        assert fake.set_experiment_calls == ["7"]

    def test_creates_named_run_when_platform_gives_none(self):
        """Without a platform parent run, a job-named experiment and run are created."""
        fake = FakeMlflow()
        config = MlflowConfig("kfp", TRACKING_URI)
        with parent_mlflow_run(fake, config, fallback_name="job-x") as run:
            assert run.info.run_id
        assert fake.set_experiment_calls == ["job-x"]
        assert fake.started[0]["run_name"] == "job-x"


class TestMlflowPatternLogger:
    """Incremental per-pattern logging behaviour."""

    @pytest.fixture
    def logger_and_mlflow(self):
        """An enabled logger bound to an open parent run on a fake MLflow."""
        fake = FakeMlflow()
        config = MlflowConfig("kfp", TRACKING_URI, experiment_id="7", run_id="parent-run-1")
        run_logger = MlflowPatternLogger(fake, config)
        with fake.start_run(run_id="parent-run-1") as parent:
            run_logger.bind_parent_run(parent)
            yield run_logger, fake

    def test_disabled_logger_is_a_noop(self):
        """With no config every method is safe to call and nothing is reported."""
        run_logger = MlflowPatternLogger(None, None)
        assert run_logger.enabled is False
        assert run_logger.configured is False
        run_logger.log_header(pipeline_name="p")
        run_logger.log_pattern(_pattern_payload())
        run_logger.finalize()
        assert run_logger.result() == (False, {})

    def test_log_header_records_params_and_tags(self, logger_and_mlflow):
        """Job-level configuration lands on the parent run."""
        run_logger, fake = logger_and_mlflow
        run_logger.log_header(
            pipeline_name="documents-rag-optimization-pipeline",
            kfp_run_id="kfp-1",
            kfp_run_name="job-x",
            preset="speed",
            optimization_metric="custom:overall_score",
            max_rag_patterns=8,
            active_evaluators=frozenset({"unitxt", "custom"}),
            embedding_models=["embed-a"],
            generation_models=["gen-a"],
        )
        parent = fake.runs["parent-run-1"]
        assert parent["params"]["preset"] == "speed"
        assert parent["params"]["optimization_metric"] == "custom:overall_score"
        assert parent["params"]["evaluators"] == "custom,unitxt"
        assert parent["params"]["max_number_of_rag_patterns"] == "8"
        assert parent["tags"]["run_type"] == "pipeline"
        assert parent["tags"]["kfp_run_name"] == "job-x"

    def test_log_header_avoids_duplicate_kfp_tags_owned_by_platform(self, logger_and_mlflow):
        """Platform KFP identity tags suppress AutoRAG's legacy duplicate tags."""
        run_logger, fake = logger_and_mlflow
        fake.runs["parent-run-1"]["tags"]["kfp.pipeline_run_id"] = "parent-run-1"
        run_logger.log_header(pipeline_name="p", kfp_run_id="run-1", kfp_run_name="job-x")

        tags = fake.runs["parent-run-1"]["tags"]
        assert "kfp_run_id" not in tags
        assert "kfp_run_name" not in tags
        assert tags["pipeline_name"] == "p"

    def test_log_pattern_creates_child_run(self, logger_and_mlflow):
        """Each evaluated pattern becomes a nested child run with params and metrics."""
        run_logger, fake = logger_and_mlflow
        run_logger.log_pattern(_pattern_payload())

        children = fake.child_runs()
        assert len(children) == 1
        child = children[0]
        assert child["name"] == "rag_pattern_1"
        assert child["metrics"]["custom_overall_score"] == 0.77
        assert child["params"]["chunking.method"] == "recursive"
        assert child["tags"]["optimization_metric"] == "custom_overall_score"
        assert fake.started[-1]["nested"] is True

    def test_tags_patterns_beyond_max_rag_patterns_as_warm_start_extras(self, logger_and_mlflow):
        """Child runs after the published slice are tagged so the UI can filter them."""
        run_logger, fake = logger_and_mlflow
        run_logger.log_header(max_rag_patterns=2)
        run_logger.log_pattern(_pattern_payload(name="p1"))
        run_logger.log_pattern(_pattern_payload(name="p2"))
        run_logger.log_pattern(_pattern_payload(name="p3"))
        children = fake.child_runs()
        assert [c["tags"]["warm_start_extra"] for c in children] == ["false", "false", "true"]
        assert [c["tags"]["published_pattern"] for c in children] == ["true", "true", "false"]
        assert children[2]["params"]["warm_start_extra"] == "true"
        run_logger.finalize()
        parent = fake.runs["parent-run-1"]
        assert parent["metrics"]["rag_pattern_count"] == 3.0
        assert parent["metrics"]["published_pattern_count"] == 2.0
        assert parent["metrics"]["warm_start_extra_count"] == 1.0

    def test_pattern_artifacts_are_not_copied_to_mlflow(self, logger_and_mlflow):
        """KFP/S3 owns output artifacts; MLflow receives params and metrics only."""
        run_logger, fake = logger_and_mlflow
        run_logger.log_pattern(_pattern_payload(), [{"question": "q", "answer": "a"}])
        assert fake.child_runs()[0]["artifacts"] == []

    def test_logs_kfp_artifact_pointers_on_child_run(self, logger_and_mlflow):
        """Child runs point to KFP/S3 output instead of duplicating output files."""
        run_logger, fake = logger_and_mlflow
        run_logger.log_pattern(_pattern_payload())
        run_logger.log_pattern_artifact_pointers("s3://bucket/run/rag_patterns", ["rag_pattern_1"])
        params = fake.child_runs()[0]["params"]
        assert params["kfp.pattern_json_uri"] == "s3://bucket/run/rag_patterns/rag_pattern_1/pattern.json"
        assert params["kfp.evaluation_results_uri"].endswith("/evaluation_results.json")
        assert fake.child_runs()[0]["artifacts"] == []

    def test_logs_per_record_manual_traces(self):
        """Each evaluation record creates semantic RAG spans under its child run."""
        fake = FakeMlflow()
        config = MlflowConfig("kfp", TRACKING_URI, experiment_id="7")
        run_logger = MlflowPatternLogger(fake, config)
        record = {
            "question": "q",
            "correct_answers": ["a"],
            "answer": "a",
            "answer_contexts": {"text": "retrieved chunk", "source": "reading-club"},
            "score": 0.9,
        }
        second_record = {"question": "q2", "answer": "a2", "score": 0.8}
        with fake.start_run(run_id="parent-run-1") as parent:
            run_logger.bind_parent_run(parent)
            run_logger.log_pattern(_pattern_payload(), [record, second_record])
        assert [span["name"] for span in fake.spans] == [
            "benchmark_request",
            "retrieval",
            "generation",
            "evaluation",
        ] * 2
        child_run_id = next(run_id for run_id, run in fake.runs.items() if run["tags"].get("run_type") == "rag_pattern")
        assert fake.spans[0]["run_id"] == child_run_id
        assert fake.spans[0]["inputs"] == {"question": "q", "correct_answers": ["a"]}
        assert fake.spans[0]["outputs"] == {
            "answer": "a",
            # Survives on root Response when artifact-backed RETRIEVER spans cannot upload.
            "retrieved_documents": [{"page_content": "retrieved chunk", "metadata": {"source": "reading-club"}}],
        }
        assert fake.spans[1]["inputs"] == {"query": "q"}
        assert fake.spans[1]["outputs"] == [{"page_content": "retrieved chunk", "metadata": {"source": "reading-club"}}]
        assert fake.spans[2]["inputs"] == {
            "question": "q",
            "contexts": [{"page_content": "retrieved chunk", "metadata": {"source": "reading-club"}}],
        }
        assert fake.spans[2]["outputs"] == {"answer": "a"}
        assert fake.spans[3]["inputs"] == {"question": "q", "correct_answers": ["a"]}
        assert fake.spans[3]["outputs"] == record
        assert [span["parent_id"] for span in fake.spans if span["parent_id"]] == [fake.spans[0]["span_id"]] * 3 + [
            fake.spans[4]["span_id"]
        ] * 3

    def test_trace_failures_do_not_fail_pattern_logging(self):
        """Tracing is best-effort, independently of the child-run metrics and params."""
        fake = FakeMlflow()
        config = MlflowConfig("kfp", TRACKING_URI, experiment_id="7")
        run_logger = MlflowPatternLogger(fake, config)
        with fake.start_run(run_id="parent-run-1") as parent:
            run_logger.bind_parent_run(parent)
            fake.start_span = mock.Mock(side_effect=RuntimeError("trace unavailable"))
            run_logger.log_pattern(_pattern_payload(), [{"question": "q"}])
        assert len(fake.child_runs()) == 1
        _, info = run_logger.result()
        assert "trace unavailable" in info["mlflow_trace_errors"]

    def test_falls_back_to_create_run_when_nesting_unsupported(self):
        """A server rejecting nested runs still gets children linked to the parent."""
        fake = FakeMlflow(nested_supported=False)
        config = MlflowConfig("kfp", TRACKING_URI, experiment_id="7", run_id="parent-run-1")
        run_logger = MlflowPatternLogger(fake, config)
        run_logger.parent_run_id = "parent-run-1"
        run_logger.log_pattern(_pattern_payload())
        created = [r for r in fake.runs.values() if r["tags"].get("mlflow.parentRunId") == "parent-run-1"]
        assert len(created) == 1

    def test_child_run_failure_does_not_propagate(self, logger_and_mlflow):
        """A tracking error is recorded but never fails the optimization step."""
        run_logger, fake = logger_and_mlflow
        with mock.patch.object(fake, "log_metrics", side_effect=RuntimeError("boom")):
            run_logger.log_pattern(_pattern_payload())
        logged, info = run_logger.result()
        assert logged is False
        assert "boom" in info["mlflow_child_run_errors"]

    def test_finalize_reports_best_pattern(self, logger_and_mlflow):
        """The best score seen across patterns is summarized on the parent run."""
        run_logger, fake = logger_and_mlflow
        fake.flush_trace_async_logging = mock.Mock()
        run_logger.log_pattern(_pattern_payload(name="p1"))
        high = _pattern_payload(name="p2")
        high["evaluation"]["metrics"][1]["scores"]["mean"] = 0.91
        run_logger.log_pattern(high)
        run_logger.finalize()

        parent = fake.runs["parent-run-1"]
        assert parent["metrics"]["best_pattern_score"] == 0.91
        assert parent["metrics"]["rag_pattern_count"] == 2.0
        assert parent["params"]["best_pattern_name"] == "p2"
        fake.flush_trace_async_logging.assert_called_once_with()

    def test_finalize_does_not_copy_leaderboard(self, logger_and_mlflow):
        """The leaderboard remains in the KFP-managed artifact store."""
        run_logger, fake = logger_and_mlflow
        run_logger.finalize()
        assert fake.runs["parent-run-1"]["artifacts"] == []

    def test_result_reports_run_url_and_children(self, logger_and_mlflow):
        """The status payload carries the deep link and child run ids."""
        run_logger, _ = logger_and_mlflow
        run_logger.log_pattern(_pattern_payload())
        logged, info = run_logger.result()
        assert logged is True
        assert info["mlflow_child_run_count"] == "1"
        assert info["mlflow_run_url"] == f"{TRACKING_URI}/#/experiments/7/runs/parent-run-1"


class TestMlflowPatternEventHandler:
    """The ai4rag callback wrapper."""

    def test_forwards_to_inner_handler_and_logs(self):
        """The wrapped handler still collects patterns; MLflow gets a copy."""
        inner = mock.Mock(patterns=[{"payload": "p"}], status_changes=[])
        run_logger = mock.Mock()
        handler = MlflowPatternEventHandler(inner, run_logger)
        payload = _pattern_payload()

        handler.on_pattern_creation(payload=payload, evaluation_results=["r"])

        inner.on_pattern_creation.assert_called_once_with(payload=payload, evaluation_results=["r"])
        run_logger.log_pattern.assert_called_once_with(payload, ["r"])
        assert handler.patterns == [{"payload": "p"}]

    def test_forwards_status_changes(self):
        """Status updates pass straight through to the wrapped handler."""
        inner = mock.Mock(patterns=[], status_changes=[])
        handler = MlflowPatternEventHandler(inner, mock.Mock())
        handler.on_status_change("info", "working", step="chunking")
        inner.on_status_change.assert_called_once_with(level="info", message="working", step="chunking")

    def test_logging_failure_does_not_break_optimization(self):
        """A broken tracking call never propagates into the ai4rag search loop."""
        inner = mock.Mock(patterns=[], status_changes=[])
        run_logger = mock.Mock()
        run_logger.log_pattern.side_effect = RuntimeError("boom")
        handler = MlflowPatternEventHandler(inner, run_logger)
        handler.on_pattern_creation(payload={}, evaluation_results=[])
        inner.on_pattern_creation.assert_called_once()

    def test_unknown_attributes_forward_to_inner(self):
        """Any other handler API ai4rag relies on still resolves."""
        inner = mock.Mock(patterns=[], status_changes=[])
        inner.custom_hook.return_value = "ok"
        handler = MlflowPatternEventHandler(inner, mock.Mock())
        assert handler.custom_hook() == "ok"


class TestExperimentRunLogger:
    """End-to-end context manager behaviour."""

    def test_yields_disabled_logger_without_platform_config(self, monkeypatch):
        """No platform config means a no-op logger and no MLflow import."""
        monkeypatch.delenv(KFP_MLFLOW_CONFIG_ENV, raising=False)
        with experiment_run_logger() as run_logger:
            assert run_logger.enabled is False
            assert run_logger.configured is False

    def test_reports_error_when_mlflow_missing(self, monkeypatch):
        """A configured-but-unimportable MLflow is reported as failed, not as off."""
        monkeypatch.setenv(KFP_MLFLOW_CONFIG_ENV, _kfp_config())
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "mlflow":
                raise ImportError("no mlflow")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", blocked_import)
        with experiment_run_logger() as run_logger:
            assert run_logger.enabled is False
            assert run_logger.configured is True
        logged, info = run_logger.result()
        assert logged is False
        assert "not installed" in info["mlflow_tracking_error"]

    def test_binds_parent_run_and_closes_it(self, monkeypatch):
        """The parent run is opened, bound to the logger, and closed on exit."""
        monkeypatch.setenv(KFP_MLFLOW_CONFIG_ENV, _kfp_config())
        fake = FakeMlflow()
        monkeypatch.setitem(__import__("sys").modules, "mlflow", fake)
        monkeypatch.setattr(
            "kfp_components.components.training.autorag.shared.mlflow_tracking.SERVICE_ACCOUNT_TOKEN_PATH",
            "/nonexistent/token",
        )
        with experiment_run_logger(run_name="job-x") as run_logger:
            assert run_logger.enabled is True
            assert run_logger.parent_run_id == "parent-run-1"
            run_logger.log_pattern(_pattern_payload())
        assert fake._stack == []
        assert len(fake.child_runs()) == 1
        assert fake.runs["parent-run-1"]["status"] == "FINISHED"

    def test_marks_parent_run_failed_when_body_raises(self, monkeypatch):
        """Optimization failures close the parent MLflow run as FAILED, not FINISHED."""
        monkeypatch.setenv(KFP_MLFLOW_CONFIG_ENV, _kfp_config())
        fake = FakeMlflow()
        monkeypatch.setitem(__import__("sys").modules, "mlflow", fake)
        monkeypatch.setattr(
            "kfp_components.components.training.autorag.shared.mlflow_tracking.SERVICE_ACCOUNT_TOKEN_PATH",
            "/nonexistent/token",
        )
        with pytest.raises(RuntimeError, match="search failed"):
            with experiment_run_logger(run_name="job-x") as run_logger:
                assert run_logger.parent_run_id == "parent-run-1"
                raise RuntimeError("search failed")
        assert fake._stack == []
        assert fake.runs["parent-run-1"]["status"] == "FAILED"

    def test_preserves_optimization_error_when_parent_close_fails(self, monkeypatch):
        """A best-effort MLflow close failure cannot hide the optimization root cause."""
        monkeypatch.setenv(KFP_MLFLOW_CONFIG_ENV, _kfp_config())
        fake = FakeMlflow()
        monkeypatch.setitem(__import__("sys").modules, "mlflow", fake)

        class FailingParentRun:
            def __enter__(self):
                return FakeRun("parent-run-1", "7")

            def __exit__(self, *exc_info):
                raise RuntimeError("MLflow close failed")

        monkeypatch.setattr(
            "kfp_components.components.training.autorag.shared.mlflow_tracking.parent_mlflow_run",
            lambda *_args, **_kwargs: FailingParentRun(),
        )

        with pytest.raises(ValueError, match="optimization failed"):
            with experiment_run_logger(run_name="job-x"):
                raise ValueError("optimization failed")

    def test_disables_tracking_when_parent_run_cannot_open(self, monkeypatch):
        """An unreachable tracking server degrades to a no-op logger with a reason."""
        monkeypatch.setenv(KFP_MLFLOW_CONFIG_ENV, _kfp_config())
        fake = FakeMlflow()
        fake.start_run = mock.Mock(side_effect=RuntimeError("unreachable"))
        monkeypatch.setitem(__import__("sys").modules, "mlflow", fake)
        with experiment_run_logger() as run_logger:
            assert run_logger.enabled is False
            run_logger.log_pattern(_pattern_payload())
        logged, info = run_logger.result()
        assert logged is False
        assert "unreachable" in info["mlflow_tracking_error"]
