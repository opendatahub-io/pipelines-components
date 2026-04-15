"""High-level integration tests for AutoGluon time series training pipeline on RHOAI.

These tests require a Red Hat OpenShift AI (RHOAI) cluster with Data Science Pipelines
enabled, and environment variables set for cluster URL, credentials, and S3 storage.
See the conftest.py in this directory for required env vars. When not set, tests
are skipped. You can set vars via a .env file (see .env.template).

Scenarios are parametrized via test_configs: each config specifies dataset
location, time-series columns, pipeline settings, and optional tags.
Filter by tags with RHOAI_TEST_CONFIG_TAGS (e.g. smoke, timeseries).
"""

import secrets
from datetime import datetime, timezone

import pytest


def _session_rhoai_integration_config():
    """Lazy import so import-guard allows only stdlib at module scope."""
    from .integration_config import RHOAI_INTEGRATION_CONFIG

    return RHOAI_INTEGRATION_CONFIG


def _session_configs_for_run():
    from .test_configs import get_test_configs_for_run

    return get_test_configs_for_run()


RHOAI_INTEGRATION_CONFIG = _session_rhoai_integration_config()
CONFIGS_FOR_RUN = _session_configs_for_run()

PIPELINE_DISPLAY_NAME = "autogluon-timeseries-training-pipeline"


def _make_automl_run_name():
    """Return a run name: automl-test-<6 hex chars>-<YYYYMMDD-HHMMSS>."""
    hex_part = secrets.token_hex(3)
    time_part = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"automl-test-{hex_part}-{time_part}"


def _run_pipeline_and_wait(client, compiled_path, arguments, timeout):
    """Submit pipeline run and wait for completion; return run_id and run detail."""
    run_name = _make_automl_run_name()
    run = client.create_run_from_pipeline_package(
        compiled_path,
        arguments=arguments,
        run_name=run_name,
    )
    run_id = run.run_id
    detail = client.wait_for_run_completion(run_id, timeout=timeout)
    return run_id, detail


def _run_succeeded(detail):
    """Return True if the run finished successfully."""
    run = getattr(detail, "run", detail)
    state = getattr(run, "state", None)
    if state is None and hasattr(run, "status"):
        state = getattr(run.status, "state", None)
    if isinstance(state, str):
        return state.upper() == "SUCCEEDED"
    return False


def _find_artifacts_in_s3(s3_client, bucket, prefix):
    """Return all object keys and common leaderboard/model keys under prefix."""
    all_keys = []
    leaderboard_keys = []
    model_keys = []
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents") or []:
                key = obj["Key"]
                all_keys.append(key)
                key_l = key.lower()
                if "leaderboard" in key_l or "html_artifact" in key_l:
                    leaderboard_keys.append(key)
                if key.endswith(".pkl") or key.endswith(".zip") or "predictor" in key_l:
                    model_keys.append(key)
    except Exception:
        pass
    return all_keys, leaderboard_keys, model_keys


@pytest.mark.integration
@pytest.mark.skipif(
    RHOAI_INTEGRATION_CONFIG is None,
    reason=(
        "RHOAI integration env not set (set RHOAI_URL, RHOAI_TOKEN, S3 vars;"
        "use SA token for Jenkins; see .env.template)"
    ),
)
@pytest.mark.parametrize("test_config", CONFIGS_FOR_RUN, ids=[c.id for c in CONFIGS_FOR_RUN])
class TestAutogluonTimeseriesPipelineIntegration:
    """Integration tests running the pipeline on RHOAI and validating outcomes."""

    def test_autogluon_timeseries_pipeline_with_config(
        self,
        test_config,
        rhoai_integration_config,
        rhoai_project,
        uploaded_datasets,
        kfp_client,
        pipeline_package_path,
        pipeline_run_timeout,
        s3_client,
    ):
        """Run pipeline for one test config; assert success and artifacts.

        Runs once with a fresh compile and once with committed ``pipeline.yaml``.
        """
        if not uploaded_datasets or not kfp_client:
            pytest.skip("Integration prerequisites not available")
        config = rhoai_integration_config
        from .test_configs import resolve_config_to_pipeline_arguments

        arguments = resolve_config_to_pipeline_arguments(test_config, uploaded_datasets, config["s3_secret_name"])
        if not arguments:
            pytest.skip(f"Dataset not available for path: {test_config.dataset_path}")

        run_id, detail = _run_pipeline_and_wait(kfp_client, pipeline_package_path, arguments, pipeline_run_timeout)
        assert _run_succeeded(detail), f"Pipeline run {run_id} did not succeed; state={getattr(detail, 'run', detail)}"

        if s3_client and config.get("s3_bucket_artifacts"):
            bucket = config["s3_bucket_artifacts"]
            prefix = f"{PIPELINE_DISPLAY_NAME}/{run_id}"
            all_keys, leaderboard_keys, model_keys = _find_artifacts_in_s3(s3_client, bucket, prefix)
            assert len(all_keys) >= 1, f"Expected at least one artifact under {prefix}; found {all_keys}"
            assert len(leaderboard_keys) >= 1, (
                f"Expected leaderboard/html artifact under {prefix}; found {leaderboard_keys}"
            )
            assert len(model_keys) >= 1, f"Expected model artifact under {prefix}; found {model_keys}"
