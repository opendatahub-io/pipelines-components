"""High-level integration tests for AutoGluon tabular training pipeline on RHOAI.

These tests require a Red Hat OpenShift AI (RHOAI) cluster with Data Science Pipelines
enabled, and environment variables set for cluster URL, credentials, and S3 storage.
See the conftest.py in this directory for required env vars. When not set, tests
are skipped. You can set vars via a .env file (see .env.template).

Scenarios are parametrized via test_configs: each config specifies dataset
location, target column, problem type, AutoML/pipeline settings, and optional
tags. Filter by tags with RHOAI_TEST_CONFIG_TAGS (e.g. smoke, regression).
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


# Configs to run this session (all, or filtered by RHOAI_TEST_CONFIG_TAGS).
RHOAI_INTEGRATION_CONFIG = _session_rhoai_integration_config()
CONFIGS_FOR_RUN = _session_configs_for_run()

# Pipeline display name in KFP (from pipeline decorator)
PIPELINE_DISPLAY_NAME = "autogluon-tabular-training-pipeline"


def _make_automl_run_name():
    """Return a run name: automl-test-<6 hex chars>-<YYYYMMDD-HHMMSS>."""
    hex_part = secrets.token_hex(3)  # 3 bytes -> 6 hex chars
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
    """List object keys under prefix; return lists of keys ending in .pkl, .ipynb, and keys containing 'leaderboard' or 'html_artifact'."""  # noqa: E501
    pkl_keys = []
    ipynb_keys = []
    leaderboard_keys = []
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents") or []:
                key = obj["Key"]
                if key.endswith(".pkl"):
                    pkl_keys.append(key)
                elif key.endswith(".ipynb"):
                    ipynb_keys.append(key)
                elif "leaderboard" in key.lower() or "html_artifact" in key.lower():
                    leaderboard_keys.append(key)
    except Exception:
        pass
    return pkl_keys, ipynb_keys, leaderboard_keys


@pytest.mark.integration
@pytest.mark.skipif(
    RHOAI_INTEGRATION_CONFIG is None,
    reason=(
        "RHOAI integration env not set (set RHOAI_URL, RHOAI_TOKEN, S3 vars;"
        "use SA token for Jenkins; see .env.template)"
    ),
)
@pytest.mark.parametrize("test_config", CONFIGS_FOR_RUN, ids=[c.id for c in CONFIGS_FOR_RUN])
class TestAutogluonPipelineIntegration:
    """Integration tests running the pipeline on RHOAI and validating outcomes."""

    def test_autogluon_pipeline_with_config(
        self,
        test_config,
        rhoai_integration_config,
        rhoai_project,
        uploaded_datasets,
        kfp_client,
        compiled_pipeline_path,
        pipeline_run_timeout,
        s3_client,
    ):
        """Run pipeline for one test config; assert success and presence of artifacts."""
        if not uploaded_datasets or not kfp_client:
            pytest.skip("Integration prerequisites not available")
        if test_config.problem_type == "timeseries":
            pytest.skip("Timeseries not yet supported by pipeline or test data")
        config = rhoai_integration_config
        from .test_configs import resolve_config_to_pipeline_arguments

        arguments = resolve_config_to_pipeline_arguments(test_config, uploaded_datasets, config["s3_secret_name"])
        if not arguments:
            pytest.skip(f"Dataset not available for path: {test_config.dataset_path}")

        run_id, detail = _run_pipeline_and_wait(kfp_client, compiled_pipeline_path, arguments, pipeline_run_timeout)
        assert _run_succeeded(detail), f"Pipeline run {run_id} did not succeed; state={getattr(detail, 'run', detail)}"

        if s3_client and config.get("s3_bucket_artifacts"):
            bucket = config["s3_bucket_artifacts"]
            prefix = f"{PIPELINE_DISPLAY_NAME}/{run_id}"
            pkl_keys, ipynb_keys, leaderboard_keys = _find_artifacts_in_s3(s3_client, bucket, prefix)
            assert len(pkl_keys) >= 1, f"Expected at least one .pkl model artifact under {prefix}; found {pkl_keys}"
            assert len(ipynb_keys) >= 1, f"Expected at least one .ipynb notebook under {prefix}; found {ipynb_keys}"
            assert len(leaderboard_keys) >= 1, (
                f"Expected leaderboard/html artifact under {prefix}; found {leaderboard_keys}"
            )
