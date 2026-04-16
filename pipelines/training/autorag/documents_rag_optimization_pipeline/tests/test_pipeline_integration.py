"""High-level integration tests for Documents RAG Optimization pipeline on RHOAI.

These tests require a Red Hat OpenShift AI (RHOAI) cluster with Data Science Pipelines
enabled, and environment variables set for cluster URL, credentials, and pipeline
parameters (secret names, bucket names, keys). See conftest.py and integration_config.py.
When not set, tests are skipped. You can set vars via a .env file (see .env.example).

Scenarios:
- Run pipeline with required parameters, validate success and optional artifacts
  (leaderboard HTML, rag_patterns, .ipynb notebooks, v1_responses_body.json) in S3 when configured.
"""

import secrets
from datetime import datetime, timezone

import pytest


def _skip_if_no_rag_integration_config():
    """Return True if integration config is not set (skip test)."""
    from .integration_config import DOCRAG_INTEGRATION_CONFIG

    return DOCRAG_INTEGRATION_CONFIG is None


# Pipeline display name in KFP (from pipeline decorator)
PIPELINE_DISPLAY_NAME = "documents-rag-optimization-pipeline"


def _make_docrag_run_name():
    """Return a run name: docrag-test-<6 hex chars>-<YYYYMMDD-HHMMSS>."""
    hex_part = secrets.token_hex(3)
    time_part = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"docrag-test-{hex_part}-{time_part}"


def _run_pipeline_and_wait(client, compiled_path, arguments, timeout):
    """Submit pipeline run and wait for completion; return run_id and run detail."""
    run_name = _make_docrag_run_name()
    run = client.create_run_from_pipeline_package(
        compiled_path,
        arguments=arguments,
        run_name=run_name,
        enable_caching=False,
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
    """List object keys under prefix.

    Returns:
        Tuple of key lists: leaderboard HTML, .ipynb notebooks, rag/pattern-related paths,
        and ``v1_responses_body.json`` files from ``prepare_responses_api_requests``.
    """
    html_keys = []
    ipynb_keys = []
    pattern_keys = []
    responses_body_keys = []
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents") or []:
                key = obj["Key"]
                if "leaderboard" in key.lower() or key.endswith(".html"):
                    html_keys.append(key)
                elif key.endswith(".ipynb"):
                    ipynb_keys.append(key)
                elif "v1_responses_body.json" in key:
                    responses_body_keys.append(key)
                elif "rag_patterns" in key or "pattern" in key.lower():
                    pattern_keys.append(key)
    except Exception:
        pass
    return html_keys, ipynb_keys, pattern_keys, responses_body_keys


def _pipeline_arguments_from_config(config):
    """Build pipeline arguments dict from integration config."""
    return {
        "test_data_secret_name": config["test_data_secret_name"],
        "test_data_bucket_name": config["test_data_bucket_name"],
        "test_data_key": config["test_data_key"],
        "input_data_secret_name": config["input_data_secret_name"],
        "input_data_bucket_name": config["input_data_bucket_name"],
        "input_data_key": config["input_data_key"],
        "llama_stack_secret_name": config["llama_stack_secret_name"],
        "llama_stack_vector_io_provider_id": config["llama_stack_vector_io_provider_id"],
    }


@pytest.mark.integration
@pytest.mark.skipif(
    _skip_if_no_rag_integration_config(),
    reason=("RHOAI integration env not set (set RHOAI_KFP_URL, RHOAI_TOKEN, pipeline params, see .env.example)"),
)
class TestDocumentsRagOptimizationPipelineIntegration:
    """Integration tests running the pipeline on RHOAI and validating outcomes."""

    def test_documents_rag_optimization_pipeline_run(
        self,
        docrag_integration_config,
        kfp_client,
        compiled_pipeline_path,
        pipeline_run_timeout,
        s3_client,
    ):
        """Run pipeline; assert success and optional presence of artifacts in S3.

        Counts leaderboard HTML, notebooks, pattern paths, and Llama Stack request body JSON files.
        """
        if not kfp_client:
            pytest.skip("Integration prerequisites not available")
        config = docrag_integration_config
        arguments = _pipeline_arguments_from_config(config)

        run_id, detail = _run_pipeline_and_wait(
            kfp_client,
            compiled_pipeline_path,
            arguments,
            pipeline_run_timeout,
        )
        assert _run_succeeded(detail), (
            f"Pipeline run {run_id} did not succeed; state={getattr(getattr(detail, 'run', detail), 'state', detail)}"
        )

        responses_body_keys: list = []
        artifact_bucket: str | None = None
        if s3_client and config.get("s3_bucket_artifacts"):
            artifact_bucket = config["s3_bucket_artifacts"]
            prefix = f"{PIPELINE_DISPLAY_NAME}/{run_id}"
            html_keys, ipynb_keys, pattern_keys, responses_body_keys = _find_artifacts_in_s3(
                s3_client, artifact_bucket, prefix
            )
            assert (
                len(html_keys) >= 1 or len(ipynb_keys) >= 1 or len(pattern_keys) >= 1 or len(responses_body_keys) >= 1
            ), (
                f"Expected at least one artifact (leaderboard, .ipynb, rag_patterns, or "
                f"v1_responses_body.json) under {prefix}; found html={len(html_keys)}, "
                f"ipynb={len(ipynb_keys)}, pattern={len(pattern_keys)}, "
                f"responses_body={len(responses_body_keys)}"
            )
