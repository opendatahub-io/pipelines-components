"""High-level integration tests for Documents RAG Optimization pipeline on RHOAI.

These tests require a Red Hat OpenShift AI (RHOAI) cluster with Data Science Pipelines
enabled, and environment variables set for cluster URL, credentials, and pipeline
parameters (secret names, bucket names, keys). See conftest.py and integration_config.py.
When not set, tests are skipped. You can set vars via a .env file (see .env.example).

Scenarios:
- Run pipeline with required parameters, validate success and optional artifacts
  (leaderboard HTML, rag_patterns, .ipynb notebooks, v1_responses_body.json) in S3 when configured.
"""

import logging
import os
import secrets
from datetime import datetime, timezone
from urllib.parse import urlparse

import pytest

LOGGER = logging.getLogger(__name__)


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


def _collect_artifact_keys(s3_client, bucket, prefix):
    """List object keys under prefix and group them by artifact type."""
    html_keys = []
    ipynb_keys = []
    pattern_keys = []
    responses_body_keys = []
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
    return html_keys, ipynb_keys, pattern_keys, responses_body_keys


def _derive_external_s3_endpoint(endpoint):
    """Derive the public MinIO route from RHOAI_URL when only a cluster-local endpoint is available."""
    if ".svc.cluster.local" not in (endpoint or ""):
        return None
    rhoai_url = os.environ.get("RHOAI_URL", "").strip()
    if not rhoai_url:
        return None
    host = urlparse(rhoai_url).hostname or ""
    _, sep, cluster_domain = host.partition(".apps.")
    if not sep or not cluster_domain:
        return None
    return f"https://minio-api-minio.apps.{cluster_domain}"


def _make_retry_s3_client(endpoint):
    """Build an ad-hoc S3 client for artifact lookup retries."""
    try:
        import boto3
    except ImportError:
        return None
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    if not endpoint or not access_key or not secret_key:
        return None
    verify_ssl = os.environ.get("KFP_VERIFY_SSL", "true").strip().lower()
    verify_ssl = verify_ssl not in ("0", "false", "no")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region,
        verify=verify_ssl,
    )


def _find_artifacts_in_s3(s3_client, bucket, prefix):
    """List object keys under prefix.

    When the primary S3 client cannot reach the endpoint (e.g. it points at a
    cluster-internal address), fall back to the external MinIO route derived
    from ``RHOAI_URL`` and retry using ``KFP_VERIFY_SSL`` for TLS verification.

    Returns:
        Tuple of key lists: leaderboard HTML, .ipynb notebooks, rag/pattern-related paths,
        and ``v1_responses_body.json`` files from ``prepare_responses_api_requests``.
    """
    try:
        return _collect_artifact_keys(s3_client, bucket, prefix)
    except Exception as exc:
        endpoint = getattr(getattr(s3_client, "meta", None), "endpoint_url", None) or ""
        LOGGER.warning(
            "Artifact lookup failed for s3://%s/%s via %s: %s",
            bucket,
            prefix,
            endpoint or "<unknown>",
            exc,
        )

        derived_endpoint = _derive_external_s3_endpoint(endpoint)
        if derived_endpoint:
            retry_client = _make_retry_s3_client(derived_endpoint)
            if retry_client is not None:
                LOGGER.info("Retrying artifact lookup via %s", derived_endpoint)
                try:
                    return _collect_artifact_keys(retry_client, bucket, prefix)
                except Exception as retry_exc:
                    LOGGER.warning("Retry via external route also failed: %s", retry_exc)
                    raise retry_exc from exc
        raise


def _pipeline_arguments_from_config(config):
    """Build pipeline arguments dict from integration config."""
    return {
        "test_data_secret_name": config["test_data_secret_name"],
        "test_data_bucket_name": config["test_data_bucket_name"],
        "test_data_key": config["test_data_key"],
        "input_data_secret_name": config["input_data_secret_name"],
        "input_data_bucket_name": config["input_data_bucket_name"],
        "input_data_key": config["input_data_key"],
        "ogx_secret_name": config["ogx_secret_name"],
        "vector_io_provider_id": config["vector_io_provider_id"],
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

        Counts leaderboard HTML, notebooks, pattern paths, and OGX request body JSON files.
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
