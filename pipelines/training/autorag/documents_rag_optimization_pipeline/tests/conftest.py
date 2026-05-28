"""Pytest fixtures for Documents RAG Optimization pipeline tests."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

_tests_dir = Path(__file__).resolve().parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))


@pytest.fixture(scope="session")
def docrag_integration_config():
    """Session-scoped RHOAI integration config from env; None if not set."""
    from integration_config import get_docrag_integration_config

    return get_docrag_integration_config()


@pytest.fixture(scope="session")
def kfp_client(docrag_integration_config):
    """Session-scoped KFP client pointing to RHOAI pipeline API."""
    if docrag_integration_config is None:
        return None
    import kfp

    host = docrag_integration_config["rhoai_kfp_url"]
    if not host.endswith("/"):
        host = host + "/"
    verify_ssl = os.environ.get("KFP_VERIFY_SSL", "true").strip().lower()
    verify_ssl = verify_ssl not in ("0", "false", "no")
    return kfp.Client(
        host=host,
        namespace=docrag_integration_config["rhoai_project"],
        existing_token=docrag_integration_config.get("rhoai_token"),
        verify_ssl=verify_ssl,
    )


@pytest.fixture(scope="session")
def compiled_pipeline_path():
    """Use an override YAML when provided, otherwise compile the pipeline locally."""
    override_path = os.environ.get("RHOAI_COMPILED_PIPELINE_PATH", "").strip()
    if override_path:
        if not Path(override_path).is_file():
            pytest.fail(f"RHOAI_COMPILED_PIPELINE_PATH does not exist: {override_path}")
        yield override_path
        return

    from kfp import compiler

    from ..pipeline import documents_rag_optimization_pipeline

    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    compiler.Compiler().compile(
        pipeline_func=documents_rag_optimization_pipeline,
        package_path=path,
    )
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture(scope="session")
def pipeline_run_timeout():
    """Timeout in seconds for waiting on a pipeline run (override via env)."""
    return int(os.environ.get("RHOAI_PIPELINE_RUN_TIMEOUT", "3600"))


@pytest.fixture(scope="session")
def s3_client(docrag_integration_config):
    """Session-scoped S3 client for artifact checks (optional)."""
    if docrag_integration_config is None or not docrag_integration_config.get("s3_endpoint"):
        return None
    try:
        import boto3
    except ImportError:
        return None
    c = docrag_integration_config
    verify_ssl = os.environ.get("KFP_VERIFY_SSL", "true").strip().lower()
    verify_ssl = verify_ssl not in ("0", "false", "no")
    return boto3.client(
        "s3",
        endpoint_url=c["s3_endpoint"],
        aws_access_key_id=c["s3_access_key"],
        aws_secret_access_key=c["s3_secret_key"],
        region_name=c["s3_region"],
        verify=verify_ssl,
    )
