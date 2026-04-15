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
def docrag_functional_config():
    """Session-scoped functional test config from env; None if not set."""
    from integration_config import get_docrag_functional_config

    return get_docrag_functional_config()


def _make_kfp_client(config):
    """Create a KFP client from a config dict; returns None if config is None."""
    if config is None:
        return None
    import kfp

    host = config["rhoai_kfp_url"]
    if not host.endswith("/"):
        host = host + "/"
    verify_ssl = os.environ.get("KFP_VERIFY_SSL", "true").strip().lower()
    verify_ssl = verify_ssl not in ("0", "false", "no")
    return kfp.Client(
        host=host,
        namespace=config["rhoai_project"],
        existing_token=config.get("rhoai_token"),
        verify_ssl=verify_ssl,
    )


def _make_s3_client(config):
    """Create a boto3 S3 client from a config dict; returns None if not configured."""
    if config is None or not config.get("s3_endpoint"):
        return None
    try:
        import boto3
    except ImportError:
        return None
    return boto3.client(
        "s3",
        endpoint_url=config["s3_endpoint"],
        aws_access_key_id=config["s3_access_key"],
        aws_secret_access_key=config["s3_secret_key"],
        region_name=config["s3_region"],
    )


@pytest.fixture(scope="session")
def kfp_client(docrag_integration_config):
    """Session-scoped KFP client for integration tests."""
    return _make_kfp_client(docrag_integration_config)


@pytest.fixture(scope="session")
def functional_kfp_client(docrag_functional_config):
    """Session-scoped KFP client for functional tests."""
    return _make_kfp_client(docrag_functional_config)


@pytest.fixture(scope="session")
def _compiled_docrag_pipeline_package_from_source():
    """Compile the Documents RAG Optimization pipeline once per session to a temp YAML file."""
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


@pytest.fixture(
    scope="session",
    params=["compile_from_source", "committed_pipeline_yaml"],
    ids=["compile-from-source", "committed-pipeline-yaml"],
)
def pipeline_package_path(request):
    """KFP pipeline package path: fresh compile or repo-root ``pipeline.yaml``.

    Integration tests run twice so both the checked-in artifact and current Python
    sources are exercised on the cluster.
    """
    if request.param == "compile_from_source":
        return request.getfixturevalue("_compiled_docrag_pipeline_package_from_source")
    committed = Path(__file__).resolve().parent.parent / "pipeline.yaml"
    if not committed.is_file():
        pytest.skip(f"Committed pipeline YAML not found: {committed}")
    return str(committed.resolve())


@pytest.fixture(scope="session")
def pipeline_run_timeout():
    """Timeout in seconds for waiting on a pipeline run (override via env)."""
    return int(os.environ.get("RHOAI_PIPELINE_RUN_TIMEOUT", "3600"))


@pytest.fixture(scope="session")
def s3_client(docrag_integration_config):
    """Session-scoped S3 client for integration test artifact checks (optional)."""
    return _make_s3_client(docrag_integration_config)


@pytest.fixture(scope="session")
def functional_s3_client(docrag_functional_config):
    """Session-scoped S3 client for functional test artifact checks (optional)."""
    return _make_s3_client(docrag_functional_config)
