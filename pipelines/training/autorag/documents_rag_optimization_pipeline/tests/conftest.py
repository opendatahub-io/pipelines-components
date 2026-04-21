"""Pytest fixtures for Documents RAG Optimization pipeline tests."""

import os
import sys
import tempfile
from pathlib import Path

import pytest
from dotenv import load_dotenv, find_dotenv

from pipelines.training.autorag.documents_rag_optimization_pipeline.tests.utils import (
    _make_kfp_client,
    _make_s3_client,
)


_tests_dir = Path(__file__).resolve().parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))


# RHOAI / KFP connection
RHOAI_KFP_URL_ENV = "RHOAI_KFP_URL"
RHOAI_TOKEN_ENV = "RHOAI_TOKEN"
RHOAI_PROJECT_ENV = "RHOAI_PROJECT_NAME"
# Pipeline parameters (Kubernetes secret names and data locations)
TEST_DATA_SECRET_ENV = "TEST_DATA_SECRET_NAME"
TEST_DATA_BUCKET_ENV = "TEST_DATA_BUCKET_NAME"
TEST_DATA_KEY_ENV = "TEST_DATA_KEY"
INPUT_DATA_SECRET_ENV = "INPUT_DATA_SECRET_NAME"
INPUT_DATA_BUCKET_ENV = "INPUT_DATA_BUCKET_NAME"
INPUT_DATA_KEY_ENV = "INPUT_DATA_KEY"
LLAMA_STACK_SECRET_ENV = "LLAMA_STACK_SECRET_NAME"
LLAMA_STACK_VECTOR_IO_PROVIDER_ENV = "LLAMA_STACK_VECTOR_IO_PROVIDER_ID"
# S3 for artifact checks (optional)
ARTIFACTS_S3_ENDPOINT_ENV = "ARTIFACTS_AWS_S3_ENDPOINT"
ARTIFACTS_S3_ACCESS_KEY_ENV = "ARTIFACTS_AWS_ACCESS_KEY_ID"
ARTIFACTS_S3_SECRET_KEY_ENV = "ARTIFACTS_AWS_SECRET_ACCESS_KEY"
ARTIFACTS_S3_REGION_ENV = "ARTIFACTS_AWS_DEFAULT_REGION"
S3_BUCKET_ARTIFACTS_ENV = "RHOAI_TEST_ARTIFACTS_BUCKET"
# Functional test: milvus provider IDs (one per milvus mode)
LLAMA_STACK_VECTOR_IO_PROVIDER_MILVUS_LITE_ENV = "LLAMA_STACK_VECTOR_IO_PROVIDER_ID_MILVUS_LITE"
LLAMA_STACK_VECTOR_IO_PROVIDER_MILVUS_STANDALONE_ENV = "LLAMA_STACK_VECTOR_IO_PROVIDER_ID_MILVUS_STANDALONE"
# Functional test: constrained model lists (JSON arrays)
DOCRAG_EMBEDDINGS_MODELS_ENV = "FUNC_TESTS_EMBEDDINGS_MODELS"
DOCRAG_GENERATION_MODELS_ENV = "FUNC_TESTS_GENERATION_MODELS"


@pytest.fixture(scope="session")
def docrag_integration_config():
    """Session-scoped RHOAI integration config from env; None if not set."""
    load_dotenv(find_dotenv(".env"))

    kfp_url = os.environ.get(RHOAI_KFP_URL_ENV) or os.environ.get("KFP_HOST")
    token = os.environ.get(RHOAI_TOKEN_ENV) or os.environ.get("KFP_TOKEN")
    project = os.environ.get(RHOAI_PROJECT_ENV) or os.environ.get("KFP_NAMESPACE")
    t_secret = os.environ.get(TEST_DATA_SECRET_ENV)
    t_bucket = os.environ.get(TEST_DATA_BUCKET_ENV)
    t_key = os.environ.get(TEST_DATA_KEY_ENV)
    i_secret = os.environ.get(INPUT_DATA_SECRET_ENV)
    i_bucket = os.environ.get(INPUT_DATA_BUCKET_ENV)
    i_key = os.environ.get(INPUT_DATA_KEY_ENV)
    llama_secret = os.environ.get(LLAMA_STACK_SECRET_ENV)
    llama_vector_io = os.environ.get(LLAMA_STACK_VECTOR_IO_PROVIDER_ENV)

    if not all([kfp_url, token, t_secret, t_bucket, t_key, i_secret, i_bucket, i_key, llama_secret, llama_vector_io]):
        return None

    endpoint = os.environ.get(ARTIFACTS_S3_ENDPOINT_ENV)
    access = os.environ.get(ARTIFACTS_S3_ACCESS_KEY_ENV)
    secret = os.environ.get(ARTIFACTS_S3_SECRET_KEY_ENV)
    region = os.environ.get(ARTIFACTS_S3_REGION_ENV, "us-east-1")
    bucket_artifacts = os.environ.get(S3_BUCKET_ARTIFACTS_ENV)

    return {
        "rhoai_kfp_url": kfp_url.strip().rstrip("/"),
        "rhoai_token": token.strip(),
        "rhoai_project": (project or "docrag-integration-test").strip(),
        "test_data_secret_name": t_secret.strip(),
        "test_data_bucket_name": t_bucket.strip(),
        "test_data_key": t_key.strip(),
        "input_data_secret_name": i_secret.strip(),
        "input_data_bucket_name": i_bucket.strip(),
        "input_data_key": i_key.strip(),
        "llama_stack_secret_name": llama_secret.strip(),
        "llama_stack_vector_io_provider_id": llama_vector_io.strip(),
        "s3_endpoint": endpoint.strip() if endpoint else None,
        "s3_access_key": access.strip() if access else None,
        "s3_secret_key": secret.strip() if secret else None,
        "s3_region": region.strip(),
        "s3_bucket_artifacts": bucket_artifacts.strip() if bucket_artifacts else None,
    }


@pytest.fixture(scope="session")
def kfp_client(docrag_integration_config):
    """Session-scoped KFP client for integration tests."""
    return _make_kfp_client(docrag_integration_config)


@pytest.fixture(scope="session")
def compiled_pipeline_path():
    """Compile the Documents RAG Optimization pipeline to a temp YAML file."""
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
    """Session-scoped S3 client for integration test artifact checks (optional)."""
    return _make_s3_client(docrag_integration_config)
