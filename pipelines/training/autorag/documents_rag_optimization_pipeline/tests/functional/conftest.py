import json
import os

from dotenv import load_dotenv, find_dotenv
import pytest

from pipelines.training.autorag.documents_rag_optimization_pipeline.tests.utils import (
    _make_kfp_client,
    _make_s3_client,
)


def get_functional_config():
    """Build functional test config from environment; None if not configured.

    Relaxed guards compared to integration config (does not require
    llama_stack_vector_io_provider_id or input_data_key since those are
    overridden per-scenario). Adds milvus provider IDs and constrained model lists.
    """
    load_dotenv(find_dotenv(".env"))

    kfp_url = os.environ.get("RHOAI_KFP_URL") or os.environ.get("KFP_HOST")
    token = os.environ.get("RHOAI_TOKEN") or os.environ.get("KFP_TOKEN")
    project = os.environ.get("RHOAI_PROJECT_NAME") or os.environ.get("KFP_NAMESPACE")
    t_secret = os.environ.get("TEST_DATA_SECRET_NAME")
    t_bucket = os.environ.get("TEST_DATA_BUCKET_NAME")
    t_key = os.environ.get("TEST_DATA_KEY")
    i_secret = os.environ.get("INPUT_DATA_SECRET_NAME")
    i_bucket = os.environ.get("INPUT_DATA_BUCKET_NAME")
    i_key = os.environ.get("INPUT_DATA_KEY")
    llama_secret = os.environ.get("LLAMA_STACK_SECRET_NAME")

    if not all([kfp_url, token, t_secret, t_bucket, t_key, i_secret, i_bucket, i_key, llama_secret]):
        return None

    endpoint = os.environ.get("ARTIFACTS_AWS_S3_ENDPOINT")
    access = os.environ.get("ARTIFACTS_AWS_ACCESS_KEY_ID")
    secret = os.environ.get("ARTIFACTS_AWS_SECRET_ACCESS_KEY")
    region = os.environ.get("ARTIFACTS_AWS_DEFAULT_REGION", "us-east-1")
    bucket_artifacts = os.environ.get("RHOAI_TEST_ARTIFACTS_BUCKET")

    # Milvus provider IDs (per milvus mode)
    vector_io_milvus_lite = os.environ.get("LLAMA_STACK_VECTOR_IO_PROVIDER_ID_MILVUS_LITE", "")
    vector_io_milvus_remote = os.environ.get("LLAMA_STACK_VECTOR_IO_PROVIDER_ID_MILVUS_REMOTE", "")

    # Constrained model lists (optional JSON arrays)
    embeddings_models_raw = os.environ.get("FUNC_TEST_EMBEDDINGS_MODELS")
    generation_models_raw = os.environ.get("FUNC_TEST_GENERATION_MODELS")
    embeddings_models = json.loads(embeddings_models_raw) if embeddings_models_raw else None
    generation_models = json.loads(generation_models_raw) if generation_models_raw else None

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
        "vector_io_provider_milvus_lite": vector_io_milvus_lite.strip(),
        "vector_io_provider_milvus_remote": vector_io_milvus_remote.strip(),
        "embeddings_models": embeddings_models,
        "generation_models": generation_models,
        "s3_endpoint": endpoint.strip() if endpoint else None,
        "s3_access_key": access.strip() if access else None,
        "s3_secret_key": secret.strip() if secret else None,
        "s3_region": region.strip(),
        "s3_bucket_artifacts": bucket_artifacts.strip() if bucket_artifacts else None,
    }


@pytest.fixture(scope="session")
def functional_env_config():
    return get_functional_config()


@pytest.fixture(scope="session")
def kfp_client_functional(functional_env_config):
    """Session-scoped KFP client for functional tests."""
    return _make_kfp_client(functional_env_config)


@pytest.fixture(scope="session")
def s3_client_functional(functional_env_config):
    """Session-scoped S3 client for functional test artifact checks (optional)."""
    return _make_s3_client(functional_env_config)
