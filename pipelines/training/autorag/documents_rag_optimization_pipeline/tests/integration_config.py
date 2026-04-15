"""RHOAI integration and functional test config: load .env and build config from environment.

Used by conftest.py (fixtures) and test_pipeline_integration.py / test_pipeline_functional.py
(skipif) so skip logic and fixtures share one source of truth. Import this module instead
of conftest to avoid resolving the repo-root conftest when running tests.
"""

import json
import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from path into os.environ."""
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if v.startswith('"') and v.endswith('"'):
                    v = v[1:-1]
                elif v.startswith("'") and v.endswith("'"):
                    v = v[1:-1]
                os.environ[k] = v


# Load .env from cwd and from this directory
_load_dotenv(Path.cwd() / ".env")
_load_dotenv(Path(__file__).resolve().parent / ".env")

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
S3_ENDPOINT_ENV = "AWS_S3_ENDPOINT"
S3_ACCESS_KEY_ENV = "AWS_ACCESS_KEY_ID"
S3_SECRET_KEY_ENV = "AWS_SECRET_ACCESS_KEY"
S3_REGION_ENV = "AWS_DEFAULT_REGION"
S3_BUCKET_ARTIFACTS_ENV = "RHOAI_TEST_ARTIFACTS_BUCKET"
# Functional test: milvus provider IDs (one per milvus mode)
LLAMA_STACK_VECTOR_IO_PROVIDER_MILVUS_LITE_ENV = "LLAMA_STACK_VECTOR_IO_PROVIDER_ID_MILVUS_LITE"
LLAMA_STACK_VECTOR_IO_PROVIDER_MILVUS_STANDALONE_ENV = "LLAMA_STACK_VECTOR_IO_PROVIDER_ID_MILVUS_STANDALONE"
# Functional test: constrained model lists (JSON arrays)
DOCRAG_EMBEDDINGS_MODELS_ENV = "DOCRAG_EMBEDDINGS_MODELS"
DOCRAG_GENERATION_MODELS_ENV = "DOCRAG_GENERATION_MODELS"


def get_docrag_integration_config():
    """Build integration config from environment; None if not configured."""
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

    endpoint = os.environ.get(S3_ENDPOINT_ENV)
    access = os.environ.get(S3_ACCESS_KEY_ENV)
    secret = os.environ.get(S3_SECRET_KEY_ENV)
    region = os.environ.get(S3_REGION_ENV, "us-east-1")
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


# Single source of truth for skipif: tests run only when this is not None.
DOCRAG_INTEGRATION_CONFIG = get_docrag_integration_config()


def get_docrag_functional_config():
    """Build functional test config from environment; None if not configured.

    Similar to get_docrag_integration_config but with relaxed guards (does not
    require llama_stack_vector_io_provider_id or input_data_key since those are
    overridden per-scenario) and additional keys for milvus provider IDs and
    constrained model lists.
    """
    kfp_url = os.environ.get(RHOAI_KFP_URL_ENV) or os.environ.get("KFP_HOST")
    token = os.environ.get(RHOAI_TOKEN_ENV) or os.environ.get("KFP_TOKEN")
    project = os.environ.get(RHOAI_PROJECT_ENV) or os.environ.get("KFP_NAMESPACE")
    t_secret = os.environ.get(TEST_DATA_SECRET_ENV)
    t_bucket = os.environ.get(TEST_DATA_BUCKET_ENV)
    t_key = os.environ.get(TEST_DATA_KEY_ENV)
    i_secret = os.environ.get(INPUT_DATA_SECRET_ENV)
    i_bucket = os.environ.get(INPUT_DATA_BUCKET_ENV)
    i_key = os.environ.get(INPUT_DATA_KEY_ENV, "")
    llama_secret = os.environ.get(LLAMA_STACK_SECRET_ENV)
    llama_vector_io = os.environ.get(LLAMA_STACK_VECTOR_IO_PROVIDER_ENV, "")

    if not all([kfp_url, token, t_secret, t_bucket, t_key, i_secret, i_bucket, llama_secret]):
        return None

    endpoint = os.environ.get(S3_ENDPOINT_ENV)
    access = os.environ.get(S3_ACCESS_KEY_ENV)
    secret = os.environ.get(S3_SECRET_KEY_ENV)
    region = os.environ.get(S3_REGION_ENV, "us-east-1")
    bucket_artifacts = os.environ.get(S3_BUCKET_ARTIFACTS_ENV)

    # Milvus provider IDs (per milvus mode)
    vector_io_milvus_lite = os.environ.get(LLAMA_STACK_VECTOR_IO_PROVIDER_MILVUS_LITE_ENV, "")
    vector_io_milvus_standalone = os.environ.get(LLAMA_STACK_VECTOR_IO_PROVIDER_MILVUS_STANDALONE_ENV, "")

    # Constrained model lists (optional JSON arrays)
    embeddings_models_raw = os.environ.get(DOCRAG_EMBEDDINGS_MODELS_ENV)
    generation_models_raw = os.environ.get(DOCRAG_GENERATION_MODELS_ENV)
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
        "llama_stack_vector_io_provider_id": llama_vector_io.strip(),
        "vector_io_provider_milvus_lite": vector_io_milvus_lite.strip(),
        "vector_io_provider_milvus_standalone": vector_io_milvus_standalone.strip(),
        "embeddings_models": embeddings_models,
        "generation_models": generation_models,
        "s3_endpoint": endpoint.strip() if endpoint else None,
        "s3_access_key": access.strip() if access else None,
        "s3_secret_key": secret.strip() if secret else None,
        "s3_region": region.strip(),
        "s3_bucket_artifacts": bucket_artifacts.strip() if bucket_artifacts else None,
    }


DOCRAG_FUNCTIONAL_CONFIG = get_docrag_functional_config()
