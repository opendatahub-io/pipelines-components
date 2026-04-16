# Documents RAG Optimization Pipeline – Tests

This directory contains unit and integration tests for the Documents RAG Optimization pipeline.

## Test types

| Test file | Type | Description |
| --------- | ------ | ------------- |
| `test_pipeline_unit.py` | Unit | Pipeline structure and interface (no cluster). |
| `test_pipeline_integration.py` | Integration | Runs the pipeline on RHOAI and optionally checks generated artifacts in S3. |

Unit tests run with the default test suite. Integration tests are **skipped** unless the required environment variables are set (see below).

## Running tests

From the repository root:

```bash
# Unit tests only
uv run python -m pytest pipelines/training/autorag/documents_rag_optimization_pipeline/tests/test_pipeline_unit.py -v

# All tests (integration skipped if env not set)
uv run python -m pytest pipelines/training/autorag/documents_rag_optimization_pipeline/tests/ -v

# Integration tests only (skip when env not set)
uv run python -m pytest pipelines/training/autorag/documents_rag_optimization_pipeline/tests/test_pipeline_integration.py -v
```

Or via the repo test runner:

```bash
uv run python -m scripts.tests.run_component_tests pipelines/training/autorag/documents_rag_optimization_pipeline
```

## Integration test setup

Integration tests require:

- A **Red Hat OpenShift AI (RHOAI)** cluster with **Data Science Pipelines** enabled.
- **Environment variables** for the KFP API URL, token, namespace, and pipeline parameters (secret names, bucket names, keys). Secrets and data must already exist in the cluster.

### 1. Configure environment

Copy the template and fill in your values:

```bash
cp .env.example .env
# Edit .env with your RHOAI URL, token, project name, and pipeline parameters.
```

Variables are loaded from a `.env` file in this directory (or from the current working directory). See `integration_config.py` for the exact keys and logic.

### 2. Required environment variables

| Variable | Description |
| -------- | ----------- |
| `RHOAI_KFP_URL` | KFP API base URL (e.g. `https://ds-pipeline-dspipeline.<domain>/pipeline`). Alternative: `KFP_HOST`. |
| `RHOAI_TOKEN` | Bearer token for API auth (e.g. `oc whoami -t`). Alternative: `KFP_TOKEN`. |
| `RHOAI_PROJECT_NAME` | KFP namespace / project (e.g. `docrag-integration-test`). Alternative: `KFP_NAMESPACE`. |
| `TEST_DATA_SECRET_NAME` | Kubernetes secret name for test data S3 credentials. |
| `TEST_DATA_BUCKET_NAME` | S3 bucket for the test data JSON file. |
| `TEST_DATA_KEY` | Object key (path) of the test data file in the bucket. |
| `INPUT_DATA_SECRET_NAME` | Kubernetes secret name for input documents S3 credentials. |
| `INPUT_DATA_BUCKET_NAME` | S3 bucket for input documents. |
| `INPUT_DATA_KEY` | Object key (path) of the input documents in the bucket. |
| `LLAMA_STACK_SECRET_NAME` | Kubernetes secret name for llama-stack API (e.g. `LLAMA_STACK_CLIENT_API_KEY`, `LLAMA_STACK_CLIENT_BASE_URL`). |
| `LLAMA_STACK_VECTOR_IO_PROVIDER_ID` | Vector I/O provider id passed to the pipeline (e.g. `milvus`); must match a provider registered in Llama Stack. |

### 3. Optional: artifact validation in S3

To assert that run artifacts (leaderboard HTML, notebooks, rag_patterns) are present in object storage, set:

| Variable | Description |
| -------- | ----------- |
| `AWS_S3_ENDPOINT` | S3-compatible endpoint (e.g. MinIO). |
| `AWS_ACCESS_KEY_ID` | Access key for artifact bucket. |
| `AWS_SECRET_ACCESS_KEY` | Secret key. |
| `AWS_DEFAULT_REGION` | Region (default `us-east-1`). |
| `RHOAI_TEST_ARTIFACTS_BUCKET` | Bucket where pipeline artifacts are stored (often same as pipeline root bucket). |

### 4. Optional: test behavior

| Variable | Description |
| -------- | ----------- |
| `RHOAI_PIPELINE_RUN_TIMEOUT` | Timeout in seconds for waiting on a run (default `3600`). |
| `KFP_VERIFY_SSL` | Set to `false` to skip TLS verification for self-signed certs. |

### 5. Optional: Llama Stack response-body artifact in S3

If S3 settings from [§3](#3-optional-artifact-validation-in-s3) are set, the integration test
checks that generated artifacts are present under the run prefix, including
`v1_responses_body.json`.

## Test layout

- **`integration_config.py`** – Loads `.env` and builds `DOCRAG_INTEGRATION_CONFIG`; used by `conftest.py` and the integration test for skip logic and config.
- **`conftest.py`** – Pytest fixtures: `docrag_integration_config`, `kfp_client`, `compiled_pipeline_path`, `pipeline_run_timeout`, `s3_client`.
- **`test_pipeline_integration.py`** – Submits the compiled pipeline with arguments from config, waits for completion, asserts success, and optionally checks for artifacts in S3.
- **`.env.example`** – Template for required and optional env vars; copy to `.env` and fill in (committable; `.env` is gitignored).

## Pipeline parameters in integration tests

The integration test builds pipeline arguments from the integration config. Only the **required** pipeline parameters are passed (secret names, bucket names, keys).
Optional pipeline parameters (e.g. `optimization_metric`, `optimization_max_rag_patterns`) can be added to `integration_config.py` and `_pipeline_arguments_from_config()` in `test_pipeline_integration.py` if needed.
