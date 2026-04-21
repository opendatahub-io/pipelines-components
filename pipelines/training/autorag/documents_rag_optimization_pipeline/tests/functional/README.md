# Documents RAG Optimization Pipeline -- Functional Tests

Parametrized functional tests that run the Documents RAG Optimization pipeline
end-to-end on a Red Hat OpenShift AI (RHOAI) cluster.  Each test scenario is
declared as a data-driven config (positive or negative), submitted to KFP, and
validated against expected outcomes.

## How it differs from integration tests

| Aspect | Integration (`../test_pipeline_integration.py`) | Functional (this directory) |
|---|---|---|
| Scenarios | Single "happy path" run | Multiple parametrized scenarios (positive and negative) |
| Parameters | Fixed from env | Per-scenario overrides (metric, provider, models, etc.) |
| Failure testing | Not covered | Negative configs assert the pipeline *fails* |
| Artifact checks | Presence only | Presence + notebook execution via papermill |
| Tag filtering | N/A | Filter scenarios by tags at runtime |

## Directory layout

```text
functional/
  conftest.py           Fixtures: env config, KFP client, S3 client
  test_configs.py       Scenario definitions (positive / negative) + TestConfig dataclass
  test_pipeline_functional.py   Pytest test class (parametrized over configs)
  utils.py              Helpers: run submission, state checks, failure diagnostics,
                        artifact validation, notebook download & execution
  .env.example          Template for required and optional env vars
  README.md             This file
```

## Prerequisites

- A **RHOAI cluster** with **Data Science Pipelines** enabled.
- Environment variables for the KFP API, bearer token, and pipeline parameters
  (secret names, bucket/key locations). All referenced Kubernetes secrets and
  data must already exist in the cluster.
- (Optional) S3 credentials and an artifacts bucket for post-run artifact and
  notebook validation.
- (Optional) The `kubernetes` Python package for pod-log fetching on failure.

## Quick start

```bash
# 1. Copy the env template and fill in your values
cp .env.example .env
# Edit .env

# 2. Run all functional tests (skipped automatically if env is not configured)
uv run python -m pytest pipelines/training/autorag/documents_rag_optimization_pipeline/tests/functional/ -v

# 3. Run only scenarios tagged "smoke"
FUNCTIONAL_TESTS_TAGS=smoke \
  uv run python -m pytest pipelines/training/autorag/documents_rag_optimization_pipeline/tests/functional/ -v

# 4. Run only negative scenarios
uv run python -m pytest \
  pipelines/training/autorag/documents_rag_optimization_pipeline/tests/functional/ \
  -v -k "negative"

# 5. Run only positive scenarios
uv run python -m pytest \
  pipelines/training/autorag/documents_rag_optimization_pipeline/tests/functional/ \
  -v -k "positive"
```

## Environment variables

### Required (tests are skipped when any of these is missing)

| Variable | Description |
|---|---|
| `RHOAI_KFP_URL` | KFP API base URL (alt: `KFP_HOST`) |
| `RHOAI_TOKEN` | Bearer token for API auth, e.g. `oc whoami -t` (alt: `KFP_TOKEN`) |
| `RHOAI_PROJECT_NAME` | KFP namespace / project (alt: `KFP_NAMESPACE`; default `docrag-integration-test`) |
| `TEST_DATA_SECRET_NAME` | K8s secret name for test-data S3 credentials |
| `TEST_DATA_BUCKET_NAME` | S3 bucket containing the test data JSON |
| `TEST_DATA_KEY` | Object key of the test data file |
| `INPUT_DATA_SECRET_NAME` | K8s secret name for input-documents S3 credentials |
| `INPUT_DATA_BUCKET_NAME` | S3 bucket for input documents |
| `INPUT_DATA_KEY` | Object key of the input documents in the bucket |
| `LLAMA_STACK_SECRET_NAME` | K8s secret name for Llama Stack API credentials |

### Milvus provider IDs (one per mode)

| Variable | Description |
|---|---|
| `LLAMA_STACK_VECTOR_IO_PROVIDER_ID_MILVUS_LITE` | Provider ID for milvus-lite scenarios |
| `LLAMA_STACK_VECTOR_IO_PROVIDER_ID_MILVUS_REMOTE` | Provider ID for milvus-remote scenarios |

### Constrained model lists (optional, JSON arrays)

| Variable | Description |
|---|---|
| `FUNC_TEST_EMBEDDINGS_MODELS` | e.g. `'["model-a","model-b"]'` |
| `FUNC_TEST_GENERATION_MODELS` | e.g. `'["model-x"]'` |

### Artifact validation (optional)

| Variable | Description |
|---|---|
| `ARTIFACTS_AWS_S3_ENDPOINT` | S3-compatible endpoint (e.g. MinIO) |
| `ARTIFACTS_AWS_ACCESS_KEY_ID` | Access key for the artifacts bucket |
| `ARTIFACTS_AWS_SECRET_ACCESS_KEY` | Secret key |
| `ARTIFACTS_AWS_DEFAULT_REGION` | Region (default `us-east-1`) |
| `RHOAI_TEST_ARTIFACTS_BUCKET` | Bucket where pipeline artifacts are stored |

### Behavior tuning (optional)

| Variable | Description |
|---|---|
| `RHOAI_PIPELINE_RUN_TIMEOUT` | Timeout in seconds (default `3600`) |
| `KFP_VERIFY_SSL` | `false` to skip TLS verification |
| `K8S_API_URL` | Override the derived Kubernetes API URL entirely |
| `K8S_API_PORT` | Override just the port (default `443` for ROSA, `6443` for OCP) |
| `FUNCTIONAL_TESTS_TAGS` | Comma-separated tags to filter scenarios (e.g. `smoke,milvus-lite`) |

## Test scenario format

Scenarios are defined as Python dicts in `test_configs.py`, split into
`_CONFIGS_NEGATIVE` and `_CONFIGS_POSITIVE`.  Each dict has:

| Key | Type | Description |
|---|---|---|
| `id` | `str` | Short identifier shown in pytest output (e.g. `TC-P-1`) |
| `description` | `str` | Human-readable summary |
| `tags` | `list[str]` | Used for runtime filtering via `FUNCTIONAL_TESTS_TAGS` |
| `expected_result` | `"pass"` or `"fail"` | Whether the pipeline run should succeed |
| `llama_stack_vector_io_provider_type` | `str \| None` | Sentinel resolved to a provider ID via env (`milvus-lite`, `milvus-remote`); `None` passes empty string (for negative tests) |
| `pipeline_params_overrides` | `dict` | Per-scenario parameter overrides (see resolution rules below) |

### Override resolution rules

Values in `pipeline_params_overrides` are resolved by `TestConfig.get_pipeline_arguments()`:

| Value | Behavior |
|---|---|
| `None` | Use the base config value from env |
| `""` | Pass an empty string explicitly |
| `"ENV"` | Read from the dedicated env var (for `embeddings_models` / `generation_models`) |
| A list (e.g. `["model"]`) | Pass directly to the pipeline |
| Any other scalar | Use as-is |

The `llama_stack_vector_io_provider_type` field is resolved separately via
`_VECTOR_IO_PROVIDER_MAP` to look up the corresponding env var.

## Pass / fail criteria

### Expected-pass scenarios

1. Pipeline run finishes with state `SUCCEEDED`
2. At least 1 pattern artifact exists in S3
3. At least 1 indexing notebook and 1 inference notebook exist in S3
4. `evaluation_results.json` exists in S3
5. A randomly selected indexing and inference notebook can be executed via
   papermill (with a mocked `input()` builtin injected)

### Expected-fail scenarios

1. Pipeline run finishes with state `FAILED` (not `SUCCEEDED`, not timeout)
2. Failure details are logged for observability

Negative scenarios use a capped timeout (`_EXPECTED_FAIL_TIMEOUT_CAP = 600s`)
since failures should surface quickly.

## Failure diagnostics

When a pipeline run fails (unexpectedly or for a negative test), `utils._collect_failure_details()`:

1. Fetches run-level and task-level metadata from the KFP v2 API
2. Lists pods matching `pipeline/runid=<run_id>` via the Kubernetes API
3. Identifies failed pods (phase `Failed` or non-zero exit code)
4. Fetches the last 100 log lines from each container of each failed pod
5. Returns a formatted report appended to the test failure message

Kubernetes API authentication is derived from the RHOAI token and KFP URL
(standard OCP and ROSA patterns are supported), with optional `K8S_API_URL` /
`K8S_API_PORT` overrides.
