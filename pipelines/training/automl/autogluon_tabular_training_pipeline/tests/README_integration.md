# AutoML integration tests (RHOAI)

High-level tests that run the **AutoGluon tabular training pipeline** (AutoML) on a **Red Hat OpenShift AI (RHOAI)** cluster and validate success and artifacts.
These tests live under `pipelines/training/automl/autogluon_tabular_training_pipeline/tests/`.
When the required environment variables are not set, the AutoML integration tests are **skipped** (unit tests in this directory still run).

## Requirements

- RHOAI cluster with Data Science Pipelines enabled. A pipeline server does not need to be running beforehand: you can have the tests create a **DataSciencePipelinesApplication** (DSPA) dynamically by setting `RHOAI_CREATE_DSPA=true`; the operator will then deploy the pipeline server in the test namespace.
- S3-compatible storage (e.g. MinIO or AWS S3) for AutoML pipeline test data and artifacts.
- **Service Account setup** is required: create a Service Account and use its token for `RHOAI_TOKEN`. See [Creating a service account for the tests](#creating-a-service-account-for-the-tests) below.
- Optional: `kubectl/oc` access (or in-cluster config) to create a test project (namespace) and S3 connection secret.

## Environment variables

Set these to enable the AutoML integration tests; otherwise they are skipped.
You can set them in the shell or via a **`.env`** file in the repo root or in this directory (`tests/`).
Copy `.env.template` to `.env` and fill in your values.
The module `integration_config.py` loads `.env` when `get_rhoai_config()` / `get_dspa_config()` run and builds the config; the same config is used for both the skip condition and the fixtures.

| Variable | Required | Description |
|----------|----------|-------------|
| `RHOAI_URL` | Yes | Base URL of the OCP cluster (e.g. `https://api.example.com`). |
| `RHOAI_KFP_URL` | No | KFP server URL where the client connects. Omit when using `RHOAI_CREATE_DSPA=true` (the route URL is resolved automatically). |
| `RHOAI_TOKEN` | Yes | API token; use a **service account token** for Jenkins/CI (long-lived, no oc or kubeconfig). |
| `RHOAI_PROJECT_NAME` | No | RHOAI Project/namespace name for the test run (default: `kfp-integration-test`). |
| `AWS_S3_ENDPOINT` | Yes | S3-compatible endpoint URL. |
| `AWS_ACCESS_KEY_ID` | Yes | S3 access key. |
| `AWS_SECRET_ACCESS_KEY` | Yes | S3 secret key. |
| `AWS_DEFAULT_REGION` | No | S3 region (default: `us-east-1`). |
| `RHOAI_TEST_DATA_BUCKET` | Yes | Bucket used for test data upload and pipeline input. |
| `RHOAI_TEST_ARTIFACTS_BUCKET` | No | Bucket where pipeline artifacts are written (default: same as data bucket). |
| `RHOAI_TEST_S3_SECRET_NAME` | No | Name of the Kubernetes secret holding S3 credentials in the project (default: `s3-connection`). |
| `RHOAI_PIPELINE_RUN_TIMEOUT` | No | Timeout in seconds for waiting on a run (default: `3600`). |
| `RHOAI_TEST_CONFIG_TAGS` | No | Comma-separated tags; if set, only test configs with at least one of these tags run (e.g. `smoke`, `regression`). See [Filtering by tags](#filtering-by-tags). |
| `RHOAI_CREATE_DSPA` | No | Set to `true` or `1` to have the tests create a DataSciencePipelinesApplication CR; the operator deploys the pipeline server. See [Creating a DataSciencePipelinesApplication CR](#creating-a-datasciencepipelinesapplication-cr-dspa). |

All required variables must be set for the AutoML integration tests to run; if any is missing, `RHOAI_INTEGRATION_CONFIG` is `None` and the tests are skipped with a reason pointing to `.env.template`.

### Authentication (service account token for Jenkins / CI)

Set **`RHOAI_TOKEN`** to an OpenShift API token.
Follow [Scenario 1](#scenario-1-automatic-project-creation) (automatic project creation) or [Scenario 2](#scenario-2-no-automatic-project-creation-existing-project) (existing project) below to create the ServiceAccount and obtain a token; in Jenkins, store the token as a secret and bind it to `RHOAI_TOKEN`.
Service account tokens are long-lived; no `oc` or kubeconfig is needed at test time.

### Creating a service account for the tests

**Required.** You must create a Service Account and use its token as `RHOAI_TOKEN` for the integration tests to authenticate to the cluster.
Choose **one** of the two scenarios below.
Both produce a token for `RHOAI_TOKEN`; no `oc` or kubeconfig is needed when running the tests.

---

#### Scenario 1: Automatic project creation

The test creates the project via OpenShift ProjectRequest (same as `oc new-project`). Set `RHOAI_PROJECT_NAME` to the project name to create; it can be a new name each run (e.g. in CI). The ServiceAccount can live in any namespace (e.g. `default`).

**1. Create the ServiceAccount** (in any namespace):

```bash
export SA_NAMESPACE=default
export SA_NAME=kfp-integration-tests
oc create serviceaccount "${SA_NAME}" -n "${SA_NAMESPACE}"
```

**2. Grant self-provisioner** so the SA can create projects:

```bash
oc adm policy add-cluster-role-to-user self-provisioner -z "${SA_NAME}" -n "${SA_NAMESPACE}"
```

**3. (Optional) If your cluster does not grant the ProjectRequest creator admin** in the new project, the test will create a RoleBinding to grant the SA admin. For that, the SA needs permission to create RoleBindings. A cluster admin runs once:

```bash
oc create clusterrole kfp-integration-tests-rolebinding-creator \
  --verb=create,get,update,patch \
  --resource=rolebindings.rbac.authorization.k8s.io

oc adm policy add-cluster-role-to-user kfp-integration-tests-rolebinding-creator \
  -z "${SA_NAME}" -n "${SA_NAMESPACE}"
```

**4. Create a token** and configure:

```bash
oc create token "${SA_NAME}" -n "${SA_NAMESPACE}" --duration=8760h
```

Set `RHOAI_TOKEN` to the printed token and `RHOAI_PROJECT_NAME` to the project name the test should create (e.g. `automl-integration-tests`). Do not create that project beforehand.

---

#### Scenario 2: No automatic project creation (existing project)

The project already exists (e.g. created with `oc new-project` or an existing RHOAI Data Science project). The ServiceAccount is created **in that project** and granted **edit** in that project.

**1. Create or use the project:**

```bash
export RHOAI_PROJECT_NAME=automl-integration-tests
oc new-project "${RHOAI_PROJECT_NAME}"
```

**2. Create the ServiceAccount in that project:**

```bash
oc create serviceaccount kfp-integration-tests -n "${RHOAI_PROJECT_NAME}"
```

**3. Grant edit in the project** (Secrets, DSPA CR, Routes):

```bash
oc adm policy add-role-to-user edit -z kfp-integration-tests -n "${RHOAI_PROJECT_NAME}"
```

**4. Create a token** and configure:

```bash
oc create token kfp-integration-tests -n "${RHOAI_PROJECT_NAME}" --duration=8760h
```

Set `RHOAI_TOKEN` to the printed token and `RHOAI_PROJECT_NAME` to that project name. No self-provisioner or extra cluster roles are required.

---

#### Token and test configuration (both scenarios)

- Create token: `oc create token <sa-name> -n <sa-namespace> [--duration=8760h]`
- If the token is in a Secret: `oc get secret <name> -n <namespace> -o jsonpath='{.data.token}' | base64 -d`
- Set `RHOAI_TOKEN` and `RHOAI_PROJECT_NAME` in `.env` or Jenkins; no `oc` or kubeconfig needed at test time.

## Test layout

- **`integration_config.py`** – Loads `.env` inside config helpers, defines `get_rhoai_config()`, and exposes `RHOAI_INTEGRATION_CONFIG` (single source of truth for AutoML integration skip and fixtures).
- **`conftest.py`** – Pytest fixtures for the AutoML pipeline tests; adds the `tests` directory to `sys.path` so `integration_config` can be imported.
  When integration config is set, a **temporary kubeconfig** is created from `RHOAI_URL` and `RHOAI_TOKEN`; the Kubernetes client uses this file instead of `~/.kube/config`.
- **`test_pipeline_integration.py`** – AutoML integration test class marked with `@pytest.mark.integration` and `@pytest.mark.skipif(RHOAI_INTEGRATION_CONFIG is None, ...)`.
  Parametrized over configs from `test_configs.json` (filterable via `RHOAI_TEST_CONFIG_TAGS`).

## Fixtures (conftest.py)

Fixtures used by the AutoML pipeline integration tests:

| Fixture | Scope | Description |
|---------|--------|-------------|
| `rhoai_integration_config` | session | Config dict from env (or `None`). |
| `integration_available` | session | `True` when config is present. |
| `temp_kubeconfig_path` | session | Temp kubeconfig file built from `RHOAI_URL` and `RHOAI_TOKEN`; used by all Kubernetes API calls so the default `~/.kube/config` is not used. Yields path or `None`. |
| `s3_client` | session | Boto3 S3 client for uploads and artifact checks; `None` if config missing or boto3 unavailable. |
| `rhoai_project` | session | Ensures Kubernetes namespace and S3 connection secret exist; skips if kubeconfig/in-cluster config unavailable. |
| `datascience_pipelines_application` | session | Optionally creates a **DataSciencePipelinesApplication** CR in the test namespace (see [Creating a DataSciencePipelinesApplication CR](#creating-a-datasciencepipelinesapplication-cr-dspa)). Yields the CR dict or `None`. |
| `uploaded_datasets` | session | Uploads dataset files from `test_configs.json` (by `dataset_path`) to S3; returns map `dataset_path` → `{"bucket", "key"}`. Empty dict when integration not configured. |
| `kfp_client` | session | KFP client pointing at RHOAI (`rhoai_kfp_url`) with token auth; `None` if config missing. |
| `pipeline_package_path` | session (parametrized) | Path passed to `create_run_from_pipeline_package`: either a **fresh compile** from `pipeline.py` or the repo **`pipeline.yaml`** (`compile-from-source` / `committed-pipeline-yaml`). Each integration scenario runs for both. |
| `pipeline_run_timeout` | function | Timeout in seconds (from `RHOAI_PIPELINE_RUN_TIMEOUT` or `3600`). |

## Test scenarios and parametrization

Scenarios are parametrized via test configs loaded from **`test_configs.json`** in this directory (optionally filtered by `RHOAI_TEST_CONFIG_TAGS`). Each configuration in the JSON array specifies:

- **Dataset location** – `dataset_path`: path to the dataset file relative to the tests directory (e.g. `data/regression.csv`). The **`uploaded_datasets`** fixture uploads each unique path to S3 and returns a map used to resolve pipeline bucket/key.
- **Target column** – `label_column` in the dataset.
- **Problem type** – `problem_type`: `"classification"`, `"regression"`, or `"timeseries"` (reserved for future use).
- **AutoML/pipeline settings** – `task_type` (`"binary"`, `"multiclass"`, `"regression"`) and `automl_settings` (e.g. `top_n`) merged into pipeline arguments.

Two integration test runs occur per config (fresh compile and committed `pipeline.yaml`).
To add a new scenario, add an object to the `test_configs.json` array with keys: `id`, `dataset_path`, `label_column`, `problem_type`, `task_type`, `automl_settings`, and optionally `tags` (list of strings for filtering).
Put the dataset file under the tests directory at `dataset_path` (e.g. `data/my_dataset.csv`); it will be uploaded to S3 once per session.

### Filtering by tags

Each config in `test_configs.json` can include a **`tags`** array (e.g. `["smoke", "regression"]`).
Set **`RHOAI_TEST_CONFIG_TAGS`** to a comma-separated list of tags to run only configs that have at least one of those tags.
If unset, all configs run.
Example: `RHOAI_TEST_CONFIG_TAGS=smoke` runs only configs tagged `smoke`; `RHOAI_TEST_CONFIG_TAGS=classification,regression` runs configs tagged either `classification` or `regression`.
You do not need to refer to config ids.

Current configs:

1. **regression** – `task_type=regression`, label `price`; tags: `regression`, `smoke`.
2. **classification_binary** – `task_type=binary`, label `target`; tags: `classification`, `binary`, `smoke`.
3. **classification_multiclass** – `task_type=multiclass`, label `target`; tags: `classification`, `multiclass`.

## Running the tests

Install dependencies for the AutoML integration tests (includes base test deps; see `test_automl` extra in `pyproject.toml`):

```bash
uv sync --extra test_automl
# or: pip install -e ".[test_automl]"
```

Run only AutoML integration tests (from repo root, change path appropriately if run from somewhere else):

```bash
uv run pytest pipelines/training/automl/autogluon_tabular_training_pipeline/tests/test_pipeline_integration.py -m integration -v
```

Run only configs with a given tag (e.g. smoke tests only):

```bash
RHOAI_TEST_CONFIG_TAGS=smoke uv run pytest pipelines/training/automl/autogluon_tabular_training_pipeline/tests/test_pipeline_integration.py -m integration -v
```

Run all AutoML pipeline tests (unit + integration; integration tests skip if env not set):

```bash
uv run pytest pipelines/training/automl/autogluon_tabular_training_pipeline/tests/ -v
```

Exclude AutoML integration tests:

```bash
uv run pytest pipelines/training/automl/autogluon_tabular_training_pipeline/tests/ -m "not integration" -v
```

### Running in Jenkins

In the Jenkins job, set environment variables from your credential store (e.g. bind `RHOAI_TOKEN` to a “Secret text” credential holding the service account token):
`RHOAI_URL`, `RHOAI_TOKEN`, `RHOAI_PROJECT_NAME`, S3 vars (`AWS_S3_ENDPOINT`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `RHOAI_TEST_DATA_BUCKET`).
Either set `RHOAI_KFP_URL` (existing pipeline server) or `RHOAI_CREATE_DSPA=true` (tests create the DSPA and the operator deploys the server).
No `oc` CLI or kubeconfig needed.

## Pipeline server: existing vs dynamic

You can run the integration tests in either of two ways:

- **Existing pipeline server:** If a pipeline server is already running in the cluster (e.g. from a Data Science Project), set `RHOAI_KFP_URL` to its API URL.
  The `rhoai_project` fixture only ensures the namespace and S3 secret exist; it does not start a server.
- **Dynamic server via DSPA:** Set `RHOAI_CREATE_DSPA=true` and the tests will create a **DataSciencePipelinesApplication** (DSPA) CR in the test namespace.
  The Data Science Pipelines Operator deploys the pipeline server; you do not need to set `RHOAI_KFP_URL` (the KFP client URL is resolved from the Route created by the operator).
  See the next section.

## Creating a DataSciencePipelinesApplication CR (DSPA)

You can have the test flow **create a DataSciencePipelinesApplication** custom resource (Red Hat OpenShift AI / Open Data Hub) in the test namespace using the Kubernetes Python client (`CustomObjectsApi`).
The Data Science Pipelines Operator (DSPO) will reconcile the CR and deploy the pipeline server in that namespace.

1. **Prerequisites:** The Data Science Pipelines Operator (or Open Data Hub operator) must be installed and the `DataSciencePipelinesApplication` CRD must exist on the cluster.
2. **Enable creation:** Set `RHOAI_CREATE_DSPA=true` (or `1`) in your environment or `.env`.
3. **Fixture:** The `datascience_pipelines_application` fixture is used by the integration tests (via the `kfp_client` dependency).
   It runs after `rhoai_project` and creates one CR named `automl-test-dspa` in the same namespace when `RHOAI_CREATE_DSPA=true`.
   It yields the created CR dict (or `None` if creation is disabled or fails).
4. **CRD identity:** Defaults are API group `datasciencepipelinesapplications.opendatahub.io`, version `v1`, plural `datasciencepipelinesapplications`.
   Override with `RHOAI_DSPA_API_GROUP`, `RHOAI_DSPA_API_VERSION`, `RHOAI_DSPA_PLURAL` if your cluster uses a different CRD.
5. **Spec:** The created CR uses a minimal `spec: {}`; the operator applies defaults. To customize (e.g. external object storage), extend the `body` in `_create_datascience_pipelines_application()` in `conftest.py` or load a spec from env/file.
6. **KFP client URL:** When `RHOAI_CREATE_DSPA=true`, the **KFP client is configured from the OpenShift Route** created by the operator.
   The test flow lists `route.openshift.io/v1` Route resources in the DSPA namespace, picks the one whose name starts with `RHOAI_DSPA_ROUTE_NAME_PREFIX` (default: `ds-pipeline`), and uses `https://<route.spec.host>` as the API URL.
   It retries for up to `RHOAI_DSPA_ROUTE_WAIT_TIMEOUT` seconds (default: 300).
   You do not need to set `RHOAI_KFP_URL` when using DSPA creation unless the route cannot be resolved (e.g. different route name); then set `RHOAI_KFP_URL` as fallback or set `RHOAI_DSPA_ROUTE_NAME_PREFIX` to match your route.

The integration test already requests `datascience_pipelines_application` indirectly (via `kfp_client`). When `RHOAI_CREATE_DSPA=true`, the DSPA CR is created before the pipeline runs; when using an existing server, omit `RHOAI_CREATE_DSPA` and set `RHOAI_KFP_URL` instead.

After the CR is created, the operator creates an OpenShift Route for the pipeline API. The `kfp_client` fixture waits up to `RHOAI_DSPA_ROUTE_WAIT_TIMEOUT` seconds (default 300) for that route and then configures the client with its URL.
