# Autogluon Timeseries Training Pipeline ✨

> ⚠️ **Stability: beta** — This asset is not yet stable and may change.

## Overview 🧾

AutoGluon time series training pipeline.

Trains AutoGluon TimeSeries models on data loaded from S3, scores candidates on a per-series temporal holdout, refits the top models on the full train portion (selection + extra splits), and aggregates metrics into a leaderboard.

**Compiled pipeline encoding:** Keep this module ASCII-only (no Unicode in docstrings or string literals). Some deployments persist compiled pipeline YAML in MySQL ``utf8`` columns, which reject multi-byte characters.

Storage strategy:

Train and test CSV splits are produced on the PVC workspace (``PipelineConfig.workspace``) so steps can read shared paths without re-downloading. The per-series test split is also exposed as a dataset artifact. S3 credentials for the initial load are supplied via the Kubernetes secret
``train_data_secret_name``.

Pipeline stages:

0. **Component stage map**: Publishes the static component-to-stage-to-step map as a KFP artifact for dashboards before data loading.

1. **Data loading & splitting** (``timeseries_data_loader``): Loads CSV from S3 (up to 100 MB), replaces ``+/-inf`` with NaN (missing targets stay for AutoGluon), requires parseable timestamps and non-null ids, deduplicates ``(id_column, timestamp_column)``, then applies a two-stage **per-series
temporal** split on ``id_column`` / ``timestamp_column``: default **80/20** train vs test per series, then **30/70** of each series' train rows into ``models_selection_train_dataset.csv`` and ``extra_train_dataset.csv`` under ``{workspace_path}/datasets/``. The test split is written to the
``sampled_test_dataset`` artifact.

2. **Model generation + full refit** (``autogluon_timeseries_models_training``): Trains multiple AutoGluon TimeSeries models on the selection split, picks top ``top_n``, and refits each selected model on the full train portion (**selection + extra** splits). The component writes all refitted models
to a single combined ``models_artifact``.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `train_data_secret_name` | `str` | `None` | Kubernetes secret name containing S3 credentials (e.g. AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT, AWS_DEFAULT_REGION). |
| `train_data_bucket_name` | `str` | `None` | S3-compatible bucket name containing the time series data file. |
| `train_data_file_key` | `str` | `None` | S3 object key of the data file (CSV or Parquet). File must include columns for item_id, timestamp, and target; optional columns for known covariates. |
| `target` | `str` | `None` | Name of the column containing the numeric values to forecast. Corresponds to :attr:`~autogluon.timeseries.TimeSeriesDataFrame` target column. |
| `id_column` | `str` | `None` | Name of the column that identifies each time series (e.g. product_id, store_id). Passed as ``id_column`` when constructing TimeSeriesDataFrame; result uses ``item_id``. |
| `timestamp_column` | `str` | `None` | Name of the column containing the timestamp/datetime for each observation. Passed as ``timestamp_column`` when constructing TimeSeriesDataFrame; result uses ``timestamp`` as the second index level. |
| `known_covariates_names` | `List[str]` | `[]` | Column names known in advance for the forecast horizon (e.g. holidays, promotions). Defaults to ``[]`` (no known covariates). See :attr:`~autogluon.timeseries.TimeSeriesPredictor.known_covariates_names`. |
| `prediction_length` | `int` | `1` | Number of time steps to forecast (horizon length). Positive integer (default: 1). |
| `top_n` | `int` | `3` | Number of top models to select for the leaderboard and output (default: 3). |
| `eval_metric` | `str` | `mean_absolute_scaled_error` | Metric for model ranking in snake_case (e.g. ``"mean_absolute_scaled_error"``, ``"weighted_quantile_loss"``) or legacy uppercase acronym form. Defaults to ``"mean_absolute_scaled_error"``. |
| `preset` | `str` | `speed` | Training quality tier. ``"speed"`` (default, 4 vCPU / 16 GiB) or ``"balanced"`` (may run more than 2x longer, 8 vCPU / 32 GiB). |
| `test_data_bucket_name` | `str` | `""` | Optional S3-compatible bucket name containing user-provided test dataset. If provided, test_data_file_key must also be specified. The test data is read with the single ``train_data_secret_name`` credential set, so it must live on the same S3 endpoint and be readable with the same credentials as the training data. |
| `test_data_file_key` | `str` | `""` | Optional S3 object key of the test CSV file (features and target column). If provided, test_data_bucket_name must also be specified. The test CSV must carry the same id/timestamp/target and ``known_covariates_names`` columns as the training data, cover the same series, and give each series at least ``prediction_length`` rows; the data loader fails on a mismatch before training starts. |

## Metadata 🗂️

- **Name**: autogluon-timeseries-training-pipeline
- **Stability**: beta
- **Managed**: Yes
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: 2.16.1
    - Name: Kubernetes, Version: >=1.28.0
- **Tags**:
  - training
  - pipeline
  - automl
  - autogluon-timeseries-training-pipeline
- **Last Verified**: 2026-07-08 12:00:00+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - DorotaDR
    - Mateusz-Switala
  - Reviewers:
    - DorotaDR

<!-- custom-content -->

### Progress and dashboard artifacts

Besides model and data artifacts below, each run publishes:

| KFP task | Output | File | Purpose |
| -------- | ------ | ---- | ------- |
| `publish-component-stage-map` | `component_stage_map` | `component_stage_map.json` | Static component-to-stage-to-step catalog for the time series pipeline (published once at run start). |
| `timeseries-data-loader` | `component_status` | `component_status.json` | Stage progress for data loading and splitting. |
| `autogluon-timeseries-models-training` | `component_status` | `component_status.json` | Stage progress for training, refit, evaluation, and leaderboard generation. |

Example artifact-store layout:

```text
<pipeline_name>/<run_id>/
├── publish-component-stage-map/<task_id>/component_stage_map/component_stage_map.json
├── timeseries-data-loader/<task_id>/component_status/component_status.json
└── autogluon-timeseries-models-training/<task_id>/component_status/component_status.json
```

See [AutoML training components README](../../../components/training/automl/README.md) for JSON field details.

#### Dashboard join keys

Dashboards join the static map (`component_stage_map.json`) to live progress (`component_status.json`) using **snake_case component ids**, not KFP task names:

| Layer | Naming | Time series data loader example |
| ----- | ------ | ------------------------------- |
| Template `components[].id` | snake_case | `timeseries_data_loader` |
| Runtime `component_status.json` → `component_id` | snake_case | `timeseries_data_loader` |
| KFP root DAG task id (compiled YAML) | kebab-case | `timeseries-data-loader` |
| KFP output parameter | snake_case | `component_status` |
| Artifact file | snake_case | `component_status.json` |

Use `component_id` (and stage `id` fields inside each file) to correlate artifacts. KFP task names are only for locating artifact paths in the store.

Canonical component ids are defined in the pipeline JSON templates under
[`run_status_templates/pipelines/`](../../../components/training/automl/shared/run_status_templates/pipelines/)
(e.g. `autogluon-timeseries-training-pipeline.json`). Legacy workspace helpers in
[`run_status.py`](../../../components/training/automl/shared/run_status.py) expose the same ids as
`COMPONENT_*` constants.

### Files stored in user storage

Pipeline outputs are written to the artifact store (S3-compatible storage configured for Kubeflow Pipelines). The layout below matches what components write and what downstream consumers expect when loading the leaderboard or a refitted model.

```text
<pipeline_name>/
└── <run_id>/
    ├── autogluon-timeseries-models-training/
    │   └── <task_id>/
    │       ├── html_artifact                    # HTML leaderboard (model names + metrics)
    │       └── models_artifact/
    │           └── <ModelName>_FULL/            # e.g. ETS_FULL (one per top-N model)
    │               ├── model.json               # Model metadata (name, location, metrics)
    │               ├── predictor/               # AutoGluon TimeSeriesPredictor files
    │               ├── metrics/
    │               │   └── metrics.json         # Evaluation metrics on test data
    │               └── notebooks/
    │                   └── automl_predictor_notebook.ipynb   # Jupyter notebook for inference
    └── timeseries-data-loader/
        └── <task_id>/
            └── sampled_test_dataset/            # Test split (S3 artifact)
```

- **autogluon-timeseries-models-training**: Writes the HTML leaderboard (`html_artifact`) and a combined models artifact containing all `<ModelName>_FULL/` subdirectories (predictor, metrics, notebook, and `model.json` per model).
- **timeseries-data-loader**: Stores the test dataset S3 artifact used for evaluation; the training splits live on the PVC workspace instead.
