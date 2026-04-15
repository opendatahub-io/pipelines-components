# Autogluon Timeseries Training Pipeline ✨

> ⚠️ **Stability: beta** — This asset is not yet stable and may change.

## Overview 🧾

AutoGluon time series training pipeline.

Trains AutoGluon TimeSeries models on data loaded from S3, scores candidates on a per-series temporal holdout, refits the top models on the full train portion (selection + extra splits), and aggregates metrics into a leaderboard.

Storage strategy:

Train and test CSV splits are produced on the PVC workspace (``PipelineConfig.workspace``) so steps can read shared paths without re-downloading. The per-series test split is also exposed as a dataset artifact. S3 credentials for the initial load are supplied via the Kubernetes secret
``train_data_secret_name``.

Pipeline stages:

1. **Data loading & splitting** (``timeseries_data_loader``): Loads CSV from S3 (up to 1GB), then applies a two-stage **per-series temporal** split on ``id_column`` / ``timestamp_column``: default **80/20** train vs test per series, then **30/70** of each series' train rows into
``models_selection_train_dataset.csv`` and ``extra_train_dataset.csv`` under ``{workspace_path}/datasets/``. The test split is written to the ``sampled_test_dataset`` artifact.

2. **Model selection** (``autogluon_timeseries_models_selection``): Trains multiple AutoGluon TimeSeries models on the selection split, scores them on the test split, and emits the top ``top_n`` models plus predictor path and configuration.

3. **Full refit** (``autogluon_timeseries_models_full_refit``, ``ParallelFor`` with parallelism 1): For each top model, fits a new predictor on **selection + extra** train data (full train portion per series), evaluates on the test split, and writes a ``_FULL`` model artifact (predictor, metrics,
notebook). Parallelism is **one** concurrent refit pod because the workspace PVC is **ReadWriteOnce**; higher parallelism would mount the same RWO volume from multiple pods and cause **Multi-Attach** errors on typical block storage.

4. **Leaderboard** (``leaderboard_evaluation``): Builds an HTML leaderboard from the refitted model metrics using the selection stage's evaluation metric.

Args: train_data_secret_name: Kubernetes secret name containing S3 credentials (e.g. AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT, AWS_DEFAULT_REGION). train_data_bucket_name: S3-compatible bucket name containing the time series data file. train_data_file_key: S3 object key of the data
file (CSV or Parquet). File must include columns for item_id, timestamp, and target; optional columns for known covariates. target: Name of the column containing the numeric values to forecast. Corresponds to :attr:`~autogluon.timeseries.TimeSeriesDataFrame` target column. id_column: Name of the
column that identifies each time series (e.g. product_id, store_id). Passed as ``id_column`` when constructing TimeSeriesDataFrame; result uses ``item_id``. timestamp_column: Name of the column containing the timestamp/datetime for each observation. Passed as ``timestamp_column`` when constructing
TimeSeriesDataFrame; result uses ``timestamp`` as the second index level. known_covariates_names: Optional list of column names known in advance for the forecast horizon (e.g. holidays, promotions). See :attr:`~autogluon.timeseries.TimeSeriesPredictor.known_covariates_names`. prediction_length:
Number of time steps to forecast (horizon length). Positive integer (default: 1). top_n: Number of top models to select for the leaderboard and output (default: 3).

Returns: This pipeline wires task outputs between components; compiled runs expose artifacts from the refit tasks (Model artifacts with predictor, metrics, notebook paths) and the leaderboard evaluation task (HTML leaderboard and aggregated metrics), subject to Kubeflow Pipelines UI and artifact
configuration.

Raises: Component and runtime failures propagate from the underlying steps (for example: S3 access or empty data from the loader, invalid inputs, AutoGluon training or evaluation errors, or resource limits in the cluster).

Example: pipeline = autogluon_timeseries_training_pipeline( train_data_secret_name="my-s3-secret", train_data_bucket_name="my-bucket", train_data_file_key="ts/sales.csv", target="sales", id_column="product_id", timestamp_column="date", known_covariates_names=["is_holiday", "promo"],
prediction_length=14, top_n=3, )

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `train_data_secret_name` | `str` | `None` |  |
| `train_data_bucket_name` | `str` | `None` |  |
| `train_data_file_key` | `str` | `None` |  |
| `target` | `str` | `None` |  |
| `id_column` | `str` | `None` |  |
| `timestamp_column` | `str` | `None` |  |
| `known_covariates_names` | `Optional[List[str]]` | `None` |  |
| `prediction_length` | `int` | `1` |  |
| `top_n` | `int` | `3` |  |

## Metadata 🗂️

- **Name**: autogluon_timeseries_training_pipeline
- **Stability**: beta
- **Managed**: Yes
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
    - Name: Kubernetes, Version: >=1.28.0
- **Tags**:
  - training
  - pipeline
  - automl
  - autogluon-timeseries-training-pipeline
- **Last Verified**: 2026-03-25 15:00:00+00:00
- **Owners**:
  - Approvers:
    - DorotaDR
    - Mateusz-Switala
  - Reviewers:
    - DorotaDR

<!-- custom-content -->
### Files stored in user storage

Pipeline outputs are written to the artifact store (S3-compatible storage configured for Kubeflow Pipelines). The layout below matches what components write and what downstream consumers expect when loading the leaderboard or a refitted model.

```text
<pipeline_name>/
└── <run_id>/
    ├── timeseries-leaderboard-evaluation/
    │   └── <task_id>/
    │       └── html_artifact                     # HTML leaderboard (model names + metrics)
    ├── autogluon-timeseries-models-full-refit/
    │   ├── <task_id>/
    │   │   └── model_artifact/
    │   │       └── <ModelName>_FULL/            # e.g. ETS_FULL (one per top-N model)
    │   │           ├── model.json               # Model metadata (name, base_model, location, metrics)
    │   │           ├── predictor/               # AutoGluon TimeSeriesPredictor files
    │   │           │   ├── predictor.pkl
    │   │           │   └── predictor_metadata.json
    │   │           ├── metrics/
    │   │           │   └── metrics.json         # Evaluation metrics on test data
    │   │           └── notebooks/
    │   │               └── automl_predictor_notebook.ipynb   # Jupyter notebook for inference
    │   └── <task_id>/
    │           └── model_artifact/
    │               └── <ModelName>_FULL/ 
    │                   └── ...
    └── timeseries-data-loader/
        └── <task_id>/
            └── sampled_test_dataset/            # Test split (S3 artifact)
```

- **timeseries-leaderboard-evaluation**: Contains the HTML leaderboard artifact summarizing all refitted model results.
- **autogluon-timeseries-models-full-refit**: Each top-N model is refitted in turn (parallelism 1 on the shared RWO workspace). Each task writes a `<ModelName>_FULL/` subdirectory containing the saved TimeSeriesPredictor, metrics, pre-filled inference notebook, and a `model.json` with model metadata.
- **timeseries-data-loader**: Stores the test dataset S3 artifact used for evaluation; the training splits live on the PVC workspace instead.
