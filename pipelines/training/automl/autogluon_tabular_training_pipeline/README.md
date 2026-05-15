# Autogluon Tabular Training Pipeline ✨

> ⚠️ **Stability: beta** — This asset is not yet stable and may change.

## Overview 🧾

AutoGluon Tabular Training Pipeline.

This pipeline implements an efficient two-stage training approach for AutoGluon tabular models that balances computational cost with model quality. The pipeline automates the complete machine learning workflow from data loading to final model evaluation.

**Storage strategy:**

Training datasets are stored on a PVC workspace (not S3 artifacts) so that all pipeline steps sharing the workspace can access them without extra downloads. Only the test dataset is written to an S3 artifact (for use by the leaderboard evaluation component). The workspace is provisioned via
``PipelineConfig.workspace``.

**Pipeline Stages:**

1. **Data Loading & Splitting**: Loads tabular (CSV) data from an S3-compatible object storage bucket using AWS credentials configured via Kubernetes secrets. The component samples the data (up to 1GB), then performs a two-stage split: *Primary split** (default 80/20): separates a *test set* (20%,
written to an S3 artifact) from the *train portion* (80%). **Secondary split** (default 30/70 of the train portion): produces ``models_selection_train_dataset.csv`` (30%, used for model selection) and ``extra_train_dataset.csv`` (70%, passed to ``refit_full`` as extra data). Both train CSVs are
written to the PVC workspace under ``{workspace_path}/datasets/``. For classification tasks the splits are stratified by the label column.

2. **Model Training & Refitting**: Trains multiple AutoGluon models on the *selection train* data using stacking (1 level) and bagging (4 folds). All models are evaluated on the test set and ranked by performance. The top N models are selected and refitted sequentially on the full training data via
``refit_full``. Each refitted model is saved with a ``_FULL`` suffix and optimized for deployment. All model artifacts are stored under a single output artifact, avoiding a ``ParallelFor`` loop in the pipeline.

3. **Leaderboard Evaluation**: Reads pre-computed metrics from the combined models artifact and generates an HTML-formatted leaderboard ranking models by their performance metrics for comparison and selection.

**Two-Stage Training Benefits:**

- **Efficient Exploration:** Initial model training uses a smaller selection-train split with efficient ensembling rather than expensive hyperparameter optimization.

- **Optimal Performance:** Final models are refitted (``refit_full``) on the predictor's training and validation data plus the extra-train split, maximizing the amount of data seen during the final fit.

- **Production-Ready:** Refitted models are AutoGluon Predictors optimized and ready for deployment.

**AutoGluon Ensembling Approach:**

The pipeline leverages AutoGluon's unique ensembling strategy that combines multiple model types using stacking and bagging rather than traditional hyperparameter optimization. This approach is more efficient and typically produces better results for tabular data by automatically:

- Training diverse model families

- Combining predictions using multi-level stacking

- Using bootstrap aggregation (bagging) for robustness

- Selecting optimal ensemble configurations

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `train_data_secret_name` | `str` | `None` | Kubernetes secret name with S3 credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT, AWS_DEFAULT_REGION). |
| `train_data_bucket_name` | `str` | `None` | S3-compatible bucket name containing the tabular data file. |
| `train_data_file_key` | `str` | `None` | S3 object key of the CSV file (features and target column). |
| `label_column` | `str` | `None` | Name of the target/label column in the dataset. |
| `task_type` | `str` | `None` | "binary", "multiclass", or "regression"; drives metrics and model types. |
| `top_n` | `int` | `3` | Number of top models to select and refit (default: 3); positive integer from range [1, 10]. |

## Metadata 🗂️

- **Name**: autogluon_tabular_training_pipeline
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
  - autogluon-tabular-training-pipeline
- **Last Verified**: 2026-05-07 12:00:00+00:00
- **Owners**:
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
  - Reviewers:
    - Mateusz-Switala
    - DorotaDR

<!-- custom-content -->
### Files stored in user storage

Pipeline outputs are written to the artifact store (S3-compatible storage configured for Kubeflow Pipelines). The layout below matches what components write and what downstream consumers expect when loading the leaderboard or a refitted model.

```text
<pipeline_name>/
└── <run_id>/
    ├── leaderboard-evaluation/
    │   └── <task_id>/
    │       └── html_artifact                     # HTML leaderboard (model names + metrics)
    ├── autogluon-models-training/
    │   └── <task_id>/
    │       └── models_artifact/
    │           ├── <ModelName>_FULL/            # e.g. LightGBM_BAG_L1_FULL (one per top-N model)
    │           │   ├── model.json               # Model metadata (name, location, metrics)
    │           │   ├── predictor/               # AutoGluon TabularPredictor files
    │           │   ├── metrics/
    │           │   │   ├── metrics.json         # model evaluation metrics (eval_metric, etc.)
    │           │   │   ├── feature_importance.json
    │           │   │   └── confusion_matrix.json  # for classification tasks only
    │           │   └── notebooks/
    │           │       └── automl_predictor_notebook.ipynb   # Jupyter notebook for inference & exploration
    │           └── <ModelName>_FULL/
    │               └──  ...
    └── automl-data-loader/
        └── <task_id>/
            └── sampled_test_dataset/            # Test split (S3 artifact)
```

- **leaderboard-evaluation**: Contains the HTML leaderboard artifact summarizing all model results.
- **autogluon-models-training**: All top-N refitted model artifacts are written under a single task, each under its own `<ModelName>_FULL/` subdirectory,
including the saved TabularPredictor, metrics, pre-filled inference notebook, and a `model.json` with model metadata (name, location paths, evaluation metrics).
- **automl-data-loader**: Stores the test dataset S3 artifact used for evaluation; the training splits live on the PVC workspace instead.

For loading:

- Load a refitted model for deployment or notebook exploration using `TabularPredictor.load(<.../models_artifact/<ModelName>_FULL/predictor>)`.
- Model metrics and feature importances are always at `metrics/` under each model directory.
- The leaderboard HTML is at `leaderboard-evaluation/<task_id>/html_artifact`.

### Model Artifact metadata

The `autogluon-models-training` component writes a single Model artifact (`models_artifact`) covering all top-N refitted models. Downstream components (e.g. leaderboard evaluation) and consumers can rely on this structure:

| Key | Type | Description |
| ----- | ------ | ----------- |
| `model_names` | `str` (JSON) | JSON-encoded list of refitted model names with `_FULL` suffix, e.g. `'["LightGBM_BAG_L1_FULL", "CatBoost_BAG_L1_FULL"]'`. |
| `context` | `dict` | Run and model context (see below). |

**`context`** contains:

| Key | Type | Description |
| ----- | ------ | ----------- |
| `data_config` | `dict` | `sampling_config` and `split_config` from the data loader component. |
| `task_type` | `str` | Problem type: `"regression"`, `"binary"`, or `"multiclass"`. |
| `label_column` | `str` | Name of the target/label column. |
| `model_config` | `dict` | Training configuration: `preset`, `eval_metric`, `time_limit`. |
| `models` | `list[dict]` | Per-model location and metrics (one entry per top-N model). |

Each entry in **`context.models`** contains:

| Key | Type | Description |
| ----- | ------ | ----------- |
| `name` | `str` | Model name with `_FULL` suffix (e.g. `"LightGBM_BAG_L1_FULL"`). |
| `location` | `dict` | Paths relative to `models_artifact.path`: `model_directory`, `predictor`, `notebook`. |
| `metrics` | `dict` | `test_data` — evaluation results dict from `evaluate_predictions` (metric names → values). |

Example (regression, top_n=2):

```json
{
  "model_names": "[\"LightGBM_BAG_L1_FULL\", \"CatBoost_BAG_L1_FULL\"]",
  "context": {
    "data_config": {
      "sampling_config": {"n_samples": 10000},
      "split_config": {"test_size": 0.2, "random_state": 42}
    },
    "task_type": "regression",
    "label_column": "price",
    "model_config": {"preset": "medium_quality", "eval_metric": "r2", "time_limit": 1800},
    "models": [
      {
        "name": "LightGBM_BAG_L1_FULL",
        "location": {
          "model_directory": "LightGBM_BAG_L1_FULL",
          "predictor": "LightGBM_BAG_L1_FULL/predictor",
          "notebook": "LightGBM_BAG_L1_FULL/notebooks/automl_predictor_notebook.ipynb"
        },
        "metrics": {
          "test_data": {"root_mean_squared_error": 0.42, "r2": 0.85}
        }
      },
      {
        "name": "CatBoost_BAG_L1_FULL",
        "location": {
          "model_directory": "CatBoost_BAG_L1_FULL",
          "predictor": "CatBoost_BAG_L1_FULL/predictor",
          "notebook": "CatBoost_BAG_L1_FULL/notebooks/automl_predictor_notebook.ipynb"
        },
        "metrics": {
          "test_data": {"root_mean_squared_error": 0.51, "r2": 0.80}
        }
      }
    ]
  }
}
```

## Usage Examples 💡

### Basic usage (regression)

Run the full two-stage pipeline with data from S3; credentials are provided via a Kubernetes secret:

```python
from kfp import dsl
from kfp_components.pipelines.training.automl.autogluon_tabular_training_pipeline import (
    autogluon_tabular_training_pipeline,
)

# Compile and run the pipeline
pipeline = autogluon_tabular_training_pipeline(
    train_data_secret_name="my-s3-secret",
    train_data_bucket_name="my-data-bucket",
    train_data_file_key="datasets/housing_prices.csv",
    label_column="price",
    task_type="regression",
    top_n=3,
)
```

### Classification (binary or multiclass)

```python
pipeline = autogluon_tabular_training_pipeline(
    train_data_secret_name="my-s3-secret",
    train_data_bucket_name="my-ml-bucket",
    train_data_file_key="data/train.csv",
    label_column="target",
    task_type="multiclass",
    top_n=5,
)
```

### Compile to YAML

```python
from kfp.compiler import Compiler
from kfp_components.pipelines.training.automl.autogluon_tabular_training_pipeline import (
    autogluon_tabular_training_pipeline,
)

Compiler().compile(
    autogluon_tabular_training_pipeline,
    package_path="autogluon_tabular_training_pipeline.yaml",
)
```

### Run pipeline using KFP SDK

Compile and submit a run using the KFP client. Configure the client for your cluster (e.g. `host`, or in-cluster auth). Pipeline parameters are passed as `arguments`:

```python
import kfp
from kfp_components.pipelines.training.automl.autogluon_tabular_training_pipeline import (
    autogluon_tabular_training_pipeline,
)

# Create client (customize host for your KFP instance)
client = kfp.Client(host="https://your-kfp-host/pipeline")

# Run the pipeline with parameters
run = client.create_run_from_pipeline_func(
    autogluon_tabular_training_pipeline,
    arguments={
        "train_data_secret_name": "my-s3-secret",
        "train_data_bucket_name": "my-data-bucket",
        "train_data_file_key": "datasets/housing_prices.csv",
        "label_column": "price",
        "task_type": "regression",
        "top_n": 3,
    },
)
print(f"Submitted run: {run.run_id}")
```

To run from a compiled YAML instead:

```python
from kfp.compiler import Compiler

Compiler().compile(
    autogluon_tabular_training_pipeline,
    package_path="autogluon_tabular_training_pipeline.yaml",
)
run = client.create_run_from_pipeline_package(
    "autogluon_tabular_training_pipeline.yaml",
    arguments={
        "train_data_secret_name": "my-s3-secret",
        "train_data_bucket_name": "my-data-bucket",
        "train_data_file_key": "datasets/housing_prices.csv",
        "label_column": "price",
        "task_type": "regression",
        "top_n": 3,
    },
)
```
