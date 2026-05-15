# Autogluon Models Training ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Train AutoGluon models, select the top N, and refit each on the full dataset.

Expects pre-cleaned CSV data from the tabular data loader (infinite values replaced, duplicates removed, missing labels dropped). Reads train/test/extra-train CSVs and validates that the label column exists in each dataset.

This component combines the model selection and full-refit stages into a single step. It trains a TabularPredictor on sampled data, ranks all models on the test set, then refits each of the top N models on the full training data in a single ``refit_full`` call. Post-refit work (predict, evaluate,
feature importance, confusion matrix, notebook generation) runs concurrently across all top-N models via ``ThreadPoolExecutor``. The deployment clone (``set_model_best`` + ``clone_for_deployment``) is serialized afterward because it mutates predictor state. All artifacts are written under a single
output artifact so the pipeline does not require a ParallelFor loop. Each model directory contains a ``model.json`` file with model metadata (name, location, metrics).

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `label_column` | `str` | `None` | Target/label column name in train and test datasets. |
| `task_type` | `str` | `None` | ML task type: ``"binary"``, ``"multiclass"``, or ``"regression"``. |
| `top_n` | `int` | `None` | Number of top models to select and refit (1-10). |
| `train_data_path` | `str` | `None` | Path to the selection-train CSV on the PVC workspace. |
| `test_data` | `dsl.Input[dsl.Dataset]` | `None` | Dataset artifact (CSV) used for leaderboard ranking and evaluation. |
| `workspace_path` | `str` | `None` | PVC workspace directory; predictor saved at ``workspace_path/autogluon_predictor``. |
| `pipeline_name` | `str` | `None` | Pipeline run name; last dash-segment stripped for the notebook. |
| `run_id` | `str` | `None` | Pipeline run ID written into the generated notebook. |
| `sample_row` | `str` | `None` | JSON array of row dicts for the notebook example input; label column is stripped. |
| `models_artifact` | `dsl.Output[dsl.Model]` | `None` | Output Model artifact containing all refitted model subdirectories. |
| `notebooks` | `dsl.EmbeddedInput[dsl.Dataset]` | `None` | Embedded notebook templates injected by the KFP runtime. |
| `sampling_config` | `Optional[dict]` | `None` | Data sampling config stored in artifact metadata. |
| `split_config` | `Optional[dict]` | `None` | Data split config stored in artifact metadata. |
| `extra_train_data_path` | `str` | `""` | Optional path to extra training CSV passed to ``refit_full``. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `NamedTuple('outputs', eval_metric=str)` | NamedTuple with ``eval_metric`` (the metric used for ranking, e.g. ``"r2"`` or ``"accuracy"``). |

## Metadata 🗂️

- **Name**: autogluon_models_training
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
- **Tags**:
  - training
  - automl
- **Last Verified**: 2026-04-21 12:00:00+00:00
- **Owners**:
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
  - Reviewers:
    - Mateusz-Switala
    - DorotaDR

<!-- custom-content -->
## Usage Examples 💡

This component is typically used inside a KFP pipeline. It depends on a PVC workspace (for predictor storage) and a test dataset artifact (for leaderboard evaluation).

### Regression

```python
from kfp import dsl
from kfp_components.components.training.automl.autogluon_models_training import autogluon_models_training


@dsl.pipeline
def my_pipeline():
    training_task = autogluon_models_training(
        label_column="price",
        task_type="regression",
        top_n=3,
        train_data_path=f"{dsl.WORKSPACE_PATH_PLACEHOLDER}/datasets/models_selection_train_dataset.csv",
        test_data=<test_dataset_artifact>,
        workspace_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        pipeline_name=dsl.PIPELINE_JOB_RESOURCE_NAME_PLACEHOLDER,
        run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
        sample_row='[{"bedrooms": 3, "sqft": 1200, "location": "urban"}]',
        extra_train_data_path=f"{dsl.WORKSPACE_PATH_PLACEHOLDER}/datasets/extra_train_dataset.csv",
    )
```

### Classification (binary or multiclass)

```python
training_task = autogluon_models_training(
    label_column="target",
    task_type="multiclass",
    top_n=5,
    train_data_path=f"{dsl.WORKSPACE_PATH_PLACEHOLDER}/datasets/models_selection_train_dataset.csv",
    test_data=<test_dataset_artifact>,
    workspace_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
    pipeline_name=dsl.PIPELINE_JOB_RESOURCE_NAME_PLACEHOLDER,
    run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
    sample_row='[{"feature_a": 1.0, "feature_b": "foo"}]',
)
```

### Loading a refitted model from the output artifact

After the pipeline run completes, load any refitted model from `models_artifact`:

```python
import json
from autogluon.tabular import TabularPredictor

# Read which models were produced
model_names = json.loads(models_artifact.metadata["model_names"])
# e.g. ["LightGBM_BAG_L1_FULL", "CatBoost_BAG_L1_FULL"]

# Load the first model
predictor = TabularPredictor.load(f"{models_artifact.path}/{model_names[0]}/predictor")
predictions = predictor.predict(new_data_df)
```

## Artifact Output Structure 📂

All refitted model artifacts are written under a single `models_artifact` output. The directory layout under `models_artifact.path` is:

```text
models_artifact/
└── <ModelName>_FULL/              # e.g. LightGBM_BAG_L1_FULL
    ├── model.json                 # Model metadata (name, location, metrics)
    ├── predictor/                 # AutoGluon TabularPredictor files (load via TabularPredictor.load())
    ├── metrics/
    │   ├── metrics.json           # Evaluation results on test data (metric names → values)
    │   ├── feature_importance.json
    │   └── confusion_matrix.json  # Classification tasks only
    └── notebooks/
        └── automl_predictor_notebook.ipynb  # Pre-filled inference notebook
```

One `<ModelName>_FULL/` directory is created for each of the top N selected models.

### `model.json`

Each model directory contains a `model.json` file with the model's metadata, matching the corresponding entry in `context.models`:

```json
{
  "name": "LightGBM_BAG_L1_FULL",
  "location": {
    "model_directory": "LightGBM_BAG_L1_FULL",
    "predictor": "LightGBM_BAG_L1_FULL/predictor",
    "notebook": "LightGBM_BAG_L1_FULL/notebooks/automl_predictor_notebook.ipynb",
    "metrics": "LightGBM_BAG_L1_FULL/metrics"
  },
  "metrics": {
    "test_data": {"root_mean_squared_error": 0.42, "r2": 0.85}
  }
}
```

This file allows downstream consumers to read model metadata directly from the filesystem without relying on artifact metadata propagation.

### `models_artifact` metadata fields

Downstream components (e.g. leaderboard evaluation) read the following fields from `models_artifact.metadata`:

| Key | Type | Description |
| --- | ---- | ----------- |
| `model_names` | `str` (JSON) | JSON-encoded list of refitted model names with `_FULL` suffix, e.g. `'["LightGBM_BAG_L1_FULL", "CatBoost_BAG_L1_FULL"]'`. |
| `context` | `dict` | Run and model context (see below). |

**`context`** contains:

| Key | Type | Description |
| --- | ---- | ----------- |
| `data_config` | `dict` | `sampling_config` and `split_config` from the data loader component. |
| `task_type` | `str` | Problem type: `"regression"`, `"binary"`, or `"multiclass"`. |
| `label_column` | `str` | Name of the target/label column. |
| `model_config` | `dict` | Training configuration: `preset`, `eval_metric`, `time_limit`. |
| `models` | `list[dict]` | Per-model location and metrics (see below). |

Each entry in **`context.models`** contains:

| Key | Type | Description |
| --- | ---- | ----------- |
| `name` | `str` | Model name with `_FULL` suffix (e.g. `"LightGBM_BAG_L1_FULL"`). |
| `location` | `dict` | Paths relative to `models_artifact.path`: `model_directory`, `predictor`, `notebook`, `metrics`. |
| `metrics` | `dict` | `test_data` — evaluation results dict from `evaluate_predictions` (metric names → values). |

Example:

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
          "notebook": "LightGBM_BAG_L1_FULL/notebooks/automl_predictor_notebook.ipynb",
          "metrics": "LightGBM_BAG_L1_FULL/metrics"
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
          "notebook": "CatBoost_BAG_L1_FULL/notebooks/automl_predictor_notebook.ipynb",
          "metrics": "CatBoost_BAG_L1_FULL/metrics"
        },
        "metrics": {
          "test_data": {"root_mean_squared_error": 0.51, "r2": 0.80}
        }
      }
    ]
  }
}
```
