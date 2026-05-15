# Autogluon Timeseries Models Full Refit ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Refit a single AutoGluon timeseries model on full training data.

This component takes a model selected during the selection phase and refits it on the full training dataset (selection + extra train data) for improved performance. The refitted model is optimized and saved for deployment. Each model directory contains a ``model.json`` file with model metadata
(name, location, metrics).

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `model_name` | `str` | `None` | Name of the model to refit. |
| `test_dataset` | `dsl.Input[dsl.Dataset]` | `None` | Test dataset artifact for evaluation. |
| `predictor_path` | `str` | `None` | Path to the predictor from selection phase. |
| `sampling_config` | `dict` | `None` | Configuration used for data sampling. |
| `split_config` | `dict` | `None` | Configuration used for data splitting. |
| `model_config` | `dict` | `None` | Model configuration from selection phase. |
| `pipeline_name` | `str` | `None` | Pipeline name for metadata. |
| `run_id` | `str` | `None` | Pipeline run ID for metadata. |
| `models_selection_train_data_path` | `str` | `None` | Path to the model-selection train split CSV (earlier segment of the train portion). |
| `extra_train_data_path` | `str` | `None` | Path to the extra train split CSV (later segment of the train portion). |
| `sample_rows` | `str` | `None` | Sample rows from test dataset as JSON string. |
| `notebooks` | `dsl.EmbeddedInput[dsl.Dataset]` | `None` | Embedded notebook templates (injected by the runtime from the component's embedded_artifact_path). |
| `model_artifact` | `dsl.Output[dsl.Model]` | `None` | Output artifact for the refitted model. |

## Usage Examples 🧪

```python
"""Example pipelines demonstrating usage of autogluon_timeseries_models_full_refit."""

from kfp import dsl
from kfp_components.components.training.automl.autogluon_timeseries_models_full_refit import (
    autogluon_timeseries_models_full_refit,
)


@dsl.pipeline(name="autogluon-timeseries-models-full-refit-example")
def example_pipeline(
    model_name: str = "WeightedEnsemble",
    predictor_path: str = "/tmp/predictor",
    sampling_config: dict = {},
    split_config: dict = {},
    model_config: dict = {},
    pipeline_name: str = "timeseries-pipeline",
    run_id: str = "run-001",
    models_selection_train_data_path: str = "/tmp/train_data",
    extra_train_data_path: str = "",
    sample_rows: str = '{"item_id": "A", "timestamp": "2024-01-01", "value": 1.0}',
):
    """Example pipeline using autogluon_timeseries_models_full_refit.

    Args:
        model_name: Name of the model to refit.
        predictor_path: Path to the predictor.
        sampling_config: Sampling configuration.
        split_config: Data split configuration.
        model_config: Model configuration.
        pipeline_name: Name of the pipeline.
        run_id: Unique run identifier.
        models_selection_train_data_path: Path to training data from model selection.
        extra_train_data_path: Path to extra training data.
        sample_rows: JSON string of sample rows.
    """
    test_dataset = dsl.importer(
        artifact_uri="gs://placeholder/test_dataset",
        artifact_class=dsl.Dataset,
    )
    autogluon_timeseries_models_full_refit(
        model_name=model_name,
        test_dataset=test_dataset.output,
        predictor_path=predictor_path,
        sampling_config=sampling_config,
        split_config=split_config,
        model_config=model_config,
        pipeline_name=pipeline_name,
        run_id=run_id,
        models_selection_train_data_path=models_selection_train_data_path,
        extra_train_data_path=extra_train_data_path,
        sample_rows=sample_rows,
    )

```

## Metadata 🗂️

- **Name**: autogluon_timeseries_models_full_refit
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
- **Tags**:
  - training
  - timeseries
  - automl
  - model-refit
- **Last Verified**: 2026-04-21 12:00:00+00:00
- **Owners**:
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
  - Reviewers:
    - Mateusz-Switala
    - DorotaDR

<!-- custom-content -->
## Artifact Output Structure 📂

The refitted model artifact is written under `model_artifact.path` with the following layout:

```text
model_artifact/
└── <ModelName>_FULL/              # e.g. ETS_FULL
    ├── model.json                 # Model metadata (name, location, metrics)
    ├── predictor/                 # AutoGluon TimeSeriesPredictor files
    │   ├── predictor.pkl
    │   └── predictor_metadata.json
    ├── metrics/
    │   └── metrics.json           # Evaluation results on test data (all available metrics)
    └── notebooks/
        └── automl_predictor_notebook.ipynb  # Pre-filled inference notebook
```

### `model.json`

Each model directory contains a `model.json` file with the model's metadata:

```json
{
  "name": "ETS_FULL",
  "location": {
    "model_directory": "ETS_FULL",
    "predictor": "ETS_FULL/predictor",
    "notebook": "ETS_FULL/notebooks/automl_predictor_notebook.ipynb",
    "metrics": "ETS_FULL/metrics"
  },
  "metrics": {
    "test_data": {"MASE": -0.85, "WAPE": -0.12, "RMSE": -150.3}
  }
}
```

This file allows downstream consumers (e.g. the timeseries leaderboard evaluation component) to read model metadata directly from the filesystem.
