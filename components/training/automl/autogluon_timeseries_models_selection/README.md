# Autogluon Timeseries Models Selection ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Train and select top N AutoGluon timeseries models based on leaderboard.

This component trains multiple AutoGluon TimeSeries models using TimeSeriesPredictor on the selection training data, evaluates them on the test set, and selects the top N performers based on the leaderboard ranking. Training uses the ``fast_training`` preset for shorter wall-clock time versus
``medium_quality`` (trade-off: accuracy).

The TimeSeriesPredictor automatically trains various model types (DeepAR, TFT, ARIMA, ETS, Theta, etc.) and ranks them by the evaluation metric. This component selects the top N models from the leaderboard for refitting on the full dataset.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `target` | `str` | `None` | Name of the target column to forecast. |
| `id_column` | `str` | `None` | Name of the column identifying each time series (item_id). |
| `timestamp_column` | `str` | `None` | Name of the timestamp/datetime column. |
| `train_data_path` | `str` | `None` | Path to the selection training CSV file. |
| `test_data` | `dsl.Input[dsl.Dataset]` | `None` | Test dataset artifact for evaluation. |
| `top_n` | `int` | `None` | Number of top models to select for refitting. |
| `workspace_path` | `str` | `None` | Workspace directory where predictor will be saved. |
| `prediction_length` | `int` | `1` | Forecast horizon (number of timesteps). |
| `known_covariates_names` | `Optional[List[str]]` | `None` | Optional list of known covariate column names. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `NamedTuple('outputs', top_models=List[str], predictor_path=str, eval_metric_name=str, model_config=dict)` | top_models list, predictor_path, eval_metric_name, model_config. |

## Usage Examples 🧪

```python
"""Example pipelines demonstrating usage of autogluon_timeseries_models_selection."""

from kfp import dsl
from kfp_components.components.training.automl.autogluon_timeseries_models_selection import (
    autogluon_timeseries_models_selection,
)


@dsl.pipeline(name="autogluon-timeseries-models-selection-example")
def example_pipeline(
    target: str = "value",
    id_column: str = "item_id",
    timestamp_column: str = "timestamp",
    train_data_path: str = "/tmp/train_data",
    top_n: int = 3,
    workspace_path: str = "/tmp/workspace",
    prediction_length: int = 1,
):
    """Example pipeline using autogluon_timeseries_models_selection.

    Args:
        target: Name of the target column.
        id_column: Name of the ID column.
        timestamp_column: Name of the timestamp column.
        train_data_path: Path to the training data.
        top_n: Number of top models to select.
        workspace_path: Path to the workspace directory.
        prediction_length: Number of time steps to predict.
    """
    test_data = dsl.importer(
        artifact_uri="gs://placeholder/test_data",
        artifact_class=dsl.Dataset,
    )
    autogluon_timeseries_models_selection(
        target=target,
        id_column=id_column,
        timestamp_column=timestamp_column,
        train_data_path=train_data_path,
        test_data=test_data.output,
        top_n=top_n,
        workspace_path=workspace_path,
        prediction_length=prediction_length,
    )

```

## Metadata 🗂️

- **Name**: autogluon_timeseries_models_selection
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
- **Tags**:
  - training
  - timeseries
  - automl
  - model-selection
- **Last Verified**: 2026-04-10 12:00:00+00:00
- **Owners**:
  - Approvers:
    - LukaszCmielowski
  - Reviewers:
    - Mateusz-Switala
    - DorotaDR
