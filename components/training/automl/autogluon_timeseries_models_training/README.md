# Autogluon Timeseries Models Training ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Train, select, and full-refit top N AutoGluon timeseries models.

This component combines model generation and full-refit into one step to avoid pipeline-level ``ParallelFor``. It trains multiple AutoGluon TimeSeries models on the selection training data, ranks them on the test set, then sequentially refits the top N models on full train data (selection split +
extra split).

Refit outputs for all selected models are written under one ``models_artifact`` in a layout compatible with ``autogluon_leaderboard_evaluation``.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `target` | `str` | `None` | Name of the target column to forecast. |
| `id_column` | `str` | `None` | Name of the column identifying each time series (item_id). |
| `timestamp_column` | `str` | `None` | Name of the timestamp/datetime column. |
| `train_data_path` | `str` | `None` | Path to the selection training CSV file. |
| `test_data` | `dsl.Input[dsl.Dataset]` | `None` | Test dataset artifact for evaluation. |
| `top_n` | `int` | `None` | Number of top models to select for full refit. |
| `workspace_path` | `str` | `None` | Workspace directory where predictor will be saved. |
| `pipeline_name` | `str` | `None` | Pipeline name used in generated notebook placeholders. |
| `run_id` | `str` | `None` | Pipeline run id used in generated notebook placeholders. |
| `models_artifact` | `dsl.Output[dsl.Model]` | `None` | Combined output artifact containing all refitted models. |
| `notebooks` | `dsl.EmbeddedInput[dsl.Dataset]` | `None` | Embedded notebook templates. |
| `sample_rows` | `str` | `[]` | Optional sample rows JSON string used in generated notebook placeholders. |
| `sampling_config` | `Optional[dict]` | `None` | Optional sampling config stored in artifact metadata. |
| `split_config` | `Optional[dict]` | `None` | Optional split config stored in artifact metadata. |
| `extra_train_data_path` | `str` | `""` | Path to extra train split for full refit. |
| `prediction_length` | `int` | `1` | Forecast horizon (number of timesteps). |
| `known_covariates_names` | `Optional[List[str]]` | `None` | Optional list of known covariate column names. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `NamedTuple('outputs', top_models=List[str], predictor_path=str, eval_metric=str, model_config=dict)` | top_models list, predictor_path, eval_metric, model_config. |

## Usage Examples 🧪

```python
"""Example pipelines demonstrating usage of autogluon_timeseries_models_training."""

from kfp import dsl
from kfp_components.components.training.automl.autogluon_timeseries_models_training import (
    autogluon_timeseries_models_training,
)


@dsl.pipeline(name="autogluon-timeseries-models-training-example")
def example_pipeline(
    target: str = "value",
    id_column: str = "item_id",
    timestamp_column: str = "timestamp",
    train_data_path: str = "/tmp/train_data",
    top_n: int = 3,
    workspace_path: str = "/tmp/workspace",
    prediction_length: int = 1,
    pipeline_name: str = "autogluon-timeseries-models-training-example",
    run_id: str = "run-001",
):
    """Example pipeline using autogluon_timeseries_models_training.

    Args:
        target: Name of the target column.
        id_column: Name of the ID column.
        timestamp_column: Name of the timestamp column.
        train_data_path: Path to the training data.
        top_n: Number of top models to select.
        workspace_path: Path to the workspace directory.
        prediction_length: Number of time steps to predict.
        pipeline_name: Pipeline name used in generated notebook placeholders.
        run_id: Run id used in generated notebook placeholders.
    """
    test_data = dsl.importer(
        artifact_uri="gs://placeholder/test_data",
        artifact_class=dsl.Dataset,
    )
    autogluon_timeseries_models_training(
        target=target,
        id_column=id_column,
        timestamp_column=timestamp_column,
        train_data_path=train_data_path,
        test_data=test_data.output,
        top_n=top_n,
        workspace_path=workspace_path,
        pipeline_name=pipeline_name,
        run_id=run_id,
        prediction_length=prediction_length,
    )

```

## Metadata 🗂️

- **Name**: autogluon_timeseries_models_training
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
    - DorotaDR
  - Reviewers:
    - Mateusz-Switala
    - DorotaDR
