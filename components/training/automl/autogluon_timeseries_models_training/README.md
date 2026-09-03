# Autogluon Timeseries Models Training ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Train, select, and full-refit top N AutoGluon timeseries models.

This component combines model generation and full-refit into one step to avoid pipeline-level ``ParallelFor``. It trains multiple AutoGluon TimeSeries models on the selection training data, ranks them on the test set, then sequentially refits the top N models on full train data (selection split +
extra split).

Refit outputs for all selected models are written under one ``models_artifact``, and a ranked HTML leaderboard is written to ``html_artifact``.

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
| `extra_train_data_path` | `str` | `None` | Path to extra train split for full refit. |
| `html_artifact` | `dsl.Output[dsl.HTML]` | `None` | Output HTML artifact containing the ranked leaderboard page. |
| `component_status` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing stage-level progress tracking for this component. |
| `uses_synthetic_id` | `bool` | `False` | True if the loader injected a synthetic ID column for two-column datasets. |
| `sample_rows` | `str` | `[]` | Sample rows JSON string used in generated notebook placeholders. |
| `sampling_config` | `Optional[dict]` | `None` | Optional sampling config stored in artifact metadata. |
| `split_config` | `Optional[dict]` | `None` | Optional split config stored in artifact metadata. |
| `prediction_length` | `int` | `1` | Forecast horizon (number of timesteps). |
| `known_covariates_names` | `Optional[List[str]]` | `None` | Optional list of known covariate column names. |
| `preset` | `str` | `speed` | Training quality tier. ``"speed"`` (default) or ``"balanced"`` (may run more than 2x longer). |
| `eval_metric` | `str` | `mean_absolute_scaled_error` | Metric for model ranking (e.g. ``"mean_absolute_scaled_error"``, ``"weighted_quantile_loss"``). Defaults to ``"mean_absolute_scaled_error"``. Legacy uppercase acronyms (e.g. ``"MASE"``) are accepted and normalized to snake_case. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `NamedTuple('outputs', top_models=List[str], predictor_path=str, eval_metric=str, model_config=dict, best_model_name=str)` | top_models list, predictor_path, eval_metric, model_config. |

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
    extra_train_data_path: str = "/tmp/extra_train_data",
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
        extra_train_data_path: Path to extra training data for full refit.
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
        extra_train_data_path=extra_train_data_path,
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
- **Last Verified**: 2026-07-08 12:00:00+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
    - Mateusz-Switala
  - Reviewers:
    - Mateusz-Switala
    - DorotaDR

<!-- custom-content -->

### Component status artifact

Writes ``component_status.json`` under ``component_status`` with ``component_id``
``autogluon_timeseries_models_training`` and training stages (``load_data``, ``model_selection`` with
steps ``feature_engineering``, ``model_training``, ``stacking``, ``evaluation``, ``refit_and_evaluate``).
Artifact metadata display name: **Timeseries Models Training Status**.

Inference notebooks are loaded from ``shared/notebook_templates/timeseries_notebook.ipynb`` at runtime (same shared package data as tabular training).

### `model.json` inference block (KServe AutoGluon time series)

Each refitted model directory includes ``model.json``. When schema construction succeeds, an ``inference`` block is added for the [KServe AutoGluon server](https://github.com/kserve/kserve/tree/master/python/autogluonserver) time-series path (**REST v1 JSON only**):

| Field | Meaning |
| ----- | ------- |
| `protocol: v1_json` | Use `POST /v1/models/{name}:predict`. |
| `prediction_length` | Forecast horizon; size known-covariate horizon rows accordingly. |
| `instances.fields` | History fields include id, timestamp, target, and each configured known covariate. History rows are **objects with scalar values** (long format). |
| `known_covariates` | Present only if the model was trained with known covariates. Horizon rows with id, timestamp, and one field per covariate (`role: known_covariate`). Covariate `datatype` is inferred from training dtypes (`integer` / `number` / `boolean` / `string`). |
| `sample_payload` | Template request. Replace placeholders such as `"<string>"`, `"<number>"`, `"<integer>"`, or `"<boolean>"` with real values (datatype-dependent for covariates). |

### Model insight artifacts (per refitted model)

Under each ``{model_name}_FULL/metrics/`` directory:

- **`metrics.json`**: Holdout test metrics from ``TimeSeriesPredictor.evaluate`` (finite values only). Uses AutoGluon's raw **higher-is-better** sign convention (error metrics such as ``mean_absolute_scaled_error`` are negated) so the HTML leaderboard ranks models correctly.
- **`back_testing.json`**: Multi-window backtest with ``per_window_metrics`` and ``series_analysis``
  (best/worst forecast timelines). Window error metrics use **natural positive** signs via
  ``filter_finite_metrics``. Best-effort after refit; omitted if backtest APIs or history are
  insufficient. Visualized in the generated inference notebook when present.
