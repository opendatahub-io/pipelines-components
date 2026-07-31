# Timeseries Data Loader ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Load and split timeseries data from S3 for AutoGluon training.

This component loads time series data from S3, samples it (up to 100 MB), applies light **cleansing** (replace ``+/-inf`` with NaN so AutoGluon can apply its own missing-value logic; require parseable timestamps and non-null ids; drop exact duplicate ``(id_column, timestamp_column)`` rows, keep
last), then performs a two-stage **per-series temporal** split for efficient AutoGluon training: 1. Primary split (default 80/20): for each distinct ``id_column`` value, the earliest (1 - test_size) fraction of rows by ``timestamp_column`` goes to the train portion and the remainder to the test set
(so every series with at least two rows contributes holdout data; single-row series stay in train only). 2. Secondary split (default 30/70 of each series' train rows): early segment to selection-train, later segment to extra-train.

The test set is written to S3 artifact, while train CSVs are written to the PVC workspace for sharing across pipeline steps.

After cleansing, at least **100** valid records must remain; otherwise the component fails with a clear error so downstream AutoGluon training does not run on datasets too small to split reliably.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `file_key` | `str` | `None` | S3 object key of the CSV file containing time series data. |
| `bucket_name` | `str` | `None` | S3 bucket name containing the file. |
| `workspace_path` | `str` | `None` | PVC workspace directory where train CSVs will be written. |
| `target` | `str` | `None` | Name of the target column to forecast. |
| `id_column` | `str` | `None` | Name of the column identifying each time series (item_id). |
| `timestamp_column` | `str` | `None` | Name of the timestamp/datetime column. |
| `sampled_test_dataset` | `dsl.Output[dsl.Dataset]` | `None` | Output dataset artifact for the test split. |
| `component_status` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing stage-level progress tracking for this component. |
| `selection_train_size` | `float` | `0.3` | Fraction of train portion for model selection (default: 0.3). |
| `prediction_length` | `int` | `1` | Forecast horizon used downstream (default: 1). Only used to fail fast when a user-provided test series is too short to be evaluated. |
| `known_covariates_names` | `Optional[List[str]]` | `None` | Covariate columns known in advance downstream (default: none). Only used to fail fast when a user-provided test dataset omits one of them. |
| `test_data_bucket_name` | `str` | `""` | S3 bucket name for user-provided test dataset (default: empty string). Read with the same credentials and endpoint as the training data. |
| `test_data_file_key` | `str` | `""` | S3 object key for user-provided test dataset (default: empty string). Must be set together with ``test_data_bucket_name``. When set, the primary temporal split is skipped and this dataset is written to ``sampled_test_dataset``. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `NamedTuple('outputs', sample_config=dict, split_config=dict, sample_rows=str, models_selection_train_data_path=str, extra_train_data_path=str, evaluation_mode=str)` | sample_config, split_config, sample_rows, models_selection_train_data_path, extra_train_data_path, evaluation_mode. |

## Usage Examples 🧪

```python
"""Example pipelines demonstrating usage of timeseries_data_loader."""

from kfp import dsl
from kfp_components.components.data_processing.automl.timeseries_data_loader import timeseries_data_loader


@dsl.pipeline(name="timeseries-data-loader-example")
def example_pipeline(
    file_key: str = "data/timeseries.csv",
    bucket_name: str = "my-bucket",
    workspace_path: str = "/tmp/workspace",
    target: str = "value",
    id_column: str = "item_id",
    timestamp_column: str = "timestamp",
    selection_train_size: float = 0.3,
):
    """Example pipeline using timeseries_data_loader.

    Args:
        file_key: S3 key of the data file.
        bucket_name: S3 bucket name.
        workspace_path: Path to the workspace directory.
        target: Name of the target column.
        id_column: Name of the ID column.
        timestamp_column: Name of the timestamp column.
        selection_train_size: Fraction of data for training.
    """
    timeseries_data_loader(
        file_key=file_key,
        bucket_name=bucket_name,
        workspace_path=workspace_path,
        target=target,
        id_column=id_column,
        timestamp_column=timestamp_column,
        selection_train_size=selection_train_size,
    )

```

## Metadata 🗂️

- **Name**: timeseries_data_loader
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
- **Tags**:
  - data-processing
  - timeseries
  - automl
  - data-loading
- **Last Verified**: 2026-05-22 00:00:00+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
  - Reviewers:
    - Mateusz-Switala
    - DorotaDR

<!-- custom-content -->

### Component status artifact

In the time series training pipeline, this component writes ``component_status.json`` under the
``component_status`` output artifact. The file includes ``component_id`` (``timeseries_data_loader``),
timestamps, and per-stage status (e.g. ``prepare_data``, ``split_and_export``).
Dashboards align stage ids with ``component_stage_map.json`` from
``publish-component-stage-map``.
