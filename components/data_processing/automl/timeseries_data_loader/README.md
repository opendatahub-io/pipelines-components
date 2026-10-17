# Timeseries Data Loader ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Load and split timeseries data from S3 for AutoGluon training.

This component loads time series data from S3, retains the latest observations per series (up to 100 MiB for the ``"speed"`` preset, up to 1 GiB for ``"balanced"``, and up to 10 GiB for ``"quality"``). Small inputs use one read; larger inputs use a bounded two-pass sampler pinned to the same S3
version or ETag. Unsorted inputs are sampled by parsed timestamps, not file order. A per-series retained row count and timestamp range is written to ``timeseries_profile.jsonl`` in the component status artifact. The loader applies light **cleansing** (replace ``+/-inf`` with NaN so AutoGluon can
apply its own missing-value logic; require parseable timestamps and non-null ids; drop exact duplicate ``(id_column, timestamp_column)`` rows, keep last), then performs a two-stage **per-series temporal** split for efficient AutoGluon training: 1. Primary split (default 80/20): for each distinct
``id_column`` value, the earliest (1 - test_size) fraction of rows by ``timestamp_column`` goes to the train portion and the remainder to the test set (so every series with at least two rows contributes holdout data; single-row series stay in train only). 2. Secondary split (default 30/70 of each
series' train rows): early segment to selection-train, later segment to extra-train.

The test set is written to an S3 artifact, while train Parquet files (selection-train and extra-train, Snappy-compressed) are written to the PVC workspace for sharing across pipeline steps.

After cleansing, at least **100** valid records must remain; otherwise the component fails with a clear error so downstream AutoGluon training does not run on datasets too small to split reliably.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `file_key` | `str` | `None` | S3 object key of the CSV file containing time series data. |
| `bucket_name` | `str` | `None` | S3 bucket name containing the file. |
| `workspace_path` | `str` | `None` | PVC workspace directory where train Parquet files will be written. |
| `target` | `str` | `None` | Name of the target column to forecast. |
| `timestamp_column` | `str` | `None` | Name of the timestamp/datetime column. |
| `sampled_test_dataset` | `dsl.Output[dsl.Dataset]` | `None` | Output dataset artifact for the test split. |
| `component_status` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing stage-level progress tracking for this component. |
| `id_column` | `str` | `""` | Name of the column identifying each time series (item_id). Pass an empty string ("") for single-series two-column datasets (timestamp + target only); the loader will inject a synthetic ID column (__synthetic_item_id) with value "item_0". |
| `selection_train_size` | `float` | `0.3` | Fraction of train portion for model selection (default: 0.3). |
| `prediction_length` | `int` | `1` | Forecast horizon used downstream (default: 1). Only used to fail fast when a user-provided test series is too short to be evaluated. |
| `known_covariates_names` | `Optional[List[str]]` | `None` | Covariate columns known in advance downstream (default: none). Only used to fail fast when a user-provided test dataset omits one of them. |
| `test_data_bucket_name` | `str` | `""` | S3 bucket name for user-provided test dataset (default: empty string). |
| `test_data_file_key` | `str` | `""` | S3 object key of the user-provided test CSV (default: empty string). |
| `preset` | `str` | `speed` | Training quality tier controlling the in-memory sampling size budget. ``"speed"`` (default) retains up to 100 MiB; ``"balanced"`` retains up to 1 GiB; and ``"quality"`` samples up to 10 GiB. User-provided test datasets are capped at 50 MiB, 100 MiB, and 1 GiB respectively using the existing head policy. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `NamedTuple('outputs', sample_config=dict, split_config=dict, sample_rows=str, models_selection_train_data_path=str, extra_train_data_path=str, effective_id_column=str, uses_synthetic_id=bool)` | sample_config, split_config, sample_rows, models_selection_train_data_path, extra_train_data_path. |

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
- **Last Verified**: 2026-10-13 00:00:00+00:00
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

### Sampling large training inputs

Training data uses `last_values_per_series`: the latest distinct timestamps of every
series are retained before the existing temporal splits. Input may be ascending,
descending, interleaved, or unsorted. Timestamp parsing, UTC normalization, fractional
years, missing-target retention, and duplicate handling (`keep last` in source order)
use the same cleansing rules as small inputs.

Inputs fitting the preset budget use one read. Larger inputs require a complete
inventory scan and a second read pinned to the same S3 version or ETag. Short series
retain all observations; longer series share a fixed row quota calculated from
conservative per-series memory bounds. Variable-width columns, dtype promotion,
synthetic IDs, and inventory overhead are included. These bounds may leave some of
the byte budget unused. A panel with too many IDs to fit its inventory or at least
one observation per series fails with a clear error instead of omitting series.

| Preset | Sample budget | Loader memory request | Loader memory limit |
| ------ | ------------- | --------------------- | ------------------- |
| `speed` | 100 MiB | 16 GiB | 64 GiB |
| `balanced` | 1 GiB | 32 GiB | 64 GiB |
| `quality` | 10 GiB | 64 GiB | 64 GiB |

The sample budget measures pandas data, rather than CSV bytes or process RSS.
Streaming and temporary compaction buffers also consume memory; they remain
bounded by the sample budget plus a CSV chunk. Split buffers are released after
concatenation, and splits reuse the already sorted sample. The 128 GiB limit of
the `quality` training task does not apply to the loader.

User-provided test CSVs retain their existing head policy and separate preset caps
(50 MiB, 100 MiB, and 1 GiB). An interrupted first training read may recover its
observed prefix, with `source_read_complete=false`; its sample describes only that
prefix. A failed second sampling read or changed S3 object fails the component.

### Retained-series profile

The existing `component_status` artifact includes `timeseries_profile.jsonl`, with
one record per retained series: `item_id`, `retained_rows`, `min_timestamp`, and
`max_timestamp`. IDs and timestamp bounds are serialized as strings; datetime
bounds are ISO timestamps normalized to UTC without timezone information.
Fractional-year bounds retain their numeric string representation.

The `prepare_data` metrics and `sample_config` include `sampling_method`,
`source_rows_read`, `source_read_complete`, `read_passes`, `series_count`, and
`profile_file`. The legacy `total_rows_loaded` and `sampled_rows` fields continue
to describe the retained, cleansed frame. The complete profile is written
incrementally into the artifact, keeping KFP parameters small.

### Component status artifact

In the time series training pipeline, this component writes ``component_status.json`` under the
``component_status`` output artifact. The file includes ``component_id`` (``timeseries_data_loader``),
timestamps, and per-stage status (e.g. ``prepare_data``, ``split_and_export``).
Dashboards align stage ids with ``component_stage_map.json`` from
``publish-component-stage-map``.
