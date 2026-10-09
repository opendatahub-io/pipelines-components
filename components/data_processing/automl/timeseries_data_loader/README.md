# Timeseries Data Loader ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Load and split timeseries data from S3 for AutoGluon training.

This component scans the complete CSV and keeps the newest observations per series, using a bounded buffer (up to 100 MiB for the ``"speed"`` preset, up to 1 GiB for ``"balanced"``, and up to 10 GiB for ``"quality"``), applies light **cleansing** (replace ``+/-inf`` with NaN so AutoGluon can apply
its own missing-value logic; require parseable timestamps and non-null ids; drop exact duplicate ``(id_column, timestamp_column)`` rows, keep last), then performs a two-stage **per-series temporal** split for efficient AutoGluon training: 1. Primary split (default 80/20): for each distinct
``id_column`` value, the earliest (1 - test_size) fraction of rows by ``timestamp_column`` goes to the train portion and the remainder to the test set (so every series with at least two rows contributes holdout data; single-row series stay in train only). 2. Secondary split (default 30/70 of each
series' train rows): early segment to selection-train, later segment to extra-train.

The test set is written to an S3 artifact, while train Parquet files (selection-train and extra-train, Snappy-compressed) are written to the PVC workspace for sharing across pipeline steps.

Unsorted input is supported: timestamps are validated before sampling and retained rows are sorted before splitting. Partial CSV reads fail. The existing component_status artifact records source and retained counts and time ranges in series_sampling_profile.json, with a summary of up to 50 series
per dataset under metadata.sampling_profile.

After cleansing, at least **100** valid records must remain; otherwise the component fails with a clear error so downstream AutoGluon training does not run on datasets too small to split reliably. At least one selection-train series must also have ``max(prediction_length + 1, 5) +
prediction_length`` observations for AutoGluon's default internal validation.

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
| `prediction_length` | `int` | `1` | Forecast horizon used downstream (default: 1). Validates that selection-train contains a series long enough for training and that user-provided test series are long enough to be evaluated. |
| `known_covariates_names` | `Optional[List[str]]` | `None` | Covariate columns known in advance downstream (default: none). Only used to fail fast when a user-provided test dataset omits one of them. |
| `test_data_bucket_name` | `str` | `""` | S3 bucket name for user-provided test dataset (default: empty string). |
| `test_data_file_key` | `str` | `""` | S3 object key of the user-provided test CSV (default: empty string). |
| `preset` | `str` | `speed` | Training quality tier controlling the sampling size budget. ``"speed"`` (default) samples up to 100 MiB; ``"balanced"`` samples up to 1 GiB; and ``"quality"`` samples up to 10 GiB. User-provided test datasets are capped at 50 MiB, 100 MiB, and 1 GiB respectively. |

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
- **Last Verified**: 2026-10-08 00:00:00+00:00
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

In the time series training pipeline, this component writes ``component_status.json`` under the
``component_status`` output artifact. The file includes ``component_id`` (``timeseries_data_loader``),
timestamps, and per-stage status (e.g. ``prepare_data``, ``split_and_export``).
Dashboards align stage ids with ``component_stage_map.json`` from
``publish-component-stage-map``.

### Sampling and source order

The `last_values_per_series` policy reads the complete CSV once and retains each series'
newest unique timestamps in a bounded buffer. Ascending, descending, shuffled, and interleaved
inputs are supported; retained rows are sorted before splitting. Conflicting `(id, timestamp)`
duplicates keep the last occurrence in file order. Invalid timestamps or incomplete reads fail
instead of silently returning an old or partial history. External test CSVs use the same policy.

Numeric-only year axes remain numeric. When a CSV mixes years and date strings, all timestamps
are normalized to dates, including previously buffered rows and source ranges. An integer year
means January 1; a fractional year represents the elapsed fraction of that calendar year.
This behavior is independent of chunk boundaries and input order.

The sampler reserves one row per series, then shares the remaining byte budget equally.
Each series has its own row limit based on its maximum observed row cost, so expensive rows
do not force every series to retain the same small number of observations. Limits only decrease
as more series or larger rows are encountered. Row costs account for stored Python values and
buffer overhead, with a small reserve for numeric dtype changes and timestamp normalization.
Per-series metadata also counts towards the budget; short series can leave some budget unused.
The `prepare_data` metrics report `sampled_buffer_estimated_bytes` alongside
`sampled_in_memory_bytes` (the final pandas frame), so the two memory representations can be
compared. The CSV parser and exporting the retained frame require additional memory.
A budget too small to retain even one observation per series
fails explicitly.

The selection-train split must contain at least one series with
`max(prediction_length + 1, 5) + prediction_length` observations for AutoGluon's default internal
validation. With the default horizon and split fractions, this requires at least 25 retained
observations in that series. The loader fails early if sampling or splitting leaves every series
too short, even when the total dataset exceeds 100 rows. Increase the sampling preset, increase
`selection_train_size`, or provide longer histories.

An observation too large to fit alone is represented by a timestamp marker. After the complete
read, history up to the newest remaining marker is removed, keeping a contiguous latest tail.
If the newest observation of a series cannot fit, the loader fails instead of returning older
data. This rule also applies to external test CSVs and respects last-occurrence duplicates.

The existing status artifact contains the full `series_sampling_profile.json`, with retained
counts and timestamp ranges for `retained`, `selection_train`, `extra_train`, and `test`.
The `retained` and `test` entries also include source counts (including duplicates), source
timestamp ranges, and `oversized_rows_seen`. All retained counts reflect final unique rows.
`component_status.json` includes up to 50 series per dataset under
`metadata.sampling_profile.datasets`, plus `series_counts`, `profiles_truncated`, and the
`profile_file` reference. No additional component parameter or output is required.
`sample_config.source_rows_seen` records the complete source count; `sampled_rows` counts
retained rows.

Run the component and pipeline regression tests from the repository root:

```bash
uv run python -m scripts.tests.run_component_tests \
  components/data_processing/automl/timeseries_data_loader \
  pipelines/training/automl/autogluon_timeseries_training_pipeline
```

Tests cover per-series tails, input order, chunk boundaries, duplicates, oversized observations,
different row costs across series, numeric timestamp profiles, full and bounded status profiles,
external test sampling, and validation beyond the old head cutoff. Pandas/Parquet checks run when
pandas and pyarrow are installed.
