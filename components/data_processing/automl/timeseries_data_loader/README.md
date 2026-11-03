# Timeseries Data Loader ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Load and split timeseries data from S3 for AutoGluon training.

This component scans the entire CSV from S3 and retains the latest observations per series (up to 100 MiB of pandas data for the ``"speed"`` preset, up to 1 GiB for ``"balanced"``, and up to 10 GiB for ``"quality"``), applies light **cleansing** (replace ``+/-inf`` with NaN so AutoGluon can apply
its own missing-value logic; require parseable timestamps and non-null ids; drop duplicate ``(id_column, timestamp_column)`` keys, keeping the last file occurrence). Timestamp order determines recency even for unsorted inputs. Small sources stay in RAM; larger sources use temporary Parquet files and
a SQLite timestamp index on the workspace. No second S3 download is required. Source/retained counts and timestamp ranges per series are saved as JSONL profiles in ``component_status``. The component then performs a two-stage **per-series temporal** split for efficient AutoGluon training: 1. Primary
split (default 80/20): for each distinct ``id_column`` value, the earliest (1 - test_size) fraction of rows by ``timestamp_column`` goes to the train portion and the remainder to the test set (so every series with at least two rows contributes holdout data; single-row series stay in train only). 2.
Secondary split (default 30/70 of each series' train rows): early segment to selection-train, later segment to extra-train.

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
| `prediction_length` | `int` | `1` | Forecast horizon used downstream (default: 1). Sampling preserves enough history for the temporal splits and at least one internal validation window when the source provides it. Also validates external test series length. |
| `known_covariates_names` | `Optional[List[str]]` | `None` | Covariate columns known in advance downstream (default: none). Only used to fail fast when a user-provided test dataset omits one of them. |
| `test_data_bucket_name` | `str` | `""` | S3 bucket name for user-provided test dataset (default: empty string). |
| `test_data_file_key` | `str` | `""` | S3 object key of the user-provided test CSV (default: empty string). |
| `preset` | `str` | `speed` | Training quality tier controlling the sampling size budget. ``"speed"`` (default) samples up to 100 MiB; ``"balanced"`` samples up to 1 GiB; and ``"quality"`` samples up to 10 GiB. User-provided test datasets are capped at 50 MiB, 100 MiB, and 1 GiB respectively. |
| `ssl_cert_path` | `Optional[str]` | `None` | Optional path to a PEM-encoded CA bundle used to verify the S3 endpoint. The filename extension is not significant; ``.pem`` and ``.crt`` are both common. It overrides ``AWS_CA_BUNDLE`` for this component; when neither is set, the system trust bundle is used. TLS verification is never disabled. |

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
- **Last Verified**: 2026-10-20 00:00:00+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - mateusz-szewczyk-dev
    - LukaszCmielowski
    - Mateusz-Switala
    - jakub-walaszczyk
  - Reviewers:
    - mateusz-szewczyk-dev
    - Mateusz-Switala

<!-- custom-content -->

### Component status artifact

In the time series training pipeline, this component writes ``component_status.json`` under the
``component_status`` output artifact. The file includes ``component_id`` (``timeseries_data_loader``),
timestamps, and per-stage status (e.g. ``prepare_data``, ``split_and_export``).
Dashboards align stage ids with ``component_stage_map.json`` from
``publish-component-stage-map``.

### Latest observations per series

The loader scans the entire CSV and keeps the most recent observations **for every ID**.
Recency is determined by parsed timestamps, independently of file order. Grouped,
interleaved, descending and unsorted panels follow the same policy. A global file
`head` or `tail` is never used for training-data sampling. IDs are read as strings,
so values such as `001` and `01` remain distinct.

Duplicate `(id, timestamp)` keys keep the last occurrence in the source, including
when duplicate records occur in different read chunks. Timestamps are normalized
before comparison; missing targets remain in the series. An interrupted read fails
the component because a partial scan cannot establish the latest history.

The preset budget caps the retained pandas DataFrame after normalization and
synthetic ID injection: 100 MiB (`speed`), 1 GiB (`balanced`), or 10 GiB (`quality`).
It is neither a CSV file-size limit nor a peak process-memory limit. Small datasets
are retained in full. For larger panels, the sampler allocates approximately equal
history lengths to each series and redistributes unused capacity from shorter
series. Actual row byte costs determine which complete tail layers fit; a remaining
partial layer is assigned in deterministic ID order. The result is checked again
against pandas deep memory usage after dtype normalization.

Sampling preserves enough observations for the configured horizon, temporal splits,
and at least one AutoGluon internal validation window when that history exists in
the source. Originally shorter series remain intact. If the budget cannot preserve
these histories for all series, the loader fails with an actionable error instead
of silently dropping IDs. The existing global minimum of 100 records still applies.

### Sources larger than the preset budget

After the RAM threshold is reached, buffered and subsequent chunks are written as
Snappy Parquet files on the workspace. A disposable SQLite index stores only series
IDs, timestamps, byte estimates and row locations. It replaces duplicate locations
and provides latest-first access without sorting the full payload. Only selected
rows are assembled into the output DataFrame; each selected Parquet part is read
at most once. The source CSV is downloaded and parsed once.

Scratch storage is capped at 16 GiB. The loader also reserves the preset sample
budget plus 256 MiB of free workspace for outputs. Workspace exhaustion fails
explicitly; it never falls back to an incomplete source prefix. Temporary files
are removed before training, including when reading, validation or sampling fails.
The pipeline's workspace is shared with training artifacts and models.

For example, a 2 GB CSV is scanned completely for all presets. `speed` retains
at most 100 MiB of its latest per-series history, `balanced` retains at most 1 GiB,
and `quality` retains everything only if its pandas representation fits 10 GiB.
Full scanning adds I/O compared with the former early-stop loader. Training/refit
presets and time budgets are unchanged, but training cost can still depend on the
number of retained series and observations.

User-provided test CSVs use the same per-series timestamp policy with their own
50 MiB / 100 MiB / 1 GiB budgets. Truncation is reported, and each retained test
series must have more observations than `prediction_length` for evaluation context.

### Sampling profiles

The existing `component_status` artifact contains `training_series_profile.jsonl`
and, when external test data is supplied, `test_series_profile.jsonl`. Each line
records a series ID, source and unique row counts, source timestamp range, retained
row count/range, and row counts/ranges of the exported splits. These files are
separate from the compact status metrics so large panels do not create oversized
pipeline parameters.

`sample_config` reports `sampling_method="latest_per_series"`, `scanned_rows`,
`source_unique_rows`, `duplicate_rows`, `sampled_rows`, `series_count`,
`sampled_in_memory_bytes`, `sample_cap_bytes`, `full_source_read`, `used_disk`,
`peak_staging_bytes`, `scan_seconds`, `sampling_seconds`, and the relative profile
filename. `cap_reached`/`truncated` describe omitted unique observations, rather
than a stopped source read. `total_rows_loaded` retains its previous meaning of
rows in the final sample. External-test sampling has a separate `test_sampling`
entry. The status metric `sample_cap_reached` remains available for dashboards.
