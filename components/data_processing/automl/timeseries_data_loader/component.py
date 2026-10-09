from typing import List, NamedTuple, Optional

from kfp import dsl
from kfp_components.utils.consts import AUTOML_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=AUTOML_IMAGE,  # noqa: E501
    install_kfp_package=False,
)
def timeseries_data_loader(
    file_key: str,
    bucket_name: str,
    workspace_path: str,
    target: str,
    timestamp_column: str,
    sampled_test_dataset: dsl.Output[dsl.Dataset],
    component_status: dsl.Output[dsl.Artifact],
    id_column: str = "",
    selection_train_size: float = 0.3,
    prediction_length: int = 1,
    known_covariates_names: Optional[List[str]] = None,
    test_data_bucket_name: str = "",
    test_data_file_key: str = "",
    preset: str = "speed",
) -> NamedTuple(
    "outputs",
    sample_config=dict,
    split_config=dict,
    sample_rows=str,
    models_selection_train_data_path=str,
    extra_train_data_path=str,
    effective_id_column=str,
    uses_synthetic_id=bool,
):
    """Load and split timeseries data from S3 for AutoGluon training.

    This component scans the complete CSV and keeps the newest observations per series,
    using a bounded buffer (up to 100 MiB for the
    ``"speed"`` preset, up to 1 GiB for ``"balanced"``, and up to 10 GiB for
    ``"quality"``),
    applies light **cleansing** (replace ``+/-inf`` with NaN so AutoGluon can apply its
    own missing-value logic; require parseable timestamps and non-null ids; drop
    exact duplicate ``(id_column, timestamp_column)`` rows, keep last), then performs a two-stage
    **per-series temporal** split for efficient AutoGluon training:
    1. Primary split (default 80/20): for each distinct ``id_column`` value, the earliest
       (1 - test_size) fraction of rows by ``timestamp_column`` goes to the train portion and
       the remainder to the test set (so every series with at least two rows contributes
       holdout data; single-row series stay in train only).
    2. Secondary split (default 30/70 of each series' train rows): early segment to
       selection-train, later segment to extra-train.

    The test set is written to an S3 artifact, while train Parquet files (selection-train
    and extra-train, Snappy-compressed) are written to the PVC workspace for sharing across
    pipeline steps.

    Unsorted input is supported: timestamps are validated before sampling and retained rows
    are sorted before splitting. Partial CSV reads fail. The existing component_status
    artifact records source and retained counts and time ranges in series_sampling_profile.json,
    with a summary of up to 50 series per dataset under metadata.sampling_profile.

    After cleansing, at least **100** valid records must remain; otherwise the component
    fails with a clear error so downstream AutoGluon training does not run on datasets too
    small to split reliably.
    At least one selection-train series must also have
    ``max(prediction_length + 1, 5) + prediction_length`` observations for AutoGluon's
    default internal validation.

    Args:
        file_key: S3 object key of the CSV file containing time series data.
        bucket_name: S3 bucket name containing the file.
        workspace_path: PVC workspace directory where train Parquet files will be written.
        target: Name of the target column to forecast.
        id_column: Name of the column identifying each time series (item_id). Pass an empty
            string ("") for single-series two-column datasets (timestamp + target only);
            the loader will inject a synthetic ID column (__synthetic_item_id) with value "item_0".
        timestamp_column: Name of the timestamp/datetime column.
        sampled_test_dataset: Output dataset artifact for the test split.
        component_status: Output artifact containing stage-level progress tracking for this component.
        selection_train_size: Fraction of train portion for model selection (default: 0.3).
        prediction_length: Forecast horizon used downstream (default: 1). Validates that
            selection-train contains a series long enough for training and that user-provided
            test series are long enough to be evaluated.
        known_covariates_names: Covariate columns known in advance downstream (default: none).
            Only used to fail fast when a user-provided test dataset omits one of them.
        test_data_bucket_name: S3 bucket name for user-provided test dataset (default: empty string).
        test_data_file_key: S3 object key of the user-provided test CSV (default: empty string).
        preset: Training quality tier controlling the sampling size budget. ``"speed"``
            (default) samples up to 100 MiB; ``"balanced"`` samples up to 1 GiB; and
            ``"quality"`` samples up to 10 GiB. User-provided test datasets are capped
            at 50 MiB, 100 MiB, and 1 GiB respectively.

    Raises:
        ValueError: If a required parameter is empty or invalid, if only one of the
            ``test_data_*`` pair is set or the test key is not a valid S3 object key, if the
            test dataset is empty, missing required or covariate columns, shares no series
            with the training data, or has a series shorter than ``prediction_length``, or if
            fewer than 100 valid records remain after cleansing, or no selection-train
            series is long enough for AutoGluon's default internal validation.

    Returns:
        NamedTuple: sample_config, split_config, sample_rows, models_selection_train_data_path,
                   extra_train_data_path.
    """
    import heapq
    import io
    import json
    import logging
    import sys
    from pathlib import Path

    import boto3
    import pandas as pd

    logger = logging.getLogger(__name__)

    def _log_dataset_stats(name, df):
        """Log the size of a dataframe: rows, columns, and in-memory bytes.

        ``memory_usage(deep=True)`` reports the pandas in-memory footprint, which
        differs from both the on-disk CSV size and the sampler's Python buffers.
        """
        try:
            n_rows = len(df)
            n_cols = df.shape[1] if hasattr(df, "shape") else len(df.columns)
            n_bytes = int(df.memory_usage(deep=True).sum())
            logger.info(
                "Dataset stats [%s]: rows=%s, columns=%s, size=%s bytes (%.2f MiB)",
                name,
                n_rows,
                n_cols,
                n_bytes,
                n_bytes / (1024**2),
            )
        except Exception as e:  # noqa: BLE001 - stats logging must never break the run
            logger.debug("Could not compute dataset stats for %s: %s", name, e)

    from kfp_components.components.training.automl.shared.component_status import ComponentStatusTracker
    from kfp_components.components.training.automl.shared.parquet_utils import stringify_mixed_object_columns
    from kfp_components.components.training.automl.shared.user_test_data import (
        raise_if_test_data_empty,
        resolve_s3_env_credentials,
        test_data_load_error,
        test_data_source_uri,
        validate_s3_env_credentials,
        validate_test_data_params,
    )

    VALID_PRESETS = {"speed", "balanced", "quality"}
    # Sampling budget per quality tier: "speed" stays small for fast runs,
    # "balanced" allows the default supported dataset size, while
    # "quality" is sized for the higher-memory training profile.
    PRESET_MAX_SIZE_BYTES = {
        "speed": 100 * 1024 * 1024,  # 100 MiB
        "balanced": 1024 * 1024 * 1024,  # 1 GiB
        "quality": 10 * 1024 * 1024 * 1024,  # 10 GiB
    }
    PRESET_TEST_DATA_MAX_SIZE_BYTES = {
        "speed": 50 * 1024 * 1024,  # 50 MiB
        "balanced": 100 * 1024 * 1024,  # 100 MiB
        "quality": 1024 * 1024 * 1024,  # 1 GiB
    }
    MIN_VALID_RECORDS_AFTER_CLEANSING = 100
    PANDAS_CHUNK_SIZE = 10000  # Rows per batch for streaming read
    DEFAULT_TEST_SIZE = 0.2
    MAX_SERIES_PROFILE = 50

    if preset not in VALID_PRESETS:
        raise ValueError(f"preset must be one of {sorted(VALID_PRESETS)}; got {preset!r}.")
    MAX_SIZE_BYTES = PRESET_MAX_SIZE_BYTES[preset]
    TEST_DATA_MAX_SIZE_BYTES = PRESET_TEST_DATA_MAX_SIZE_BYTES[preset]

    SYNTHETIC_ITEM_ID_COLUMN = "__synthetic_item_id"
    SYNTHETIC_ITEM_ID_VALUE = "item_0"

    # Input validation
    for param, value in (
        ("bucket_name", bucket_name),
        ("file_key", file_key),
        ("workspace_path", workspace_path),
        ("target", target),
        ("timestamp_column", timestamp_column),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{param} must be a non-empty string.")
    if not isinstance(id_column, str):
        raise ValueError("id_column must be a string.")
    if id_column == "":
        pass  # Empty string is valid for two-column datasets
    elif not id_column.strip():
        raise ValueError(
            "id_column must be an empty string or a non-empty column name (whitespace-only values are not allowed)."
        )
    if selection_train_size <= 0 or selection_train_size >= 1:
        raise ValueError("selection_train_size must be in a range 0 to 1.")

    if file_key.startswith("/") or file_key.endswith("/") or "//" in file_key:
        raise ValueError("file_key must be a valid S3 object key and must not start/end with '/' or contain '//'.")

    # Validate workspace_path to prevent path traversal
    workspace_path_obj = Path(workspace_path)
    if not workspace_path_obj.is_absolute():
        raise ValueError(f"workspace_path must be an absolute path; got {workspace_path!r}")

    workspace_path_resolved = workspace_path_obj.resolve()
    try:
        workspace_path_resolved.relative_to(workspace_path_obj)
    except ValueError:
        raise ValueError(
            f"workspace_path resolves outside the trusted workspace boundary: "
            f"{workspace_path!r} -> {workspace_path_resolved}"
        )

    if prediction_length <= 0:
        raise ValueError("prediction_length must be greater than 0.")

    test_data_bucket_name, test_data_file_key = validate_test_data_params(test_data_bucket_name, test_data_file_key)

    status = ComponentStatusTracker(component_status.path, "timeseries_data_loader")
    with status:
        status.set_metadata(display_name="Timeseries Data Loader Status")
        component_status.metadata["display_name"] = "Timeseries Data Loader Status"
        status.record("prepare_data", "started")

        logger.info(
            "Sampling size budget: preset=%s, max_size=%s bytes (%.0f MiB)",
            preset,
            MAX_SIZE_BYTES,
            MAX_SIZE_BYTES / (1024**2),
        )

        def get_s3_client(verify=True):
            """Create and return an S3 client using credentials from environment variables."""
            credentials = resolve_s3_env_credentials()
            validate_s3_env_credentials(credentials)

            return boto3.client(
                "s3",
                endpoint_url=credentials["endpoint_url"],
                region_name=credentials["region_name"],
                aws_access_key_id=credentials["access_key"],
                aws_secret_access_key=credentials["secret_key"],
                verify=verify,
            )

        def _year_as_iso(value):
            """Convert year tokens to calendar dates, leaving other date formats for pandas."""
            try:
                year = float(value)
            except (TypeError, ValueError):
                return value
            if year != year:  # Preserve NaN for the existing invalid-timestamp check.
                return value
            if not 1800 <= year <= 2200:
                raise ValueError(
                    f"Column {timestamp_column!r} contains numeric values outside the fractional year range "
                    "(1800-2200). If these are Unix timestamps, convert them to ISO date strings upstream."
                )
            start = pd.Timestamp(year=int(year), month=1, day=1)
            end = pd.Timestamp(year=int(year) + 1, month=1, day=1)
            return (start + (end - start) * (year - int(year))).isoformat()

        def load_timeseries_data_truncate(
            bucket_name,
            file_key,
            max_size_bytes,
            chunk_size,
            truncation_report=None,
            fail_on_partial_read: bool = False,
        ):
            """Read to EOF and retain the newest unique timestamps per series.

            Reserve one row per series, then share the remaining bytes equally.
            Each series' row limit uses its maximum row cost and only decreases as
            more series or larger rows are discovered, so history need not be recovered. Input
            order does not affect the retained timestamp window. Partial reads fail.
            Rows too large to fit alone leave timestamp markers; the final history
            starts after the newest remaining marker, preserving a contiguous tail.
            An oversized latest observation fails instead of returning older history.
            Numeric-only time axes stay numeric. Encountering a date switches all
            timestamp keys and source ranges to datetimes for the rest of the CSV.
            ``fail_on_partial_read`` also permits an empty test frame for the caller.
            """
            from botocore.exceptions import SSLError

            s3_client = get_s3_client()
            try:
                response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
            except SSLError:
                logger.warning(
                    "SSL error when downloading s3://%s/%s, retrying with verify=False",
                    bucket_name,
                    file_key,
                )
                no_verify_client = get_s3_client(verify=False)
                response = no_verify_client.get_object(Bucket=bucket_name, Key=file_key)
            buffers, costs, source_stats = {}, {}, {}
            total_cost, total_rows_read = 0, 0
            extra_bytes_per_series, minimum_extra_bytes = 0, 0
            metadata_bytes = 0
            sampled = False
            datetime_timestamps = False
            columns = None
            sampling_id = id_column or SYNTHETIC_ITEM_ID_COLUMN
            inject_id = sampling_id == SYNTHETIC_ITEM_ID_COLUMN
            with io.TextIOWrapper(response["Body"], encoding="utf-8") as text_stream:
                try:
                    for chunk_df in pd.read_csv(
                        text_stream, chunksize=chunk_size, dtype=None if inject_id else {sampling_id: "string"}
                    ):
                        total_rows_read += len(chunk_df)
                        if not len(chunk_df):
                            continue
                        columns = list(chunk_df.columns)
                        required = {timestamp_column, target} | (set() if inject_id else {id_column})
                        missing = required - set(columns)
                        if missing:
                            dataset = "test dataset" if fail_on_partial_read else "dataset"
                            raise ValueError(f"Missing required columns in {dataset}: {missing}.")
                        if inject_id:
                            chunk_df[sampling_id] = SYNTHETIC_ITEM_ID_VALUE
                        if (
                            not datetime_timestamps
                            and not pd.to_numeric(chunk_df[timestamp_column], errors="coerce").notna().all()
                        ):
                            datetime_timestamps = True
                            # Convert earlier numeric history once, including oversized markers.
                            for heap, rows in buffers.values():
                                converted = {}
                                for timestamp, payload in rows.items():
                                    timestamp = pd.to_datetime(_year_as_iso(timestamp), utc=True).tz_localize(None)
                                    if payload is not None:
                                        payload[columns.index(timestamp_column)] = timestamp.isoformat()
                                    converted[timestamp] = payload
                                heap[:] = sorted(converted)
                                rows.clear()
                                rows.update(converted)
                            for stats in source_stats.values():
                                for bound in ("source_timestamp_min", "source_timestamp_max"):
                                    stats[bound] = pd.to_datetime(_year_as_iso(stats[bound]), utc=True).tz_localize(
                                        None
                                    )
                        chunk_df = _clean_timeseries_dataframe(
                            chunk_df,
                            sampling_id,
                            timestamp_column,
                            logger,
                            deduplicate=False,
                            normalize_years=datetime_timestamps,
                        )
                        id_index = list(chunk_df.columns).index(sampling_id)
                        ts_index = list(chunk_df.columns).index(timestamp_column)
                        for row in chunk_df.itertuples(index=False, name=None):
                            item_id, timestamp = str(row[id_index]), row[ts_index]
                            cost = (
                                192
                                + sys.getsizeof(row)
                                # Reserve datetime conversion space only for timestamps.
                                # Other cells need only their stored size; 32 bytes covers
                                # numeric dtype changes between CSV chunks.
                                + sum(
                                    max(128 if index == ts_index else 32, sys.getsizeof(value))
                                    for index, value in enumerate(row)
                                )
                            )
                            oversized = cost > max_size_bytes - 132
                            if oversized:
                                # Keep only a timestamp marker; never hide a gap in the final tail.
                                cost = 192 + sys.getsizeof(row) + 128 * len(row)
                            if item_id not in buffers:
                                metadata_bytes += 1024 + sys.getsizeof(item_id)
                                if metadata_bytes > max_size_bytes:
                                    raise ValueError("Series metadata exceeds the sampling budget; increase preset.")
                                buffers[item_id] = ([], {})
                                source_stats[item_id] = {
                                    "source_rows_seen": 0,
                                    "source_timestamp_min": timestamp,
                                    "source_timestamp_max": timestamp,
                                    "oversized_rows_seen": 0,
                                }
                            stats = source_stats[item_id]
                            stats["source_rows_seen"] += 1
                            stats["source_timestamp_min"] = min(stats["source_timestamp_min"], timestamp)
                            stats["source_timestamp_max"] = max(stats["source_timestamp_max"], timestamp)
                            stats["oversized_rows_seen"] += int(oversized)
                            if cost > costs.get(item_id, 0):
                                total_cost += cost - costs.get(item_id, 0)
                                costs[item_id] = cost
                                remaining_bytes = max_size_bytes - 132 - metadata_bytes - total_cost
                                if remaining_bytes < 0:
                                    raise ValueError(
                                        "Sampling budget cannot retain one row per series; increase preset."
                                    )
                                extra_bytes_per_series = remaining_bytes // len(costs)
                                minimum_extra_bytes = max(minimum_extra_bytes, (len(buffers[item_id][0]) - 1) * cost)
                                # Scan all buffers only when a retained history needs trimming.
                                if extra_bytes_per_series < minimum_extra_bytes:
                                    minimum_extra_bytes = 0
                                    for series_id, (heap, rows) in buffers.items():
                                        limit = 1 + extra_bytes_per_series // costs[series_id]
                                        while len(heap) > limit:
                                            del rows[heapq.heappop(heap)]
                                            sampled = True
                                        minimum_extra_bytes = max(
                                            minimum_extra_bytes, (len(heap) - 1) * costs[series_id]
                                        )
                            # Return the original columns for the existing ID injection
                            # and cleansing paths. Preserve parsed time as the heap key.
                            payload = None
                            if not oversized:
                                payload = list(row[: len(columns)])
                                if hasattr(timestamp, "isoformat"):
                                    payload[ts_index] = timestamp.isoformat()
                            heap, rows = buffers[item_id]
                            row_limit = 1 + extra_bytes_per_series // costs[item_id]
                            if timestamp in rows:
                                rows[timestamp] = payload
                            elif len(heap) < row_limit:
                                heapq.heappush(heap, timestamp)
                                rows[timestamp] = payload
                            else:
                                sampled = True
                                if timestamp > heap[0]:
                                    del rows[heapq.heapreplace(heap, timestamp)]
                                    rows[timestamp] = payload
                            minimum_extra_bytes = max(minimum_extra_bytes, (len(heap) - 1) * costs[item_id])
                except Exception as e:
                    raise ValueError(f"Error reading CSV from S3: {e}") from e
            for heap, rows in buffers.values():
                cutoff = max((timestamp for timestamp, payload in rows.items() if payload is None), default=None)
                if cutoff is not None:
                    while heap and heap[0] <= cutoff:
                        del rows[heapq.heappop(heap)]
                    sampled = True
                    if not heap:
                        raise ValueError(
                            "Sampling budget cannot retain the latest observation of a series; increase preset."
                        )
            if truncation_report is not None:
                truncation_report.update(
                    truncated=sampled,
                    cap_reached=sampled,
                    source_rows_seen=total_rows_read,
                    input_complete=True,
                    sampled_buffer_estimated_bytes=132
                    + metadata_bytes
                    + sum(len(rows) * costs[item_id] for item_id, (_, rows) in buffers.items()),
                    source_series={
                        item_id: {
                            **stats,
                            "source_timestamp_min": str(stats["source_timestamp_min"]),
                            "source_timestamp_max": str(stats["source_timestamp_max"]),
                        }
                        for item_id, stats in source_stats.items()
                    },
                )
            if not buffers:
                if fail_on_partial_read:
                    return pd.DataFrame()
                raise ValueError("No data was loaded from S3. The file may be empty or inaccessible.")
            return pd.DataFrame(
                [rows[timestamp] for _, rows in buffers.values() for timestamp in sorted(rows)], columns=columns
            )

        def _clean_timeseries_dataframe(data, id_col, ts_col, log, deduplicate=True, normalize_years=False):
            """Prepare panel data without dropping rows for missing targets (AutoGluon handles NaNs).

            Per time-series practice, **do not** drop rows for null/NaN targets or non-finite values
            after mapping ``+/-inf`` to NaN: removing observations creates irregular grids and breaks frequency
            inference unless callers set ``freq`` explicitly. See AutoGluon TimeSeries missing-value
            handling per model family.

            This step: replace ``+/-inf`` with NaN; parse timestamps and **fail** if any are invalid;
            **fail** if any ``id_col`` or timestamp is null; ``drop_duplicates`` on ``(id_col, ts_col)``
            (keep last) only for true duplicate keys. Sampling disables deduplication
            to account for every source row before the timestamp buffers keep the last value.
            """
            rows_in = len(data)
            if rows_in == 0:
                return data

            out = data.replace([float("inf"), float("-inf")], float("nan"))

            # Detect and handle numeric fractional year timestamps
            ts_series = out[ts_col]
            non_null_ts = ts_series[ts_series.notna()]

            # Check if all non-null timestamps are numeric
            is_numeric = pd.to_numeric(non_null_ts, errors="coerce").notna().all() if len(non_null_ts) > 0 else False
            if normalize_years and pd.to_numeric(non_null_ts, errors="coerce").notna().any():
                out[ts_col] = ts_series.map(_year_as_iso)
                is_numeric = False

            if is_numeric:
                # Convert to numeric
                numeric_ts = pd.to_numeric(ts_series, errors="coerce")
                non_null_numeric = numeric_ts[numeric_ts.notna()]

                # Check if values look like fractional years (reasonable year range: 1800-2200)
                if len(non_null_numeric) > 0:
                    min_val = non_null_numeric.min()
                    max_val = non_null_numeric.max()

                    if 1800 <= min_val <= 2200 and 1800 <= max_val <= 2200:
                        # Treat as fractional years - keep as numeric for sorting
                        # AutoGluon will handle conversion when loading the data
                        log.info(
                            "Timestamp column %r contains numeric values in year range [%.2f, %.2f]; "
                            "treating as fractional years (kept as numeric for sorting).",
                            ts_col,
                            min_val,
                            max_val,
                        )
                        out[ts_col] = numeric_ts
                    else:
                        # Numeric but not in year range - likely Unix timestamps or invalid
                        raise ValueError(
                            f"Column {ts_col!r} contains numeric values outside the fractional year range "
                            f"(1800-2200): min={min_val:.2f}, max={max_val:.2f}. "
                            "If these are Unix timestamps, convert them to ISO date strings upstream. "
                            "If these are fractional years, ensure values are in a reasonable range."
                        )
                else:
                    # All nulls, let pd.to_datetime handle it
                    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce", utc=True, format="mixed").dt.tz_localize(
                        None
                    )
            else:
                # Not all numeric - use standard datetime parsing.
                # utc=True normalizes tz-aware strings (e.g. ISO 8601 with Z suffix) to UTC
                # before tz_localize(None) strips timezone info, producing tz-naive datetime64[ns].
                # This prevents pandas from writing tz-aware strings to CSV that AutoGluon
                # cannot read back as datetime64 via TimeSeriesDataFrame.from_data_frame().
                out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce", utc=True, format="mixed").dt.tz_localize(
                    None
                )

            if out[id_col].isna().any():
                raise ValueError(
                    f"Column {id_col!r} contains null values. Fix the input data; do not drop rows here, "
                    "as that can break regular frequency expected by AutoGluon TimeSeries."
                )
            bad_ts = int(out[ts_col].isna().sum())
            if bad_ts:
                raise ValueError(
                    f"Column {ts_col!r} has {bad_ts} value(s) that could not be parsed as datetimes. "
                    "Fix the input data. Dropping those rows would create irregular series and can break "
                    "AutoGluon frequency inference (set TimeSeriesPredictor(freq=...) or regularize upstream)."
                )

            # Keep the last occurrence in file order before sorting the unique keys.
            before_dedupe = len(out)
            if deduplicate:
                out = out.drop_duplicates(subset=[id_col, ts_col], keep="last")
                out = out.sort_values(by=[id_col, ts_col])
            dropped_dupes = before_dedupe - len(out)
            if dropped_dupes:
                log.info(
                    "Timeseries cleansing: dropped %s duplicate rows on (%s, %s), keep=last.",
                    dropped_dupes,
                    id_col,
                    ts_col,
                )

            rows_out = len(out)
            log.info("Timeseries cleansing: rows in=%s out=%s (target NaNs retained for AutoGluon).", rows_in, rows_out)
            if rows_out == 0:
                raise ValueError("After removing duplicate (id, timestamp) pairs, the dataset has no rows left.")

            return out.reset_index(drop=True)

        status.record(
            "prepare_data",
            "running",
            metrics={"source": f"s3://{bucket_name}/{file_key}"},
        )
        sampling_report = {}
        df = load_timeseries_data_truncate(
            bucket_name,
            file_key,
            MAX_SIZE_BYTES,
            PANDAS_CHUNK_SIZE,
            truncation_report=sampling_report,
        )

        # Reject collision with reserved synthetic-ID column (CWE-20)
        # Check if any user-specified column uses the reserved name before we try to inject it.
        reserved_columns = {timestamp_column, target}
        if id_column:
            reserved_columns.add(id_column)
        if SYNTHETIC_ITEM_ID_COLUMN in reserved_columns:
            raise ValueError(
                f"Column name {SYNTHETIC_ITEM_ID_COLUMN!r} is reserved for synthetic ID injection. "
                f"Please rename your timestamp, target, or id_column to avoid collision."
            )

        uses_synthetic_id = False
        if id_column == "":
            # Two-column mode: dataset must have exactly timestamp + target columns.
            required_columns = {timestamp_column, target}
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(
                    f"Missing required columns in dataset: {missing_columns}. Available columns: {list(df.columns)}"
                )
            if len(df.columns) != 2:
                raise ValueError(
                    f"When id_column is not provided, the dataset must have exactly 2 columns "
                    f"(timestamp + target), but found {len(df.columns)} columns: {list(df.columns)}. "
                    f"Provide id_column to identify the series column for datasets with more than 2 columns."
                )
            # Check for collision with existing columns in the dataset
            if SYNTHETIC_ITEM_ID_COLUMN in df.columns:
                raise ValueError(
                    f"Dataset already contains a column named {SYNTHETIC_ITEM_ID_COLUMN!r}. "
                    f"This name is reserved for synthetic ID injection. Please rename the existing column."
                )
            df[SYNTHETIC_ITEM_ID_COLUMN] = SYNTHETIC_ITEM_ID_VALUE
            id_column = SYNTHETIC_ITEM_ID_COLUMN
            uses_synthetic_id = True
            logger.info(
                "Two-column dataset detected; injected synthetic item ID column %r with value %r.",
                SYNTHETIC_ITEM_ID_COLUMN,
                SYNTHETIC_ITEM_ID_VALUE,
            )
        else:
            required_columns = {id_column, timestamp_column, target}
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(
                    f"Missing required columns in dataset: {missing_columns}. Available columns: {list(df.columns)}"
                )

        if len(df) == 0:
            raise ValueError(
                "The loaded dataset has no data rows. Provide at least one row per time series "
                f"with columns {sorted({id_column, timestamp_column, target})}."
            )

        df = _clean_timeseries_dataframe(df, id_column, timestamp_column, logger)
        stringify_mixed_object_columns(df)
        _log_dataset_stats("loaded (after cleansing)", df)

        n_valid = len(df)
        if n_valid < MIN_VALID_RECORDS_AFTER_CLEANSING:
            raise ValueError(
                f"After data cleansing, only {n_valid} valid record(s) remain; "
                f"at least {MIN_VALID_RECORDS_AFTER_CLEANSING} are required for AutoML training. "
                "Provide a larger dataset or fix invalid timestamps, null ids, and duplicate keys."
            )

        status.record(
            "prepare_data",
            "completed",
            metrics={
                "rows": n_valid,
                "sampled_rows": n_valid,
                "sampled_in_memory_bytes": int(df.memory_usage(deep=True).sum()),
                "sampled_buffer_estimated_bytes": sampling_report["sampled_buffer_estimated_bytes"],
                "sample_cap_bytes": MAX_SIZE_BYTES,
                "sample_cap_reached": bool(sampling_report.get("cap_reached")),
                "sampling_method": "last_values_per_series",
                "preset": preset,
                "source_rows_seen": sampling_report["source_rows_seen"],
                "input_complete": True,
            },
        )
        status.record("split_and_export", "started")

        if not sampled_test_dataset.uri or not sampled_test_dataset.uri.endswith(".parquet"):
            sampled_test_dataset.uri = (sampled_test_dataset.uri or "sampled_test_dataset") + ".parquet"

        # Create workspace datasets directory (use validated resolved path)
        datasets_dir = workspace_path_resolved / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)

        # Stable ordering for downstream I/O (redundant if cleanse already sorted; kept for clarity)
        df = df.sort_values(by=[id_column, timestamp_column]).reset_index(drop=True)

        def _early_late_split(group: pd.DataFrame, early_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
            """Split one series by time: first ``early_fraction`` of rows (by time) vs remainder.

            Ensures at least one row in each side when len >= 2. A single-row series is kept
            entirely in the early (train/selection) part.
            """
            g = group.sort_values(by=timestamp_column)
            n = len(g)
            if n == 0:
                return g.iloc[:0].copy(), g.iloc[:0].copy()
            if n == 1:
                return g.copy(), g.iloc[:0].copy()
            split_idx = int(n * early_fraction)
            split_idx = max(1, min(split_idx, n - 1))
            return g.iloc[:split_idx].copy(), g.iloc[split_idx:].copy()

        def _concat_sorted(parts: list, sort_by: list) -> pd.DataFrame:
            if not parts:
                return pd.DataFrame(columns=df.columns)
            out = pd.concat(parts, ignore_index=True)
            return out.sort_values(by=sort_by).reset_index(drop=True)

        has_user_test_data = bool(test_data_file_key)

        if has_user_test_data:
            test_data_source = test_data_source_uri(test_data_bucket_name, test_data_file_key)

            # Download user test data from S3 (AC5: distinguishable error message)
            truncation_report = {}
            try:
                user_test_df = load_timeseries_data_truncate(
                    test_data_bucket_name,
                    test_data_file_key,
                    TEST_DATA_MAX_SIZE_BYTES,
                    PANDAS_CHUNK_SIZE,
                    truncation_report=truncation_report,
                    fail_on_partial_read=True,
                )
            except Exception as e:
                raise test_data_load_error(test_data_source, e) from e

            # Validate non-empty (AC4). Runs before the column checks: a header-only CSV
            # comes back with no columns at all, so "no data rows" is the accurate report.
            raise_if_test_data_empty(len(user_test_df), test_data_source)

            # Validate required columns. In synthetic-id mode the user cannot supply the
            # reserved id column -- the training path above rejects it by name -- so
            # require only the real columns and inject the id afterwards.
            required_columns = {timestamp_column, target}
            if not uses_synthetic_id:
                required_columns.add(id_column)
            missing_columns = required_columns - set(user_test_df.columns)
            if missing_columns:
                raise ValueError(
                    f"Missing required columns in test dataset: {missing_columns}. "
                    f"Available columns: {list(user_test_df.columns)}"
                )
            if uses_synthetic_id:
                user_test_df[SYNTHETIC_ITEM_ID_COLUMN] = SYNTHETIC_ITEM_ID_VALUE

            if truncation_report.get("truncated"):
                logger.warning(
                    "Test dataset %s was truncated to %s row(s), retaining the newest timestamps per series. "
                    "Evaluation covers only these observations; see the component status sampling profile.",
                    test_data_source,
                    len(user_test_df),
                )
                status.record(
                    "split_and_export",
                    "running",
                    metrics={
                        "truncated": True,
                        "test_rows": len(user_test_df),
                        "max_size_bytes": TEST_DATA_MAX_SIZE_BYTES,
                    },
                )

            # Fail fast on covariates AutoGluon will demand at predict time, hours into the run.
            missing_covariates = [c for c in (known_covariates_names or []) if c not in set(user_test_df.columns)]
            if missing_covariates:
                raise ValueError(
                    f"Missing known covariate column(s) in test dataset: {missing_covariates}. "
                    f"Available columns: {list(user_test_df.columns)}"
                )

            # Apply same cleansing
            user_test_df = _clean_timeseries_dataframe(user_test_df, id_column, timestamp_column, logger)

            if not user_test_df[target].notna().any():
                raise ValueError(
                    f"Test dataset has no observed values in target column {target!r}. "
                    f"Source: {test_data_source}. Provide at least one non-null target for evaluation."
                )

            # A test set that shares no series with the training data cannot be scored:
            # every forecast would be for an item the predictor never saw.
            # Compare as strings: pandas infers dtypes per file, so a training frame with
            # mixed ids (object) and a numeric-only test frame (int64) would otherwise
            # look disjoint and reject a perfectly valid test set.
            train_item_ids = {str(item_id) for item_id, _ in df.groupby(id_column, sort=False)}
            test_series_lengths = {
                str(item_id): len(series_df) for item_id, series_df in user_test_df.groupby(id_column, sort=False)
            }
            unknown_item_ids = sorted(i for i in test_series_lengths if i not in train_item_ids)
            if len(unknown_item_ids) == len(test_series_lengths):
                raise ValueError(
                    f"Test dataset shares no {id_column!r} values with the training data "
                    f"(test ids: {unknown_item_ids[:10]}). Provide a test dataset for the same time series."
                )
            if unknown_item_ids:
                # Counts only: a series id can be an email address or a customer number.
                logger.warning(
                    "Test dataset contains %s %r value(s) absent from the training data. "
                    "Those series cannot be forecast and are expected to be skipped downstream.",
                    len(unknown_item_ids),
                    id_column,
                )

            # AutoGluon reserves the last prediction_length steps of each series as ground
            # truth and needs at least one historical step before them, so a series of
            # exactly prediction_length rows fails in leaderboard()/evaluate() -- hours
            # into the run -- rather than here.
            short_series = sorted(
                item_id for item_id, length in test_series_lengths.items() if length <= prediction_length
            )
            if short_series:
                raise ValueError(
                    f"Test dataset series too short for prediction_length ({prediction_length}): "
                    f"{short_series[:10]}. Each series needs more than {prediction_length} row(s) -- the "
                    "forecast horizon plus at least one historical step -- to be evaluated."
                )

            # Write user test data to artifact
            stringify_mixed_object_columns(user_test_df)
            user_test_df.to_parquet(sampled_test_dataset.path, index=False)

            # Skip primary temporal split -- use ALL data for secondary split
            selection_parts = []
            extra_parts = []
            for _, series_df in df.groupby(id_column, sort=False):
                sel, ext = _early_late_split(series_df, selection_train_size)
                selection_parts.append(sel)
                extra_parts.append(ext)

            selection_train_df = _concat_sorted(selection_parts, [id_column, timestamp_column])
            extra_train_df = _concat_sorted(extra_parts, [id_column, timestamp_column])

            if len(selection_train_df) == 0:
                raise ValueError(
                    "Secondary split produced an empty selection-train dataset. "
                    "Increase rows per time series and/or selection_train_size."
                )

            # Extract tail rows from test data for preview
            test_data_for_sample = user_test_df

            split_config_out = {
                "test_size": 0.0,
                "selection_train_size": selection_train_size,
            }

        else:
            test_size = DEFAULT_TEST_SIZE

            train_parts: list = []
            test_parts: list = []
            for _, series_df in df.groupby(id_column, sort=False):
                tr, te = _early_late_split(series_df, 1.0 - test_size)
                train_parts.append(tr)
                test_parts.append(te)

            train_df = _concat_sorted(train_parts, [id_column, timestamp_column])
            test_df = _concat_sorted(test_parts, [id_column, timestamp_column])

            selection_parts: list = []
            extra_parts: list = []
            for _, series_train in train_df.groupby(id_column, sort=False):
                sel, ext = _early_late_split(series_train, selection_train_size)
                selection_parts.append(sel)
                extra_parts.append(ext)

            selection_train_df = _concat_sorted(selection_parts, [id_column, timestamp_column])
            extra_train_df = _concat_sorted(extra_parts, [id_column, timestamp_column])

            # Validate split outputs:
            if len(train_df) == 0:
                raise ValueError(
                    "Primary temporal split produced no train rows. The dataset may be too small for "
                    "the configured splits. Add more rows per time series, or reduce test_size "
                    f"(default is {DEFAULT_TEST_SIZE})."
                )
            if len(selection_train_df) == 0:
                raise ValueError(
                    "Secondary split produced an empty selection-train dataset; "
                    "models_selection_train_dataset.parquet would be empty and downstream training would fail. "
                    "Increase rows per time series and/or selection_train_size, or reduce test_size so "
                    "each series has enough train rows for the selection segment."
                )

            # Save test dataset to artifact
            test_df.to_parquet(sampled_test_dataset.path, index=False)

            test_data_for_sample = test_df

            split_config_out = {
                "test_size": test_size,
                "selection_train_size": selection_train_size,
            }

        minimum_selection_rows = max(prediction_length + 1, 5) + prediction_length
        longest_selection_series = max(
            (len(series) for _, series in selection_train_df.groupby(id_column, sort=False)), default=0
        )
        if longest_selection_series < minimum_selection_rows:
            raise ValueError(
                f"Selection-train has no series with at least {minimum_selection_rows} observations required "
                f"for prediction_length={prediction_length} and AutoGluon's internal validation; "
                f"the longest has {longest_selection_series}. Increase preset to retain more history, "
                "increase selection_train_size, or provide longer series."
            )

        _log_dataset_stats("split: selection_train", selection_train_df)
        _log_dataset_stats("split: extra_train", extra_train_df)
        _log_dataset_stats(
            "split: test (user-provided)" if has_user_test_data else "split: test",
            test_data_for_sample,
        )

        # Common post-split: write selection-train and extra-train data to workspace as
        # Snappy-compressed Parquet (typed + compressed, materially smaller on disk than CSV).
        selection_path = datasets_dir / "models_selection_train_dataset.parquet"
        extra_path = datasets_dir / "extra_train_dataset.parquet"
        selection_train_df.to_parquet(selection_path, index=False)
        extra_train_df.to_parquet(extra_path, index=False)

        split_export_metrics = {
            "test_size": split_config_out["test_size"],
            "selection_train_size": selection_train_size,
            "selection_train_rows": len(selection_train_df),
            "extra_train_rows": len(extra_train_df),
            "test_rows": len(test_data_for_sample),
            "selection_train_disk_bytes": selection_path.stat().st_size,
            "extra_train_disk_bytes": extra_path.stat().st_size,
        }
        if has_user_test_data:
            split_export_metrics["user_test_source"] = test_data_source
            split_export_metrics["test_rows"] = len(user_test_df)
            split_export_metrics["truncated"] = bool(truncation_report.get("truncated"))

        status.record("split_and_export", "completed", metrics=split_export_metrics)

        # Extract tail rows for downstream preview (ISO timestamps when supported; JSON to avoid NaN issues)
        sample_tail = test_data_for_sample.tail(min(5, len(test_data_for_sample)))
        if hasattr(sample_tail, "to_dict"):
            from kfp_components.components.training.automl.shared.timeseries_notebook_utils import (
                _json_records,
            )

            sample_rows = json.dumps(_json_records(sample_tail))
        else:
            sample_rows = sample_tail.to_json(orient="records")

        def _series_profile(data, source_series=None):
            """Record retained row counts and normalized time ranges without expanding the component API."""
            return [
                {
                    "series_id": str(item_id),
                    "rows": len(series),
                    "timestamp_min": str(series[timestamp_column].min()),
                    "timestamp_max": str(series[timestamp_column].max()),
                    **(source_series or {}).get(str(item_id), {}),
                }
                for item_id, series in data.groupby(id_column, sort=False)
            ]

        profile = {
            "sampling_method": "last_values_per_series",
            "source_rows_seen": sampling_report["source_rows_seen"],
            "input_complete": True,
            "datasets": {
                "retained": _series_profile(df, sampling_report["source_series"]),
                "selection_train": _series_profile(selection_train_df),
                "extra_train": _series_profile(extra_train_df),
                "test": _series_profile(
                    test_data_for_sample,
                    truncation_report["source_series"] if has_user_test_data else sampling_report["source_series"],
                ),
            },
        }
        profile_name = "series_sampling_profile.json"
        status_dir = Path(component_status.path)
        status_dir.mkdir(parents=True, exist_ok=True)
        (status_dir / profile_name).write_text(json.dumps(profile), encoding="utf-8")
        status.set_metadata(
            sampling_profile={
                **profile,
                "profile_file": profile_name,
                "series_counts": {name: len(entries) for name, entries in profile["datasets"].items()},
                "profiles_truncated": {
                    name: len(entries) > MAX_SERIES_PROFILE for name, entries in profile["datasets"].items()
                },
                "datasets": {name: entries[:MAX_SERIES_PROFILE] for name, entries in profile["datasets"].items()},
            }
        )
        sample_config = {
            "sampling_method": "last_values_per_series",
            "total_rows_loaded": len(df),
            "sampled_rows": len(df),
            "source_rows_seen": sampling_report["source_rows_seen"],
        }

        logger.info(
            "Timeseries loader: %s rows from s3://%s/%s; split selection=%s extra=%s test=%s",
            len(df),
            bucket_name,
            file_key,
            len(selection_train_df),
            len(extra_train_df),
            len(test_data_for_sample),
        )

        return NamedTuple(
            "outputs",
            sample_config=dict,
            split_config=dict,
            sample_rows=str,
            models_selection_train_data_path=str,
            extra_train_data_path=str,
            effective_id_column=str,
            uses_synthetic_id=bool,
        )(
            sample_config=sample_config,
            split_config=split_config_out,
            sample_rows=sample_rows,
            models_selection_train_data_path=str(selection_path),
            extra_train_data_path=str(extra_path),
            effective_id_column=id_column,
            uses_synthetic_id=uses_synthetic_id,
        )


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        timeseries_data_loader,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
