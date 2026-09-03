from typing import NamedTuple

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

    This component loads time series data from S3, samples it (up to 100 MB),
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

    The test set is written to S3 artifact, while train CSVs are written
    to the PVC workspace for sharing across pipeline steps.

    After cleansing, at least **100** valid records must remain; otherwise the component
    fails with a clear error so downstream AutoGluon training does not run on datasets too
    small to split reliably.

    Args:
        file_key: S3 object key of the CSV file containing time series data.
        bucket_name: S3 bucket name containing the file.
        workspace_path: PVC workspace directory where train CSVs will be written.
        target: Name of the target column to forecast.
        id_column: Name of the column identifying each time series (item_id). Pass an empty
            string ("") for single-series two-column datasets (timestamp + target only);
            the loader will inject a synthetic ID column (__synthetic_item_id) with value "item_0".
        timestamp_column: Name of the timestamp/datetime column.
        sampled_test_dataset: Output dataset artifact for the test split.
        component_status: Output artifact containing stage-level progress tracking for this component.
        selection_train_size: Fraction of train portion for model selection (default: 0.3).

    Returns:
        NamedTuple: sample_config, split_config, sample_rows, models_selection_train_data_path, extra_train_data_path.
    """
    import io
    import json
    import logging
    import os
    from pathlib import Path

    import boto3
    import pandas as pd

    logger = logging.getLogger(__name__)

    MAX_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB limit in bytes
    MIN_VALID_RECORDS_AFTER_CLEANSING = 100
    PANDAS_CHUNK_SIZE = 10000  # Rows per batch for streaming read
    DEFAULT_TEST_SIZE = 0.2

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

    from kfp_components.components.training.automl.shared.component_status import ComponentStatusTracker

    status = ComponentStatusTracker(component_status.path, "timeseries_data_loader")
    with status:
        status.set_metadata(display_name="Timeseries Data Loader Status")
        component_status.metadata["display_name"] = "Timeseries Data Loader Status"
        status.record("prepare_data", "started")

        def get_s3_client(verify=True):
            """Create and return an S3 client using credentials from environment variables."""
            access_key = os.environ.get("AWS_ACCESS_KEY_ID")
            secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
            endpoint_url = os.environ.get("AWS_S3_ENDPOINT")
            region_name = os.environ.get("AWS_DEFAULT_REGION")

            if (access_key and not secret_key) or (secret_key and not access_key):
                raise ValueError(
                    "S3 credentials misconfigured: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must either "
                    "both be set and non-empty, or both be unset. "
                    "Check the Kubernetes secret or environment configuration."
                )
            if not access_key and not secret_key:
                raise ValueError(
                    "S3 credentials missing: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be provided via "
                    "a Kubernetes secret or environment configuration when using s3:// dataset URIs."
                )

            if not endpoint_url:
                raise ValueError(
                    "S3 credentials missing: AWS_S3_ENDPOINT must be provided via "
                    "a Kubernetes secret or environment configuration."
                )

            return boto3.client(
                "s3",
                endpoint_url=endpoint_url,
                region_name=region_name,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                verify=verify,
            )

        def load_timeseries_data_truncate(bucket_name, file_key, max_size_bytes, chunk_size):
            """Load time series CSV from S3, truncating to max_size_bytes while preserving order."""
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
            text_stream = io.TextIOWrapper(response["Body"], encoding="utf-8")

            chunk_list = []
            accumulated_size = 0
            total_rows_read = 0

            try:
                for chunk_df in pd.read_csv(text_stream, chunksize=chunk_size):
                    chunk_memory = chunk_df.memory_usage(deep=True).sum()

                    if accumulated_size + chunk_memory > max_size_bytes:
                        remaining_bytes = max_size_bytes - accumulated_size
                        if remaining_bytes <= 0:
                            break
                        bytes_per_row = chunk_memory / len(chunk_df) if len(chunk_df) > 0 else 0
                        if bytes_per_row > 0:
                            rows_to_take = int(remaining_bytes / bytes_per_row)
                            if rows_to_take > 0:
                                chunk_df = chunk_df.head(rows_to_take)
                                chunk_list.append(chunk_df)
                                total_rows_read += len(chunk_df)
                        break

                    chunk_list.append(chunk_df)
                    accumulated_size += chunk_memory
                    total_rows_read += len(chunk_df)

                    if accumulated_size >= max_size_bytes:
                        break

            except Exception as e:
                if not chunk_list:
                    raise ValueError(f"Error reading CSV from S3: {str(e)}") from e

            if not chunk_list:
                raise ValueError("No data was loaded from S3. The file may be empty or inaccessible.")

            logger.debug(
                "S3 chunk read: %s rows (~%.2f MB)",
                total_rows_read,
                accumulated_size / (1024**2),
            )
            return pd.concat(chunk_list, ignore_index=True)

        def _clean_timeseries_dataframe(data, id_col, ts_col, log):
            """Prepare panel data without dropping rows for missing targets (AutoGluon handles NaNs).

            Per time-series practice, **do not** drop rows for null/NaN targets or non-finite values
            after mapping ``+/-inf`` to NaN: removing observations creates irregular grids and breaks frequency
            inference unless callers set ``freq`` explicitly. See AutoGluon TimeSeries missing-value
            handling per model family.

            This step: replace ``+/-inf`` with NaN; parse timestamps and **fail** if any are invalid;
            **fail** if any ``id_col`` or timestamp is null; ``drop_duplicates`` on ``(id_col, ts_col)``
            (keep last) only for true duplicate keys.
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
                    out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce", utc=True).dt.tz_localize(None)
            else:
                # Not all numeric - use standard datetime parsing.
                # utc=True normalizes tz-aware strings (e.g. ISO 8601 with Z suffix) to UTC
                # before tz_localize(None) strips timezone info, producing tz-naive datetime64[ns].
                # This prevents pandas from writing tz-aware strings to CSV that AutoGluon
                # cannot read back as datetime64 via TimeSeriesDataFrame.from_data_frame().
                out[ts_col] = pd.to_datetime(out[ts_col], errors="coerce", utc=True).dt.tz_localize(None)

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

            # Sort by (id, timestamp) BEFORE deduplication so that keep="last" means
            # "keep the last row in chronological order" (after sorting), not "keep the last row in file order".
            # This ensures we retain the chronologically latest observation for each (id, timestamp) pair.
            out = out.sort_values(by=[id_col, ts_col])
            before_dedupe = len(out)
            out = out.drop_duplicates(subset=[id_col, ts_col], keep="last")
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
        df = load_timeseries_data_truncate(bucket_name, file_key, MAX_SIZE_BYTES, PANDAS_CHUNK_SIZE)

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

        n_valid = len(df)
        if n_valid < MIN_VALID_RECORDS_AFTER_CLEANSING:
            raise ValueError(
                f"After data cleansing, only {n_valid} valid record(s) remain; "
                f"at least {MIN_VALID_RECORDS_AFTER_CLEANSING} are required for AutoML training. "
                "Provide a larger dataset or fix invalid timestamps, null ids, and duplicate keys."
            )

        status.record("prepare_data", "completed", metrics={"rows": n_valid})
        status.record("split_and_export", "started")

        # Create workspace datasets directory (use validated resolved path)
        datasets_dir = workspace_path_resolved / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)

        # Stable ordering for downstream I/O (redundant if cleanse already sorted; kept for clarity)
        df = df.sort_values(by=[id_column, timestamp_column]).reset_index(drop=True)

        test_size = DEFAULT_TEST_SIZE

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
                "models_selection_train_dataset.csv would be empty and downstream training would fail. "
                "Increase rows per time series and/or selection_train_size, or reduce test_size so "
                "each series has enough train rows for the selection segment."
            )

        # Save test dataset to artifact
        test_df.to_csv(sampled_test_dataset.path, index=False)

        selection_path = datasets_dir / "models_selection_train_dataset.csv"
        extra_path = datasets_dir / "extra_train_dataset.csv"

        selection_train_df.to_csv(selection_path, index=False)
        extra_train_df.to_csv(extra_path, index=False)

        status.record(
            "split_and_export",
            "completed",
            metrics={"test_size": test_size, "selection_train_size": selection_train_size},
        )

        logger.info(
            "Timeseries loader: %s rows from s3://%s/%s; split selection=%s extra=%s test=%s",
            len(df),
            bucket_name,
            file_key,
            len(selection_train_df),
            len(extra_train_df),
            len(test_df),
        )

        # Create sample config and split config
        sample_config = {"sampling_method": "first_n_rows", "total_rows_loaded": len(df), "sampled_rows": len(df)}

        split_config = {
            "test_size": test_size,
            "selection_train_size": selection_train_size,
        }

        # Sample rows for downstream use (ISO timestamps when supported; JSON string to avoid NaN issues)
        sample_tail = test_df.tail(min(5, len(test_df)))
        if hasattr(sample_tail, "to_dict"):
            from kfp_components.components.training.automl.shared.timeseries_notebook_utils import (
                _json_records,
            )

            sample_rows = json.dumps(_json_records(sample_tail))
        else:
            sample_rows = sample_tail.to_json(orient="records")

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
            split_config=split_config,
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
