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
    ssl_cert_path: Optional[str] = None,
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

    This component scans the entire CSV from S3 and retains the latest observations
    per series (up to 100 MiB of pandas data for the
    ``"speed"`` preset, up to 1 GiB for ``"balanced"``, and up to 10 GiB for
    ``"quality"``),
    applies light **cleansing** (replace ``+/-inf`` with NaN so AutoGluon can apply its
    own missing-value logic; require parseable timestamps and non-null ids; drop
    duplicate ``(id_column, timestamp_column)`` keys, keeping the last file occurrence).
    Timestamp order determines recency even for unsorted inputs. Small sources stay
    in RAM; larger sources use temporary Parquet files and a SQLite timestamp index
    on the workspace. No second S3 download is required. Source/retained counts and
    timestamp ranges per series are saved as JSONL profiles in ``component_status``.
    The component then performs a two-stage
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

    After cleansing, at least **100** valid records must remain; otherwise the component
    fails with a clear error so downstream AutoGluon training does not run on datasets too
    small to split reliably.

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
        prediction_length: Forecast horizon used downstream (default: 1). Sampling preserves
            enough history for the temporal splits and at least one internal validation
            window when the source provides it. Also validates external test series length.
        known_covariates_names: Covariate columns known in advance downstream (default: none).
            Only used to fail fast when a user-provided test dataset omits one of them.
        test_data_bucket_name: S3 bucket name for user-provided test dataset (default: empty string).
        test_data_file_key: S3 object key of the user-provided test CSV (default: empty string).
        preset: Training quality tier controlling the sampling size budget. ``"speed"``
            (default) samples up to 100 MiB; ``"balanced"`` samples up to 1 GiB; and
            ``"quality"`` samples up to 10 GiB. User-provided test datasets are capped
            at 50 MiB, 100 MiB, and 1 GiB respectively.
        ssl_cert_path: Optional path to a PEM-encoded CA bundle used to verify the S3
            endpoint. The filename extension is not significant; ``.pem`` and ``.crt``
            are both common. It overrides ``AWS_CA_BUNDLE`` for this component; when
            neither is set, the system trust bundle is used. TLS verification is never
            disabled.

    Raises:
        ValueError: If a required parameter is empty or invalid, if only one of the
            ``test_data_*`` pair is set or the test key is not a valid S3 object key, if the
            test dataset is empty, missing required or covariate columns, shares no series
            with the training data, or has a series shorter than ``prediction_length``, or if
            fewer than 100 valid records remain after cleansing.

    Returns:
        NamedTuple: sample_config, split_config, sample_rows, models_selection_train_data_path,
                   extra_train_data_path.
    """
    import io
    import json
    import logging
    import os
    from pathlib import Path

    import boto3
    import pandas as pd

    logger = logging.getLogger(__name__)

    def _log_dataset_stats(name, df):
        """Log the size of a dataframe: rows, columns, and in-memory bytes.

        ``memory_usage(deep=True)`` reports the pandas in-memory footprint, which
        differs from the on-disk CSV size; it is the figure that matters for the
        sampling budget and the pod memory limit.
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
    from kfp_components.components.training.automl.shared.timeseries_sampling import (
        PANDAS_CHUNK_SIZE,
        PRESET_MAX_SIZE_BYTES,
        PRESET_TEST_DATA_MAX_SIZE_BYTES,
        add_split_profile,
        minimum_retained_rows,
        sample_latest_timeseries,
    )
    from kfp_components.components.training.automl.shared.user_test_data import (
        raise_if_test_data_empty,
        report_test_data_truncation,
        resolve_s3_env_credentials,
        test_data_load_error,
        test_data_source_uri,
        validate_s3_env_credentials,
        validate_test_data_params,
    )

    VALID_PRESETS = {"speed", "balanced", "quality"}
    MIN_VALID_RECORDS_AFTER_CLEANSING = 100
    DEFAULT_TEST_SIZE = 0.2

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
    if not 0 < selection_train_size < 1:
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

    if not isinstance(prediction_length, int) or prediction_length <= 0:
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

        effective_ssl_cert_path = (ssl_cert_path or os.environ.get("AWS_CA_BUNDLE") or "").strip() or None

        def get_s3_client():
            """Create and return an S3 client using credentials from environment variables.

            TLS certificate verification is always enabled: the client uses the PEM CA
            bundle from ``ssl_cert_path`` or ``AWS_CA_BUNDLE`` when set and the system
            trust bundle otherwise.
            """
            credentials = resolve_s3_env_credentials()
            validate_s3_env_credentials(credentials)

            return boto3.client(
                "s3",
                endpoint_url=credentials["endpoint_url"],
                region_name=credentials["region_name"],
                aws_access_key_id=credentials["access_key"],
                aws_secret_access_key=credentials["secret_key"],
                verify=effective_ssl_cert_path or True,
            )

        def _clean_timeseries_dataframe(data, id_col, ts_col, log):
            """Prepare panel data without dropping rows for missing targets (AutoGluon handles NaNs).

            Per time-series practice, **do not** drop rows for null/NaN targets or non-finite values
            after mapping ``+/-inf`` to NaN: removing observations creates irregular grids and breaks frequency
            inference unless callers set ``freq`` explicitly. See AutoGluon TimeSeries missing-value
            handling per model family.

            This step: replace ``+/-inf`` with NaN; parse timestamps and **fail** if any are invalid;
            **fail** if any ``id_col`` or timestamp is null. The sampler deduplicates
            keys across the entire source after this normalization.
            """
            if len(data) == 0:
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

            # Deduplication and ordering happen across the complete source in the
            # sampler. Doing this per chunk loses cross-chunk duplicates and order.
            return out.reset_index(drop=True)

        status.record(
            "prepare_data",
            "running",
            metrics={"source": f"s3://{bucket_name}/{file_key}"},
        )
        reserved_columns = {timestamp_column, target, id_column}
        if SYNTHETIC_ITEM_ID_COLUMN in reserved_columns:
            raise ValueError(
                f"Column name {SYNTHETIC_ITEM_ID_COLUMN!r} is reserved for synthetic ID injection. "
                "Please rename your timestamp, target, or id_column to avoid collision."
            )
        uses_synthetic_id = id_column == ""
        if uses_synthetic_id:
            id_column = SYNTHETIC_ITEM_ID_COLUMN

        def _load_latest(bucket, key, budget, profile_name, *, is_test=False):
            from botocore.exceptions import SSLError

            try:
                response = get_s3_client().get_object(Bucket=bucket, Key=key)
            except SSLError as exc:
                certificate_source = effective_ssl_cert_path or "the system trust bundle"
                raise RuntimeError(
                    "S3 TLS certificate validation failed when downloading "
                    f"s3://{bucket}/{key}. Verify that ssl_cert_path="
                    f"{certificate_source!r} is a PEM CA bundle that trusts the S3 endpoint. "
                    "Do not disable certificate verification."
                ) from exc

            def _prepared_chunks(stream):
                try:
                    reader = pd.read_csv(
                        stream,
                        chunksize=PANDAS_CHUNK_SIZE,
                        dtype=None if uses_synthetic_id else {id_column: str},
                    )
                except pd.errors.EmptyDataError:
                    return
                with reader:
                    while True:
                        try:
                            chunk = next(reader)
                        except StopIteration:
                            return
                        except Exception as exc:
                            # A partial scan cannot establish the latest observations.
                            raise ValueError(f"Error reading CSV from S3: {exc}") from exc
                        if not len(chunk):
                            continue
                        required_columns = {timestamp_column, target}
                        if not uses_synthetic_id:
                            required_columns.add(id_column)
                        missing = required_columns - set(chunk.columns)
                        if missing:
                            kind = "test dataset" if is_test else "dataset"
                            raise ValueError(
                                f"Missing required columns in {kind}: {missing}. "
                                f"Available columns: {list(chunk.columns)}"
                            )
                        if uses_synthetic_id:
                            if not is_test and len(chunk.columns) != 2:
                                raise ValueError(
                                    "When id_column is not provided, the dataset must have exactly 2 columns "
                                    f"(timestamp + target), but found {len(chunk.columns)} columns: "
                                    f"{list(chunk.columns)}. Provide id_column for panel datasets."
                                )
                            if SYNTHETIC_ITEM_ID_COLUMN in chunk.columns:
                                raise ValueError(
                                    f"Dataset already contains a column named {SYNTHETIC_ITEM_ID_COLUMN!r}. "
                                    "This name is reserved for synthetic ID injection."
                                )
                            chunk[id_column] = SYNTHETIC_ITEM_ID_VALUE
                        chunk = _clean_timeseries_dataframe(chunk, id_column, timestamp_column, logger)
                        stringify_mixed_object_columns(chunk)
                        yield chunk

            minimum_rows = (
                prediction_length + 1
                if is_test
                else minimum_retained_rows(prediction_length, selection_train_size, bool(test_data_file_key))
            )
            with io.TextIOWrapper(response["Body"], encoding="utf-8") as stream:
                return sample_latest_timeseries(
                    _prepared_chunks(stream),
                    id_column=id_column,
                    timestamp_column=timestamp_column,
                    max_size_bytes=budget,
                    workspace_path=workspace_path_resolved,
                    profile_path=Path(component_status.path) / profile_name,
                    minimum_rows_per_series=minimum_rows,
                )

        df, sampling_report = _load_latest(bucket_name, file_key, MAX_SIZE_BYTES, "training_series_profile.jsonl")
        if len(df) == 0:
            raise ValueError("No data was loaded from S3. The file may be empty or inaccessible.")
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
                "sample_cap_bytes": MAX_SIZE_BYTES,
                "sample_cap_reached": bool(sampling_report.get("cap_reached")),
                **sampling_report,
                "sampling_method": "latest_per_series",
                "preset": preset,
            },
        )
        status.record("split_and_export", "started")

        if not sampled_test_dataset.uri or not sampled_test_dataset.uri.endswith(".parquet"):
            sampled_test_dataset.uri = (sampled_test_dataset.uri or "sampled_test_dataset") + ".parquet"

        # Create workspace datasets directory (use validated resolved path)
        datasets_dir = workspace_path_resolved / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)

        def _temporal_split(frame: pd.DataFrame, early_fraction: float):
            """Split already sorted series in vectorized operations; singletons stay in train."""
            groups = frame.groupby(id_column, sort=False)
            sizes = groups[timestamp_column].transform("size")
            cutoffs = (sizes * early_fraction).astype(int).clip(lower=1)
            cutoffs = cutoffs.where(cutoffs < sizes, (sizes - 1).clip(lower=1))
            early = groups.cumcount() < cutoffs
            return frame[early].copy(), frame[~early].copy()

        has_user_test_data = bool(test_data_file_key)

        if has_user_test_data:
            test_data_source = test_data_source_uri(test_data_bucket_name, test_data_file_key)

            # Download user test data from S3 (AC5: distinguishable error message)
            try:
                user_test_df, truncation_report = _load_latest(
                    test_data_bucket_name,
                    test_data_file_key,
                    TEST_DATA_MAX_SIZE_BYTES,
                    "test_series_profile.jsonl",
                    is_test=True,
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
                report_test_data_truncation(
                    status, logger, test_data_source, len(user_test_df), TEST_DATA_MAX_SIZE_BYTES
                )

            # Fail fast on covariates AutoGluon will demand at predict time, hours into the run.
            missing_covariates = [c for c in (known_covariates_names or []) if c not in set(user_test_df.columns)]
            if missing_covariates:
                raise ValueError(
                    f"Missing known covariate column(s) in test dataset: {missing_covariates}. "
                    f"Available columns: {list(user_test_df.columns)}"
                )

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
            selection_train_df, extra_train_df = _temporal_split(df, selection_train_size)

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

            train_df, test_df = _temporal_split(df, 1.0 - test_size)
            selection_train_df, extra_train_df = _temporal_split(train_df, selection_train_size)

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

        profile_splits = {"selection_train": selection_train_df, "extra_train": extra_train_df}
        if not has_user_test_data:
            profile_splits["test"] = test_data_for_sample
        add_split_profile(
            Path(component_status.path) / sampling_report["series_profile"],
            profile_splits,
            id_column,
            timestamp_column,
        )
        if has_user_test_data:
            add_split_profile(
                Path(component_status.path) / truncation_report["series_profile"],
                {"test": user_test_df},
                id_column,
                timestamp_column,
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

        sample_config = dict(sampling_report)
        if has_user_test_data:
            sample_config["test_sampling"] = truncation_report

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
