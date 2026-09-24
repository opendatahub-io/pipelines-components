from typing import NamedTuple, Optional

from kfp import dsl
from kfp_components.utils.consts import AUTOML_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=AUTOML_IMAGE,  # noqa: E501
    install_kfp_package=False,
)
def automl_data_loader(  # noqa: D417
    file_key: str,
    bucket_name: str,
    workspace_path: str,
    label_column: str,
    sampled_test_dataset: dsl.Output[dsl.Dataset],
    component_status: dsl.Output[dsl.Artifact],
    sampling_method: Optional[str] = None,
    task_type: str = "regression",
    split_config: Optional[dict] = None,
    selection_train_size: float = 0.3,
    test_data_bucket_name: str = "",
    test_data_file_key: str = "",
    preset: str = "speed",
) -> NamedTuple(
    "outputs",
    sample_config=dict,
    split_config=dict,
    sample_row=str,
    models_selection_train_data_path=str,
    extra_train_data_path=str,
):
    """AutoML Data Loader component.

    Loads tabular (CSV) data from S3 in batches, sampling up to a preset-dependent
    size budget (``"speed"``: 100 MB, ``"balanced"``: 1 GB, ``"heavy"``: 10 GB), then splits the sampled
    data into test, selection-train, and extra-train sets.

    The component reads data in chunks to efficiently handle large files without
    loading the entire dataset into memory at once. After sampling, it performs
    a two-stage split:

    1. **Primary split** (default 80/20): separates a *test set* (20%, written as
       Parquet to the ``sampled_test_dataset`` artifact) from the *train portion* (80%).

    2. **Secondary split** (default 30/70 of the train portion): produces
       ``models_selection_train_dataset.parquet`` (30%, used for model selection) and
       ``extra_train_dataset.parquet`` (70%, passed to ``refit_full`` as extra data).
       Both are written to the PVC workspace under ``{workspace_path}/datasets/`` as
       Snappy-compressed Parquet so the workspace never holds a full-size CSV copy of
       either split.

    For **regression** tasks the split is random; for **binary** and **multiclass**
    tasks the split is **stratified** by the label column by default.

    Rows with a missing label (NaN / empty in ``label_column``) are dropped after load
    and before splitting, so regression runs do not propagate null targets into splits
    or the ``sample_row`` JSON (stratified sampling already dropped per chunk; this
    applies the same rule to random and first-n-rows paths).

    After cleansing (infinity replacement, duplicate removal, and label drop), at least
    **100** valid records must remain; otherwise the component fails with a clear error
    so downstream AutoML training does not run on datasets too small to split reliably.

    After sampling, **+/- infinity** values in the frame are replaced with **NaN** (same
    idea as AutoAI ``loadXy``), then **full-row duplicates** are dropped before the
    label drop and train/test split.

    Authentication uses AWS-style credentials provided via environment variables
    (e.g. from a Kubernetes secret).

    Args:
        file_key: S3 object key of the CSV file.
        bucket_name: S3 bucket name containing the file.
        workspace_path: PVC workspace directory where train Parquet files will be written.
        label_column: Name of the label/target column in the dataset.
        sampled_test_dataset: Output dataset artifact for the test split.
        component_status: Output artifact containing stage-level progress tracking for this component.
        sampling_method: "first_n_rows", "stratified", or "random"; if None, derived from task_type.
        task_type: "binary", "multiclass", or "regression" (default); used when sampling_method is None.
        split_config: Split configuration dictionary. Available keys: "test_size" (float), "random_state" (int), "stratify" (bool).
        selection_train_size: Fraction of the train portion used for model selection (default 0.3).
        test_data_bucket_name: S3 bucket name for user-provided test dataset (default: empty string).
        test_data_file_key: S3 object key of the user-provided test CSV (default: empty string).
        preset: Training quality tier controlling the sampling size budget. ``"speed"``
            (default) samples up to 100 MB; ``"balanced"`` samples up to 1 GB; and
            ``"heavy"`` samples up to 10 GB. User-provided test datasets are capped at
            50 MB, 100 MB, and 1 GB respectively.

    Raises:
        ValueError: If sampling_method, task_type, or preset is invalid, if required parameters are missing,
            if only one of the ``test_data_*`` pair is set or the test key is not a valid S3 object key,
            if the test dataset is empty or missing the label or a training feature column, or if
            fewer than 100 valid records remain after data cleansing.

    Returns:
        NamedTuple: Contains sample config, split config, a sample row, and paths to
            selection-train and extra-train Parquet files.
    """  # noqa: E501
    import io
    import logging
    import math
    import os
    import tempfile
    import time
    from typing import Any

    import boto3
    import pandas as pd
    from kfp_components.components.training.automl.shared.component_status import ComponentStatusTracker
    from kfp_components.components.training.automl.shared.parquet_utils import stringify_mixed_object_columns
    from kfp_components.components.training.automl.shared.user_test_data import (
        raise_if_test_data_empty,
        report_test_data_truncation,
        resolve_s3_env_credentials,
        test_data_load_error,
        test_data_source_uri,
        validate_s3_env_credentials,
        validate_test_data_params,
    )

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
                "Dataset stats [%s]: rows=%s, columns=%s, size=%s bytes (%.2f MB)",
                name,
                n_rows,
                n_cols,
                n_bytes,
                n_bytes / (1024**2),
            )
        except Exception as e:  # noqa: BLE001 - stats logging must never break the run
            logger.debug("Could not compute dataset stats for %s: %s", name, e)

    VALID_PRESETS = {"speed", "balanced", "heavy"}
    # Sampling budget per quality tier: "speed" stays small for fast runs,
    # "balanced" allows the full supported dataset size.
    PRESET_MAX_SIZE_BYTES = {
        "speed": 100 * 1024 * 1024,  # 100 MB
        "balanced": 1024 * 1024 * 1024,  # 1 GB
        "heavy": 10 * 1024 * 1024 * 1024,  # 10 GB
    }
    PRESET_TEST_DATA_MAX_SIZE_BYTES = {
        "speed": 50 * 1024 * 1024,  # 50 MiB
        "balanced": 100 * 1024 * 1024,  # 100 MiB
        "heavy": 1024 * 1024 * 1024,  # 1 GiB
    }
    MIN_VALID_RECORDS_AFTER_CLEANSING = 100
    PANDAS_CHUNK_SIZE = 10000  # Rows per batch for streaming read
    SAMPLE_COMPACTION_CHUNKS = 10
    # Keep pending input well below the retained sample budget: compaction creates
    # temporary copies of both while concatenating and downsampling.
    SAMPLE_COMPACTION_MAX_PENDING_FRACTION = 0.1
    # The multipart fast path stages only moderately sized source objects on node-local
    # storage. Larger or unknown-size objects retain the streaming path so the sample
    # cap is not mistaken for a source-file disk reservation.
    LOCAL_DOWNLOAD_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
    DEFAULT_RANDOM_STATE = 42
    VALID_SAMPLING_METHODS = {"first_n_rows", "stratified", "random"}
    VALID_TASK_TYPES = {"binary", "multiclass", "regression"}

    if preset not in VALID_PRESETS:
        raise ValueError(f"preset must be one of {sorted(VALID_PRESETS)}; got {preset!r}.")
    MAX_SIZE_BYTES = PRESET_MAX_SIZE_BYTES[preset]
    TEST_DATA_MAX_SIZE_BYTES = PRESET_TEST_DATA_MAX_SIZE_BYTES[preset]

    # Input validation
    for param, value in (
        ("bucket_name", bucket_name),
        ("file_key", file_key),
        ("workspace_path", workspace_path),
        ("label_column", label_column),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{param} must be a non-empty string.")

    if task_type not in VALID_TASK_TYPES:
        raise ValueError(f"task_type must be one of {VALID_TASK_TYPES}; got {task_type!r}.")
    if split_config is not None and not isinstance(split_config, dict):
        raise ValueError("split_config must be a dictionary with possible keys test_size, random_state, stratify.")
    if isinstance(split_config, dict):
        test_size = split_config.get("test_size")
        if test_size is not None and (not isinstance(test_size, (int, float)) or test_size <= 0 or test_size >= 1):
            raise ValueError("split_config['test_size'] must be a number in (0, 1) when provided.")
        random_state = split_config.get("random_state")
        if random_state is not None and (not isinstance(random_state, int)):
            raise ValueError("split_config['random_state'] must be an integer when provided.")
        stratify = split_config.get("stratify")
        if stratify is not None and not isinstance(stratify, bool):
            raise ValueError("split_config['stratify'] must be a boolean when provided.")
    if not isinstance(selection_train_size, (int, float)):
        raise ValueError("selection_train_size must be a numerical value.")
    elif selection_train_size <= 0 or selection_train_size >= 1:
        raise ValueError("selection_train_size must be in a range 0 to 1.")

    test_data_bucket_name, test_data_file_key = validate_test_data_params(test_data_bucket_name, test_data_file_key)

    # Initialize status tracker
    status = ComponentStatusTracker(component_status.path, "automl_data_loader")
    with status:
        status.set_metadata(display_name="Data Loader Status")
        component_status.metadata["display_name"] = "Data Loader Status"
        status.record("prepare_data", "started")

        logger.info(
            "Sampling size budget: preset=%s, max_size=%s bytes (%.0f MB)",
            preset,
            MAX_SIZE_BYTES,
            MAX_SIZE_BYTES / (1024**2),
        )

        if sampling_method is None:
            if task_type in ("binary", "multiclass"):
                sampling_method = "stratified"
            else:
                sampling_method = "random"
            logger.info("Sampling method derived from task_type=%s: using %s", task_type, sampling_method)
        else:
            if sampling_method not in VALID_SAMPLING_METHODS:
                raise ValueError(
                    f"sampling_method must be one of {VALID_SAMPLING_METHODS} or None; got {sampling_method!r}."
                )
            if sampling_method == "stratified" and task_type not in ("binary", "multiclass"):
                raise ValueError(
                    "Stratified sampling is only available when task_type is "
                    "'binary' or 'multiclass' (classification tasks). "
                    f"Got task_type='{task_type}'."
                )
            logger.info("Performing sampling: method=%s", sampling_method)

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

        def _sample_first_n_rows(
            text_stream, chunk_size, max_size_bytes, truncation_report=None, fail_on_partial_read=False
        ):
            """Take rows from the start of the stream until the size limit is reached.

            When ``truncation_report`` is a dict, its ``"truncated"`` key is set to True if
            the size limit stopped the read before the stream was exhausted. The signal is
            conservative: a stream whose rows add up to exactly ``max_size_bytes`` is also
            reported, since the read stops without proving no rows follow.

            ``fail_on_partial_read`` makes a mid-stream error fatal instead of returning the
            rows read so far. Test data must fail closed: a partial holdout set would be
            scored as if it were complete.
            """
            chunk_list = []
            accumulated_size = 0
            sampling_metrics = {"batches_read": 0, "source_rows_read": 0, "sampling_compactions": 0}

            def _mark_truncated():
                if truncation_report is not None:
                    truncation_report["truncated"] = True

            try:
                for chunk_df in pd.read_csv(text_stream, chunksize=chunk_size):
                    sampling_metrics["batches_read"] += 1
                    sampling_metrics["source_rows_read"] += len(chunk_df)
                    chunk_memory = chunk_df.memory_usage(deep=True).sum()

                    if accumulated_size + chunk_memory > max_size_bytes:
                        _mark_truncated()
                        remaining_bytes = max_size_bytes - accumulated_size
                        if remaining_bytes <= 0:
                            # No remaining budget; do not take any more rows
                            break
                        bytes_per_row = chunk_memory / len(chunk_df) if len(chunk_df) > 0 else 0
                        if bytes_per_row > 0:
                            rows_to_take = int(remaining_bytes / bytes_per_row)
                            if rows_to_take > 0:
                                chunk_df = chunk_df.head(rows_to_take)
                                chunk_list.append(chunk_df)
                        break

                    chunk_list.append(chunk_df)
                    accumulated_size += chunk_memory

                    if accumulated_size >= max_size_bytes:
                        _mark_truncated()
                        break
            except MemoryError:
                raise
            except Exception as e:
                if not chunk_list or fail_on_partial_read:
                    raise ValueError(f"Error reading CSV from S3: {str(e)}") from e
                logger.warning("Partial CSV read, keeping the %s chunk(s) read so far: %s", len(chunk_list), e)
                _mark_truncated()

            return pd.concat(chunk_list, ignore_index=True) if chunk_list else pd.DataFrame(), sampling_metrics

        def _sample_stratified(csv_source, chunk_size, max_size_bytes, label_column):
            """Merge batches and subsample proportionally by target column to stay under the size limit."""
            subsampled_data = None
            pending_chunks = []
            pending_memory = 0
            sampling_metrics = {"batches_read": 0, "source_rows_read": 0, "sampling_compactions": 0}

            def compact_pending():
                """Merge pending chunks once, rather than copying the sample per chunk."""
                nonlocal subsampled_data, pending_chunks, pending_memory
                if not pending_chunks:
                    return
                sampling_metrics["sampling_compactions"] += 1
                frames = ([subsampled_data] if subsampled_data is not None else []) + pending_chunks
                combined_data = pd.concat(frames, ignore_index=True)
                pending_chunks = []
                pending_memory = 0
                combined_memory = combined_data.memory_usage(deep=True).sum()
                if combined_memory <= max_size_bytes:
                    subsampled_data = combined_data
                    return
                sampling_frac = max_size_bytes / combined_memory
                subsampled_data = (
                    combined_data.groupby(label_column, group_keys=False)
                    .apply(lambda x: x.sample(frac=sampling_frac, random_state=DEFAULT_RANDOM_STATE))
                    .reset_index(drop=True)
                )

            try:
                for chunk_df in pd.read_csv(csv_source, chunksize=chunk_size):
                    sampling_metrics["batches_read"] += 1
                    sampling_metrics["source_rows_read"] += len(chunk_df)
                    if label_column not in chunk_df.columns:
                        raise ValueError(
                            f"Target column '{label_column}' not found in the dataset. "
                            f"Available columns: {list(chunk_df.columns)}"
                        )
                    chunk_df = chunk_df.dropna(subset=[label_column])
                    if chunk_df.empty:
                        continue

                    pending_chunks.append(chunk_df)
                    pending_memory += chunk_df.memory_usage(deep=True).sum()
                    if (
                        pending_memory >= max_size_bytes * SAMPLE_COMPACTION_MAX_PENDING_FRACTION
                        or len(pending_chunks) >= SAMPLE_COMPACTION_CHUNKS
                    ):
                        compact_pending()

                compact_pending()

            except MemoryError:
                raise
            except Exception as e:
                logger.debug("Error reading CSV and stratified sampling: %s", e, exc_info=True)
                # Preserve rows parsed after the last compaction when recovery is allowed.
                compact_pending()
                if subsampled_data is None or subsampled_data.empty:
                    raise ValueError(f"Error reading CSV from S3: {str(e)}") from e

            if subsampled_data is None:
                return pd.DataFrame(), sampling_metrics
            return (
                subsampled_data.sample(frac=1, random_state=DEFAULT_RANDOM_STATE).reset_index(drop=True),
                sampling_metrics,
            )

        def _sample_random(csv_source, chunk_size, max_size_bytes):
            """Iterate all batches, merge with accumulated data, randomly subsample when over the limit."""
            subsampled_data = None
            pending_chunks = []
            pending_memory = 0
            sampling_metrics = {"batches_read": 0, "source_rows_read": 0, "sampling_compactions": 0}

            def compact_pending():
                """Merge pending chunks once, rather than copying the sample per chunk."""
                nonlocal subsampled_data, pending_chunks, pending_memory
                if not pending_chunks:
                    return
                sampling_metrics["sampling_compactions"] += 1
                frames = ([subsampled_data] if subsampled_data is not None else []) + pending_chunks
                data = pd.concat(frames, ignore_index=True)
                pending_chunks = []
                pending_memory = 0
                combined_memory = data.memory_usage(deep=True).sum()
                if combined_memory <= max_size_bytes:
                    subsampled_data = data
                    return
                sampling_frac = max_size_bytes / combined_memory
                subsampled_data = data.sample(frac=sampling_frac, random_state=DEFAULT_RANDOM_STATE).reset_index(
                    drop=True
                )

            try:
                for chunk_df in pd.read_csv(csv_source, chunksize=chunk_size):
                    sampling_metrics["batches_read"] += 1
                    sampling_metrics["source_rows_read"] += len(chunk_df)
                    pending_chunks.append(chunk_df)
                    pending_memory += chunk_df.memory_usage(deep=True).sum()
                    if (
                        pending_memory >= max_size_bytes * SAMPLE_COMPACTION_MAX_PENDING_FRACTION
                        or len(pending_chunks) >= SAMPLE_COMPACTION_CHUNKS
                    ):
                        compact_pending()

                compact_pending()

                return subsampled_data if subsampled_data is not None else pd.DataFrame(), sampling_metrics

            except MemoryError:
                raise
            except Exception as e:
                # Preserve rows parsed after the last compaction when recovery is allowed.
                compact_pending()
                if subsampled_data is None or subsampled_data.empty:
                    raise ValueError(f"Error reading CSV from S3: {str(e)}") from e
                return subsampled_data, sampling_metrics

        def _sample_csv_source(
            csv_source,
            max_size_bytes,
            sampling_method,
            label_column,
            truncation_report=None,
            fail_on_partial_read=False,
        ):
            """Sample one CSV source with the configured strategy."""
            if sampling_method == "stratified":
                return _sample_stratified(csv_source, PANDAS_CHUNK_SIZE, max_size_bytes, label_column)
            if sampling_method == "random":
                return _sample_random(csv_source, PANDAS_CHUNK_SIZE, max_size_bytes)
            return _sample_first_n_rows(
                csv_source,
                PANDAS_CHUNK_SIZE,
                max_size_bytes,
                truncation_report=truncation_report,
                fail_on_partial_read=fail_on_partial_read,
            )

        def _try_load_local_representative_sample(
            s3_client,
            bucket_name,
            file_key,
            max_size_bytes,
            sampling_method,
            label_column,
            fallback_report=None,
        ) -> tuple[Any, dict[str, Any]] | None:
            """Download and sample a bounded source locally, or return None to stream it."""
            local_path = None
            try:
                from boto3.s3.transfer import TransferConfig

                descriptor, local_path = tempfile.mkstemp(prefix="automl-source-", suffix=".csv")
                os.close(descriptor)
                transfer_started = time.monotonic()
                s3_client.download_file(
                    bucket_name,
                    file_key,
                    local_path,
                    Config=TransferConfig(
                        multipart_threshold=8 * 1024 * 1024,
                        multipart_chunksize=16 * 1024 * 1024,
                        max_concurrency=10,
                        use_threads=True,
                    ),
                )
                download_seconds = round(time.monotonic() - transfer_started, 3)
            except Exception as exc:  # noqa: BLE001 - retain streaming compatibility
                logger.warning("Multipart S3 download unavailable; falling back to streaming read: %s", exc)
                if local_path:
                    try:
                        os.unlink(local_path)
                    except FileNotFoundError:
                        pass
                if fallback_report is not None:
                    fallback_report["multipart_fallback"] = type(exc).__name__
                return None

            try:
                data, sampling_metrics = _sample_csv_source(
                    local_path,
                    max_size_bytes,
                    sampling_method,
                    label_column,
                )
                return data, {
                    "s3_transfer_mode": "multipart_local",
                    "download_seconds": download_seconds,
                    **sampling_metrics,
                }
            finally:
                try:
                    os.unlink(local_path)
                except FileNotFoundError:
                    pass

        def _load_streamed_sample(
            s3_client,
            bucket_name,
            file_key,
            max_size_bytes,
            sampling_method,
            label_column,
            io_metrics,
            truncation_report=None,
            fail_on_partial_read=False,
        ) -> tuple[Any, dict[str, Any]]:
            """Stream one S3 object through the selected batch sampler."""
            from botocore.exceptions import SSLError

            transfer_started = time.monotonic()
            try:
                response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
            except SSLError:
                logger.warning(
                    "SSL error when downloading s3://%s/%s, retrying with verify=False",
                    bucket_name,
                    file_key,
                )
                response = get_s3_client(verify=False).get_object(Bucket=bucket_name, Key=file_key)

            if "source_bytes" not in io_metrics:
                source_bytes = response.get("ContentLength")
                if isinstance(source_bytes, int) and source_bytes >= 0:
                    io_metrics["source_bytes"] = source_bytes

            data, sampling_metrics = _sample_csv_source(
                io.TextIOWrapper(response["Body"], encoding="utf-8"),
                max_size_bytes,
                sampling_method,
                label_column,
                truncation_report=truncation_report,
                fail_on_partial_read=fail_on_partial_read,
            )
            io_metrics.update(sampling_metrics)
            io_metrics["download_and_parse_seconds"] = round(time.monotonic() - transfer_started, 3)
            return data, io_metrics

        def load_data_in_batches(
            s3_client,
            bucket_name,
            file_key,
            max_size_bytes,
            sampling_method,
            label_column,
            truncation_report=None,
            fail_on_partial_read: bool = False,
        ) -> tuple[Any, dict[str, Any]]:
            """Load CSV from S3 in batches and return a sampled dataframe using the chosen strategy.

            ``truncation_report`` is only honoured by the ``first_n_rows`` strategy; the
            subsampling strategies keep a representative sample of the whole stream.

            ``fail_on_partial_read`` makes a mid-stream read error fatal instead of returning
            a partial dataframe.
            """
            if sampling_method == "stratified" and label_column is None:
                raise ValueError("label_column must be provided when sampling_method='stratified'")

            io_metrics = {"s3_transfer_mode": "streaming"}

            # Representative samplers inspect the entire source. For moderately sized
            # objects, stage it locally through boto3's concurrent multipart transfer
            # manager so pandas can parse a local file. Never stage an unbounded
            # object: source files above the node-local budget stay on the streaming
            # path and are still sampled progressively in pandas batches.
            if sampling_method in {"random", "stratified"}:
                try:
                    source_bytes = s3_client.head_object(Bucket=bucket_name, Key=file_key)["ContentLength"]
                    if not isinstance(source_bytes, int) or source_bytes < 0:
                        raise ValueError("head_object returned an invalid ContentLength")
                    io_metrics["source_bytes"] = source_bytes
                except Exception as exc:  # noqa: BLE001 - retain compatibility with S3-compatible stores
                    logger.info("Could not determine S3 object size; using streaming read: %s", exc)
                    io_metrics["multipart_skipped"] = type(exc).__name__
                else:
                    if source_bytes > LOCAL_DOWNLOAD_MAX_BYTES:
                        logger.info(
                            "Source object is %s bytes, above the %s-byte local download budget; using streaming read.",
                            source_bytes,
                            LOCAL_DOWNLOAD_MAX_BYTES,
                        )
                        io_metrics["multipart_skipped"] = "source_too_large"
                    else:
                        local_result = _try_load_local_representative_sample(
                            s3_client,
                            bucket_name,
                            file_key,
                            max_size_bytes,
                            sampling_method,
                            label_column,
                            fallback_report=io_metrics,
                        )
                        if local_result is not None:
                            data, local_metrics = local_result
                            io_metrics.update(local_metrics)
                            return data, io_metrics

            return _load_streamed_sample(
                s3_client,
                bucket_name,
                file_key,
                max_size_bytes,
                sampling_method,
                label_column,
                io_metrics,
                truncation_report=truncation_report,
                fail_on_partial_read=fail_on_partial_read,
            )

        status.record(
            "prepare_data",
            "running",
            metrics={"sampling_method": sampling_method, "source": f"s3://{bucket_name}/{file_key}"},
        )
        s3_client = get_s3_client()
        load_started = time.monotonic()
        sampled_dataframe, load_metrics = load_data_in_batches(
            s3_client,
            bucket_name,
            file_key,
            max_size_bytes=MAX_SIZE_BYTES,
            sampling_method=sampling_method,
            label_column=label_column,
        )
        load_metrics["load_and_sample_seconds"] = round(time.monotonic() - load_started, 3)
        load_metrics["rows_before_cleansing"] = len(sampled_dataframe)

        cleansing_started = time.monotonic()
        if label_column not in sampled_dataframe.columns:
            raise ValueError(
                f"Label column {label_column!r} not found in the dataset. "
                f"Available columns: {list(sampled_dataframe.columns)}"
            )

        sampled_dataframe.replace([math.inf, -math.inf], float("nan"), inplace=True)

        n_before_dedup = len(sampled_dataframe)
        sampled_dataframe.drop_duplicates(inplace=True)
        n_dup_dropped = n_before_dedup - len(sampled_dataframe)
        if n_dup_dropped:
            logger.info("Dropped %s full-row duplicate(s) (%s rows remaining).", n_dup_dropped, len(sampled_dataframe))

        if sampled_dataframe.empty:
            raise ValueError(
                "No valid data rows remain after replacing infinite values and dropping duplicates. "
                "The source CSV may contain only infinite/NaN values or duplicate rows."
            )

        n_before_drop = len(sampled_dataframe)
        sampled_dataframe = sampled_dataframe.dropna(subset=[label_column])
        n_dropped = n_before_drop - len(sampled_dataframe)
        if n_dropped:
            logger.info(
                "Dropped %s row(s) with missing label in column %r before splitting (loaded %s rows, %s remaining).",
                n_dropped,
                label_column,
                n_before_drop,
                len(sampled_dataframe),
            )
        if sampled_dataframe.empty:
            raise ValueError(
                f"No rows remain after removing missing values in label column {label_column!r}. "
                "Ensure the dataset has at least one row with a non-null label (e.g. empty cells in the target column)."
            )

        n_valid = len(sampled_dataframe)
        if n_valid < MIN_VALID_RECORDS_AFTER_CLEANSING:
            raise ValueError(
                f"After data cleansing, only {n_valid} valid record(s) remain; "
                f"at least {MIN_VALID_RECORDS_AFTER_CLEANSING} are required for AutoML training. "
                "Provide a larger dataset or reduce missing labels, duplicates, and invalid values."
            )

        n_samples = n_valid
        logger.info(
            "Read %d rows from s3://%s/%s (sampling_method=%s)",
            n_samples,
            bucket_name,
            file_key,
            sampling_method,
        )
        stringify_mixed_object_columns(sampled_dataframe)
        _log_dataset_stats("loaded (after cleansing)", sampled_dataframe)
        sampled_in_memory_bytes = int(sampled_dataframe.memory_usage(deep=True).sum())
        status.record(
            "prepare_data",
            "completed",
            metrics={
                "rows": n_samples,
                "duplicates_dropped": n_dup_dropped,
                "labels_dropped": n_dropped,
                "source_rows_scanned": load_metrics.get("source_rows_read", 0),
                "sampled_rows": n_samples,
                "sampled_in_memory_bytes": sampled_in_memory_bytes,
                "sample_cap_bytes": MAX_SIZE_BYTES,
                "cleansing_seconds": round(time.monotonic() - cleansing_started, 3),
                **load_metrics,
            },
        )

        status.record("split_and_export", "started")
        split_export_started = time.monotonic()

        # --- Train/test split ---
        from pathlib import Path

        from sklearn.model_selection import train_test_split

        DEFAULT_TEST_SIZE = 0.2
        DEFAULT_SPLIT_RANDOM_STATE = 42

        split_config = split_config or {}
        test_size = split_config.get("test_size", DEFAULT_TEST_SIZE)
        random_state = split_config.get("random_state", DEFAULT_SPLIT_RANDOM_STATE)

        if not sampled_test_dataset.uri or not sampled_test_dataset.uri.endswith(".parquet"):
            sampled_test_dataset.uri = (sampled_test_dataset.uri or "sampled_test_dataset") + ".parquet"

        # Common setup: workspace directory and stratification flag
        datasets_dir = Path(workspace_path) / "datasets"
        datasets_dir.mkdir(parents=True, exist_ok=True)
        stratify_effective = task_type != "regression" and split_config.get("stratify", True)

        has_user_test_data = bool(test_data_file_key)

        if has_user_test_data:
            test_data_source = test_data_source_uri(test_data_bucket_name, test_data_file_key)

            # Download user test data from S3 (AC5: distinguishable error message)
            truncation_report = {}
            test_s3_client = get_s3_client()
            try:
                user_test_df, user_test_load_metrics = load_data_in_batches(
                    test_s3_client,
                    test_data_bucket_name,
                    test_data_file_key,
                    max_size_bytes=TEST_DATA_MAX_SIZE_BYTES,
                    sampling_method="first_n_rows",
                    label_column=label_column,
                    truncation_report=truncation_report,
                    fail_on_partial_read=True,
                )
            except Exception as e:
                raise test_data_load_error(test_data_source, e) from e
            logger.info("User test dataset load metrics: %s", user_test_load_metrics)

            # Validate test data is non-empty (AC4)
            raise_if_test_data_empty(len(user_test_df), test_data_source)

            if truncation_report.get("truncated"):
                report_test_data_truncation(
                    status, logger, test_data_source, len(user_test_df), TEST_DATA_MAX_SIZE_BYTES
                )

            # Validate label column exists in test data
            if label_column not in user_test_df.columns:
                raise ValueError(
                    f"Label column {label_column!r} not found in test dataset. "
                    f"Available columns: {list(user_test_df.columns)}"
                )

            # Fail fast on a train/test schema mismatch: AutoGluon would otherwise only
            # surface it inside predict/evaluate, after the full fit has already run.
            train_feature_columns = [c for c in sampled_dataframe.columns if c != label_column]
            test_columns = set(user_test_df.columns)
            missing_features = [c for c in train_feature_columns if c not in test_columns]
            if missing_features:
                raise ValueError(
                    f"Test dataset is missing feature column(s) present in the training data: "
                    f"{missing_features}. Test data columns: {list(user_test_df.columns)}. "
                    "Provide a test dataset with the same feature columns as the training data."
                )
            extra_features = [c for c in user_test_df.columns if c != label_column and c not in train_feature_columns]
            if extra_features:
                logger.warning(
                    "Test dataset has column(s) not present in the training data: %s. "
                    "They are dropped before evaluation.",
                    extra_features,
                )
                # Drop rather than carry: these columns would otherwise reach the
                # sampled_test_dataset artifact and the notebook sample payload built from
                # user_test_df.head(1), advertising features the predictor cannot accept.
                user_test_df = user_test_df.drop(columns=extra_features)

            # Apply same cleansing as training data
            user_test_df.replace([math.inf, -math.inf], float("nan"), inplace=True)
            user_test_df.drop_duplicates(inplace=True)
            user_test_df = user_test_df.dropna(subset=[label_column])

            if user_test_df.empty:
                raise ValueError(
                    "Test dataset has no valid rows after cleansing "
                    "(infinite value replacement, duplicate removal, label NaN drop). "
                    f"Source: {test_data_source}."
                )

            # Write user test data to the sampled_test_dataset artifact
            stringify_mixed_object_columns(user_test_df)
            user_test_df.to_parquet(sampled_test_dataset.path, index=False)

            # Skip primary holdout -- use all sampled training rows for the secondary split.
            selection_X = sampled_dataframe.drop(columns=[label_column], inplace=False)
            selection_y = sampled_dataframe[label_column]
            test_sample_df = user_test_df
            effective_test_size = 0.0

        else:
            X = sampled_dataframe.drop(columns=[label_column], inplace=False)
            y = sampled_dataframe[label_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                stratify=(y if stratify_effective else None),
                random_state=random_state,
            )

            selection_X = X_train
            selection_y = y_train
            test_sample_df = pd.concat([X_test, y_test], axis=1)
            test_sample_df.to_parquet(sampled_test_dataset.path, index=False)
            effective_test_size = test_size

        X_sel, X_extra, y_sel, y_extra = train_test_split(
            selection_X,
            selection_y,
            test_size=(1 - selection_train_size),
            stratify=(selection_y if stratify_effective else None),
            random_state=random_state,
        )

        X_y_sel = pd.concat([X_sel, y_sel], axis=1)
        X_y_extra = pd.concat([X_extra, y_extra], axis=1)

        _log_dataset_stats("split: selection_train", X_y_sel)
        _log_dataset_stats("split: extra_train", X_y_extra)
        _log_dataset_stats(
            "split: test (user-provided)" if has_user_test_data else "split: test",
            test_sample_df,
        )

        if len(X_y_sel) == 0:
            raise ValueError(
                "Secondary split produced an empty selection-train dataset; "
                "models_selection_train_dataset.parquet would be empty and downstream training would fail. "
                "Increase training data size and/or selection_train_size."
            )

        sample_row = test_sample_df.head(1).to_json(orient="records")

        split_config_out = {
            "test_size": effective_test_size,
            "random_state": random_state,
            "stratify": stratify_effective,
        }

        # Common post-split: write selection-train and extra-train data to workspace as
        # Snappy-compressed Parquet (typed + compressed, materially smaller on disk than CSV).
        models_selection_train_data_path = str(datasets_dir / "models_selection_train_dataset.parquet")
        extra_train_data_path = str(datasets_dir / "extra_train_dataset.parquet")
        X_y_sel.to_parquet(models_selection_train_data_path, index=False)
        X_y_extra.to_parquet(extra_train_data_path, index=False)

        split_export_metrics = {
            "test_size": split_config_out["test_size"],
            "selection_train_size": selection_train_size,
            "stratify": stratify_effective,
            "selection_train_rows": len(X_y_sel),
            "extra_train_rows": len(X_y_extra),
            "test_rows": len(test_sample_df),
            "selection_train_disk_bytes": Path(models_selection_train_data_path).stat().st_size,
            "extra_train_disk_bytes": Path(extra_train_data_path).stat().st_size,
            "split_and_export_seconds": round(time.monotonic() - split_export_started, 3),
        }
        if has_user_test_data:
            split_export_metrics["user_test_source"] = test_data_source
            split_export_metrics["test_rows"] = len(user_test_df)
            split_export_metrics["truncated"] = bool(truncation_report.get("truncated"))

        status.record("split_and_export", "completed", metrics=split_export_metrics)

        return NamedTuple(
            "outputs",
            sample_config=dict,
            split_config=dict,
            sample_row=str,
            models_selection_train_data_path=str,
            extra_train_data_path=str,
        )(
            sample_config={"n_samples": n_samples},
            split_config=split_config_out,
            sample_row=sample_row,
            models_selection_train_data_path=models_selection_train_data_path,
            extra_train_data_path=extra_train_data_path,
        )


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        automl_data_loader,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
