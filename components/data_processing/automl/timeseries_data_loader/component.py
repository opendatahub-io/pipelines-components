from typing import NamedTuple

from kfp import dsl
from kfp_components.utils.consts import AUTOML_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=AUTOML_IMAGE,  # noqa: E501
)
def timeseries_data_loader(
    file_key: str,
    bucket_name: str,
    workspace_path: str,
    target: str,
    id_column: str,
    timestamp_column: str,
    sampled_test_dataset: dsl.Output[dsl.Dataset],
    data_source: str = "s3",
    pvc_data_path: str = "",
    selection_train_size: float = 0.3,
) -> NamedTuple(
    "outputs",
    sample_config=dict,
    split_config=dict,
    sample_rows=str,
    models_selection_train_data_path=str,
    extra_train_data_path=str,
):
    """Load and split timeseries data from S3 or PVC for AutoGluon training.

    This component loads time series data from S3 or PVC, samples it (up to 100 MB),
    and performs a two-stage **per-series temporal** split for efficient AutoGluon training:
    1. Primary split (default 80/20): for each distinct ``id_column`` value, the earliest
       (1 - test_size) fraction of rows by ``timestamp_column`` goes to the train portion and
       the remainder to the test set (so every series with at least two rows contributes
       holdout data; single-row series stay in train only).
    2. Secondary split (default 30/70 of each series' train rows): early segment to
       selection-train, later segment to extra-train.

    The test set is written to S3 artifact, while train CSVs are written
    to the PVC workspace for sharing across pipeline steps.

    **Data Source Configuration:**
    - When ``data_source="s3"``: reads from S3 using AWS credentials from environment variables
    - When ``data_source="pvc"``: reads from PVC workspace filesystem

    Args:
        file_key: S3 object key of the CSV file (required when data_source="s3").
        bucket_name: S3 bucket name containing the file (required when data_source="s3").
        workspace_path: PVC workspace directory where train CSVs will be written.
        target: Name of the target column to forecast.
        id_column: Name of the column identifying each time series (item_id).
        timestamp_column: Name of the timestamp/datetime column.
        sampled_test_dataset: Output dataset artifact for the test split.
        data_source: Data source type ("s3" or "pvc"). Default is "s3".
        pvc_data_path: Path to CSV file on PVC (required when data_source="pvc").
            Can be absolute (/path/to/file.csv) or relative to workspace_path.
        selection_train_size: Fraction of train portion for model selection (default: 0.3).

    Raises:
        ValueError: If required parameters are missing or invalid.
        FileNotFoundError: If PVC data file not found when data_source="pvc".

    Returns:
        NamedTuple: sample_config, split_config, sample_rows, models_selection_train_data_path, extra_train_data_path.
    """
    import io
    import logging
    import os
    from pathlib import Path

    import boto3
    import pandas as pd

    logger = logging.getLogger(__name__)

    MAX_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB limit in bytes
    PANDAS_CHUNK_SIZE = 10000  # Rows per batch for streaming read
    DEFAULT_TEST_SIZE = 0.2
    VALID_DATA_SOURCES = {"s3", "pvc"}

    # Data source validation
    if data_source not in VALID_DATA_SOURCES:
        raise ValueError(f"data_source must be one of {VALID_DATA_SOURCES}; got {data_source!r}.")

    if data_source == "pvc":
        if not pvc_data_path or not pvc_data_path.strip():
            raise ValueError("pvc_data_path must be provided when data_source='pvc'")

    # Input validation
    for param, value in (
        ("workspace_path", workspace_path),
        ("target", target),
        ("id_column", id_column),
        ("timestamp_column", timestamp_column),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{param} must be a non-empty string.")

    # S3-specific validation (only when using S3)
    if data_source == "s3":
        for param, value in (
            ("bucket_name", bucket_name),
            ("file_key", file_key),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{param} must be a non-empty string when data_source='s3'.")
        if file_key.startswith("/") or file_key.endswith("/") or "//" in file_key:
            raise ValueError("file_key must be a valid S3 object key and must not start/end with '/' or contain '//'.")

    if selection_train_size <= 0 or selection_train_size >= 1:
        raise ValueError("selection_train_size must be in a range 0 to 1.")

    def get_s3_client(verify=True):
        """Create and return an S3 client using credentials from environment variables."""
        access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        endpoint_url = os.environ.get("AWS_S3_ENDPOINT")
        region_name = os.environ.get("AWS_DEFAULT_REGION")

        if (access_key and not secret_key) or (secret_key and not access_key):
            raise ValueError(
                "S3 credentials misconfigured: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must either "
                "both be set and non-empty, or both be unset. Check the Kubernetes secret or environment configuration."
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

    def _load_pvc_timeseries_data(file_path, max_size_bytes, chunk_size):
        """Load timeseries CSV from PVC, truncating to max_size_bytes."""
        with open(file_path, "r", encoding="utf-8") as f:
            chunk_list = []
            accumulated_size = 0
            total_rows_read = 0

            try:
                for chunk_df in pd.read_csv(f, chunksize=chunk_size, low_memory=False):
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
                    raise ValueError(f"Error reading CSV from PVC: {str(e)}") from e

            if not chunk_list:
                raise ValueError("No data was loaded from PVC. The file may be empty or inaccessible.")

            logger.debug(
                "PVC chunk read: %s rows (~%.2f MB)",
                total_rows_read,
                accumulated_size / (1024**2),
            )
            return pd.concat(chunk_list, ignore_index=True)

    # Load data from S3 or PVC based on data_source
    if data_source == "s3":
        df = load_timeseries_data_truncate(bucket_name, file_key, MAX_SIZE_BYTES, PANDAS_CHUNK_SIZE)
        logger.info("Read %d rows from s3://%s/%s", len(df), bucket_name, file_key)

    elif data_source == "pvc":
        # Resolve PVC path (support both absolute and relative paths)
        pvc_file_path = (
            os.path.join(workspace_path, pvc_data_path) if not pvc_data_path.startswith("/") else pvc_data_path
        )
        if not os.path.exists(pvc_file_path):
            raise FileNotFoundError(f"PVC data file not found: {pvc_file_path}")

        logger.info("Loading timeseries data from PVC: %s", pvc_file_path)
        df = _load_pvc_timeseries_data(pvc_file_path, MAX_SIZE_BYTES, PANDAS_CHUNK_SIZE)
        logger.info("Read %d rows from PVC file %s", len(df), pvc_file_path)

    required_columns = {id_column, timestamp_column, target}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in dataset: {missing_columns}. Available columns: {list(df.columns)}"
        )

    if len(df) == 0:
        raise ValueError(
            "The loaded dataset has no data rows. Provide at least one row per time series "
            f"with columns {sorted(required_columns)}."
        )

    # Create workspace datasets directory
    datasets_dir = Path(workspace_path) / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    # Stable ordering for downstream I/O
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

    # Sample row for downstream use (JSON string to avoid NaN issues)
    sample_rows = test_df.tail(min(5, len(test_df))).to_json(orient="records")

    return NamedTuple(
        "outputs",
        sample_config=dict,
        split_config=dict,
        sample_rows=str,
        models_selection_train_data_path=str,
        extra_train_data_path=str,
    )(
        sample_config=sample_config,
        split_config=split_config,
        sample_rows=sample_rows,
        models_selection_train_data_path=str(selection_path),
        extra_train_data_path=str(extra_path),
    )


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        timeseries_data_loader,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
