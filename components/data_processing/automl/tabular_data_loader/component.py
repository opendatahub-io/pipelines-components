from typing import NamedTuple, Optional

from kfp import dsl
from kfp_components.utils.consts import AUTOML_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=AUTOML_IMAGE,  # noqa: E501
)
def automl_data_loader(  # noqa: D417
    file_key: str,
    bucket_name: str,
    workspace_path: str,
    label_column: str,
    sampled_test_dataset: dsl.Output[dsl.Dataset],
    sampling_method: Optional[str] = None,
    task_type: str = "regression",
    split_config: Optional[dict] = None,
    selection_train_size: float = 0.3,
) -> NamedTuple(
    "outputs",
    sample_config=dict,
    split_config=dict,
    sample_row=str,
    models_selection_train_data_path=str,
    extra_train_data_path=str,
):
    """Automl Data Loader component.

    Loads tabular (CSV) data from S3 in batches, sampling up to 100 MB of data,
    then splits the sampled data into test, selection-train, and extra-train sets.

    The component reads data in chunks to efficiently handle large files without
    loading the entire dataset into memory at once. After sampling, it performs
    a two-stage split:

    1. **Primary split** (default 80/20): separates a *test set* (20%, written to
       the ``sampled_test_dataset`` S3 artifact) from the *train portion* (80%).

    2. **Secondary split** (default 30/70 of the train portion): produces
       ``models_selection_train_dataset.csv`` (30%, used for model selection) and
       ``extra_train_dataset.csv`` (70%, passed to ``refit_full`` as extra data).
       Both are written to the PVC workspace under ``{workspace_path}/datasets/``.

    For **regression** tasks the split is random; for **binary** and **multiclass**
    tasks the split is **stratified** by the label column by default.

    Rows with a missing label (NaN / empty in ``label_column``) are dropped after load
    and before splitting, so regression runs do not propagate null targets into splits
    or the ``sample_row`` JSON (stratified sampling already dropped per chunk; this
    applies the same rule to random and first-n-rows paths).

    After sampling, **+/- infinity** values in the frame are replaced with **NaN** (same
    idea as AutoAI ``loadXy``), then **full-row duplicates** are dropped before the
    label drop and train/test split.

    Authentication uses AWS-style credentials provided via environment variables
    (e.g. from a Kubernetes secret).

    Args:
        file_key: S3 object key of the CSV file.
        bucket_name: S3 bucket name containing the file.
        workspace_path: PVC workspace directory where train CSVs will be written.
        label_column: Name of the label/target column in the dataset.
        sampled_test_dataset: Output dataset artifact for the test split.
        sampling_method: "first_n_rows", "stratified", or "random"; if None, derived from task_type.
        task_type: "binary", "multiclass", or "regression" (default); used when sampling_method is None.
        split_config: Split configuration dictionary. Available keys: "test_size" (float), "random_state" (int), "stratify" (bool).
        selection_train_size: Fraction of the train portion used for model selection (default 0.3).

    Raises:
        ValueError: If sampling_method or task_type is invalid, or if required parameters are missing.

    Returns:
        NamedTuple: Contains sample config, split config, a sample row, and paths to selection-train and extra-train CSVs.
    """  # noqa: E501
    import io
    import logging
    import math
    import os

    import boto3
    import pandas as pd

    logger = logging.getLogger(__name__)

    MAX_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB limit in bytes
    PANDAS_CHUNK_SIZE = 10000  # Rows per batch for streaming read
    DEFAULT_RANDOM_STATE = 42
    VALID_SAMPLING_METHODS = {"first_n_rows", "stratified", "random"}
    VALID_TASK_TYPES = {"binary", "multiclass", "regression"}

    # Input validation
    for param, value in (
        ("bucket_name", bucket_name),
        ("file_key", file_key),
        ("workspace_path", workspace_path),
        ("label_column", label_column),
    ):
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"{param} must be a non-empty string.")

    if task_type not in VALID_TASK_TYPES:
        raise ValueError(f"task_type must be one of {VALID_TASK_TYPES}; got {task_type!r}.")
    if split_config is not None and not isinstance(split_config, dict):
        raise TypeError("split_config must be a dictionary with possible keys test_size, random_state, stratify.")
    if isinstance(split_config, dict):
        test_size = split_config.get("test_size")
        if test_size is not None and (not isinstance(test_size, (int, float)) or test_size <= 0 or test_size >= 1):
            raise TypeError("split_config['test_size'] must be a number in (0, 1) when provided.")
        random_state = split_config.get("random_state")
        if random_state is not None and (not isinstance(random_state, int)):
            raise TypeError("split_config['random_state'] must be an integer when provided.")
        stratify = split_config.get("stratify")
        if stratify is not None and not isinstance(stratify, bool):
            raise TypeError("split_config['stratify'] must be a boolean when provided.")
    if not isinstance(selection_train_size, (int, float)):
        raise TypeError("selection_train_size must be a numerical value.")
    elif selection_train_size <= 0 or selection_train_size >= 1:
        raise ValueError("selection_train_size must be in a range 0 to 1.")

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

    def _sample_first_n_rows(text_stream, chunk_size, max_size_bytes):
        """Take rows from the start of the stream until the size limit is reached."""
        chunk_list = []
        accumulated_size = 0

        try:
            for chunk_df in pd.read_csv(text_stream, chunksize=chunk_size):
                chunk_memory = chunk_df.memory_usage(deep=True).sum()

                if accumulated_size + chunk_memory > max_size_bytes:
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
                    break
        except Exception as e:
            if not chunk_list:
                raise ValueError(f"Error reading CSV from S3: {str(e)}") from e

        return pd.concat(chunk_list, ignore_index=True) if chunk_list else pd.DataFrame()

    def _sample_stratified(text_stream, chunk_size, max_size_bytes, label_column):
        """Merge batches and subsample proportionally by target column to stay under the size limit."""
        subsampled_data = None

        try:
            for chunk_df in pd.read_csv(text_stream, chunksize=chunk_size):
                if label_column not in chunk_df.columns:
                    raise ValueError(
                        f"Target column '{label_column}' not found in the dataset. "
                        f"Available columns: {list(chunk_df.columns)}"
                    )
                chunk_df = chunk_df.dropna(subset=[label_column])
                if chunk_df.empty:
                    continue

                combined_data = (
                    pd.concat([subsampled_data, chunk_df], ignore_index=True)
                    if subsampled_data is not None
                    else chunk_df
                )
                combined_memory = combined_data.memory_usage(deep=True).sum()

                if combined_memory <= max_size_bytes:
                    subsampled_data = combined_data
                else:
                    sampling_frac = max_size_bytes / combined_memory
                    subsampled_data = (
                        combined_data.groupby(label_column, group_keys=False)
                        .apply(lambda x: x.sample(frac=sampling_frac, random_state=DEFAULT_RANDOM_STATE))
                        .reset_index(drop=True)
                    )

        except Exception as e:
            logger.debug("Error reading CSV and stratified sampling: %s", e, exc_info=True)
            if subsampled_data is None or subsampled_data.empty:
                raise ValueError(f"Error reading CSV from S3: {str(e)}") from e

        if subsampled_data is None:
            return pd.DataFrame()
        return subsampled_data.sample(frac=1, random_state=DEFAULT_RANDOM_STATE).reset_index(drop=True)

    def _sample_random(text_stream, chunk_size, max_size_bytes):
        """Iterate all batches, merge with accumulated data, randomly subsample when over the limit."""
        subsampled_data = None

        try:
            for chunk_df in pd.read_csv(text_stream, chunksize=chunk_size):
                data = (
                    pd.concat([subsampled_data, chunk_df], ignore_index=True)
                    if subsampled_data is not None
                    else chunk_df
                )
                combined_memory = data.memory_usage(deep=True).sum()

                if combined_memory <= max_size_bytes:
                    subsampled_data = data
                else:
                    sampling_frac = max_size_bytes / combined_memory
                    subsampled_data = data.sample(frac=sampling_frac, random_state=DEFAULT_RANDOM_STATE).reset_index(
                        drop=True
                    )

            return subsampled_data if subsampled_data is not None else pd.DataFrame()

        except Exception as e:
            if subsampled_data is None or subsampled_data.empty:
                raise ValueError(f"Error reading CSV from S3: {str(e)}") from e
            return subsampled_data

    def load_data_in_batches(
        s3_client,
        bucket_name,
        file_key,
        max_size_bytes,
        sampling_method,
        label_column,
    ):
        """Load CSV from S3 in batches and return a sampled dataframe using the chosen strategy."""
        from botocore.exceptions import SSLError

        if sampling_method == "stratified" and label_column is None:
            raise ValueError("label_column must be provided when sampling_method='stratified'")

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

        if sampling_method == "stratified":
            return _sample_stratified(text_stream, PANDAS_CHUNK_SIZE, max_size_bytes, label_column)
        if sampling_method == "random":
            return _sample_random(text_stream, PANDAS_CHUNK_SIZE, max_size_bytes)
        return _sample_first_n_rows(text_stream, PANDAS_CHUNK_SIZE, max_size_bytes)

    s3_client = get_s3_client()
    sampled_dataframe = load_data_in_batches(
        s3_client,
        bucket_name,
        file_key,
        max_size_bytes=MAX_SIZE_BYTES,
        sampling_method=sampling_method,
        label_column=label_column,
    )

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

    n_samples = len(sampled_dataframe)
    logger.info("Read %d rows from s3://%s/%s (sampling_method=%s)", n_samples, bucket_name, file_key, sampling_method)

    # --- Train/test split ---
    from pathlib import Path

    from sklearn.model_selection import train_test_split

    DEFAULT_TEST_SIZE = 0.2
    DEFAULT_SPLIT_RANDOM_STATE = 42

    split_config = split_config or {}
    test_size = split_config.get("test_size", DEFAULT_TEST_SIZE)
    random_state = split_config.get("random_state", DEFAULT_SPLIT_RANDOM_STATE)

    if not sampled_test_dataset.uri or not sampled_test_dataset.uri.endswith(".csv"):
        sampled_test_dataset.uri = (sampled_test_dataset.uri or "sampled_test_dataset") + ".csv"

    # Features and target
    X = sampled_dataframe.drop(columns=[label_column], inplace=False)
    y = sampled_dataframe[label_column]

    stratify_effective = task_type != "regression" and split_config.get("stratify", True)

    # Primary split: train vs test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=(y if stratify_effective else None),
        random_state=random_state,
    )

    # Secondary split: selection train vs extra train
    X_sel, X_extra, y_sel, y_extra = train_test_split(
        X_train,
        y_train,
        test_size=(1 - selection_train_size),
        stratify=(y_train if stratify_effective else None),
        random_state=random_state,
    )

    X_y_sel = pd.concat([X_sel, y_sel], axis=1)
    X_y_extra = pd.concat([X_extra, y_extra], axis=1)
    X_y_test = pd.concat([X_test, y_test], axis=1)

    # Write selection train and extra train to PVC workspace
    datasets_dir = Path(workspace_path) / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    models_selection_train_data_path = str(datasets_dir / "models_selection_train_dataset.csv")
    extra_train_data_path = str(datasets_dir / "extra_train_dataset.csv")
    X_y_sel.to_csv(models_selection_train_data_path, index=False)
    X_y_extra.to_csv(extra_train_data_path, index=False)

    # Write test to S3 artifact
    X_y_test.to_csv(sampled_test_dataset.path, index=False)

    # Sample row for downstream use (JSON string to avoid NaN issues)
    sample_row = X_y_test.head(1).to_json(orient="records")

    return NamedTuple(
        "outputs",
        sample_config=dict,
        split_config=dict,
        sample_row=str,
        models_selection_train_data_path=str,
        extra_train_data_path=str,
    )(
        sample_config={"n_samples": n_samples},
        split_config={
            "test_size": test_size,
            "random_state": random_state,
            "stratify": stratify_effective,
        },
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
