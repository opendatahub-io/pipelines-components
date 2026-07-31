"""Shared helpers for the optional user-provided test dataset.

Both AutoML data loaders (tabular and timeseries) accept an optional
``test_data_bucket_name`` / ``test_data_file_key`` pair. When set, the referenced
CSV replaces the loader's own holdout split. The parameter validation, error
reporting, and truncation accounting are identical in both loaders, so they live
here instead of being duplicated.

Usage:
    test_bucket, test_key = validate_test_data_params(test_data_bucket_name, test_data_file_key)
    if test_key:
        source = test_data_source_uri(test_bucket, test_key)
        try:
            df = load(...)
        except Exception as e:
            raise test_data_load_error(source, e) from e
        raise_if_test_data_empty(len(df), source)
"""

from __future__ import annotations


def validate_test_data_params(test_data_bucket_name: str, test_data_file_key: str) -> tuple[str, str]:
    """Validate the optional test-data parameter pair and return the stripped values.

    Both parameters default to an empty string, meaning "no user-provided test data".
    Supplying one without the other is an error, and the object key gets the same
    hygiene check the loaders apply to the training ``file_key``.

    Args:
        test_data_bucket_name: S3 bucket name for the user-provided test dataset.
        test_data_file_key: S3 object key for the user-provided test dataset.

    Returns:
        tuple: ``(bucket, key)`` with surrounding whitespace stripped. Both are
            empty strings when no test data was requested.

    Raises:
        ValueError: If only one of the pair is set, or the key is not a well-formed
            S3 object key.
    """
    bucket = (test_data_bucket_name or "").strip()
    key = (test_data_file_key or "").strip()

    if key and not bucket:
        raise ValueError("test_data_bucket_name must be provided when test_data_file_key is set.")
    if bucket and not key:
        raise ValueError("test_data_file_key must be provided when test_data_bucket_name is set.")
    if key and (key.startswith("/") or key.endswith("/") or "//" in key):
        raise ValueError(
            "test_data_file_key must be a valid S3 object key and must not start/end with '/' or contain '//'."
        )

    return bucket, key


def test_data_source_uri(bucket: str, key: str) -> str:
    """Return the ``s3://bucket/key`` URI used in log lines, status records, and errors."""
    return f"s3://{bucket}/{key}"


def test_data_load_error(source: str, error: BaseException) -> ValueError:
    """Build the error raised when the test dataset cannot be read.

    The message identifies the *test* dataset so it is distinguishable from a
    failure on the training data, and it carries the underlying error verbatim
    rather than guessing at a cause (a permissions failure is not a bad path).
    """
    return ValueError(f"Failed to load user-provided test dataset from {source}: {error}")


def raise_if_test_data_empty(row_count: int, source: str) -> None:
    """Raise when the test dataset carries no data rows.

    Args:
        row_count: Number of rows loaded from the test dataset.
        source: ``s3://bucket/key`` URI of the test dataset.

    Raises:
        ValueError: If ``row_count`` is zero.
    """
    if row_count == 0:
        raise ValueError(
            "Test dataset contains no data rows (only headers). "
            f"Source: {source}. Provide a test dataset with at least one data row."
        )


def report_test_data_truncation(status, logger, source: str, rows_loaded: int, max_size_bytes: int) -> None:
    """Warn that the test dataset was truncated and record it on the status stage.

    The loaders read S3 in chunks and stop at ``max_size_bytes``. For training data
    that is deliberate sampling, but for a test set it means metrics are reported
    against an arbitrary leading-row prefix, so it must be visible to the user
    rather than buried at debug level.

    Args:
        status: :class:`ComponentStatusTracker` for the running component.
        logger: Logger used by the calling component.
        source: ``s3://bucket/key`` URI of the test dataset.
        rows_loaded: Number of rows that were kept.
        max_size_bytes: Size cap that triggered the truncation.
    """
    logger.warning(
        "Test dataset %s exceeds the %.0f MB load limit and was truncated to the first %s row(s). "
        "Evaluation metrics will be computed on that leading-row prefix only, and rows (for panel "
        "data, whole trailing series) beyond the limit are not evaluated. "
        "Provide a smaller test dataset to evaluate all of it.",
        source,
        max_size_bytes / (1024**2),
        rows_loaded,
    )
    status.record(
        "load_test_data",
        "running",
        truncated=True,
        rows_loaded=rows_loaded,
        max_size_bytes=max_size_bytes,
    )
