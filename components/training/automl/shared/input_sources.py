"""Shared input-source helpers for the AutoML data loaders: format detection and Parquet reading.

Both AutoML data loaders (tabular and timeseries) historically read CSV sources
from S3. Parquet input support needs the same pieces in both loaders, so they
live here instead of being duplicated (see ``user_test_data.py`` and
``parquet_utils.py`` for the same rationale):

* **Format detection** (:func:`detect_input_source_format`): decide ``parquet``
  vs ``csv`` from the object key's extension, sniffing the leading magic bytes
  through a ranged GET when the extension is inconclusive. Detection never
  raises: an inconclusive probe falls back to CSV, the loaders' historical
  behavior.
* **Staging** (:func:`staged_parquet_source`): Parquet metadata (the footer) sits
  at the *end* of the file, so unlike CSV a Parquet object cannot be parsed as a
  forward-only stream. It must be downloaded to seekable local storage first.
* **Chunked reading** (:func:`iter_parquet_batches`): yield bounded pandas
  batches from a staged Parquet file, mirroring ``pd.read_csv(..., chunksize=)``
  so the loaders' samplers can consume either format the same way. Corrupted or
  malformed files raise :class:`ParquetSourceError` with a message that names
  the file and the underlying problem.

boto3 and pyarrow are imported lazily inside the functions that need them: this
module is imported by components that run in the AutoML image (where both are
present) and by unit tests that mock them (where they may be absent).
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

logger = logging.getLogger(__name__)

# Every Parquet file starts and ends with these four bytes. The header magic
# identifies the format; the trailing magic brackets the footer metadata.
PARQUET_MAGIC = b"PAR1"
PARQUET_SUFFIXES = (".parquet", ".pq")
CSV_SUFFIXES = (".csv", ".tsv", ".txt")

FORMAT_PARQUET = "parquet"
FORMAT_CSV = "csv"

# Staging budgets for Parquet sources, mirroring the loaders' existing local
# download limits: node-local (container ephemeral) storage for moderate
# objects, the pipeline workspace PVC for larger ones.
NODE_LOCAL_STAGING_MAX_BYTES = 2 * 1024**3  # 2 GiB
WORKSPACE_STAGING_MAX_BYTES = 16 * 1024**3  # 16 GiB

# Smallest structurally possible Parquet file: header magic + footer length
# field (4 bytes) + footer magic.
_MINIMUM_PARQUET_BYTES = 2 * len(PARQUET_MAGIC) + 4


class ParquetSourceError(ValueError):
    """A Parquet input could not be read because it is corrupted or malformed.

    Subclasses :class:`ValueError` so loaders treat it like any other invalid
    input (including the user-test-data error wrapper), while samplers re-raise
    it verbatim instead of wrapping it in a generic CSV read error.
    """


def is_parquet_object_key(file_key: str) -> bool:
    """Return True when the S3 object key's extension identifies a Parquet file.

    Case-insensitive: ``Data.PARQUET`` and ``data.parquet`` are both Parquet
    keys. Extensions are a *hint* consumed by :func:`detect_input_source_format`;
    the file's magic bytes remain the authority once it is staged.
    """
    return (file_key or "").lower().endswith(PARQUET_SUFFIXES)


def detect_input_source_format(s3_client, bucket_name: str, file_key: str, *, component_logger=None) -> str:
    """Return ``FORMAT_PARQUET`` or ``FORMAT_CSV`` for the referenced S3 object.

    The object key's extension decides when it is conclusive (``.parquet`` /
    ``.pq`` vs ``.csv`` / ``.tsv`` / ``.txt``); keys with any other extension
    are sniffed by reading the first bytes through a ranged GET, so a valid
    Parquet file uploaded without a Parquet extension is still detected and
    trained from rather than failing with an opaque CSV parse error.

    Known CSV extensions skip the probe entirely, keeping the CSV request
    pattern byte-for-byte identical to the pre-Parquet loaders. A failed or
    inconclusive probe (store without Range support, permissions error, empty
    object) falls back to ``FORMAT_CSV`` and never raises: format detection
    must not turn a working CSV load into a failure.

    Args:
        s3_client: boto3 S3 client used only for the ranged sniff.
        bucket_name: S3 bucket name.
        file_key: S3 object key.
        component_logger: Loader logger; defaults to this module's logger.

    Returns:
        str: ``FORMAT_PARQUET`` or ``FORMAT_CSV``.
    """
    log = component_logger or logger
    key = (file_key or "").lower()
    if key.endswith(PARQUET_SUFFIXES):
        return FORMAT_PARQUET
    if key.endswith(CSV_SUFFIXES):
        return FORMAT_CSV
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key, Range=f"bytes=0-{len(PARQUET_MAGIC) - 1}")
        body = response["Body"]
        try:
            leading = body.read(len(PARQUET_MAGIC))
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
        if leading == PARQUET_MAGIC:
            log.info(
                "Object %s has no Parquet extension but starts with the Parquet magic bytes; reading it as Parquet.",
                file_key,
            )
            return FORMAT_PARQUET
    except Exception as exc:  # noqa: BLE001 - detection must not break CSV loads
        log.info("Could not sniff the leading bytes of %s; assuming CSV: %s", file_key, exc)
    return FORMAT_CSV


@contextmanager
def staged_parquet_source(s3_client, bucket_name: str, file_key: str, *, workspace_dir=None, source_label=None):
    """Download a Parquet object to readable local storage; clean up on exit.

    Objects up to :data:`NODE_LOCAL_STAGING_MAX_BYTES` (2 GiB) are staged on
    node-local temporary storage. Larger objects are staged on ``workspace_dir``
    (the pipeline's PVC workspace) up to :data:`WORKSPACE_STAGING_MAX_BYTES`
    (16 GiB), matching the timeseries sampler's spool budget. When the size
    cannot be determined the object is staged node-local, best effort.

    Download failures propagate to the caller (botocore errors are not wrapped)
    so loaders keep their existing TLS/ClientError handling; the staged file is
    always removed, including when sampling raises.

    Args:
        s3_client: boto3 S3 client used for ``head_object``/``download_file``.
        bucket_name: S3 bucket name.
        file_key: S3 object key.
        workspace_dir: Optional workspace directory (Path or str) used to stage
            objects too large for node-local storage.
        source_label: URI shown in error messages; defaults to
            ``s3://bucket_name/file_key``.

    Yields:
        SimpleNamespace with ``path`` (local file), ``source_bytes`` (int or
        None when the size could not be determined), and
        ``staged_on_workspace`` (bool).

    Raises:
        ValueError: If the object is larger than the workspace staging budget,
            or larger than node-local storage with no ``workspace_dir`` given.
    """
    label = source_label or f"s3://{bucket_name}/{file_key}"
    source_bytes = None
    try:
        response = s3_client.head_object(Bucket=bucket_name, Key=file_key)
        content_length = response.get("ContentLength") if hasattr(response, "get") else None
        if isinstance(content_length, int) and content_length >= 0:
            source_bytes = content_length
    except Exception as exc:  # noqa: BLE001 - the size probe is best effort
        logger.info("Could not determine the size of %s; staging it without a size check: %s", label, exc)

    if source_bytes is not None and source_bytes > WORKSPACE_STAGING_MAX_BYTES:
        raise ValueError(
            f"The Parquet file {label} is {source_bytes} bytes, above the "
            f"{WORKSPACE_STAGING_MAX_BYTES}-byte staging limit. Reduce or partition the dataset."
        )

    staged_on_workspace = False
    staging_dir = None
    if source_bytes is None or source_bytes <= NODE_LOCAL_STAGING_MAX_BYTES:
        pass  # node-local temporary storage
    elif workspace_dir is not None:
        staging_dir = Path(workspace_dir)
        staging_dir.mkdir(parents=True, exist_ok=True)
        staged_on_workspace = True
    else:
        raise ValueError(
            f"The Parquet file {label} is {source_bytes} bytes, above the "
            f"{NODE_LOCAL_STAGING_MAX_BYTES}-byte node-local staging budget, and no workspace "
            "directory is available for larger files."
        )

    descriptor, path = tempfile.mkstemp(prefix="automl-parquet-", suffix=".parquet", dir=staging_dir)
    os.close(descriptor)
    try:
        s3_client.download_file(bucket_name, file_key, path)
        yield SimpleNamespace(path=path, source_bytes=source_bytes, staged_on_workspace=staged_on_workspace)
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def iter_parquet_batches(path, batch_rows: int, *, source_label=None):
    """Open a local Parquet file eagerly and yield its rows as pandas DataFrames.

    The file is validated and opened when called (not on first iteration), so a
    corrupted or malformed file fails before any sampling work starts. Row
    batches mirror ``pd.read_csv(..., chunksize=batch_rows)``: bounded by
    ``batch_rows`` and decoded lazily, so budget-driven samplers that stop early
    do not decode the whole file.

    Both the leading and trailing Parquet magic bytes are checked for precise
    diagnostics, then pyarrow validates the footer metadata and decodes row
    groups. Any failure raises :class:`ParquetSourceError` naming the file and
    the underlying problem, so users see *why* the file was rejected rather
    than a bare pyarrow stack trace.

    Args:
        path: Local path of the Parquet file (as staged by
            :func:`staged_parquet_source`).
        batch_rows: Maximum number of rows per yielded dataframe.
        source_label: URI shown in error messages; defaults to ``str(path)``.

    Returns:
        Generator yielding one pandas DataFrame per batch. A structurally valid
        file with zero data rows yields nothing, matching pandas' behavior for
        a header-only CSV.

    Raises:
        ParquetSourceError: If the file is empty, is not Parquet, is truncated,
            or cannot be decoded.
    """
    import pyarrow.parquet as pq  # lazy: heavy native dependency, mocked in unit tests

    label = source_label or str(path)
    parquet_path = Path(path)
    size = parquet_path.stat().st_size
    if size == 0:
        raise ParquetSourceError(
            f"The Parquet file {label} is empty (0 bytes). "
            "The upload may have failed or the source file has no content. "
            "Provide a Parquet file with data rows, or upload the data as CSV."
        )
    if size < _MINIMUM_PARQUET_BYTES:
        raise ParquetSourceError(
            f"The Parquet file {label} is only {size} byte(s) and is too small to be a valid "
            "Parquet file. The file is corrupted or was truncated during upload. "
            "Re-upload a valid Parquet file, or convert the data to CSV."
        )

    with parquet_path.open("rb") as handle:
        header = handle.read(len(PARQUET_MAGIC))
        handle.seek(-len(PARQUET_MAGIC), os.SEEK_END)
        footer = handle.read(len(PARQUET_MAGIC))
    if header != PARQUET_MAGIC:
        raise ParquetSourceError(
            f"The file {label} is not a valid Parquet file: it does not start with the Parquet "
            f"magic bytes ({PARQUET_MAGIC!r}). The file is corrupted, or it is not Parquet at all "
            "(for example a CSV or text file renamed to .parquet). Re-export the data as Parquet "
            "(for example with pandas DataFrame.to_parquet), or upload it as CSV."
        )
    if footer != PARQUET_MAGIC:
        raise ParquetSourceError(
            f"The Parquet file {label} appears truncated: its footer magic bytes are missing or "
            "damaged. A Parquet file must end with the same four bytes it starts with, so the "
            "upload or transfer was cut short. Re-upload the complete file, or convert the data to CSV."
        )

    try:
        parquet_file = pq.ParquetFile(parquet_path)
    except Exception as exc:  # noqa: BLE001 - surface every pyarrow open failure uniformly
        raise ParquetSourceError(
            f"Failed to read the Parquet file {label}: the file is corrupted or malformed "
            "(damaged footer, unsupported Parquet version, or unreadable schema). "
            f"Underlying error: {exc}. Re-upload a valid Parquet file, or convert the data to CSV."
        ) from exc

    def _batches():
        try:
            for record_batch in parquet_file.iter_batches(batch_size=batch_rows):
                yield record_batch.to_pandas()
        except Exception as exc:  # noqa: BLE001 - surface every decode failure uniformly
            raise ParquetSourceError(
                f"Failed to read the Parquet file {label}: the file is corrupted or malformed "
                f"(a row group could not be decoded). Underlying error: {exc}. "
                "Re-upload a valid Parquet file, or convert the data to CSV."
            ) from exc
        finally:
            close = getattr(parquet_file, "close", None)
            if callable(close):
                close()

    return _batches()
