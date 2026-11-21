"""Unit tests for the shared input-source helpers (format detection, Parquet staging and reading).

pandas and pyarrow run for real so the Parquet error paths are exercised against
actual (deliberately corrupted) files; S3 clients are plain mocks.
"""

import io
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
from kfp_components.components.training.automl.shared.input_sources import (
    FORMAT_CSV,
    FORMAT_PARQUET,
    NODE_LOCAL_STAGING_MAX_BYTES,
    PARQUET_MAGIC,
    WORKSPACE_STAGING_MAX_BYTES,
    detect_input_source_format,
    is_parquet_object_key,
    iter_parquet_batches,
    staged_parquet_source,
)


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a dataframe to an in-memory Parquet file."""
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _simple_frame(rows: int = 250) -> pd.DataFrame:
    """Deterministic frame with dtypes Parquet round-trips exactly (int64, object)."""
    return pd.DataFrame({"feature_1": range(rows), "label": [str(i % 3) for i in range(rows)]})


class TestIsParquetObjectKey:
    """The object key's extension identifies Parquet sources case-insensitively."""

    @pytest.mark.parametrize(
        ("file_key", "expected"),
        [
            ("data/train.parquet", True),
            ("data/train.pq", True),
            ("data/TRAIN.PARQUET", True),
            ("data/train.Parquet", True),
            ("data/train.csv", False),
            ("data/train.tsv", False),
            ("data/train", False),
            ("data/train.parquet.gz", False),
            ("", False),
        ],
    )
    def test_key_classification(self, file_key, expected):
        """Keys with Parquet extensions (any case) are Parquet; everything else is not."""
        assert is_parquet_object_key(file_key) is expected


class TestDetectInputSourceFormat:
    """Conclusive extensions decide without S3 calls; anything else is sniffed."""

    def test_conclusive_extensions_skip_the_s3_probe(self):
        """Known Parquet and CSV extensions return without any S3 request."""
        s3 = mock.MagicMock()
        assert detect_input_source_format(s3, "bucket", "data/train.parquet") == FORMAT_PARQUET
        assert detect_input_source_format(s3, "bucket", "data/train.pq") == FORMAT_PARQUET
        assert detect_input_source_format(s3, "bucket", "data/train.csv") == FORMAT_CSV
        assert detect_input_source_format(s3, "bucket", "data/train.tsv") == FORMAT_CSV
        s3.get_object.assert_not_called()
        s3.head_object.assert_not_called()

    def test_unknown_extension_with_parquet_magic_is_parquet(self):
        """A Parquet file without a Parquet extension is detected by its leading magic bytes."""
        s3 = mock.MagicMock()
        s3.get_object.return_value = {"Body": io.BytesIO(PARQUET_MAGIC + b"rest-of-file")}

        assert detect_input_source_format(s3, "bucket", "data/export") == FORMAT_PARQUET

        s3.get_object.assert_called_once_with(Bucket="bucket", Key="data/export", Range="bytes=0-3")

    def test_unknown_extension_with_csv_content_is_csv(self):
        """Non-Parquet leading bytes fall back to CSV."""
        s3 = mock.MagicMock()
        s3.get_object.return_value = {"Body": io.BytesIO(b"a,b,c\n1,2,3\n")}

        assert detect_input_source_format(s3, "bucket", "data/export") == FORMAT_CSV

    def test_empty_object_is_csv(self):
        """A zero-byte object sniffs as CSV and fails later in the CSV reader, as before."""
        s3 = mock.MagicMock()
        s3.get_object.return_value = {"Body": io.BytesIO(b"")}

        assert detect_input_source_format(s3, "bucket", "data/export") == FORMAT_CSV

    def test_probe_failure_falls_back_to_csv(self):
        """A failing ranged GET (no Range support, permissions, ...) never breaks the load."""
        s3 = mock.MagicMock()
        s3.get_object.side_effect = RuntimeError("Range not supported")

        assert detect_input_source_format(s3, "bucket", "data/export") == FORMAT_CSV


class TestIterParquetBatches:
    """Chunked Parquet reading with descriptive corruption errors."""

    def test_yields_bounded_batches_matching_the_source(self, tmp_path):
        """Batches respect batch_rows and concatenate back to the source frame."""
        path = tmp_path / "train.parquet"
        path.write_bytes(_parquet_bytes(_simple_frame(250)))

        batches = list(iter_parquet_batches(path, 100, source_label="s3://bucket/train.parquet"))

        assert [len(batch) for batch in batches] == [100, 100, 50]
        combined = pd.concat(batches, ignore_index=True)
        pd.testing.assert_frame_equal(combined, _simple_frame(250))
        # Typed Parquet columns survive the round trip (CSV re-inference is not involved).
        assert combined["feature_1"].dtype == "int64"

    def test_zero_row_file_yields_no_batches(self, tmp_path):
        """A schema-only Parquet file yields nothing, like a header-only CSV."""
        path = tmp_path / "empty_rows.parquet"
        frame = pd.DataFrame({"a": pd.Series([], dtype="int64"), "b": pd.Series([], dtype="float64")})
        path.write_bytes(_parquet_bytes(frame))

        assert list(iter_parquet_batches(path, 100)) == []

    def test_missing_file_raises(self, tmp_path):
        """A missing local file surfaces a clear error (e.g. a failed staging download)."""
        with pytest.raises(FileNotFoundError):
            iter_parquet_batches(tmp_path / "absent.parquet", 100)

    def test_empty_file_is_rejected_descriptively(self, tmp_path):
        """A zero-byte Parquet upload is rejected naming the file and the problem."""
        path = tmp_path / "train.parquet"
        path.write_bytes(b"")

        with pytest.raises(ValueError, match="s3://bucket/train.parquet") as exc_info:
            iter_parquet_batches(path, 100, source_label="s3://bucket/train.parquet")

        assert "empty (0 bytes)" in str(exc_info.value)
        assert isinstance(exc_info.value, ValueError)

    def test_non_parquet_bytes_are_rejected_descriptively(self, tmp_path):
        """A file that is not Parquet at all (e.g. CSV renamed to .parquet) is rejected."""
        path = tmp_path / "train.parquet"
        path.write_bytes(b"a,b,c\n1,2,3\n")

        with pytest.raises(ValueError, match="s3://bucket/train.parquet") as exc_info:
            iter_parquet_batches(path, 100, source_label="s3://bucket/train.parquet")

        message = str(exc_info.value)
        assert "is not a valid Parquet file" in message
        assert "does not start with the Parquet magic bytes" in message
        assert "renamed to .parquet" in message

    def test_truncated_parquet_is_rejected_descriptively(self, tmp_path):
        """A Parquet file cut short (damaged or missing footer) is rejected."""
        payload = _parquet_bytes(_simple_frame(50))
        path = tmp_path / "train.parquet"
        path.write_bytes(payload[: len(payload) // 2])

        with pytest.raises(ValueError, match="s3://bucket/train.parquet") as exc_info:
            iter_parquet_batches(path, 100, source_label="s3://bucket/train.parquet")

        message = str(exc_info.value)
        assert "appears truncated" in message
        assert "footer magic bytes are missing or damaged" in message

    def test_corrupt_metadata_is_rejected_descriptively(self, tmp_path):
        """Header and trailer magic with garbage between them surfaces the wrapped pyarrow error."""
        path = tmp_path / "train.parquet"
        path.write_bytes(PARQUET_MAGIC + b"garbage-metadata-bytes" + PARQUET_MAGIC)

        with pytest.raises(ValueError, match="s3://bucket/train.parquet") as exc_info:
            iter_parquet_batches(path, 100, source_label="s3://bucket/train.parquet")

        message = str(exc_info.value)
        assert "corrupted or malformed" in message
        assert "Underlying error" in message

    def test_tiny_file_is_rejected_descriptively(self, tmp_path):
        """A file too small to structurally be Parquet is rejected as truncated."""
        path = tmp_path / "train.parquet"
        path.write_bytes(PARQUET_MAGIC)

        with pytest.raises(ValueError, match="too small to be a valid") as exc_info:
            iter_parquet_batches(path, 100, source_label="s3://bucket/train.parquet")

        assert "corrupted or was truncated during upload" in str(exc_info.value)


class TestStagedParquetSource:
    """Parquet objects are downloaded to seekable local storage and cleaned up."""

    @staticmethod
    def _s3_with_payload(payload: bytes, content_length=None):
        s3 = mock.MagicMock()
        s3.head_object.return_value = {"ContentLength": len(payload) if content_length is None else content_length}
        s3.download_file.side_effect = lambda bucket, key, filename: open(filename, "wb").write(payload)
        return s3

    def test_small_object_stages_node_locally_and_cleans_up(self, tmp_path):
        """Objects within the node-local budget are staged outside the workspace and removed after use."""
        payload = _parquet_bytes(_simple_frame(10))
        s3 = self._s3_with_payload(payload)
        workspace = tmp_path / "workspace"

        with staged_parquet_source(s3, "bucket", "train.parquet", workspace_dir=workspace) as staged:
            staged_path = Path(staged.path)
            assert staged.source_bytes == len(payload)
            assert staged.staged_on_workspace is False
            assert staged_path.read_bytes() == payload

        assert not staged_path.exists()
        s3.download_file.assert_called_once()
        # The workspace was never touched for a small object.
        assert not workspace.exists()

    def test_small_object_stages_a_readable_parquet_file(self, tmp_path):
        """The staged file is a real Parquet file the chunk reader can consume."""
        payload = _parquet_bytes(_simple_frame(10))
        s3 = self._s3_with_payload(payload)

        with staged_parquet_source(s3, "bucket", "train.parquet") as staged:
            batches = list(iter_parquet_batches(staged.path, 4))

        assert [len(batch) for batch in batches] == [4, 4, 2]

    def test_large_object_stages_on_the_workspace(self, tmp_path):
        """Objects above the node-local budget are staged on the provided workspace directory."""
        payload = _parquet_bytes(_simple_frame(5))
        s3 = self._s3_with_payload(payload, content_length=NODE_LOCAL_STAGING_MAX_BYTES + 1)
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        with staged_parquet_source(s3, "bucket", "train.parquet", workspace_dir=workspace) as staged:
            staged_path = Path(staged.path)
            assert staged.staged_on_workspace is True
            assert staged.source_bytes == NODE_LOCAL_STAGING_MAX_BYTES + 1
            assert staged_path.is_relative_to(workspace)
            assert staged_path.read_bytes() == payload

        assert not staged_path.exists()

    def test_large_object_without_workspace_is_rejected(self):
        """No workspace directory and an oversized object fails with a clear error."""
        s3 = self._s3_with_payload(b"x", content_length=NODE_LOCAL_STAGING_MAX_BYTES + 1)

        with pytest.raises(ValueError, match="above the") as exc_info:
            with staged_parquet_source(s3, "bucket", "train.parquet"):
                pass

        assert "node-local staging budget" in str(exc_info.value)
        s3.download_file.assert_not_called()

    def test_object_above_workspace_budget_is_rejected(self, tmp_path):
        """Objects beyond the workspace staging limit are refused before any download."""
        s3 = self._s3_with_payload(b"x", content_length=WORKSPACE_STAGING_MAX_BYTES + 1)

        with pytest.raises(ValueError, match="above the") as exc_info:
            with staged_parquet_source(s3, "bucket", "train.parquet", workspace_dir=tmp_path):
                pass

        assert "staging limit" in str(exc_info.value)
        s3.download_file.assert_not_called()

    def test_unknown_size_stages_node_locally(self, tmp_path):
        """A failed size probe still allows staging (best effort), with source_bytes unknown."""
        payload = _parquet_bytes(_simple_frame(3))
        s3 = mock.MagicMock()
        s3.head_object.side_effect = RuntimeError("HEAD not supported")
        s3.download_file.side_effect = lambda bucket, key, filename: open(filename, "wb").write(payload)
        workspace = tmp_path / "workspace"

        with staged_parquet_source(s3, "bucket", "train.parquet", workspace_dir=workspace) as staged:
            assert staged.source_bytes is None
            assert staged.staged_on_workspace is False
            assert open(staged.path, "rb").read() == payload

    def test_invalid_content_length_is_treated_as_unknown(self, tmp_path):
        """A non-integer ContentLength (e.g. a mock or a broken store) degrades to unknown size."""
        payload = _parquet_bytes(_simple_frame(3))
        s3 = mock.MagicMock()
        s3.head_object.return_value = {"ContentLength": object()}
        s3.download_file.side_effect = lambda bucket, key, filename: open(filename, "wb").write(payload)

        with staged_parquet_source(s3, "bucket", "train.parquet") as staged:
            assert staged.source_bytes is None

    def test_download_failure_propagates_and_cleans_up(self):
        """Download errors propagate unwrapped (loaders keep their TLS/ClientError handling)."""
        s3 = mock.MagicMock()
        s3.head_object.return_value = {"ContentLength": 128}
        s3.download_file.side_effect = RuntimeError("NoSuchKey")

        with pytest.raises(RuntimeError, match="NoSuchKey"):
            with staged_parquet_source(s3, "bucket", "train.parquet"):
                pass

        s3.download_file.assert_called_once()

    def test_sampling_error_still_cleans_up(self, tmp_path):
        """An exception raised while sampling inside the context removes the staged file."""
        payload = _parquet_bytes(_simple_frame(3))
        s3 = self._s3_with_payload(payload)

        with pytest.raises(ValueError, match="boom"):
            with staged_parquet_source(s3, "bucket", "train.parquet") as staged:
                staged_path = Path(staged.path)
                raise ValueError("boom")

        assert not staged_path.exists()
