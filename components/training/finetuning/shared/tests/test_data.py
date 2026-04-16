"""Unit tests for the shared data utilities (tarfile extraction)."""

import io
import logging
import tarfile

import pytest

from ..data import _extract_tar


@pytest.fixture
def log():
    """Create a test logger."""
    return logging.getLogger("test_data")


def _create_tar(path, members):
    """Create a tar archive with the given member name/content pairs.

    Args:
        path: Filesystem path for the tar file.
        members: List of (name, content_bytes) tuples.
    """
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


class TestExtractTar:
    """Tests for _extract_tar safe extraction."""

    def test_extracts_model_files(self, log, tmp_path):
        """Valid models/ members are extracted correctly."""
        img_dir = tmp_path / "img"
        img_dir.mkdir()
        out_dir = tmp_path / "out"

        _create_tar(
            str(img_dir / "layer.tar.gz"),
            [
                ("models/config.json", b'{"key": "value"}'),
                ("models/weights.bin", b"\x00\x01\x02"),
                ("other/skip.txt", b"ignored"),
            ],
        )

        result = _extract_tar(str(img_dir), str(out_dir), log)

        assert sorted(result) == ["models/config.json", "models/weights.bin"]
        assert (out_dir / "models" / "config.json").read_bytes() == b'{"key": "value"}'
        assert (out_dir / "models" / "weights.bin").read_bytes() == b"\x00\x01\x02"

    def test_skips_non_tar_files(self, log, tmp_path):
        """Non-tar files in the image directory are silently skipped."""
        img_dir = tmp_path / "img"
        img_dir.mkdir()
        out_dir = tmp_path / "out"

        (img_dir / "random_blob").write_bytes(b"not a tar file")

        result = _extract_tar(str(img_dir), str(out_dir), log)
        assert result == []

    def test_skips_json_and_manifest_files(self, log, tmp_path):
        """JSON files and manifest are skipped before tar open."""
        img_dir = tmp_path / "img"
        img_dir.mkdir()
        out_dir = tmp_path / "out"

        (img_dir / "config.json").write_bytes(b"{}")
        (img_dir / "manifest").write_bytes(b"manifest data")
        (img_dir / "index.json").write_bytes(b"{}")

        result = _extract_tar(str(img_dir), str(out_dir), log)
        assert result == []

    def test_path_traversal_raises_filter_error(self, log, tmp_path):
        """A tar member with path traversal must raise FilterError, not be silently swallowed."""
        img_dir = tmp_path / "img"
        img_dir.mkdir()
        out_dir = tmp_path / "out"

        _create_tar(
            str(img_dir / "malicious.tar.gz"),
            [("models/../../etc/evil", b"pwned")],
        )

        with pytest.raises(tarfile.FilterError):
            _extract_tar(str(img_dir), str(out_dir), log)

        assert not (tmp_path / "etc" / "evil").exists()

    def test_models_prefix_traversal_raises_filter_error(self, log, tmp_path):
        """A models/-prefixed member with path traversal must raise FilterError."""
        img_dir = tmp_path / "img"
        img_dir.mkdir()
        out_dir = tmp_path / "out"

        _create_tar(
            str(img_dir / "traversal.tar.gz"),
            [("models/../../../etc/passwd", b"root")],
        )

        with pytest.raises(tarfile.FilterError):
            _extract_tar(str(img_dir), str(out_dir), log)

        assert not (tmp_path / "etc" / "passwd").exists()

    def test_empty_image_dir(self, log, tmp_path):
        """An empty image directory returns an empty list."""
        img_dir = tmp_path / "img"
        img_dir.mkdir()
        out_dir = tmp_path / "out"

        result = _extract_tar(str(img_dir), str(out_dir), log)
        assert result == []

    def test_mixed_valid_and_non_tar_layers(self, log, tmp_path):
        """Valid tar layers are extracted even when mixed with non-tar blobs."""
        img_dir = tmp_path / "img"
        img_dir.mkdir()
        out_dir = tmp_path / "out"

        _create_tar(
            str(img_dir / "layer1.tar.gz"),
            [("models/model.bin", b"model-data")],
        )
        (img_dir / "sha256_blob").write_bytes(b"not a tar")

        result = _extract_tar(str(img_dir), str(out_dir), log)
        assert result == ["models/model.bin"]
