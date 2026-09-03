"""Unit tests for the shared data utilities."""

import json
import logging
from unittest import mock

import pytest

from ..data import (
    prepare_jsonl,
    resolve_dataset,
)


@pytest.fixture
def log():
    """Create a test logger."""
    return logging.getLogger("test_data")


@pytest.fixture
def mock_datasets():
    """Provide a mock 'datasets' module for local imports in data.py."""
    mod = mock.MagicMock()
    with mock.patch.dict("sys.modules", {"datasets": mod}):
        yield mod


class TestResolveDataset:
    """Tests for resolve_dataset."""

    def test_existing_dir_is_reused(self, log, tmp_path, mock_datasets):
        """Non-empty output directory is reused without re-downloading."""
        out_dir = tmp_path / "ds"
        out_dir.mkdir()
        (out_dir / "data.jsonl").write_text("{}")

        resolve_dataset(None, str(out_dir), log)
        assert (out_dir / "data.jsonl").exists()

    def test_artifact_dir_is_copied(self, log, tmp_path, mock_datasets):
        """Artifact directory is copied to the output directory."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "train.jsonl").write_text('{"text":"hello"}')
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        inp = mock.MagicMock()
        inp.path = str(src_dir)

        resolve_dataset(inp, str(out_dir), log)
        assert (out_dir / "train.jsonl").read_text() == '{"text":"hello"}'

    def test_artifact_file_is_copied(self, log, tmp_path, mock_datasets):
        """Artifact file is copied to the output directory."""
        src_file = tmp_path / "data.jsonl"
        src_file.write_text('{"text":"hello"}')
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        inp = mock.MagicMock()
        inp.path = str(src_file)

        resolve_dataset(inp, str(out_dir), log)
        assert (out_dir / "data.jsonl").read_text() == '{"text":"hello"}'

    def test_artifact_file_without_extension_gets_default_name(self, log, tmp_path, mock_datasets):
        """File without extension is renamed to train.jsonl."""
        src_file = tmp_path / "mydata"
        src_file.write_text('{"text":"data"}')
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        inp = mock.MagicMock()
        inp.path = str(src_file)

        resolve_dataset(inp, str(out_dir), log)
        assert (out_dir / "train.jsonl").exists()

    def test_pvc_metadata_dir_is_copied(self, log, tmp_path, mock_datasets):
        """PVC directory from metadata is copied to the output directory."""
        pvc_dir = tmp_path / "pvc_ds"
        pvc_dir.mkdir()
        (pvc_dir / "train.jsonl").write_text('{"text":"pvc"}')
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        inp = mock.MagicMock()
        inp.path = None
        inp.metadata = {"pvc_path": str(pvc_dir)}

        resolve_dataset(inp, str(out_dir), log)
        assert (out_dir / "train.jsonl").read_text() == '{"text":"pvc"}'

    def test_remote_json_loads_via_hf_datasets(self, log, tmp_path, mock_datasets):
        """Remote JSONL URL is loaded via HuggingFace datasets."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        inp = mock.MagicMock()
        inp.path = None
        inp.metadata = {"artifact_path": "https://example.com/data.jsonl"}

        mock_ds = mock.MagicMock()
        mock_datasets.load_dataset.return_value = mock_ds

        resolve_dataset(inp, str(out_dir), log)
        mock_datasets.load_dataset.assert_called_once_with(
            "json", data_files="https://example.com/data.jsonl", split="train"
        )
        mock_ds.save_to_disk.assert_called_once_with(str(out_dir))

    def test_hf_repo_id_loads_via_hf_datasets(self, log, tmp_path, mock_datasets):
        """HuggingFace repo ID is loaded via datasets library."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        inp = mock.MagicMock()
        inp.path = None
        inp.metadata = {"artifact_path": "tatsu-lab/alpaca"}

        mock_ds = mock.MagicMock()
        mock_datasets.load_dataset.return_value = mock_ds

        resolve_dataset(inp, str(out_dir), log)
        mock_datasets.load_dataset.assert_called_once_with("tatsu-lab/alpaca", split="train")

    def test_no_source_raises_value_error(self, log, tmp_path, mock_datasets):
        """ValueError is raised when no dataset source is resolvable."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        with pytest.raises(ValueError, match="No dataset provided"):
            resolve_dataset(None, str(out_dir), log)


class TestPrepareJsonl:
    """Tests for prepare_jsonl."""

    def test_exports_dataset_to_jsonl(self, log, tmp_path, mock_datasets):
        """Dataset is exported to JSONL via to_json."""
        ds_dir = str(tmp_path / "ds")
        jsonl_path = str(tmp_path / "out.jsonl")

        mock_ds = mock.MagicMock()
        mock_ds.__contains__ = lambda self, key: key == "train"
        mock_ds.__getitem__ = lambda self, key: mock_ds
        mock_datasets.load_from_disk.return_value = mock_ds

        prepare_jsonl(ds_dir, jsonl_path, log)
        mock_ds.to_json.assert_called_once_with(jsonl_path, lines=True)

    def test_falls_back_to_manual_write_on_attribute_error(self, log, tmp_path, mock_datasets):
        """Falls back to manual JSONL write when to_json raises AttributeError."""
        ds_dir = str(tmp_path / "ds")
        jsonl_path = str(tmp_path / "out.jsonl")

        mock_ds = mock.MagicMock()
        mock_ds.__contains__ = lambda self, key: key == "train"
        mock_ds.__getitem__ = lambda self, key: mock_ds
        mock_ds.to_json.side_effect = AttributeError("no to_json")
        mock_ds.__iter__ = lambda self: iter([{"text": "a"}, {"text": "b"}])
        mock_datasets.load_from_disk.return_value = mock_ds

        prepare_jsonl(ds_dir, jsonl_path, log)

        with open(jsonl_path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"text": "a"}

    def test_handles_dict_dataset_with_train_split(self, log, tmp_path, mock_datasets):
        """Dict-type dataset with 'train' split is handled correctly."""
        ds_dir = str(tmp_path / "ds")
        jsonl_path = str(tmp_path / "out.jsonl")

        train_split = mock.MagicMock()
        mock_datasets.load_from_disk.return_value = {"train": train_split}

        prepare_jsonl(ds_dir, jsonl_path, log)
        train_split.to_json.assert_called_once_with(jsonl_path, lines=True)

    def test_logs_warning_on_failure(self, log, tmp_path, mock_datasets):
        """Warning is logged when load_from_disk raises an exception."""
        ds_dir = str(tmp_path / "ds")
        jsonl_path = str(tmp_path / "out.jsonl")

        mock_datasets.load_from_disk.side_effect = Exception("corrupt")

        with mock.patch.object(log, "warning") as mock_warn:
            prepare_jsonl(ds_dir, jsonl_path, log)
        mock_warn.assert_called_once()
        assert "JSONL export failed" in mock_warn.call_args[0][0]
