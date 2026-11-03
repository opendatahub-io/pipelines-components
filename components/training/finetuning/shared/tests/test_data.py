"""Unit tests for the shared data utilities."""

import io
import json
import logging
import os
import tarfile
from unittest import mock

import pytest

from ..data import (
    _extract_tar,
    _find_hf_model,
    _get_oci_auth,
    _skopeo_copy,
    download_oci_model,
    normalize_hf_dataset_uri,
    prepare_jsonl,
    resolve_dataset,
)


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


class TestNormalizeHFDatasetURI:
    """Tests for direct Training Hub Hugging Face inputs."""

    def test_splits_scheme_and_config(self):
        """Split the pipeline URI into repository and config arguments."""
        assert normalize_hf_dataset_uri("hf://Agent-Ark/Toucan-1.5M:Qwen3", "") == (
            "Agent-Ark/Toucan-1.5M",
            "Qwen3",
        )

    def test_preserves_config_for_dataset_without_suffix(self):
        """Keep the configured subset when the URI has no suffix."""
        assert normalize_hf_dataset_uri("hf://org/dataset", "train") == ("org/dataset", "train")

    def test_preserves_non_huggingface_path(self):
        """Leave local paths unchanged for the materialized data path."""
        assert normalize_hf_dataset_uri("/mnt/pipeline/datasets/train.jsonl", "Qwen3") == (
            "/mnt/pipeline/datasets/train.jsonl",
            "Qwen3",
        )


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


class TestFindHfModel:
    """Tests for _find_hf_model."""

    def test_finds_valid_hf_model_dir(self, tmp_path):
        """Directory with config.json, weights, and tokenizer is found."""
        model_dir = tmp_path / "models" / "my_model"
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")
        (model_dir / "model.safetensors").write_bytes(b"")
        (model_dir / "tokenizer.json").write_text("{}")

        result = _find_hf_model(str(tmp_path))
        assert result == str(model_dir)

    def test_returns_none_when_no_model(self, tmp_path):
        """Returns None when no HF model structure exists."""
        (tmp_path / "random.txt").write_text("nothing")
        assert _find_hf_model(str(tmp_path)) is None

    def test_requires_config_and_weights_and_tokenizer(self, tmp_path):
        """Returns None when tokenizer is missing."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        (model_dir / "model.safetensors").write_bytes(b"")
        assert _find_hf_model(str(tmp_path)) is None


class TestGetOciAuth:
    """Tests for _get_oci_auth."""

    def test_returns_none_when_unset(self, log):
        """Returns None when env var is not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            assert _get_oci_auth(log) is None

    def test_returns_raw_json_when_valid(self, log):
        """Returns raw JSON string when valid Docker config is provided."""
        auth = json.dumps({"auths": {"registry.io": {"auth": "abc"}}})
        with mock.patch.dict(os.environ, {"OCI_PULL_SECRET_MODEL_DOWNLOAD": auth}):
            result = _get_oci_auth(log)
        assert result == auth

    def test_raises_on_invalid_json(self, log):
        """Raises ValueError for non-JSON content."""
        with mock.patch.dict(os.environ, {"OCI_PULL_SECRET_MODEL_DOWNLOAD": "not-json"}):
            with pytest.raises(ValueError, match="not valid JSON"):
                _get_oci_auth(log)

    def test_raises_on_missing_auths(self, log):
        """Raises ValueError when 'auths' key is absent."""
        with mock.patch.dict(os.environ, {"OCI_PULL_SECRET_MODEL_DOWNLOAD": "{}"}):
            with pytest.raises(ValueError, match="non-empty 'auths' field"):
                _get_oci_auth(log)

    def test_raises_on_empty_auths(self, log):
        """Raises ValueError when 'auths' dict is empty."""
        auth = json.dumps({"auths": {}})
        with mock.patch.dict(os.environ, {"OCI_PULL_SECRET_MODEL_DOWNLOAD": auth}):
            with pytest.raises(ValueError, match="non-empty 'auths' field"):
                _get_oci_auth(log)


class TestSkopeoCopy:
    """Tests for _skopeo_copy."""

    def test_runs_skopeo_without_auth(self, log, tmp_path):
        """Skopeo is invoked without --authfile when no auth is provided."""
        dest = str(tmp_path / "dest")
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(returncode=0)
            _skopeo_copy("registry.io/model:latest", dest, None, log)
        args = mock_run.call_args[0][0]
        assert args[:2] == ["skopeo", "copy"]
        assert "--authfile" not in args
        assert "docker://registry.io/model:latest" in args

    def test_runs_skopeo_with_auth(self, log, tmp_path):
        """Skopeo is invoked with --authfile when auth is provided."""
        dest = str(tmp_path / "dest")
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(returncode=0)
            _skopeo_copy("registry.io/model:latest", dest, '{"auths":{}}', log)
        args = mock_run.call_args[0][0]
        assert "--authfile" in args

    def test_raises_on_failure(self, log, tmp_path):
        """RuntimeError is raised when skopeo exits with non-zero code."""
        dest = str(tmp_path / "dest")
        with mock.patch("subprocess.run") as mock_run:
            proc = mock.MagicMock(returncode=1, stderr="pull denied")
            proc.check_returncode.side_effect = RuntimeError("skopeo failed")
            mock_run.return_value = proc
            with pytest.raises(RuntimeError):
                _skopeo_copy("registry.io/model:latest", dest, None, log)

    def test_cleans_up_authfile(self, log, tmp_path):
        """Temporary auth file is cleaned up after skopeo run."""
        dest = str(tmp_path / "dest")
        created_files = []

        original_unlink = os.unlink

        def track_unlink(path):
            created_files.append(path)
            original_unlink(path)

        with (
            mock.patch("subprocess.run", return_value=mock.MagicMock(returncode=0)),
            mock.patch("os.unlink", side_effect=track_unlink),
        ):
            _skopeo_copy("ref", dest, '{"auths":{}}', log)
        assert len(created_files) == 1


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


class TestDownloadOciModel:
    """Tests for download_oci_model."""

    def test_orchestrates_skopeo_extract_and_find(self, log, tmp_path):
        """Skopeo copy, tar extraction, and HF model finding are orchestrated."""
        pvc_path = str(tmp_path)
        model_ref = "oci://registry.io/model:v1"

        with (
            mock.patch("components.training.finetuning.shared.data._get_oci_auth", return_value=None),
            mock.patch("components.training.finetuning.shared.data._skopeo_copy") as mock_skopeo,
            mock.patch("components.training.finetuning.shared.data._extract_tar"),
            mock.patch("components.training.finetuning.shared.data._find_hf_model", return_value="/pvc/model/hf"),
            mock.patch("os.path.isdir", return_value=True),
        ):
            result = download_oci_model(model_ref, pvc_path, log)

        mock_skopeo.assert_called_once()
        assert mock_skopeo.call_args[0][0] == "registry.io/model:v1"
        assert result == "/pvc/model/hf"

    def test_returns_mod_out_when_no_hf_model_found(self, log, tmp_path):
        """Falls back to model output directory when no HF model is found."""
        pvc_path = str(tmp_path)
        model_ref = "oci://registry.io/model:v1"

        with (
            mock.patch("components.training.finetuning.shared.data._get_oci_auth", return_value=None),
            mock.patch("components.training.finetuning.shared.data._skopeo_copy"),
            mock.patch("components.training.finetuning.shared.data._extract_tar"),
            mock.patch("components.training.finetuning.shared.data._find_hf_model", return_value=None),
            mock.patch("os.path.isdir", return_value=False),
        ):
            result = download_oci_model(model_ref, pvc_path, log)

        assert result == os.path.join(pvc_path, "model")
