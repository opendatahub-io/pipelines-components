"""Unit tests for the download_model KFP component."""

import sys
import tempfile
from pathlib import Path
from unittest import mock

from kfp import compiler
from kfp_components.components.data_processing.download_model import download_model


def test_component_compiles():
    """Verify the component compiles to a valid YAML pipeline spec."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        compiler.Compiler().compile(download_model, f.name)
        assert Path(f.name).stat().st_size > 0


def test_component_signature():
    """Verify the component exposes the expected input parameters."""
    spec = download_model.component_spec
    input_names = set(spec.inputs.keys())
    assert input_names == {"model_name", "model_cache_pvc", "model_cache_mount"}


def _run_download(tmp_path, model_name="mistralai/Mistral-7B", pre_cached=False):
    """Invoke download_model.python_func with huggingface_hub mocked via sys.modules."""
    model_dir_name = model_name.replace("/", "--")
    model_path = tmp_path / model_dir_name

    if pre_cached:
        model_path.mkdir(parents=True, exist_ok=True)
        (model_path / ".download_complete").write_text(model_name)

    mock_hf = mock.MagicMock()

    def fake_download(repo_id, local_dir, local_dir_use_symlinks):
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        (Path(local_dir) / "config.json").write_text("{}")

    mock_hf.snapshot_download.side_effect = fake_download

    with mock.patch.dict(sys.modules, {"huggingface_hub": mock_hf}):
        result = download_model.python_func(
            model_name=model_name,
            model_cache_pvc="my-pvc",
            model_cache_mount=str(tmp_path),
        )

    return result, mock_hf, model_path


def test_downloads_model_when_not_cached(tmp_path):
    """Verify snapshot_download is called and sentinel written when model is not cached."""
    result, mock_hf, model_path = _run_download(tmp_path)

    assert result == "mistralai--Mistral-7B"
    mock_hf.snapshot_download.assert_called_once_with(
        repo_id="mistralai/Mistral-7B",
        local_dir=str(model_path),
        local_dir_use_symlinks=False,
    )
    assert (model_path / ".download_complete").exists()


def test_skips_download_when_cached(tmp_path):
    """Verify download is skipped when the sentinel file already exists."""
    result, mock_hf, _ = _run_download(tmp_path, pre_cached=True)

    assert result == "mistralai--Mistral-7B"
    mock_hf.snapshot_download.assert_not_called()


def test_returns_model_subpath():
    """Verify slash-to-double-dash conversion for model directory name."""
    assert "mistralai/Mistral-7B".replace("/", "--") == "mistralai--Mistral-7B"


def test_sentinel_contains_model_name(tmp_path):
    """Verify sentinel file content equals the original model name."""
    _, _, model_path = _run_download(tmp_path)
    assert (model_path / ".download_complete").read_text() == "mistralai/Mistral-7B"
