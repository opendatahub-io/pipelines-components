"""Tests for shared/output.py — model persistence, metrics extraction, loss plotting."""

import json
import logging
import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from components.training.finetuning.shared.output import (
    extract_metrics_from_jsonl,
    find_model_dir,
    normalize_tokenizer_config,
    persist_model,
    plot_training_loss,
    preserve_tokenizer_files,
)


@pytest.fixture
def log():
    """Create a test logger."""
    return logging.getLogger("test_output")


class TestFindModelDir:
    """Tests for find_model_dir checkpoint discovery."""

    def test_nonexistent_root_returns_none(self, tmp_path):
        """Non-existent root directory returns None."""
        result = find_model_dir(str(tmp_path / "nonexistent"))
        assert result is None

    def test_empty_root_returns_none(self, tmp_path):
        """Empty root directory returns None."""
        result = find_model_dir(str(tmp_path))
        assert result is None

    def test_single_checkpoint_with_config_json(self, tmp_path):
        """Checkpoint containing config.json is discovered."""
        ckpt = tmp_path / "checkpoint-1"
        ckpt.mkdir()
        (ckpt / "config.json").write_text("{}")
        result = find_model_dir(str(tmp_path))
        assert result == str(ckpt)

    def test_returns_most_recent_checkpoint_by_mtime(self, tmp_path):
        """Most recently modified checkpoint is returned."""
        old = tmp_path / "checkpoint-1"
        old.mkdir()
        (old / "config.json").write_text("{}")
        new = tmp_path / "checkpoint-2"
        new.mkdir()
        (new / "config.json").write_text("{}")
        os.utime(new / "config.json", (2_000_000_000, 2_000_000_000))
        result = find_model_dir(str(tmp_path))
        assert result == str(new)

    def test_fallback_to_subdir_mtime_when_no_config_json(self, tmp_path):
        """Falls back to newest subdirectory when no config.json exists."""
        sub = tmp_path / "model_dir"
        sub.mkdir()
        result = find_model_dir(str(tmp_path))
        assert result == str(sub)


class TestExtractMetricsFromJsonl:
    """Tests for extract_metrics_from_jsonl metric parsing."""

    def test_missing_file_returns_empty(self, tmp_path):
        """Missing file returns empty metrics and loss list."""
        met, loss = extract_metrics_from_jsonl(str(tmp_path / "nonexistent.jsonl"))
        assert met == {}
        assert loss == []

    def test_single_line_with_loss(self, tmp_path):
        """Single JSONL line with loss is parsed correctly."""
        f = tmp_path / "metrics.jsonl"
        f.write_text(json.dumps({"loss": 1.5, "step": 10}) + "\n")
        met, loss = extract_metrics_from_jsonl(str(f))
        assert loss == [1.5]
        assert met["final_loss"] == 1.5
        assert met["min_loss"] == 1.5
        assert met["step"] == 10.0

    def test_multiple_lines_accumulate_loss(self, tmp_path):
        """Multiple lines accumulate loss values."""
        f = tmp_path / "metrics.jsonl"
        lines = [
            json.dumps({"loss": 2.0}),
            json.dumps({"loss": 1.5}),
            json.dumps({"loss": 1.0}),
        ]
        f.write_text("\n".join(lines) + "\n")
        met, loss = extract_metrics_from_jsonl(str(f))
        assert loss == [2.0, 1.5, 1.0]
        assert met["final_loss"] == 1.0
        assert met["min_loss"] == 1.0

    def test_first_value_wins_for_duplicate_metric_keys(self, tmp_path):
        """First occurrence of a metric key wins over duplicates."""
        f = tmp_path / "metrics.jsonl"
        f.write_text(json.dumps({"step": 1}) + "\n" + json.dumps({"step": 99}) + "\n")
        met, _ = extract_metrics_from_jsonl(str(f))
        assert met["step"] == 1.0

    def test_malformed_line_is_skipped(self, tmp_path):
        """Malformed JSON lines are silently skipped."""
        f = tmp_path / "metrics.jsonl"
        f.write_text("not-valid-json\n" + json.dumps({"loss": 0.5}) + "\n")
        met, loss = extract_metrics_from_jsonl(str(f))
        assert loss == [0.5]

    def test_non_numeric_loss_is_skipped(self, tmp_path):
        """Non-numeric loss values are skipped."""
        f = tmp_path / "metrics.jsonl"
        f.write_text(json.dumps({"loss": "not-a-number"}) + "\n")
        _, loss = extract_metrics_from_jsonl(str(f))
        assert loss == []

    def test_avg_loss_field_treated_as_loss(self, tmp_path):
        """The avg_loss field is treated as a loss value."""
        f = tmp_path / "metrics.jsonl"
        f.write_text(json.dumps({"avg_loss": 0.8}) + "\n")
        met, loss = extract_metrics_from_jsonl(str(f))
        assert loss == [0.8]
        assert met["final_loss"] == 0.8


class TestPersistModel:
    """Tests for persist_model artifact persistence."""

    def test_raises_when_no_model_found(self, tmp_path, log):
        """Raises RuntimeError when no model checkpoint exists."""
        output_model = mock.MagicMock()
        with pytest.raises(RuntimeError, match="No model found"):
            persist_model(
                str(tmp_path / "empty_ckpts"),
                str(tmp_path / "pvc"),
                "base-model",
                output_model,
                log,
            )

    def test_copies_model_to_pvc_and_artifact(self, tmp_path, log):
        """Model files are copied to PVC and artifact paths."""
        ckpt = tmp_path / "ckpts" / "checkpoint-1"
        ckpt.mkdir(parents=True)
        (ckpt / "config.json").write_text("{}")
        (ckpt / "model.bin").write_bytes(b"weights")
        pvc = tmp_path / "pvc"
        pvc.mkdir()
        output_model = mock.MagicMock()
        output_model.path = str(tmp_path / "artifact")
        output_model.metadata = {}

        persist_model(str(tmp_path / "ckpts"), str(pvc), "MyModel", output_model, log)

        assert (pvc / "final_model" / "config.json").exists()
        assert (pvc / "final_model" / "model.bin").exists()
        assert output_model.metadata["model_name"] == "MyModel"
        assert output_model.metadata["pvc_model_dir"] == str(pvc / "final_model")
        assert output_model.metadata["pvc_model_subpath"] == "final_model"

    def test_persists_model_to_run_scoped_subpath(self, tmp_path, log):
        """Model persistence supports isolated subpaths on a shared PVC."""
        ckpt = tmp_path / "ckpts" / "checkpoint-1"
        ckpt.mkdir(parents=True)
        (ckpt / "config.json").write_text("{}")
        pvc = tmp_path / "pvc"
        pvc.mkdir()
        output_model = mock.MagicMock()
        output_model.path = str(tmp_path / "artifact")
        output_model.metadata = {}

        persist_model(
            str(tmp_path / "ckpts"),
            str(pvc),
            "MyModel",
            output_model,
            log,
            model_subpath="final_model/run-123",
        )

        assert (pvc / "final_model" / "run-123" / "config.json").exists()
        assert output_model.metadata["pvc_model_subpath"] == "final_model/run-123"

    def test_sets_artifact_name(self, tmp_path, log):
        """Artifact name is set from the base model name."""
        ckpt = tmp_path / "ckpts" / "final"
        ckpt.mkdir(parents=True)
        (ckpt / "config.json").write_text("{}")
        pvc = tmp_path / "pvc"
        pvc.mkdir()
        output_model = mock.MagicMock()
        output_model.path = str(tmp_path / "artifact")

        persist_model(str(tmp_path / "ckpts"), str(pvc), "granite-7b", output_model, log)

        assert output_model.name == "granite-7b-checkpoint"


class TestPreserveTokenizerFiles:
    """Tests for preserving tokenizer files required by serving runtimes."""

    def test_copies_resolved_tokenizer_files(self, tmp_path, log):
        """Copies vocabulary files referenced by the tokenizer metadata."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "vocab.json").write_text("{}")
        (source / "merges.txt").write_text("#version: 0.2\n")
        destination = tmp_path / "destination"
        destination.mkdir()
        tokenizer = SimpleNamespace(
            init_kwargs={
                "vocab_file": str(source / "vocab.json"),
                "merges_file": str(source / "merges.txt"),
                "chat_template": "not-a-file",
            }
        )

        preserve_tokenizer_files(tokenizer, str(destination), log)

        assert (destination / "vocab.json").read_text() == "{}"
        assert (destination / "merges.txt").read_text() == "#version: 0.2\n"


class TestNormalizeTokenizerConfig:
    """Tests for Transformers-compatible special-token metadata."""

    def test_converts_legacy_special_token_list(self, tmp_path, log):
        """Converts additional_special_tokens to named extra tokens."""
        config_path = tmp_path / "tokenizer_config.json"
        config_path.write_text(
            json.dumps({"additional_special_tokens": ["<a>", "<b>"], "tokenizer_class": "Qwen2Tokenizer"})
        )

        normalize_tokenizer_config(str(tmp_path), log)

        config = json.loads(config_path.read_text())
        assert "additional_special_tokens" not in config
        assert config["extra_special_tokens"] == {
            "extra_special_token_0": "<a>",
            "extra_special_token_1": "<b>",
        }


class TestPlotTrainingLoss:
    """Tests for plot_training_loss HTML output."""

    @pytest.fixture(autouse=True)
    def _mock_matplotlib(self):
        """Patch matplotlib in sys.modules so tests run without it installed."""
        mock_mpl = mock.MagicMock()
        mock_plt = mock.MagicMock()
        mock_mpl.pyplot = mock_plt
        with mock.patch.dict(
            sys.modules,
            {"matplotlib": mock_mpl, "matplotlib.pyplot": mock_plt},
        ):
            self._mock_plt = mock_plt
            yield

    def test_empty_loss_writes_no_data_html(self, tmp_path):
        """Empty loss list produces a 'No loss data' HTML page."""
        path = str(tmp_path / "loss.html")
        plot_training_loss([], path)
        with open(path) as f:
            content = f.read()
        assert "No loss data" in content

    def test_valid_loss_writes_html_with_base64_image(self, tmp_path):
        """Non-empty loss list produces HTML with a base64-encoded image."""
        mock_fig = mock.MagicMock()
        mock_ax = mock.MagicMock()
        self._mock_plt.subplots.return_value = (mock_fig, mock_ax)

        mock_buf = mock.MagicMock()
        mock_buf.read.return_value = b"fake_png_data"
        mock_buf.seek = mock.MagicMock()

        path = str(tmp_path / "loss.html")
        with (
            mock.patch("io.BytesIO", return_value=mock_buf),
            mock.patch("base64.b64encode", return_value=b"FAKE_BASE64_IMG"),
        ):
            plot_training_loss([2.0, 1.5, 1.0], path)

        with open(path) as f:
            content = f.read()
        assert "data:image/png;base64," in content
        assert "<!DOCTYPE html>" in content
