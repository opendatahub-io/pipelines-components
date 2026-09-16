"""Unit tests for OCI utils components."""

import os

import pytest

from ..component import copy_oci_model_to_pvc, is_oci_uri, passthrough_uri


class TestIsOciUri:
    """Tests for is_oci_uri component."""

    def test_oci_uri_returns_true(self):
        """Verify OCI URIs are detected."""
        assert is_oci_uri.python_func("oci://registry.redhat.io/model:latest") == "true"

    def test_hf_uri_returns_false(self):
        """Verify HuggingFace IDs are not detected as OCI."""
        assert is_oci_uri.python_func("Qwen/Qwen2.5-1.5B-Instruct") == "false"

    def test_empty_string_returns_false(self):
        """Verify empty string is not detected as OCI."""
        assert is_oci_uri.python_func("") == "false"


class TestPassthroughUri:
    """Tests for passthrough_uri component."""

    def test_returns_value_unchanged(self):
        """Verify value passes through unchanged."""
        assert passthrough_uri.python_func("Qwen/Qwen2.5-1.5B-Instruct") == "Qwen/Qwen2.5-1.5B-Instruct"


class TestCopyOciModelToPvc:
    """Tests for copy_oci_model_to_pvc component."""

    def test_rejects_symlinks(self, tmp_path):
        """Verify symlinks in model contents are rejected (CWE-59)."""
        src = tmp_path / "models"
        src.mkdir()
        (src / "config.json").write_text("{}")
        (src / "evil.json").symlink_to("/var/run/secrets/kubernetes.io/serviceaccount/token")

        pvc = tmp_path / "pvc"
        pvc.mkdir()

        model = type("Model", (), {"path": str(src)})()

        with pytest.raises(RuntimeError, match="Refusing symlink"):
            copy_oci_model_to_pvc.python_func(model=model, pvc_mount_path=str(pvc))

    def test_copies_regular_files(self, tmp_path):
        """Verify regular model files are copied to PVC."""
        src = tmp_path / "models"
        src.mkdir()
        (src / "config.json").write_text('{"model_type": "gpt2"}')
        (src / "model.safetensors").write_bytes(b"\x00" * 100)
        (src / "tokenizer.json").write_text('{"version": "1.0"}')
        (src / "pytorch_model.bin").write_bytes(b"\x00" * 100)

        pvc = tmp_path / "pvc"
        pvc.mkdir()

        model = type("Model", (), {"path": str(src)})()

        copy_oci_model_to_pvc.python_func(model=model, pvc_mount_path=str(pvc))
        assert os.path.exists(os.path.join(str(pvc), "model", "config.json"))

    def test_raises_if_src_missing(self, tmp_path):
        """Verify error is raised when model path does not exist."""
        pvc = tmp_path / "pvc"
        pvc.mkdir()

        model = type("Model", (), {"path": str(tmp_path / "nonexistent")})()

        with pytest.raises(RuntimeError, match="Model path not found"):
            copy_oci_model_to_pvc.python_func(model=model, pvc_mount_path=str(pvc))
