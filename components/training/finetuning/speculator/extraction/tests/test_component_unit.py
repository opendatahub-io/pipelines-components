"""Unit tests for the Speculator extraction component."""

import inspect
import sys
from unittest import mock

import pytest

from ..component import extract_hidden_states


def _mock_run_speculator(monkeypatch):
    run_speculator = mock.Mock(return_value="ok")
    monkeypatch.setitem(sys.modules, "speculator", mock.MagicMock(run_speculator=run_speculator))
    return run_speculator


class TestSpeculatorExtractionComponent:
    """Tests for the fixed extraction component interface."""

    def test_component_function_exists(self):
        """Verify the extraction component is exposed as a KFP component."""
        assert callable(extract_hidden_states)
        assert hasattr(extract_hidden_states, "python_func")

    def test_component_has_extraction_parameters(self):
        """Verify extraction-specific parameters are exposed."""
        params = inspect.signature(extract_hidden_states.python_func).parameters
        for name in ("verifier_model", "dataset_name", "output_hidden_states", "vllm_endpoint"):
            assert name in params

    def test_component_has_fixed_mode(self):
        """Verify extraction does not expose a mode selector."""
        assert "mode" not in inspect.signature(extract_hidden_states.python_func).parameters

    def test_managed_mode_uses_pvc_verifier_when_download_is_disabled(self, monkeypatch):
        """Managed extraction uses the configured PVC model when reusing a cached verifier."""
        run_speculator = mock.Mock(return_value="ok")
        fake_speculator = mock.MagicMock(run_speculator=run_speculator)
        monkeypatch.setitem(sys.modules, "speculator", fake_speculator)

        result = extract_hidden_states.python_func(
            verifier_model="Qwen/Qwen2.5-1.5B-Instruct",
            verifier_model_pvc="models/Qwen3-0.6B",
            output_dir="speculator-output",
            pvc_path="/workspace",
            dataset_name="ultrachat",
            vllm_source="managed",
            download_model=False,
        )

        assert result == "ok"
        values = run_speculator.call_args.args[1]
        assert values["verifier_model"] == "models/Qwen3-0.6B"

    def test_managed_mode_uses_hf_verifier_directly_without_cached_path(self, monkeypatch):
        """Managed extraction keeps a direct HF model ID when no PVC path is set."""
        run_speculator = _mock_run_speculator(monkeypatch)

        extract_hidden_states.python_func(
            verifier_model="Qwen/Qwen3-0.6B",
            output_dir="speculator-output",
            pvc_path="/workspace",
            dataset_name="ultrachat",
            vllm_source="managed",
            download_model=False,
        )

        values = run_speculator.call_args.args[1]
        assert values["verifier_model"] == "Qwen/Qwen3-0.6B"

    def test_managed_mode_keeps_hf_verifier_when_download_is_enabled(self, monkeypatch):
        """Managed extraction keeps the HF verifier when the pipeline will download it."""
        run_speculator = mock.Mock(return_value="ok")
        monkeypatch.setitem(
            sys.modules,
            "speculator",
            mock.MagicMock(run_speculator=run_speculator),
        )

        extract_hidden_states.python_func(
            verifier_model="Qwen/Qwen3-0.6B",
            verifier_model_pvc="models/Qwen--Qwen3-0.6B",
            output_dir="speculator-output",
            pvc_path="/workspace",
            dataset_name="ultrachat",
            vllm_source="managed",
            download_model=True,
        )

        values = run_speculator.call_args.args[1]
        assert values["verifier_model"] == "Qwen/Qwen3-0.6B"

    @pytest.mark.parametrize("endpoint", [None, "", "   "])
    def test_remote_mode_requires_a_vllm_endpoint(self, tmp_path, endpoint):
        """Remote extraction fails early instead of silently selecting managed vLLM."""
        with pytest.raises(ValueError, match="vllm_endpoint is required when vllm_source='remote'"):
            extract_hidden_states.python_func(
                verifier_model="models/Qwen3-0.6B",
                output_dir="speculator-output",
                pvc_path="/workspace",
                dataset_name="ultrachat",
                persistent_pvc="shared",
                persistent_mount_path=str(tmp_path / "shared"),
                vllm_source="remote",
                vllm_endpoint=endpoint,
            )

    def test_managed_mode_rejects_an_external_vllm_endpoint(self):
        """A configured endpoint cannot be ignored by managed extraction."""
        with pytest.raises(ValueError, match="must be omitted when vllm_source='managed'"):
            extract_hidden_states.python_func(
                verifier_model="Qwen/Qwen2.5-1.5B-Instruct",
                output_dir="speculator-output",
                pvc_path="/workspace",
                dataset_name="ultrachat",
                vllm_source="managed",
                vllm_endpoint="http://vllm.example.test:8000",
            )

    def test_remote_mode_trims_endpoint_whitespace(self, monkeypatch):
        """A non-empty remote endpoint is forwarded without surrounding whitespace."""
        run_speculator = _mock_run_speculator(monkeypatch)

        extract_hidden_states.python_func(
            verifier_model="pvc://shared/models/Qwen3-0.6B",
            output_dir="speculator-output",
            pvc_path="/workspace",
            dataset_name="ultrachat",
            persistent_pvc="shared",
            vllm_source="remote",
            vllm_endpoint=" http://vllm.example.test:8000 ",
        )

        values = run_speculator.call_args.args[1]
        assert values["vllm_endpoint"] == "http://vllm.example.test:8000"

    def test_relative_jsonl_path_resolves_under_shared_pvc(self, tmp_path, monkeypatch):
        """Relative JSONL paths become Trainer PVC URIs rooted at the shared mount."""
        pvc_root = tmp_path / "shared"
        dataset_path = pvc_root / "datasets" / "train.jsonl"
        dataset_path.parent.mkdir(parents=True)
        dataset_path.write_text('{"question":"test"}\n')
        run_speculator = _mock_run_speculator(monkeypatch)

        result = extract_hidden_states.python_func(
            verifier_model="Qwen/Qwen2.5-1.5B-Instruct",
            output_dir="speculator-output",
            pvc_path="/workspace",
            dataset_name="datasets/train.jsonl",
            persistent_pvc="shared",
            persistent_mount_path=str(pvc_root),
        )

        assert result == "ok"
        values = run_speculator.call_args.args[1]
        assert values["dataset_name"] == "pvc://shared/datasets/train.jsonl"

    def test_absolute_json_path_under_shared_pvc_resolves_relatively(self, tmp_path, monkeypatch):
        """Absolute paths inside the shared mount are converted to claim-relative URIs."""
        pvc_root = tmp_path / "shared"
        dataset_path = pvc_root / "data.json"
        pvc_root.mkdir()
        dataset_path.write_text('{"question":"test"}\n')
        run_speculator = _mock_run_speculator(monkeypatch)

        extract_hidden_states.python_func(
            verifier_model="Qwen/Qwen2.5-1.5B-Instruct",
            output_dir="speculator-output",
            pvc_path="/workspace",
            dataset_name=str(dataset_path),
            persistent_pvc="shared",
            persistent_mount_path=str(pvc_root),
        )

        values = run_speculator.call_args.args[1]
        assert values["dataset_name"] == "pvc://shared/data.json"

    @pytest.mark.parametrize(
        "dataset_name",
        ["../outside.jsonl", "/outside/data.jsonl", "pvc://shared/../outside.jsonl"],
    )
    def test_local_dataset_paths_cannot_escape_shared_mount(self, tmp_path, dataset_name):
        """Relative traversal and absolute paths outside the mounted PVC are rejected."""
        pvc_root = tmp_path / "shared"
        pvc_root.mkdir()

        with pytest.raises(ValueError, match="remain within persistent_mount_path"):
            extract_hidden_states.python_func(
                verifier_model="Qwen/Qwen2.5-1.5B-Instruct",
                output_dir="speculator-output",
                pvc_path="/workspace",
                dataset_name=dataset_name,
                persistent_pvc="shared",
                persistent_mount_path=str(pvc_root),
            )

    def test_missing_local_dataset_fails_before_starting_trainer(self, tmp_path):
        """A local-looking dataset path must exist on the shared PVC."""
        pvc_root = tmp_path / "shared"
        pvc_root.mkdir()

        with pytest.raises(FileNotFoundError, match="dataset file not found on persistent_pvc"):
            extract_hidden_states.python_func(
                verifier_model="Qwen/Qwen2.5-1.5B-Instruct",
                output_dir="speculator-output",
                pvc_path="/workspace",
                dataset_name="datasets/missing.jsonl",
                persistent_pvc="shared",
                persistent_mount_path=str(pvc_root),
            )

    def test_existing_pvc_uri_is_passed_through(self, tmp_path, monkeypatch):
        """Existing Trainer PVC URIs retain their claim-based semantics."""
        run_speculator = _mock_run_speculator(monkeypatch)

        extract_hidden_states.python_func(
            verifier_model="Qwen/Qwen2.5-1.5B-Instruct",
            output_dir="speculator-output",
            pvc_path="/workspace",
            dataset_name="pvc://other-claim/datasets/train.jsonl",
            persistent_pvc="shared",
            persistent_mount_path=str(tmp_path / "shared"),
        )

        values = run_speculator.call_args.args[1]
        assert values["dataset_name"] == "pvc://other-claim/datasets/train.jsonl"

    def test_pvc_uri_for_shared_claim_is_validated_at_mount_root(self, tmp_path, monkeypatch):
        """A shared-claim URI is checked against the mounted file before Trainer starts."""
        pvc_root = tmp_path / "shared"
        dataset_path = pvc_root / "datasets" / "train.jsonl"
        dataset_path.parent.mkdir(parents=True)
        dataset_path.write_text('{"question":"test"}\n')
        run_speculator = _mock_run_speculator(monkeypatch)

        extract_hidden_states.python_func(
            verifier_model="Qwen/Qwen2.5-1.5B-Instruct",
            output_dir="speculator-output",
            pvc_path="/workspace",
            dataset_name="pvc://shared/datasets/train.jsonl",
            persistent_pvc="shared",
            persistent_mount_path=str(pvc_root),
        )

        values = run_speculator.call_args.args[1]
        assert values["dataset_name"] == "pvc://shared/datasets/train.jsonl"
