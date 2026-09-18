"""Tests for the Speculator runtime."""

import inspect

from .. import speculator


class TestSpeculatorRuntime:
    """Tests for shared mode dispatch."""

    def test_runtime_function_exists(self):
        """Verify the runtime entry point exists."""
        assert callable(speculator.run_speculator)

    def test_runtime_supports_all_modes(self):
        """Verify all SDK modes are represented in the runtime."""
        source = inspect.getsource(speculator.run_speculator)
        for mode in ("data_only", "train_only", "offline", "online"):
            assert mode in source

    def test_runtime_publishes_checkpoint_best(self):
        """Verify Speculator publishes its best checkpoint, not an arbitrary one."""
        assert '"checkpoint_best"' in inspect.getsource(speculator.run_speculator)

    def test_runtime_validates_hidden_states_output(self):
        """Verify data-only mode validates the extracted hidden-state directory."""
        source = inspect.getsource(speculator.run_speculator)
        assert "os.path.isdir(hidden_states_dir)" in source
        assert "Check the TrainJob logs" in source

    def test_runtime_warns_without_hidden_states_artifact(self):
        """Verify PVC-only hidden-state output is explicitly reported."""
        source = inspect.getsource(speculator.run_speculator)
        assert "log.warning(" in source
        assert "no KFP output artifact is wired" in source

    def test_runtime_rejects_empty_vllm_endpoint(self):
        """Verify an empty endpoint cannot silently select managed mode."""
        source = inspect.getsource(speculator.run_speculator)
        assert 'values.get("vllm_endpoint") == ""' in source
        assert "must be omitted/None for managed vLLM" in source

    def test_runtime_validates_raw_verifier_for_external_vllm(self):
        """Verify external vLLM validation checks the original verifier input."""
        source = inspect.getsource(speculator.run_speculator)
        assert 'raw_verifier_model = values["verifier_model"]' in source
        assert 'not raw_verifier_model.startswith("pvc://")' in source
