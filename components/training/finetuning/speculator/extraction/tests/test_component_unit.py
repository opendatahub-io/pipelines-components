"""Unit tests for the Speculator extraction component."""

import inspect

from ..component import extract_speculator


class TestSpeculatorExtractionComponent:
    """Tests for the fixed extraction component interface."""

    def test_component_function_exists(self):
        """Verify the extraction component is exposed as a KFP component."""
        assert callable(extract_speculator)
        assert hasattr(extract_speculator, "python_func")

    def test_component_has_extraction_parameters(self):
        """Verify extraction-specific parameters are exposed."""
        params = inspect.signature(extract_speculator.python_func).parameters
        for name in ("verifier_model", "dataset_name", "output_hidden_states", "vllm_endpoint"):
            assert name in params

    def test_component_has_fixed_mode(self):
        """Verify extraction does not expose a mode selector."""
        assert "mode" not in inspect.signature(extract_speculator.python_func).parameters
