"""Unit tests for the Speculator online component."""

import inspect

from ..component import online_speculator


class TestSpeculatorOnlineComponent:
    """Tests for the fixed online component interface."""

    def test_component_function_exists(self):
        """Verify the online component is exposed as a KFP component."""
        assert callable(online_speculator)
        assert hasattr(online_speculator, "python_func")

    def test_component_has_online_parameters(self):
        """Verify online extraction and training parameters are exposed."""
        params = inspect.signature(online_speculator.python_func).parameters
        for name in ("verifier_model", "dataset_name", "output_model", "output_metrics"):
            assert name in params

    def test_component_has_fixed_mode(self):
        """Verify online training does not expose a mode selector."""
        assert "mode" not in inspect.signature(online_speculator.python_func).parameters
