"""Unit tests for the Speculator training component."""

import inspect

from ..component import train_speculator_mode


class TestSpeculatorTrainingComponent:
    """Tests for the fixed training component interface."""

    def test_component_function_exists(self):
        """Verify the training component is exposed as a KFP component."""
        assert callable(train_speculator_mode)
        assert hasattr(train_speculator_mode, "python_func")

    def test_component_has_training_parameters(self):
        """Verify training-specific parameters are exposed."""
        params = inspect.signature(train_speculator_mode.python_func).parameters
        for name in ("verifier_model", "hidden_states_path", "output_model", "output_metrics"):
            assert name in params

    def test_component_has_fixed_mode(self):
        """Verify training does not expose a mode selector."""
        assert "mode" not in inspect.signature(train_speculator_mode.python_func).parameters
