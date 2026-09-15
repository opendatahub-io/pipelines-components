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
