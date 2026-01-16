"""Unit tests for the lm_eval component."""

from unittest import mock

from ..component import universal_llm_evaluator


class TestLmEvalUnitTests:
    """Unit tests for component logic."""

    def test_component_function_exists(self):
        """Test that the component function is properly imported."""
        assert callable(universal_llm_evaluator)
        assert hasattr(universal_llm_evaluator, "python_func")

    def test_component_has_expected_parameters(self):
        """Test that the component has expected input parameters."""
        import inspect

        sig = inspect.signature(universal_llm_evaluator.python_func)
        params = list(sig.parameters.keys())

        expected_params = [
            "output_metrics",
            "output_results",
            "output_samples",
            "task_names",
            "model_path",
            "model_artifact",
            "eval_dataset",
            "batch_size",
            "limit",
            "log_samples",
        ]

        for param in expected_params:
            assert param in params, f"Expected parameter '{param}' not found in component"

    def test_component_default_values(self):
        """Test that the component has expected default values."""
        import inspect

        sig = inspect.signature(universal_llm_evaluator.python_func)
        params = sig.parameters

        assert params["batch_size"].default == "auto"
        assert params["limit"].default == -1
        assert params["log_samples"].default is True
        assert params["verbosity"].default == "INFO"
        assert params["custom_eval_max_tokens"].default == 256

    def test_component_supports_model_sources(self):
        """Test that the component supports both model_path and model_artifact."""
        import inspect

        sig = inspect.signature(universal_llm_evaluator.python_func)
        params = sig.parameters

        # Both model sources should be optional (have defaults of None)
        assert params["model_path"].default is None
        assert "model_artifact" in params

    def test_component_docstring_describes_evaluation_types(self):
        """Test that the component docstring describes supported evaluation types."""
        docstring = universal_llm_evaluator.python_func.__doc__

        # Should mention both benchmark and custom evaluation
        assert "benchmark" in docstring.lower() or "lm-eval" in docstring.lower()
        assert "custom" in docstring.lower() or "holdout" in docstring.lower()
