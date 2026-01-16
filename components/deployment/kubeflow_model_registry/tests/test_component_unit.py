"""Unit tests for the kubeflow_model_registry component."""

from unittest import mock

from ..component import kubeflow_model_registry


class TestKubeflowModelRegistryUnitTests:
    """Unit tests for component logic."""

    def test_component_function_exists(self):
        """Test that the component function is properly imported."""
        assert callable(kubeflow_model_registry)
        assert hasattr(kubeflow_model_registry, "python_func")

    def test_component_has_expected_parameters(self):
        """Test that the component has expected input parameters."""
        import inspect

        sig = inspect.signature(kubeflow_model_registry.python_func)
        params = list(sig.parameters.keys())

        expected_params = [
            "pvc_mount_path",
            "input_model",
            "input_metrics",
            "eval_metrics",
            "registry_address",
            "registry_port",
            "model_name",
            "model_version",
            "model_format_name",
            "author",
        ]

        for param in expected_params:
            assert param in params, f"Expected parameter '{param}' not found in component"

    def test_component_default_values(self):
        """Test that the component has expected default values."""
        import inspect

        sig = inspect.signature(kubeflow_model_registry.python_func)
        params = sig.parameters

        assert params["registry_port"].default == 8080
        assert params["model_name"].default == "fine-tuned-model"
        assert params["model_version"].default == "1.0.0"
        assert params["model_format_name"].default == "pytorch"
        assert params["author"].default == "pipeline"

    def test_component_supports_provenance_fields(self):
        """Test that the component supports provenance/lineage tracking."""
        import inspect

        sig = inspect.signature(kubeflow_model_registry.python_func)
        params = list(sig.parameters.keys())

        # Provenance fields
        assert "source_pipeline_name" in params
        assert "source_pipeline_run_id" in params
        assert "source_pipeline_run_name" in params
        assert "source_namespace" in params

    @mock.patch.dict("sys.modules", {"model_registry": mock.MagicMock()})
    def test_component_with_mocked_registry(self):
        """Test component with mocked Model Registry client."""
        # The component would need full execution context
        # For now verify the component definition is valid
        assert kubeflow_model_registry.python_func is not None

    def test_component_returns_string(self):
        """Test that the component is annotated to return a string."""
        import inspect

        sig = inspect.signature(kubeflow_model_registry.python_func)

        # The component should return a string (model_id)
        assert sig.return_annotation == str or "str" in str(sig.return_annotation)
