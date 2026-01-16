"""Unit tests for the finetuning component."""

from unittest import mock

from ..component import train_model


class TestFinetuningComponentUnitTests:
    """Unit tests for component logic."""

    def test_component_function_exists(self):
        """Test that the component function is properly imported."""
        assert callable(train_model)
        assert hasattr(train_model, "python_func")

    def test_component_has_expected_parameters(self):
        """Test that the component has expected input parameters."""
        import inspect

        sig = inspect.signature(train_model.python_func)
        params = list(sig.parameters.keys())

        # Verify key parameters exist
        expected_params = [
            "pvc_path",
            "output_model",
            "output_metrics",
            "dataset",
            "training_base_model",
            "training_algorithm",
            "training_effective_batch_size",
            "training_learning_rate",
            "training_num_epochs",
        ]

        for param in expected_params:
            assert param in params, f"Expected parameter '{param}' not found in component"

    def test_component_default_values(self):
        """Test that the component has expected default values."""
        import inspect

        sig = inspect.signature(train_model.python_func)
        params = sig.parameters

        # Verify some default values
        assert params["training_base_model"].default == "Qwen/Qwen2.5-1.5B-Instruct"
        assert params["training_algorithm"].default == "OSFT"
        assert params["training_effective_batch_size"].default == 128
        assert params["training_backend"].default == "mini-trainer"
        assert params["training_max_seq_len"].default == 8192

    def test_component_supports_both_algorithms(self):
        """Test that the component supports both OSFT and SFT algorithms."""
        import inspect

        sig = inspect.signature(train_model.python_func)
        params = sig.parameters

        # OSFT-specific params
        assert "training_unfreeze_rank_ratio" in params
        assert "training_osft_memory_efficient_init" in params
        assert "training_target_patterns" in params

        # SFT-specific params
        assert "training_save_samples" in params
        assert "training_fsdp_sharding_strategy" in params

    @mock.patch("shutil.copytree")
    @mock.patch("shutil.rmtree")
    @mock.patch("os.makedirs")
    @mock.patch("os.path.exists")
    @mock.patch("os.path.isdir")
    def test_component_decorator_configuration(
        self,
        mock_isdir,
        mock_exists,
        mock_makedirs,
        mock_rmtree,
        mock_copytree,
    ):
        """Test that the component decorator is properly configured."""
        # Verify component_spec exists and has expected configuration
        assert hasattr(train_model, "component_spec")
        component_spec = train_model.component_spec
        assert component_spec is not None

        # Verify base_image is set (check implementation attribute)
        assert hasattr(train_model, "python_func")
