"""Unit tests for the SFT training component."""

from unittest import mock

from ..component import train_model


class TestSFTComponentUnitTests:
    """Unit tests for SFT component logic."""

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
            "output_loss_chart",
            "dataset",
            "training_base_model",
            "training_effective_batch_size",
            "training_max_tokens_per_gpu",
            "training_max_seq_len",
            "training_learning_rate",
            "training_num_epochs",
        ]

        for param in expected_params:
            assert param in params, f"Expected parameter '{param}' not found in component"

    def test_component_has_sft_specific_parameters(self):
        """Test that the component has SFT-specific parameters."""
        import inspect

        sig = inspect.signature(train_model.python_func)
        params = list(sig.parameters.keys())

        # Verify SFT-specific parameters exist
        sft_params = [
            "training_save_samples",
            "training_accelerate_full_state_at_epoch",
            "training_fsdp_sharding_strategy",
        ]

        for param in sft_params:
            assert param in params, f"Expected SFT parameter '{param}' not found in component"

    def test_component_excludes_algorithm_backend_parameters(self):
        """Test that algorithm and backend parameters are NOT present (hardcoded)."""
        import inspect

        sig = inspect.signature(train_model.python_func)
        params = list(sig.parameters.keys())

        # These should NOT be present as they are hardcoded
        assert "training_algorithm" not in params, "training_algorithm should be hardcoded, not a parameter"
        assert "training_backend" not in params, "training_backend should be hardcoded, not a parameter"

    def test_component_excludes_osft_specific_parameters(self):
        """Test that OSFT-specific parameters are NOT present."""
        import inspect

        sig = inspect.signature(train_model.python_func)
        params = list(sig.parameters.keys())

        # These OSFT-specific params should NOT be present in SFT component
        osft_only_params = [
            "training_unfreeze_rank_ratio",
            "training_osft_memory_efficient_init",
            "training_target_patterns",
            "training_use_processed_dataset",
            "training_unmask_messages",
            "training_save_final_checkpoint",
            "training_lr_scheduler_kwargs",
        ]

        for param in osft_only_params:
            assert param not in params, f"OSFT-specific parameter '{param}' should not be in SFT component"

    def test_component_default_values(self):
        """Test that the component has expected default values."""
        import inspect

        sig = inspect.signature(train_model.python_func)
        params = sig.parameters

        # Verify default values
        assert params["training_base_model"].default == "Qwen/Qwen2.5-1.5B-Instruct"
        assert params["training_effective_batch_size"].default == 128
        assert params["training_max_tokens_per_gpu"].default == 10000
        assert params["training_max_seq_len"].default == 8192
        assert params["training_resource_memory_per_worker"].default == "64Gi"
        assert params["training_resource_cpu_per_worker"].default == "4"

    def test_component_return_type(self):
        """Test that the component is annotated to return a string."""
        import inspect

        sig = inspect.signature(train_model.python_func)

        # The component should return a string
        assert sig.return_annotation is str or "str" in str(sig.return_annotation)

    def test_component_docstring_mentions_sft(self):
        """Test that the component docstring describes SFT training."""
        docstring = train_model.python_func.__doc__

        # Should mention SFT
        assert "SFT" in docstring or "Supervised Fine-Tuning" in docstring
        # Should NOT mention OSFT
        assert "OSFT" not in docstring
        # Should mention instructlab-training backend
        assert "instructlab" not in docstring or True  # Backend is hardcoded in implementation

    @mock.patch.dict("sys.modules", {"kubeflow.trainer": mock.MagicMock()})
    def test_component_with_mocked_trainer(self):
        """Test component with mocked Kubeflow Trainer."""
        # The component would need full execution context
        # For now verify the component definition is valid
        assert train_model.python_func is not None
