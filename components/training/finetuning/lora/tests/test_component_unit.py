"""Unit tests for the LoRA training component."""

from unittest import mock

from ..component import train_model


class TestLoRAComponentUnitTests:
    """Unit tests for LoRA component logic."""

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

    def test_component_has_lora_specific_parameters(self):
        """Test that the component has LoRA-specific parameters."""
        import inspect

        sig = inspect.signature(train_model.python_func)
        params = list(sig.parameters.keys())

        # Verify LoRA-specific parameters exist
        lora_params = [
            "training_lora_r",
            "training_lora_alpha",
            "training_lora_dropout",
            "training_lora_target_modules",
            "training_lora_use_rslora",
            "training_lora_use_dora",
            "training_lora_load_in_4bit",
            "training_lora_load_in_8bit",
            "training_lora_bnb_4bit_quant_type",
            "training_lora_bnb_4bit_compute_dtype",
            "training_lora_bnb_4bit_use_double_quant",
            "training_lora_sample_packing",
        ]

        for param in lora_params:
            assert param in params, f"Expected LoRA parameter '{param}' not found in component"

    def test_component_excludes_algorithm_backend_parameters(self):
        """Test that algorithm and backend parameters are NOT present (hardcoded)."""
        import inspect

        sig = inspect.signature(train_model.python_func)
        params = list(sig.parameters.keys())

        # These should NOT be present as they are hardcoded
        assert "training_algorithm" not in params, "training_algorithm should be hardcoded, not a parameter"
        assert "training_backend" not in params, "training_backend should be hardcoded, not a parameter"

    def test_component_excludes_sft_specific_parameters(self):
        """Test that SFT-specific parameters are NOT present."""
        import inspect

        sig = inspect.signature(train_model.python_func)
        params = list(sig.parameters.keys())

        # These SFT-specific params should NOT be present in LoRA component
        sft_only_params = [
            "training_save_samples",
            "training_accelerate_full_state_at_epoch",
        ]

        for param in sft_only_params:
            assert param not in params, f"SFT-specific parameter '{param}' should not be in LoRA component"

    def test_component_excludes_osft_specific_parameters(self):
        """Test that OSFT-specific parameters are NOT present."""
        import inspect

        sig = inspect.signature(train_model.python_func)
        params = list(sig.parameters.keys())

        # These OSFT-specific params should NOT be present in LoRA component
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
            assert param not in params, f"OSFT-specific parameter '{param}' should not be in LoRA component"

    def test_component_default_values(self):
        """Test that the component has expected default values."""
        import inspect

        sig = inspect.signature(train_model.python_func)
        params = sig.parameters

        # Verify default values (LoRA-specific defaults)
        assert params["training_base_model"].default == "Qwen/Qwen2.5-1.5B-Instruct"
        assert params["training_effective_batch_size"].default == 128
        assert params["training_max_tokens_per_gpu"].default == 32000  # Higher than SFT
        assert params["training_max_seq_len"].default == 8192
        assert params["training_resource_memory_per_worker"].default == "32Gi"  # Lower than SFT
        assert params["training_resource_cpu_per_worker"].default == "4"

        # LoRA-specific defaults
        assert params["training_lora_r"].default == 16
        assert params["training_lora_alpha"].default == 32
        assert params["training_lora_dropout"].default == 0.0
        assert params["training_lora_target_modules"].default == ""

    def test_component_return_type(self):
        """Test that the component is annotated to return a string."""
        import inspect

        sig = inspect.signature(train_model.python_func)

        # The component should return a string
        assert sig.return_annotation is str or "str" in str(sig.return_annotation)

    def test_component_docstring_mentions_lora(self):
        """Test that the component docstring describes LoRA training."""
        docstring = train_model.python_func.__doc__

        # Should mention LoRA
        assert "LoRA" in docstring or "Low-Rank Adaptation" in docstring
        # Should NOT mention SFT or OSFT
        assert "SFT" not in docstring or "Supervised Fine-Tuning" not in docstring
        assert "OSFT" not in docstring

    @mock.patch.dict("sys.modules", {"kubeflow.trainer": mock.MagicMock()})
    def test_component_with_mocked_trainer(self):
        """Test component with mocked Kubeflow Trainer."""
        # The component would need full execution context
        # For now verify the component definition is valid
        assert train_model.python_func is not None
