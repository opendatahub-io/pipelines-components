"""Unit tests for the LoRA GRPO training component."""

from unittest import mock

from ..component import train_model


class TestLoRAGRPOComponentUnitTests:
    """Unit tests for LoRA GRPO component logic."""

    def test_component_function_exists(self):
        """Test that the component function is properly imported."""
        assert callable(train_model)
        assert hasattr(train_model, "python_func")

    def test_component_has_expected_parameters(self):
        """Test that the component has expected input parameters."""
        import inspect

        sig = inspect.signature(train_model.python_func)
        params = list(sig.parameters.keys())

        expected_params = [
            "pvc_path",
            "output_model",
            "output_metrics",
            "dataset",
            "training_base_model",
        ]

        for param in expected_params:
            assert param in params, f"Expected parameter '{param}' not found in component"

    def test_component_has_grpo_specific_parameters(self):
        """Test that the component has GRPO-specific parameters."""
        import inspect

        sig = inspect.signature(train_model.python_func)
        params = list(sig.parameters.keys())

        grpo_params = [
            "training_num_iterations",
            "training_group_size",
            "training_prompt_batch_size",
            "training_n_train",
            "training_gpu_memory_utilization",
            "training_enforce_eager",
            "training_data_config",
            "training_lora_r",
            "training_lora_alpha",
        ]

        for param in grpo_params:
            assert param in params, f"Expected GRPO parameter '{param}' not found in component"

    def test_component_excludes_algorithm_backend_parameters(self):
        """Test that algorithm and backend parameters are NOT present (hardcoded)."""
        import inspect

        sig = inspect.signature(train_model.python_func)
        params = list(sig.parameters.keys())

        assert "training_algorithm" not in params, "training_algorithm should be hardcoded, not a parameter"
        assert "training_backend" not in params, "training_backend should be hardcoded, not a parameter"

    def test_component_excludes_sft_specific_parameters(self):
        """Test that SFT-specific parameters are NOT present."""
        import inspect

        sig = inspect.signature(train_model.python_func)
        params = list(sig.parameters.keys())

        sft_only_params = [
            "training_effective_batch_size",
            "training_max_tokens_per_gpu",
            "training_max_seq_len",
            "training_num_epochs",
            "training_save_samples",
            "training_accelerate_full_state_at_epoch",
            "training_fsdp_sharding_strategy",
            "training_checkpoint_at_epoch",
        ]

        for param in sft_only_params:
            assert param not in params, f"SFT-specific parameter '{param}' should not be in LoRA GRPO component"

        assert "output_loss_chart" not in params, (
            "output_loss_chart should not be in LoRA GRPO component (reward chart is in grpo_eval)"
        )

    def test_component_excludes_lora_sft_specific_parameters(self):
        """Test that LoRA SFT-specific parameters are NOT present."""
        import inspect

        sig = inspect.signature(train_model.python_func)
        params = list(sig.parameters.keys())

        lora_sft_only_params = [
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

        for param in lora_sft_only_params:
            assert param not in params, f"LoRA SFT-specific parameter '{param}' should not be in LoRA GRPO component"

    def test_component_excludes_osft_specific_parameters(self):
        """Test that OSFT-specific parameters are NOT present."""
        import inspect

        sig = inspect.signature(train_model.python_func)
        params = list(sig.parameters.keys())

        osft_only_params = [
            "training_unfreeze_rank_ratio",
            "training_osft_memory_efficient_init",
            "training_target_patterns",
        ]

        for param in osft_only_params:
            assert param not in params, f"OSFT-specific parameter '{param}' should not be in LoRA GRPO component"

    def test_component_excludes_multi_gpu_parameters(self):
        """Test that multi-GPU/multi-node parameters are NOT present (ART is single-GPU)."""
        import inspect

        sig = inspect.signature(train_model.python_func)
        params = list(sig.parameters.keys())

        multi_gpu_params = [
            "training_resource_num_procs_per_worker",
            "training_resource_num_workers",
            "training_enable_model_splitting",
        ]

        for param in multi_gpu_params:
            assert param not in params, (
                f"Multi-GPU parameter '{param}' should not be in LoRA GRPO component (ART is single-GPU)"
            )

    def test_component_default_values(self):
        """Test that the component has expected default values."""
        import inspect

        sig = inspect.signature(train_model.python_func)
        params = sig.parameters

        assert params["training_base_model"].default == "Qwen/Qwen3-4B"
        assert params["training_num_iterations"].default == 5
        assert params["training_group_size"].default == 4
        assert params["training_prompt_batch_size"].default == 50
        assert params["training_n_train"].default == 200
        assert params["training_gpu_memory_utilization"].default == 0.45
        assert params["training_enforce_eager"].default is True
        assert params["training_data_config"].default == "Qwen3"
        assert params["training_lora_r"].default == 16
        assert params["training_lora_alpha"].default == 8
        assert params["training_resource_memory_per_worker"].default == "64Gi"
        assert params["training_resource_cpu_per_worker"].default == "4"
        assert params["training_resource_gpu_per_worker"].default == 1

    def test_component_return_type(self):
        """Test that the component is annotated to return a string."""
        import inspect

        sig = inspect.signature(train_model.python_func)
        assert sig.return_annotation is str or "str" in str(sig.return_annotation)

    def test_component_docstring_mentions_grpo(self):
        """Test that the component docstring describes LoRA GRPO training."""
        docstring = train_model.python_func.__doc__

        assert "GRPO" in docstring
        assert "LoRA" in docstring
        assert "ART" in docstring
        assert "SFT" not in docstring
        assert "OSFT" not in docstring

    def test_component_base_image(self):
        """Test that the component uses the correct CPU base image."""
        decorator_kwargs = train_model.component_spec.implementation.container
        assert "odh-th-torch-cpu-py312" in decorator_kwargs.image

    @mock.patch.dict("sys.modules", {"kubeflow.trainer": mock.MagicMock()})
    def test_component_with_mocked_trainer(self):
        """Test component with mocked Kubeflow Trainer."""
        assert train_model.python_func is not None
