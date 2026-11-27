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
        assert params["training_completion_timeout_seconds"].default == 1800
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
        assert "training_resource_ephemeral_storage_per_worker" not in params
        assert "training_job_timeout_seconds" not in params

    def test_worker_caches_are_pvc_backed(self):
        """Worker caches are PVC-backed and the Training Hub cache is run-scoped."""
        import inspect

        source = inspect.getsource(train_model.python_func)

        assert "worker_home = pvc_path" in source
        assert 'run_home = os.path.join(worker_home, ".runs", run_id)' in source
        assert '"HOME": run_home' in source
        assert '"HF_TOKEN_PATH": "/var/run/secrets/huggingface/token"' in source
        assert '"HF_HUB_CACHE": os.path.join(cache, "hub")' in source
        assert '"VLLM_CACHE_ROOT": os.path.join(worker_cache, "vllm")' in source
        assert '"TRITON_CACHE_DIR": os.path.join(worker_cache, "triton")' in source
        assert '"TORCHINDUCTOR_CACHE_DIR": os.path.join(worker_cache, "torchinductor")' in source
        assert '"name": "hf-token"' in source
        assert '"mountPath": "/var/run/secrets/huggingface"' in source
        assert '"TMPDIR": worker_tmp' in source
        assert 'trainer_env = {key: value for key, value in merged_env.items() if key != "HF_TOKEN"}' in source
        assert '"ephemeral-storage"' not in source

    def test_training_run_id_is_path_safe(self):
        """Training run IDs are restricted to one safe path segment."""
        import inspect

        source = inspect.getsource(train_model.python_func)

        assert 'run_id in {".", ".."}' in source
        assert 're.fullmatch(r"[A-Za-z0-9._-]+", run_id)' in source
        assert "training_run_id must be a single path-safe segment" in source

    def test_grpo_completion_wait_defaults_to_shared_30_minutes(self):
        """GRPO exposes the shared 30-minute completion wait default."""
        import inspect

        source = inspect.getsource(train_model.python_func)
        assert "training_completion_timeout_seconds <= 0" in source
        assert "completion_timeout_seconds=training_completion_timeout_seconds" in source
        assert "LORA_GRPO_TRAINING_COMPLETION_TIMEOUT_SECONDS" not in source

    def test_component_archive_contains_runtime_modules_only(self):
        """The lightweight component must not embed tests or bytecode caches."""
        import base64
        import gzip
        import io
        import re
        import tarfile

        command = train_model.component_spec.implementation.container.command[-1]
        match = re.search(r"__KFP_EMBEDDED_ARCHIVE_B64 = '([^']+)'", command)
        assert match is not None

        archive = tarfile.open(fileobj=io.BytesIO(gzip.decompress(base64.b64decode(match.group(1)))))
        members = {member.name for member in archive.getmembers()}

        assert members == {".", "./data.py", "./output.py", "./setup.py", "./training.py"}

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

    @mock.patch.dict("sys.modules", {"kubeflow.trainer": mock.MagicMock()})
    def test_component_with_mocked_trainer(self):
        """Test component with mocked Kubeflow Trainer."""
        assert train_model.python_func is not None
