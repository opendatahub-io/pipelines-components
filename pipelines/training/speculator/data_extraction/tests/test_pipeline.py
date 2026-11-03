"""Tests for the managed Speculator data-only pipeline."""

from kfp import compiler

from ..pipeline import speculator_hidden_state_extraction_pipeline

PUBLIC_INPUTS = [
    "p01_persistent_pvc",
    "p02_verifier_model",
    "p03_verifier_model_pvc",
    "p04_download_model",
    "p05_dataset_name",
    "p06_output_dir",
    "p07_hidden_states_path",
    "p08_max_samples",
    "p09_regenerate_responses",
    "p10_vllm_mode",
    "p11_vllm_endpoint",
    "p12_vllm_resource_gpu",
    "p13_vllm_resource_memory",
    "p14_vllm_gpu_memory_utilization",
    "p15_vllm_readiness_timeout_minutes",
    "p16_total_seq_len",
    "p17_speculator_type",
    "p18_target_layer_ids",
    "p19_data_extraction_concurrency",
    "p20_packages_to_install",
    "p21_pip_index_urls",
    "p22_job_envs",
    "p23_job_runtime",
]


class TestSpeculatorHiddenStateExtractionPipeline:
    """Basic tests for the data-only pipeline."""

    def test_pipeline_function_exists(self):
        """Test that the pipeline function is defined."""
        assert callable(speculator_hidden_state_extraction_pipeline)

    def test_verifier_model_inputs_do_not_assume_cached_models(self):
        """The pipeline does not assume a model was previously downloaded to the PVC."""
        inputs = speculator_hidden_state_extraction_pipeline.component_spec.inputs

        assert inputs["p01_persistent_pvc"].default is None
        assert inputs["p02_verifier_model"].default == ""
        assert inputs["p03_verifier_model_pvc"].default == ""

    def test_public_input_names_preserve_the_ui_order(self):
        """Zero-padded input names keep KFP's sorted parameter list grouped."""
        inputs = speculator_hidden_state_extraction_pipeline.component_spec.inputs

        assert list(inputs) == PUBLIC_INPUTS

    def test_vllm_mode_is_the_public_control(self):
        """The pipeline exposes the deployment mode, not an ambiguous source name."""
        inputs = speculator_hidden_state_extraction_pipeline.component_spec.inputs

        assert inputs["p10_vllm_mode"].default == "managed"
        assert "vllm_source" not in inputs

    def test_extraction_controls_have_concise_public_names(self):
        """Extraction controls use names appropriate for the pipeline UI."""
        inputs = speculator_hidden_state_extraction_pipeline.component_spec.inputs

        assert inputs["p08_max_samples"].default == 16
        assert inputs["p09_regenerate_responses"].default is False
        assert "data_extraction_max_samples" not in inputs
        assert "data_extraction_regenerate_responses" not in inputs

    def test_job_runtime_controls_have_clear_public_names(self):
        """The pipeline exposes job controls without Trainer-specific prefixes."""
        inputs = speculator_hidden_state_extraction_pipeline.component_spec.inputs

        assert inputs["p22_job_envs"].default == "NCCL_DEBUG=INFO,PYTHONUNBUFFERED=1"
        assert inputs["p23_job_runtime"].default == "vllm-extract-cuda"
        assert "training_envs" not in inputs
        assert "training_runtime" not in inputs

    def test_progression_metrics_are_not_public_controls(self):
        """Progress tracking uses the extraction component's defaults."""
        inputs = speculator_hidden_state_extraction_pipeline.component_spec.inputs

        assert "enable_progression_tracking" not in inputs
        assert "metrics_port" not in inputs
        assert "metrics_poll_interval_seconds" not in inputs

    def test_pipeline_compiles(self, tmp_path):
        """Test that the pipeline compiles successfully."""
        import yaml

        output_path = tmp_path / "pipeline.yaml"
        compiler.Compiler().compile(
            pipeline_func=speculator_hidden_state_extraction_pipeline,
            package_path=str(output_path),
        )
        assert output_path.exists()
        compiled = output_path.read_text()
        assert "speculator-hidden-state-extraction-pipeline" in compiled
        assert "verifier hidden states for speculative decoding training" in compiled
        assert "Name of the existing PVC in the pipeline namespace" in compiled
        assert "Required when using ``p03_verifier_model_pvc``" in compiled
        assert "Re-generate the assistant responses in the dataset" in compiled
        assert "managed-vllm" in compiled
        assert "remote-vllm" in compiled
        assert "extract-hidden-states" in compiled
        assert "vllm-connected" not in compiled
        assert "vllm-disconnected" not in compiled
        assert "extract-speculator" not in compiled

        pipeline_spec = next(yaml.safe_load_all(compiled))
        compiled_inputs = pipeline_spec["root"]["inputDefinitions"]["parameters"]
        assert list(compiled_inputs) == PUBLIC_INPUTS
        target_layer_description = compiled_inputs["p18_target_layer_ids"]["description"]
        assert "Required when using" in target_layer_description
        assert "p03_verifier_model_pvc" in target_layer_description
        assert "p02_verifier_model" in target_layer_description
        assert "managed mode" in target_layer_description
        assert "download_model" in target_layer_description

        def has_mode_mapping(value):
            if isinstance(value, dict):
                source = value.get("vllm_source")
                parameter = source.get("componentInputParameter", "") if isinstance(source, dict) else ""
                if parameter.endswith("vllm_mode"):
                    return True
                return any(has_mode_mapping(child) for child in value.values())
            if isinstance(value, list):
                return any(has_mode_mapping(child) for child in value)
            return False

        assert has_mode_mapping(pipeline_spec)
