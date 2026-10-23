"""Tests for the LoRA GRPO minimal pipeline."""

from kfp import compiler

from ..pipeline import lora_grpo_minimal_pipeline


class TestLoraGrpoMinimalPipeline:
    """Basic tests for the LoRA GRPO minimal pipeline."""

    def test_pipeline_function_exists(self):
        """The minimal pipeline is importable and callable."""
        assert callable(lora_grpo_minimal_pipeline)

    def test_completion_timeout_is_exposed_with_quick_start_default(self):
        """Users can adjust the TrainJob completion wait for their workload."""
        parameter = lora_grpo_minimal_pipeline.component_spec.inputs["phase_02_train_opt_completion_timeout_seconds"]
        assert parameter.default == 7200

    def test_dataset_config_is_combined_with_dataset_uri(self):
        """The minimal interface uses the inline Hugging Face URI config."""
        assert "phase_02_train_opt_data_config" not in lora_grpo_minimal_pipeline.component_spec.inputs

    def test_s3_connection_secret_uses_the_quick_start_name(self):
        """The minimal interface uses the conventional RHOAI S3 Secret."""
        assert "phase_01_dataset_opt_data_connection_name" not in lora_grpo_minimal_pipeline.component_spec.inputs

    def test_minimum_deployment_replicas_are_fixed_for_quick_start(self):
        """The quick-start deployment keeps one predictor replica warm."""
        assert "phase_05_deploy_opt_min_replicas" not in lora_grpo_minimal_pipeline.component_spec.inputs

    def test_pipeline_compiles(self, tmp_path):
        """The minimal pipeline compiles successfully."""
        output_path = tmp_path / "pipeline.yaml"
        compiler.Compiler().compile(
            pipeline_func=lora_grpo_minimal_pipeline,
            package_path=str(output_path),
        )
        assert output_path.exists()
        assert output_path.stat().st_size > 0
