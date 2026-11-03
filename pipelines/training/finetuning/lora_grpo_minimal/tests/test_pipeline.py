"""Tests for the LoRA GRPO minimal pipeline."""

import pytest
from kfp import compiler

from ..pipeline import lora_grpo_minimal_pipeline, validate_replica_settings


class TestLoraGrpoMinimalPipeline:
    """Basic tests for the LoRA GRPO minimal pipeline."""

    def test_pipeline_function_exists(self):
        """The minimal pipeline is importable and callable."""
        assert callable(lora_grpo_minimal_pipeline)

    @pytest.mark.parametrize("min_replicas,max_replicas", [(0, 1), (2, 1)])
    def test_replica_validation_rejects_invalid_bounds(self, min_replicas, max_replicas):
        """Serving replicas must satisfy the KServe min/max relationship."""
        validate_replica_settings.python_func(min_replicas=1, max_replicas=1)
        with pytest.raises(ValueError, match="1 <= min_replicas <= max_replicas"):
            validate_replica_settings.python_func(
                min_replicas=min_replicas,
                max_replicas=max_replicas,
            )

    def test_completion_timeout_is_exposed_with_quick_start_default(self):
        """Users can adjust the TrainJob completion wait for their workload."""
        parameter = lora_grpo_minimal_pipeline.component_spec.inputs["p15_training_completion_timeout_seconds"]
        assert parameter.default == 7200

    def test_dataset_config_is_combined_with_dataset_uri(self):
        """The minimal interface uses the inline Hugging Face URI config."""
        inputs = lora_grpo_minimal_pipeline.component_spec.inputs
        assert "p28_training_data_config" not in inputs
        assert all(not name.startswith("phase_") for name in inputs)

    def test_s3_connection_secret_is_configurable(self):
        """The minimal interface keeps the conventional RHOAI S3 default."""
        parameter = lora_grpo_minimal_pipeline.component_spec.inputs["p08_dataset_data_connection_name"]
        assert parameter.default == "s3-secret"

    def test_minimum_deployment_replicas_are_fixed_for_quick_start(self):
        """The quick-start deployment keeps one predictor replica warm."""
        inputs = lora_grpo_minimal_pipeline.component_spec.inputs
        assert "p39_serving_min_replicas" not in inputs
        assert "p25_serving_max_replicas" in inputs

    def test_pipeline_compiles(self, tmp_path):
        """The minimal pipeline compiles successfully."""
        output_path = tmp_path / "pipeline.yaml"
        compiler.Compiler().compile(
            pipeline_func=lora_grpo_minimal_pipeline,
            package_path=str(output_path),
        )
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_dataset_preparation_depends_on_replica_validation(self, tmp_path):
        """Invalid replica settings fail before dataset preparation can run."""
        output_path = tmp_path / "pipeline.yaml"
        compiler.Compiler().compile(
            pipeline_func=lora_grpo_minimal_pipeline,
            package_path=str(output_path),
        )
        compiled = output_path.read_text()

        assert "- validate-replica-settings" in compiled
