"""Tests for LoRA GRPO pipeline."""

import pytest
from kfp import compiler

from ..pipeline import lora_grpo_pipeline, validate_replica_settings


class TestLoraGrpoPipeline:
    """Basic tests for LoRA GRPO pipeline."""

    def test_pipeline_function_exists(self):
        """Test that the pipeline function is properly defined."""
        assert callable(lora_grpo_pipeline)

    @pytest.mark.parametrize("min_replicas,max_replicas", [(0, 1), (2, 1)])
    def test_replica_validation_rejects_invalid_bounds(self, min_replicas, max_replicas):
        """Serving replicas must satisfy the KServe min/max relationship."""
        validate_replica_settings.python_func(min_replicas=1, max_replicas=1)
        with pytest.raises(ValueError, match="1 <= min_replicas <= max_replicas"):
            validate_replica_settings.python_func(
                min_replicas=min_replicas,
                max_replicas=max_replicas,
            )

    def test_tokenizer_mode_is_required(self):
        """Deployment tokenizer mode must be selected for the target model."""
        parameter = lora_grpo_pipeline.component_spec.inputs["p06_tokenizer_mode"]
        assert parameter.default is None
        assert parameter.optional is False

    def test_training_ephemeral_storage_is_not_pipeline_configurable(self):
        """The TrainJob worker's local storage reservation is an internal setting."""
        assert "p02_train_opt_ephemeral_storage" not in lora_grpo_pipeline.component_spec.inputs

    def test_pipeline_uses_training_hub_for_hugging_face_data(self):
        """The pipeline exposes dataset preparation controls."""
        inputs = lora_grpo_pipeline.component_spec.inputs
        assert "p02_dataset_uri" in inputs
        for required_input in (
            "p02_dataset_uri",
            "p03_training_model",
        ):
            assert inputs[required_input].default is None
            assert inputs[required_input].optional is False
        assert inputs["p28_training_data_config"].default == ""
        assert inputs["p28_training_data_config"].optional is True
        assert inputs["p07_dataset_max_invalid_row_ratio"].default == 0.1
        assert inputs["p08_dataset_data_connection_name"].default == "s3-secret"
        assert all(not name.startswith("phase_") for name in inputs)
        assert "data_connection_name" not in inputs
        assert "p02_dataset_data_split" not in inputs
        assert "p02_dataset_subset" not in inputs
        assert "p02_dataset_use_hf_direct" not in inputs

    def test_training_timeout_is_not_a_pipeline_parameter(self):
        """The GRPO-specific timeout does not change the public pipeline interface."""
        assert "p02_training_job_timeout_seconds" not in lora_grpo_pipeline.component_spec.inputs

    def test_training_completion_timeout_defaults_to_two_hours(self):
        """The pipeline allows up to two hours for TrainJob completion by default."""
        parameter = lora_grpo_pipeline.component_spec.inputs["p15_training_completion_timeout_seconds"]
        assert parameter.default == 7200
        assert parameter.optional is True

    def test_pipeline_compiles(self, tmp_path):
        """Test that the pipeline compiles successfully."""
        output_path = tmp_path / "pipeline.yaml"
        compiler.Compiler().compile(
            pipeline_func=lora_grpo_pipeline,
            package_path=str(output_path),
        )
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_dataset_preparation_wires_training_inputs(self, tmp_path):
        """The compiled graph prepares data and passes its outputs to training."""
        output_path = tmp_path / "pipeline.yaml"
        compiler.Compiler().compile(
            pipeline_func=lora_grpo_pipeline,
            package_path=str(output_path),
        )
        compiled = output_path.read_text()

        assert "prepare_grpo_dataset" in compiled
        assert "resolved_data_path" in compiled
        assert "resolved_data_config" in compiled
        assert "training_use_hf_direct" in compiled
        assert "componentInputParameter: p28_training_data_config" in compiled
        assert "AWS_ACCESS_KEY_ID" in compiled
        assert "secretNameParameter" in compiled
        assert "p15_training_completion_timeout_seconds" in compiled
        assert "LORA_GRPO_TRAINING_COMPLETION_TIMEOUT_SECONDS" not in compiled
        assert "model_run_id" in compiled
        assert "final_model/{{$.pipeline_job_uuid}}" not in compiled
        assert "- validate-replica-settings" in compiled

    def test_registry_sdk_pin_is_grpo_specific(self, tmp_path):
        """The GRPO artifact pins the registry SDK without changing the shared component."""
        output_path = tmp_path / "pipeline.yaml"
        compiler.Compiler().compile(
            pipeline_func=lora_grpo_pipeline,
            package_path=str(output_path),
        )
        compiled = output_path.read_text()

        assert "model-registry==0.3.9" in compiled
        assert "model-registry==0.3.4" not in compiled
