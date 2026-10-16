"""Tests for LoRA GRPO pipeline."""

from kfp import compiler

from ..pipeline import lora_grpo_pipeline


class TestLoraGrpoPipeline:
    """Basic tests for LoRA GRPO pipeline."""

    def test_pipeline_function_exists(self):
        """Test that the pipeline function is properly defined."""
        assert callable(lora_grpo_pipeline)

    def test_tokenizer_mode_defaults_to_auto(self):
        """Deployment tokenizer mode is configurable and generic by default."""
        parameter = lora_grpo_pipeline.component_spec.inputs["phase_05_deploy_opt_tokenizer_mode"]
        assert parameter.default == "auto"

    def test_training_ephemeral_storage_is_not_pipeline_configurable(self):
        """The TrainJob worker's local storage reservation is an internal setting."""
        assert "phase_02_train_opt_ephemeral_storage" not in lora_grpo_pipeline.component_spec.inputs

    def test_pipeline_uses_training_hub_for_hugging_face_data(self):
        """The pipeline exposes a Hugging Face dataset URI directly to training."""
        inputs = lora_grpo_pipeline.component_spec.inputs
        assert "phase_01_dataset_man_data_uri" in inputs
        assert "phase_01_dataset_man_data_split" not in inputs
        assert "phase_01_dataset_opt_subset" not in inputs
        assert "phase_01_dataset_opt_use_hf_direct" not in inputs

    def test_training_timeout_is_not_a_pipeline_parameter(self):
        """The GRPO-specific timeout does not change the public pipeline interface."""
        assert "phase_02_train_opt_job_timeout_seconds" not in lora_grpo_pipeline.component_spec.inputs

    def test_pipeline_compiles(self, tmp_path):
        """Test that the pipeline compiles successfully."""
        output_path = tmp_path / "pipeline.yaml"
        compiler.Compiler().compile(
            pipeline_func=lora_grpo_pipeline,
            package_path=str(output_path),
        )
        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_training_uses_hugging_face_loader(self, tmp_path):
        """The compiled graph has one direct-HF training task and no dataset task."""
        output_path = tmp_path / "pipeline.yaml"
        compiler.Compiler().compile(
            pipeline_func=lora_grpo_pipeline,
            package_path=str(output_path),
        )
        compiled = output_path.read_text()

        assert "dataset_download" not in compiled
        assert "direct-huggingface-dataset" not in compiled
        assert "materialize-dataset" not in compiled
        assert "training_use_hf_direct" in compiled
        assert "Agent-Ark/Toucan-1.5M:Qwen3" in compiled
        assert "LORA_GRPO_TRAINING_COMPLETION_TIMEOUT_SECONDS" in compiled

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
