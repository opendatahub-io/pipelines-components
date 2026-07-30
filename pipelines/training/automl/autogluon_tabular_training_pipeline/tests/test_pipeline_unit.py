"""Tests for the autogluon_tabular_training_pipeline pipeline."""

import tempfile
from pathlib import Path

import pytest
from kfp import compiler
from kfp_components.utils.pipeline_dag_tasks import (
    assert_compiled_pipeline_root_dag_task_ids,
)

from ..pipeline import autogluon_tabular_training_pipeline

# Root DAG task IDs from ``root.dag.tasks`` (fresh compile). Update when the graph changes.
_EXPECTED_ROOT_DAG_TASK_IDS = (
    "condition-branches-1",
    "automl-data-loader",
    "publish-component-stage-map",
)


class TestAutogluonTabularTrainingPipelineUnitTests:
    """Unit tests for pipeline logic."""

    def test_pipeline_function_exists(self):
        """Test that the pipeline function is properly imported."""
        assert callable(autogluon_tabular_training_pipeline)

    def test_pipeline_compiles(self):
        """Test that the pipeline compiles successfully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            compiler.Compiler().compile(
                pipeline_func=autogluon_tabular_training_pipeline,
                package_path=tmp_path,
            )
            assert Path(tmp_path).exists()
        except Exception as e:
            pytest.fail(f"Pipeline compilation failed: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_pipeline_signature(self):
        """Test that the pipeline has the expected parameters."""
        # KFP pipelines expose parameters via component_spec.inputs, not inspect.signature
        expected_params = {
            "train_data_secret_name",
            "train_data_bucket_name",
            "train_data_file_key",
            "label_column",
            "task_type",
            "top_n",
            "positive_class",
            "preset",
            "eval_metric",
            "test_data_secret_name",
            "test_data_bucket_name",
            "test_data_file_key",
        }
        inputs = autogluon_tabular_training_pipeline.component_spec.inputs
        params = set(inputs.keys())
        assert params == expected_params, f"Pipeline params {params} != expected {expected_params}"
        assert inputs["top_n"].default == 3
        assert inputs["preset"].default == "speed"
        assert inputs["eval_metric"].default == ""
        assert inputs["test_data_secret_name"].default == ""
        assert inputs["test_data_bucket_name"].default == ""
        assert inputs["test_data_file_key"].default == ""

    def test_compiled_pipeline_has_expected_inputs(self):
        """Test that the compiled pipeline YAML contains expected pipeline inputs."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            compiler.Compiler().compile(
                pipeline_func=autogluon_tabular_training_pipeline,
                package_path=tmp_path,
            )
            content = Path(tmp_path).read_text()
            for name in (
                "train_data_secret_name",
                "train_data_bucket_name",
                "train_data_file_key",
                "label_column",
                "task_type",
                "top_n",
                "positive_class",
                "preset",
                "eval_metric",
                "test_data_bucket_name",
                "test_data_file_key",
            ):
                assert name in content, f"Expected pipeline input '{name}' in compiled YAML"
        except Exception as e:
            pytest.fail(f"Pipeline compilation or validation failed: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_compiled_pipeline_root_dag_task_ids(self):
        """Root-level step IDs are stable; renames or add/remove steps require updating expectations."""
        assert_compiled_pipeline_root_dag_task_ids(
            pipeline_func=autogluon_tabular_training_pipeline,
            expected_task_ids=_EXPECTED_ROOT_DAG_TASK_IDS,
        )

    def test_compiled_pipeline_yaml_is_ascii_only(self):
        """PipelineRuntimeManifest storage requires ASCII-only compiled YAML (MySQL utf8)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            compiler.Compiler().compile(
                pipeline_func=autogluon_tabular_training_pipeline,
                package_path=tmp_path,
            )
            content = Path(tmp_path).read_bytes()
            try:
                content.decode("ascii")
            except UnicodeDecodeError as exc:
                pytest.fail(f"Compiled pipeline YAML must be ASCII-only: {exc}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_compiled_pipeline_wires_loader_outputs_to_training_task(self):
        """Training step consumes data-loader outputs (sample_row, paths, configs)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            compiler.Compiler().compile(
                pipeline_func=autogluon_tabular_training_pipeline,
                package_path=tmp_path,
            )
            content = Path(tmp_path).read_text(encoding="utf-8")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        assert "dependentTasks:" in content
        assert "autogluon-models-training:" in content
        assert "automl-data-loader" in content
        assert "producerTask: automl-data-loader" in content
        assert "outputParameterKey: sample_row" in content
        assert "outputParameterKey: models_selection_train_data_path" in content
        assert "outputParameterKey: split_config" in content
        assert "outputParameterKey: sample_config" in content
        assert "outputParameterKey: extra_train_data_path" in content
        assert "positive_class" in content

    def test_compiled_pipeline_wires_eval_metric_to_training_task(self):
        """eval_metric pipeline input is forwarded into the training task."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            compiler.Compiler().compile(
                pipeline_func=autogluon_tabular_training_pipeline,
                package_path=tmp_path,
            )
            content = Path(tmp_path).read_text(encoding="utf-8")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        assert "componentInputParameter: eval_metric" in content
        assert "best_model_name:\n          parameterType: STRING" in content

    def test_compiled_pipeline_wires_preset_to_training_task(self):
        """Preset pipeline input is forwarded into the training task; good_quality branch has higher resources."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            compiler.Compiler().compile(
                pipeline_func=autogluon_tabular_training_pipeline,
                package_path=tmp_path,
            )
            content = Path(tmp_path).read_text(encoding="utf-8")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        assert "componentInputParameter: preset" in content
        assert "condition-branches-1" in content

    def test_compiled_pipeline_declares_speed_and_balanced_resource_tiers(self):
        """Speed and balanced preset branches request different training CPU/memory."""
        from kfp_components.utils.pipeline_task_resources import (
            assert_executor_resources,
            compile_executor_resources,
        )

        from .pipeline_resource_expectations import AUTOML_TABULAR_EXECUTOR_RESOURCES

        actual = compile_executor_resources(autogluon_tabular_training_pipeline)
        assert_executor_resources(
            actual,
            {
                "autogluon-models-training": AUTOML_TABULAR_EXECUTOR_RESOURCES["autogluon-models-training"],
                "autogluon-models-training-2": AUTOML_TABULAR_EXECUTOR_RESOURCES["autogluon-models-training-2"],
            },
            pipeline_name="autogluon_tabular_training_pipeline (training tiers only)",
            allow_extra=True,
        )

    def test_compiled_pipeline_data_loader_declares_task_type_and_label(self):
        """Tabular data loader component exposes task_type and label_column inputs."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            compiler.Compiler().compile(
                pipeline_func=autogluon_tabular_training_pipeline,
                package_path=tmp_path,
            )
            content = Path(tmp_path).read_text(encoding="utf-8")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        assert "comp-automl-data-loader:" in content
        assert "task_type:" in content
        assert "label_column:" in content
        assert "defaultValue: regression" in content

    def test_compiled_pipeline_wires_test_data_params_to_data_loader(self):
        """Test that pipeline wires test_data_bucket_name and test_data_file_key to data loader."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            compiler.Compiler().compile(
                pipeline_func=autogluon_tabular_training_pipeline,
                package_path=tmp_path,
            )
            content = Path(tmp_path).read_text(encoding="utf-8")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Verify test data parameters are passed to data loader
        assert "test_data_bucket_name:" in content
        assert "test_data_file_key:" in content
        assert "componentInputParameter: test_data_bucket_name" in content
        assert "componentInputParameter: test_data_file_key" in content

    def test_compiled_pipeline_wires_evaluation_mode_to_training(self):
        """Test that evaluation_mode output from data loader is wired to training component."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            compiler.Compiler().compile(
                pipeline_func=autogluon_tabular_training_pipeline,
                package_path=tmp_path,
            )
            content = Path(tmp_path).read_text(encoding="utf-8")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Verify evaluation_mode is wired from data loader to training
        assert "evaluation_mode:" in content
        assert "outputParameterKey: evaluation_mode" in content
        assert "producerTask: automl-data-loader" in content
