"""Tests for the autogluon_timeseries_training_pipeline pipeline."""

import tempfile
from pathlib import Path

import pytest
from kfp import compiler
from kfp_components.utils.pipeline_dag_tasks import (
    assert_compiled_pipeline_root_dag_task_ids,
)

from ..pipeline import autogluon_timeseries_training_pipeline

_EXPECTED_ROOT_DAG_TASK_IDS = (
    "condition-branches-1",
    "publish-component-stage-map",
    "timeseries-data-loader",
)


class TestAutogluonTimeseriesTrainingPipelineUnitTests:
    """Unit tests for pipeline logic."""

    def test_pipeline_function_exists(self):
        """Test that the pipeline function is properly imported."""
        assert callable(autogluon_timeseries_training_pipeline)

    def test_pipeline_compiles(self):
        """Test that the pipeline compiles successfully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            compiler.Compiler().compile(
                pipeline_func=autogluon_timeseries_training_pipeline,
                package_path=tmp_path,
            )
            assert Path(tmp_path).exists()
        except Exception as e:
            pytest.fail(f"Pipeline compilation failed: {e}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_pipeline_signature(self):
        """Test that the pipeline has the expected parameters and defaults."""
        expected_params = {
            "train_data_secret_name",
            "train_data_bucket_name",
            "train_data_file_key",
            "target",
            "id_column",
            "timestamp_column",
            "known_covariates_names",
            "prediction_length",
            "top_n",
            "preset",
            "eval_metric",
            "test_data_bucket_name",
            "test_data_file_key",
        }
        inputs = autogluon_timeseries_training_pipeline.component_spec.inputs
        params = set(inputs.keys())
        assert params == expected_params, f"Pipeline params {params} != expected {expected_params}"
        assert inputs["id_column"].default == ""
        assert inputs["prediction_length"].default == 1
        assert inputs["top_n"].default == 3
        assert inputs["known_covariates_names"].default == []
        assert inputs["preset"].default == "speed"
        assert inputs["eval_metric"].default == "mean_absolute_scaled_error"
        assert inputs["test_data_bucket_name"].default == ""
        assert inputs["test_data_file_key"].default == ""

    def test_compiled_pipeline_has_expected_inputs(self):
        """Test that compiled pipeline YAML contains expected pipeline input names."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            compiler.Compiler().compile(
                pipeline_func=autogluon_timeseries_training_pipeline,
                package_path=tmp_path,
            )
            content = Path(tmp_path).read_text()
            for name in (
                "train_data_secret_name",
                "train_data_bucket_name",
                "train_data_file_key",
                "target",
                "id_column",
                "timestamp_column",
                "known_covariates_names",
                "prediction_length",
                "top_n",
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
            pipeline_func=autogluon_timeseries_training_pipeline,
            expected_task_ids=_EXPECTED_ROOT_DAG_TASK_IDS,
        )

    def test_compiled_pipeline_yaml_is_ascii_only(self):
        """PipelineRuntimeManifest storage requires ASCII-only compiled YAML (MySQL utf8)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            compiler.Compiler().compile(
                pipeline_func=autogluon_timeseries_training_pipeline,
                package_path=tmp_path,
            )
            content_bytes = Path(tmp_path).read_bytes()
            try:
                content = content_bytes.decode("ascii")
            except UnicodeDecodeError as exc:
                pytest.fail(f"Compiled pipeline YAML must be ASCII-only: {exc}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        assert "componentInputParameter: eval_metric" in content
        assert "best_model_name:\n          parameterType: STRING" in content
        assert "componentInputParameter: preset" in content

    def test_compiled_pipeline_wires_preset_to_training_task(self):
        """Preset pipeline input is forwarded into the training task; medium_quality branch has higher resources."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            compiler.Compiler().compile(
                pipeline_func=autogluon_timeseries_training_pipeline,
                package_path=tmp_path,
            )
            content = Path(tmp_path).read_text(encoding="utf-8")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        assert "componentInputParameter: preset" in content
        assert "condition-branches-1" in content

    def test_compiled_pipeline_wires_mlflow_inputs_to_training(self):
        """MLflow inputs are forwarded into the training task.

        The training task now logs to MLflow; the standalone mlflow-logger step no longer exists.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            compiler.Compiler().compile(
                pipeline_func=autogluon_timeseries_training_pipeline,
                package_path=tmp_path,
            )
            content = Path(tmp_path).read_text(encoding="utf-8")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        assert "automl-mlflow-logger" not in content
        assert "exec-autogluon-timeseries-models-training:" in content
        assert "exec-autogluon-timeseries-models-training-2:" in content

    def test_compiled_pipeline_declares_speed_and_balanced_resource_tiers(self):
        """Speed and balanced preset branches request different training CPU/memory."""
        from kfp_components.utils.pipeline_task_resources import (
            assert_executor_resources,
            compile_executor_resources,
        )

        from .pipeline_resource_expectations import AUTOML_TIMESERIES_EXECUTOR_RESOURCES

        actual = compile_executor_resources(autogluon_timeseries_training_pipeline)
        assert_executor_resources(
            actual,
            {
                "autogluon-timeseries-models-training": AUTOML_TIMESERIES_EXECUTOR_RESOURCES[
                    "autogluon-timeseries-models-training"
                ],
                "autogluon-timeseries-models-training-2": AUTOML_TIMESERIES_EXECUTOR_RESOURCES[
                    "autogluon-timeseries-models-training-2"
                ],
            },
            pipeline_name="autogluon_timeseries_training_pipeline (training tiers only)",
            allow_extra=True,
        )

    def test_compiled_pipeline_wires_test_data_params_to_data_loader(self):
        """Test that pipeline wires test_data_bucket_name and test_data_file_key to data loader."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            compiler.Compiler().compile(
                pipeline_func=autogluon_timeseries_training_pipeline,
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

    def test_compiled_pipeline_uses_single_train_secret_mount(self):
        """Train secret is mounted once; test data reuses AWS_* via component fallback."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            compiler.Compiler().compile(
                pipeline_func=autogluon_timeseries_training_pipeline,
                package_path=tmp_path,
            )
            content = Path(tmp_path).read_text(encoding="utf-8")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        assert "condition-1" not in content
        assert "TEST_DATA_AWS_ACCESS_KEY_ID" not in content
        train_secret_block = content.split("envVar: AWS_ACCESS_KEY_ID", 1)[1]
        assert "optional: true" in train_secret_block[:500]
        assert "componentInputParameter: train_data_secret_name" in train_secret_block[:500]

