"""Tests for the autogluon_timeseries_training_pipeline pipeline."""

import json
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
            "test_data_secret_name",
            "test_data_bucket_name",
            "test_data_file_key",
        }
        inputs = autogluon_timeseries_training_pipeline.component_spec.inputs
        params = set(inputs.keys())
        assert params == expected_params, f"Pipeline params {params} != expected {expected_params}"
        assert inputs["prediction_length"].default == 1
        assert inputs["top_n"].default == 3
        assert inputs["known_covariates_names"].default == []
        assert inputs["preset"].default == "speed"
        assert inputs["eval_metric"].default == "mean_absolute_scaled_error"
        assert inputs["test_data_secret_name"].default == ""
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
                "test_data_secret_name",
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

    def test_compiled_pipeline_wires_evaluation_mode_to_training(self):
        """Test that evaluation_mode output from data loader is wired to training component."""
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

        # Verify evaluation_mode is wired from data loader to training
        assert "evaluation_mode:" in content
        assert "outputParameterKey: evaluation_mode" in content
        assert "producerTask: timeseries-data-loader" in content


class TestTimeseriesTestConfigs:
    """Unit tests for test_configs.json loading (integration configs live in autox-ci)."""

    def test_load_configs_rejects_blank_eval_metric(self, tmp_path):
        """Blank eval_metric values fail at config load time."""
        from . import test_configs

        bad = tmp_path / "configs.json"
        bad.write_text(
            json.dumps(
                [
                    {
                        "id": "cfg-1",
                        "dataset_path": "data/timeseries_sales.csv",
                        "target": "target",
                        "id_column": "item_id",
                        "timestamp_column": "timestamp",
                        "known_covariates_names": ["promo"],
                        "prediction_length": 2,
                        "top_n": 2,
                        "tags": [],
                        "eval_metric": "   ",
                    }
                ]
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=r"test_configs\.json\[0\] 'eval_metric'"):
            test_configs._load_configs(bad)

    def test_load_configs_rejects_non_string_eval_metric(self, tmp_path):
        """Non-string eval_metric values fail at config load time."""
        from . import test_configs

        bad = tmp_path / "configs.json"
        bad.write_text(
            json.dumps(
                [
                    {
                        "id": "cfg-1",
                        "dataset_path": "data/timeseries_sales.csv",
                        "target": "target",
                        "id_column": "item_id",
                        "timestamp_column": "timestamp",
                        "known_covariates_names": ["promo"],
                        "prediction_length": 2,
                        "top_n": 2,
                        "tags": [],
                        "eval_metric": 123,
                    }
                ]
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=r"test_configs\.json\[0\] 'eval_metric'"):
            test_configs._load_configs(bad)

    def test_load_configs_accepts_valid_eval_metric(self, tmp_path):
        """Valid eval_metric is stripped and forwarded through pipeline arguments."""
        from . import test_configs

        path = tmp_path / "configs.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "id": "cfg-1",
                        "dataset_path": "data/timeseries_sales.csv",
                        "target": "target",
                        "id_column": "item_id",
                        "timestamp_column": "timestamp",
                        "known_covariates_names": ["promo"],
                        "prediction_length": 2,
                        "top_n": 2,
                        "tags": [],
                        "eval_metric": " WQL ",
                    }
                ]
            ),
            encoding="utf-8",
        )
        loaded = test_configs._load_configs(path)
        assert len(loaded) == 1
        assert loaded[0].eval_metric == "WQL"
        args = loaded[0].get_pipeline_arguments("bucket", "key", "secret")
        assert args["eval_metric"] == "WQL"

    def test_load_configs_absent_known_covariates_uses_pipeline_default(self, tmp_path):
        """When known_covariates_names is absent from JSON, pipeline arguments omit the key so the
        pipeline default ([]) applies — this exercises the bug path fixed in RHOAIENG-71419.
        """  # noqa: D205
        from . import test_configs

        path = tmp_path / "configs.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "id": "cfg-no-covariates",
                        "dataset_path": "data/timeseries_sales.csv",
                        "target": "target",
                        "id_column": "item_id",
                        "timestamp_column": "timestamp",
                        "prediction_length": 2,
                        "top_n": 2,
                        "tags": [],
                    }
                ]
            ),
            encoding="utf-8",
        )
        loaded = test_configs._load_configs(path)
        assert len(loaded) == 1
        assert loaded[0].known_covariates_names is None
        args = loaded[0].get_pipeline_arguments("bucket", "key", "secret")
        assert "known_covariates_names" not in args

    def test_load_configs_explicit_known_covariates_included_in_args(self, tmp_path):
        """When known_covariates_names is provided in JSON, it is forwarded into pipeline arguments."""
        from . import test_configs

        path = tmp_path / "configs.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "id": "cfg-with-covariates",
                        "dataset_path": "data/timeseries_sales.csv",
                        "target": "target",
                        "id_column": "item_id",
                        "timestamp_column": "timestamp",
                        "known_covariates_names": ["promo", "holiday"],
                        "prediction_length": 2,
                        "top_n": 2,
                        "tags": [],
                    }
                ]
            ),
            encoding="utf-8",
        )
        loaded = test_configs._load_configs(path)
        assert loaded[0].known_covariates_names == ["promo", "holiday"]
        args = loaded[0].get_pipeline_arguments("bucket", "key", "secret")
        assert args["known_covariates_names"] == ["promo", "holiday"]
