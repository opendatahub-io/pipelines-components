"""Unit tests for pipeline executor resource helpers and tabular tier matrix."""

import pytest
from kfp_components.utils.pipeline_task_resources import (
    ExecutorResources,
    assert_executor_resources,
    compile_executor_resources,
    normalize_executor_name,
)

from ..pipeline import autogluon_tabular_training_pipeline
from .pipeline_resource_expectations import (
    AUTOML_TABULAR_EXECUTOR_RESOURCES,
    TRAINING_BALANCED_RESOURCES,
    TRAINING_HEAVY_RESOURCES,
    TRAINING_SPEED_RESOURCES,
)


class TestPipelineTaskResourcesHelpers:
    """Tests for compile/assert helpers in pipeline_task_resources."""

    def test_normalize_executor_name_strips_prefix(self):
        """Executor keys drop the ``exec-`` prefix for stable task names."""
        assert normalize_executor_name("exec-automl-data-loader") == "automl-data-loader"

    def test_assert_executor_resources_detects_cpu_change(self):
        """Mismatched CPU requests fail with the task name in the error."""
        actual = {
            "exec-automl-data-loader": ExecutorResources("2", "8Gi", "32", "64Gi"),
        }
        expected = {
            "automl-data-loader": ExecutorResources("4", "8Gi", "32", "64Gi"),
        }
        with pytest.raises(AssertionError, match="automl-data-loader"):
            assert_executor_resources(actual, expected, pipeline_name="test-pipeline")

    def test_assert_executor_resources_allow_extra_executors(self):
        """Partial expected maps can ignore additional executors when allow_extra is set."""
        actual = {
            "exec-automl-data-loader": ExecutorResources("2", "8Gi", "32", "64Gi"),
            "exec-leaderboard-evaluation": ExecutorResources("1", "4Gi", "32", "64Gi"),
        }
        expected = {
            "automl-data-loader": ExecutorResources("2", "8Gi", "32", "64Gi"),
        }
        assert_executor_resources(actual, expected, pipeline_name="test-pipeline", allow_extra=True)


class TestAutogluonTabularPipelineResourceRequirements:
    """Tabular pipeline declares preset-dependent training tiers plus shared loader/leaderboard tiers."""

    def test_tabular_pipeline_executor_resources(self):
        """All tabular pipeline executors match the declared CPU/memory matrix."""
        assert_executor_resources(
            compile_executor_resources(autogluon_tabular_training_pipeline),
            AUTOML_TABULAR_EXECUTOR_RESOURCES,
            pipeline_name="autogluon_tabular_training_pipeline",
        )

    def test_preset_branches_declare_ordered_training_tiers(self):
        """Speed, balanced, and heavy branches declare the intended resources."""
        actual = compile_executor_resources(autogluon_tabular_training_pipeline)
        training_resources = [resources for name, resources in actual.items() if "models-training" in name]
        assert len(training_resources) == 3
        assert TRAINING_SPEED_RESOURCES in training_resources
        assert TRAINING_BALANCED_RESOURCES in training_resources
        assert TRAINING_HEAVY_RESOURCES in training_resources
        speed = TRAINING_SPEED_RESOURCES
        balanced = TRAINING_BALANCED_RESOURCES
        heavy = TRAINING_HEAVY_RESOURCES
        assert float(speed.cpu_request) < float(balanced.cpu_request) < float(heavy.cpu_request)
        assert int(speed.memory_request.removesuffix("Gi")) < int(balanced.memory_request.removesuffix("Gi"))
        assert int(balanced.memory_request.removesuffix("Gi")) < int(heavy.memory_request.removesuffix("Gi"))
