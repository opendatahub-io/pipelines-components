"""Test configurations for parametrized AutoML pipeline integration tests.

Configurations are loaded from test_configs.json in this directory. Each entry
specifies dataset location, target column, problem type, AutoML/pipeline
settings, and optional tags for filtering. Use RHOAI_TEST_CONFIG_TAGS (comma-
separated) to run only configs that have at least one of the given tags.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Problem types supported by the pipeline; "timeseries" reserved for future use.
ProblemType = str  # "classification" | "regression" | "timeseries"

_CONFIGS_JSON = Path(__file__).resolve().parent / "test_configs.json"


def _require_nonempty_str(value: Any, field: str, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"test_configs.json[{index}] {field!r} must be a non-empty string; got {value!r}.")
    return value.strip()


@dataclass
class PipelineConfig:
    """Single test configuration for one pipeline run.

    Attributes:
        id: Short identifier for the config (used in pytest parametrize ids).
        dataset_path: Path to the dataset file relative to the tests directory (e.g.
            "data/regression.csv"). The file is uploaded to S3 by the uploaded_datasets
            fixture; this path is the key into that mapping.
        label_column: Name of the target/label column in the dataset.
        problem_type: High-level problem kind: "classification", "regression", or "timeseries".
        task_type: Pipeline task_type argument: "binary", "multiclass", or "regression".
        automl_settings: Extra pipeline/experiment settings (e.g. top_n). Merged into
            pipeline arguments; must match pipeline parameter names.
        tags: Optional list of tags for filtering (e.g. ["smoke", "regression"]).
            Use RHOAI_TEST_CONFIG_TAGS to run only configs matching at least one tag.
    """

    id: str
    dataset_path: str
    label_column: str
    problem_type: ProblemType
    task_type: str
    automl_settings: dict[str, Any]
    tags: list[str]

    def get_pipeline_arguments(
        self,
        train_data_bucket_name: str,
        train_data_file_key: str,
        train_data_secret_name: str,
    ) -> dict[str, Any]:
        """Build pipeline arguments dict for this config."""
        return {
            "train_data_secret_name": train_data_secret_name,
            "train_data_bucket_name": train_data_bucket_name,
            "train_data_file_key": train_data_file_key,
            "label_column": self.label_column,
            "task_type": self.task_type,
            **self.automl_settings,
        }


def _load_configs(config_path: Path | None = None) -> list[PipelineConfig]:
    """Load test configs from JSON and return PipelineConfig instances.

    Args:
        config_path: Optional path to a JSON array of config objects; defaults to
            ``test_configs.json`` beside this module.
    """
    path = config_path if config_path is not None else _CONFIGS_JSON
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"test_configs.json must be a JSON array; got {type(data).__name__}")
    configs = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"test_configs.json[{i}] must be an object; got {type(item).__name__}")
        try:
            raw_tags = item.get("tags")
            if raw_tags is None:
                tags = []
            elif isinstance(raw_tags, list):
                tags = [str(t) for t in raw_tags]
            else:
                raise ValueError(f"test_configs.json[{i}] 'tags' must be a list; got {type(raw_tags).__name__}")
            configs.append(
                PipelineConfig(
                    id=_require_nonempty_str(item.get("id"), "id", i),
                    dataset_path=_require_nonempty_str(item.get("dataset_path"), "dataset_path", i),
                    label_column=_require_nonempty_str(item.get("label_column"), "label_column", i),
                    problem_type=_require_nonempty_str(item.get("problem_type"), "problem_type", i),
                    task_type=_require_nonempty_str(item.get("task_type"), "task_type", i),
                    automl_settings=item.get("automl_settings") or {},
                    tags=tags,
                )
            )
        except KeyError as e:
            raise ValueError(f"test_configs.json[{i}] missing required key {e}") from e
    return configs


# Environment variable: comma-separated list of tags; if set, only configs with
# at least one of these tags are run (used by get_test_configs_for_run).
TEST_CONFIG_TAGS_ENV = "RHOAI_TEST_CONFIG_TAGS"


def get_test_configs_for_run() -> list[PipelineConfig]:
    """Return configs to run for this session, optionally filtered by tags.

    If RHOAI_TEST_CONFIG_TAGS is set to a comma-separated list of tags, only
    configs that have at least one of those tags are returned. Otherwise all
    configs are returned.
    """
    raw = os.environ.get(TEST_CONFIG_TAGS_ENV)
    if not raw or not raw.strip():
        return TEST_CONFIGS
    allowed = {t.strip().lower() for t in raw.split(",") if t.strip()}
    if not allowed:
        return TEST_CONFIGS
    return [c for c in TEST_CONFIGS if any(t.lower() in allowed for t in c.tags)]


def resolve_config_to_pipeline_arguments(
    config: PipelineConfig,
    uploaded_datasets: dict[str, dict[str, str]] | None,
    secret_name: str,
) -> dict[str, Any] | None:
    """Resolve a PipelineConfig and uploaded datasets to pipeline arguments.

    Args:
        config: The test configuration.
        uploaded_datasets: Map from dataset_path to {"bucket": str, "key": str}
            (from the uploaded_datasets fixture).
        secret_name: S3 secret name to pass to the pipeline.

    Returns:
        Pipeline arguments dict, or None if uploaded_datasets is missing or
        config.dataset_path is not in the map.
    """
    if not uploaded_datasets:
        return None
    location = uploaded_datasets.get(config.dataset_path)
    if not location:
        return None
    return config.get_pipeline_arguments(location["bucket"], location["key"], secret_name)


# ---------------------------------------------------------------------------
# Test configurations loaded from JSON (parametrize integration tests over these)
# ---------------------------------------------------------------------------

TEST_CONFIGS: list[PipelineConfig] = _load_configs()
