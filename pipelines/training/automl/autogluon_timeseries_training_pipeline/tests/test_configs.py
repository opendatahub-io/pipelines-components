"""Test configurations for parametrized AutoGluon time series pipeline integration tests.

Configurations are loaded from test_configs.json in this directory. Each entry
specifies dataset location, time series column names, pipeline settings, and
optional tags for filtering. Use RHOAI_TEST_CONFIG_TAGS (comma-separated) to run
only configs that have at least one of the given tags.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CONFIGS_JSON = Path(__file__).resolve().parent / "test_configs.json"


@dataclass
class TestConfig:
    """Single test configuration for one pipeline run."""

    __test__ = False

    id: str
    dataset_path: str
    target: str
    id_column: str
    timestamp_column: str
    known_covariates_names: list[str]
    prediction_length: int
    top_n: int
    tags: list[str]
    preset: str = "fast_training"
    eval_metric: str = "MASE"

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
            "target": self.target,
            "id_column": self.id_column,
            "timestamp_column": self.timestamp_column,
            "known_covariates_names": self.known_covariates_names,
            "prediction_length": self.prediction_length,
            "top_n": self.top_n,
            "preset": self.preset,
            "eval_metric": self.eval_metric,
        }


def _load_configs() -> list[TestConfig]:
    """Load test configs from test_configs.json and return TestConfig instances."""
    raw = _CONFIGS_JSON.read_text(encoding="utf-8")
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
            known_covariates_names = item.get("known_covariates_names") or []
            if not isinstance(known_covariates_names, list):
                raise ValueError(
                    f"test_configs.json[{i}] 'known_covariates_names' must be a list; got {type(known_covariates_names).__name__}"  # noqa: E501
                )
            configs.append(
                TestConfig(
                    id=item["id"],
                    dataset_path=item["dataset_path"],
                    target=item["target"],
                    id_column=item["id_column"],
                    timestamp_column=item["timestamp_column"],
                    known_covariates_names=[str(x) for x in known_covariates_names],
                    prediction_length=int(item.get("prediction_length", 1)),
                    top_n=int(item.get("top_n", 3)),
                    tags=tags,
                    preset=str(item.get("preset", "fast_training")),
                    eval_metric=str(item.get("eval_metric", "MASE")),
                )
            )
        except KeyError as e:
            raise ValueError(f"test_configs.json[{i}] missing required key {e}") from e
    return configs


TEST_CONFIG_TAGS_ENV = "RHOAI_TEST_CONFIG_TAGS"


def get_test_configs_for_run() -> list[TestConfig]:
    """Return configs to run for this session, optionally filtered by tags."""
    raw = os.environ.get(TEST_CONFIG_TAGS_ENV)
    if not raw or not raw.strip():
        return TEST_CONFIGS
    allowed = {t.strip().lower() for t in raw.split(",") if t.strip()}
    if not allowed:
        return TEST_CONFIGS
    return [c for c in TEST_CONFIGS if any(t.lower() in allowed for t in c.tags)]


def resolve_config_to_pipeline_arguments(
    config: TestConfig,
    uploaded_datasets: dict[str, dict[str, str]] | None,
    secret_name: str,
) -> dict[str, Any] | None:
    """Resolve a TestConfig and uploaded datasets to pipeline arguments."""
    if not uploaded_datasets:
        return None
    location = uploaded_datasets.get(config.dataset_path)
    if not location:
        return None
    return config.get_pipeline_arguments(location["bucket"], location["key"], secret_name)


TEST_CONFIGS: list[TestConfig] = _load_configs()
