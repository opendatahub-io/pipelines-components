"""Test configurations for parametrized functional tests of the Documents RAG Optimization pipeline.

Configurations are loaded from test_configs.json in this directory. Each entry
specifies pipeline parameter overrides, expected result (pass/fail), and optional
tags for filtering. Use RHOAI_TEST_CONFIG_TAGS (comma-separated) to run only
configs that have at least one of the given tags.
"""

import os
from dataclasses import dataclass, field
from typing import Any

_CONFIGS = [
    # {
    #     "id": "TC-A-1",
    #     "description": "faithfulness metric, vector_db_id unset => pipeline must fail",
    #     "tags": ["smoke", "negative", "milvus-lite"],
    #     "expected_result": "fail",
    #     "pipeline_params_overrides": {
    #         "optimization_metric": "faithfulness",
    #         "optimization_max_rag_patterns": 6,
    #         "llama_stack_vector_io_provider_id": "",
    #         "input_data_key": null,
    #         "embeddings_models": null,
    #         "generation_models": null
    #     }
    # },
    # {
    #     "id": "TC-A-2",
    #     "description": "answer_correctness, milvus-lite provider, 4 patterns, default models",
    #     "tags": ["smoke", "positive", "milvus-lite"],
    #     "expected_result": "pass",
    #     "pipeline_params_overrides": {
    #         "optimization_metric": "answer_correctness",
    #         "optimization_max_rag_patterns": 4,
    #         "llama_stack_vector_io_provider_id": "milvus-lite",
    #         "input_data_key": None,
    #         "embeddings_models": None,
    #         "generation_models": None,
    #     }
    # },
    {
        "id": "TC-A-3",
        "description": "faithfulness, milvus-standalone provider, 8 patterns, constrained models",
        "tags": ["positive", "milvus-standalone", "nightly"],
        "expected_result": "pass",
        "pipeline_params_overrides": {
            "optimization_metric": "faithfulness",
            "optimization_max_rag_patterns": 12,
            "llama_stack_vector_io_provider_id": "milvus-standalone",
            "input_data_key": None,
            "embeddings_models": "ENV",
            "generation_models": "ENV"
        }
    }
]

# Milvus provider ID resolution: maps sentinel values in JSON to env var keys
# in the functional config dict.
_VECTOR_IO_PROVIDER_MAP = {
    "milvus-lite": "vector_io_provider_milvus_lite",
    "milvus-standalone": "vector_io_provider_milvus_standalone",
}


@dataclass
class TestConfig:
    """Single test configuration for one pipeline run.

    Attributes:
        id: Short identifier for the config (used in pytest parametrize ids).
        description: Human-readable summary of the test scenario.
        tags: Optional list of tags for filtering (e.g. ["smoke", "positive"]).
            Use RHOAI_TEST_CONFIG_TAGS to run only configs matching at least one tag.
        expected_result: "pass" or "fail" — whether the pipeline run should succeed.
        pipeline_params_overrides: Keys matching pipeline parameter names. Values
            are resolved against the base config using these rules:
            - null/None: use base config value from env
            - "": pass empty string explicitly
            - "ENV": read from dedicated env var (for model lists)
            - "milvus-lite"/"milvus-standalone": read provider ID from corresponding env var
            - any other value: use as-is
    """

    __test__ = False  # prevent pytest collection

    id: str
    description: str
    tags: list[str]
    expected_result: str
    pipeline_params_overrides: dict[str, Any] = field(default_factory=dict)

    def get_pipeline_arguments(self, base_config: dict) -> dict[str, Any]:
        """Build pipeline arguments dict by merging base config with overrides.

        Args:
            base_config: Functional config dict from integration_config.get_docrag_functional_config().

        Returns:
            Pipeline arguments dict ready for KFP submission.
        """
        # Start with required parameters from base config
        arguments: dict[str, Any] = {
            "test_data_secret_name": base_config["test_data_secret_name"],
            "test_data_bucket_name": base_config["test_data_bucket_name"],
            "test_data_key": base_config["test_data_key"],
            "input_data_secret_name": base_config["input_data_secret_name"],
            "input_data_bucket_name": base_config["input_data_bucket_name"],
            "llama_stack_secret_name": base_config["llama_stack_secret_name"],
        }

        # Resolve overrides
        overrides = self.pipeline_params_overrides

        # llama_stack_vector_io_provider_id
        vio_override = overrides.get("llama_stack_vector_io_provider_id")
        if vio_override is None:
            arguments["llama_stack_vector_io_provider_id"] = base_config["llama_stack_vector_io_provider_id"]
        elif vio_override in _VECTOR_IO_PROVIDER_MAP:
            config_key = _VECTOR_IO_PROVIDER_MAP[vio_override]
            arguments["llama_stack_vector_io_provider_id"] = base_config.get(config_key, "")
        else:
            arguments["llama_stack_vector_io_provider_id"] = vio_override

        # input_data_key
        idk_override = overrides.get("input_data_key")
        if idk_override is None:
            arguments["input_data_key"] = base_config.get("input_data_key", "")
        else:
            arguments["input_data_key"] = idk_override

        # optimization_metric
        metric_override = overrides.get("optimization_metric")
        if metric_override is not None:
            arguments["optimization_metric"] = metric_override

        # optimization_max_rag_patterns
        max_patterns_override = overrides.get("optimization_max_rag_patterns")
        if max_patterns_override is not None:
            arguments["optimization_max_rag_patterns"] = max_patterns_override

        # embeddings_models
        emb_override = overrides.get("embeddings_models")
        if emb_override == "ENV":
            emb_val = base_config.get("embeddings_models")
            if emb_val is not None:
                arguments["embeddings_models"] = emb_val
        elif emb_override is not None:
            arguments["embeddings_models"] = emb_override

        # generation_models
        gen_override = overrides.get("generation_models")
        if gen_override == "ENV":
            gen_val = base_config.get("generation_models")
            if gen_val is not None:
                arguments["generation_models"] = gen_val
        elif gen_override is not None:
            arguments["generation_models"] = gen_override

        return arguments


def _load_configs() -> list[TestConfig]:
    """Load test configs from test_configs.json and return TestConfig instances."""
    data = _CONFIGS
    configs = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"test_configs[{i}] must be a dict; got {type(item).__name__}")
        try:
            raw_tags = item.get("tags")
            if raw_tags is None:
                tags = []
            elif isinstance(raw_tags, list):
                tags = [str(t) for t in raw_tags]
            else:
                raise ValueError(f"test_configs[{i}] 'tags' must be a list; got {type(raw_tags).__name__}")

            expected_result = item["expected_result"]
            if expected_result not in ("pass", "fail"):
                raise ValueError(
                    f"test_configs[{i}] 'expected_result' must be 'pass' or 'fail'; got '{expected_result}'"
                )

            configs.append(
                TestConfig(
                    id=item["id"],
                    description=item.get("description", ""),
                    tags=tags,
                    expected_result=expected_result,
                    pipeline_params_overrides=item.get("pipeline_params_overrides") or {},
                )
            )
        except KeyError as e:
            raise ValueError(f"test_configs[{i}] missing required key {e}") from e
    return configs


# Environment variable: comma-separated list of tags; if set, only configs with
# at least one of these tags are run (used by get_test_configs_for_run).
TEST_CONFIG_TAGS_ENV = "RHOAI_TEST_CONFIG_TAGS"


def get_test_configs_for_run() -> list[TestConfig]:
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


# ---------------------------------------------------------------------------
# Test configurations loaded from JSON (parametrize functional tests over these)
# ---------------------------------------------------------------------------

TEST_CONFIGS: list[TestConfig] = _load_configs()
