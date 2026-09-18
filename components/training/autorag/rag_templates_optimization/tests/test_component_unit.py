"""Unit tests for the rag_templates_optimization component.

ai4rag is mocked via ``sys.modules`` so the pure orchestration (metric
resolution, evaluator wiring, artifact generation) is exercised without the
real (heavy) library or any network access.
"""

import inspect
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from unittest import mock

import pytest

from ..component import rag_templates_optimization

MOCKED_ENV_VARIABLES = {
    "MAAS_BASE_URL": "https://maas.example.com/v1",
    "MAAS_API_KEY": "test-api-key",
    "MILVUS_URI": "https://milvus.example.com:19530",
}


@dataclass(frozen=True)
class _RAGMetric:
    """Stand-in for ``ai4rag.evaluator.metric.RAGMetric``."""

    name: str
    evaluator: Literal["unitxt", "judge", "ragas", "custom"]
    description: str = ""


class _Metrics:
    """Stand-in for ``ai4rag.evaluator.metric.Metrics``, mirroring its real members.

    Mirrors the production set closely enough to exercise ambiguous names
    (``faithfulness`` scored by both unitxt and RAGAS), an evaluator this
    component never runs (``judge``), and the ``custom`` aggregate.
    """

    ANSWER_CORRECTNESS = _RAGMetric("answer_correctness", "unitxt")
    FAITHFULNESS = _RAGMetric("faithfulness", "unitxt")
    CONTEXT_CORRECTNESS = _RAGMetric("context_correctness", "unitxt")
    JUDGE_ANSWER_RELEVANCE = _RAGMetric("answer_relevance", "judge")
    RAGAS_FAITHFULNESS = _RAGMetric("faithfulness", "ragas")
    RAGAS_ANSWER_RELEVANCY = _RAGMetric("answer_relevancy", "ragas")
    RAGAS_CONTEXT_PRECISION = _RAGMetric("context_precision", "ragas")
    RAGAS_CONTEXT_RECALL = _RAGMetric("context_recall", "ragas")
    OVERALL_SCORE = _RAGMetric("overall_score", "custom")

    def __iter__(self):
        return (v for v in vars(type(self)).values() if isinstance(v, _RAGMetric))


def _make_ai4rag_mocks() -> SimpleNamespace:
    """Build mock ai4rag modules wired into sys.modules for the component body."""
    experiment_cls = mock.MagicMock(name="AI4RAGExperiment")
    gam_opt_settings_cls = mock.MagicMock(name="GAMOptSettings")
    unitxt_evaluator_cls = mock.MagicMock(name="UnitxtEvaluator")
    ragas_evaluator_cls = mock.MagicMock(name="RagasEvaluator")
    get_vector_store_config = mock.MagicMock(name="get_vector_store_config")
    get_foundation_models = mock.MagicMock(name="get_foundation_models", return_value=[mock.MagicMock(model_id="fm-0")])
    get_embedding_models = mock.MagicMock(name="get_embedding_models", return_value=[mock.MagicMock(model_id="em-0")])
    parameter_cls = mock.MagicMock(name="Parameter")
    search_space_cls = mock.MagicMock(name="AI4RAGSearchSpace")
    build_leaderboard_html = mock.MagicMock(name="build_leaderboard_html", return_value="<html></html>")
    generate_notebook_from_template = mock.MagicMock(name="generate_notebook_from_template")
    load_docling_documents = mock.MagicMock(name="load_docling_documents", return_value=["doc"])
    create_maas_client = mock.MagicMock(name="create_maas_client")
    event_handler_cls = mock.MagicMock(name="KFPEventHandler")
    event_handler_cls.return_value.patterns = []
    # Real pandas/numpy must never be imported inside a `sys.modules` patch: exiting the
    # patch restores the *entire* dict snapshot, purging whatever got cached during the
    # `with` block (including numpy's C extension) and breaking the next test's import.
    pandas_mock = mock.MagicMock(name="pandas")

    experiment_module = mock.MagicMock()
    experiment_module.AI4RAGExperiment = experiment_cls

    gam_opt_module = mock.MagicMock()
    gam_opt_module.GAMOptSettings = gam_opt_settings_cls

    evaluator_module = mock.MagicMock()
    evaluator_module.BaseEvaluator = mock.MagicMock(name="BaseEvaluator")
    evaluator_module.UnitxtEvaluator = unitxt_evaluator_cls
    evaluator_module.RagasEvaluator = ragas_evaluator_cls

    metric_module = mock.MagicMock()
    metric_module.Metrics = _Metrics()
    metric_module.RAGMetric = _RAGMetric

    embedding_model_module = mock.MagicMock()
    embedding_model_module.OpenAIEmbeddingModel = mock.MagicMock(name="OpenAIEmbeddingModel")

    foundation_model_module = mock.MagicMock()
    foundation_model_module.OpenAIFoundationModel = mock.MagicMock(name="OpenAIFoundationModel")

    vector_store_module = mock.MagicMock()
    vector_store_module.get_vector_store_config = get_vector_store_config

    search_space_models_module = mock.MagicMock()
    search_space_models_module.get_foundation_models = get_foundation_models
    search_space_models_module.get_embedding_models = get_embedding_models

    parameter_module = mock.MagicMock()
    parameter_module.Parameter = parameter_cls

    search_space_module = mock.MagicMock()
    search_space_module.AI4RAGSearchSpace = search_space_cls

    assets_generator_module = mock.MagicMock()
    assets_generator_module.build_leaderboard_html = build_leaderboard_html
    assets_generator_module.generate_notebook_from_template = generate_notebook_from_template

    maas_client_module = mock.MagicMock()
    maas_client_module.create_maas_client = create_maas_client

    docling_io_module = mock.MagicMock()
    docling_io_module.load_docling_documents = load_docling_documents

    event_handler_module = mock.MagicMock()
    event_handler_module.KFPEventHandler = event_handler_cls

    ai4rag_top = mock.MagicMock()
    # A real handler avoids `record.levelno >= hdlr.level` blowing up on a
    # MagicMock when the component logs through it.
    ai4rag_top.handler = logging.NullHandler()

    modules = {
        "ai4rag": ai4rag_top,
        "ai4rag.core": mock.MagicMock(),
        "ai4rag.core.experiment": mock.MagicMock(),
        "ai4rag.core.experiment.experiment": experiment_module,
        "ai4rag.core.hpo": mock.MagicMock(),
        "ai4rag.core.hpo.gam_opt": gam_opt_module,
        "ai4rag.evaluator": evaluator_module,
        "ai4rag.evaluator.metric": metric_module,
        "ai4rag.rag": mock.MagicMock(),
        "ai4rag.rag.embedding": mock.MagicMock(),
        "ai4rag.rag.embedding.openai_model": embedding_model_module,
        "ai4rag.rag.foundation_models": mock.MagicMock(),
        "ai4rag.rag.foundation_models.openai_model": foundation_model_module,
        "ai4rag.rag.vector_store": vector_store_module,
        "ai4rag.search_space": mock.MagicMock(),
        "ai4rag.search_space.prepare": mock.MagicMock(),
        "ai4rag.search_space.prepare.models": search_space_models_module,
        "ai4rag.search_space.src": mock.MagicMock(),
        "ai4rag.search_space.src.parameter": parameter_module,
        "ai4rag.search_space.src.search_space": search_space_module,
        "ai4rag.utils": mock.MagicMock(),
        "ai4rag.assets_generator": assets_generator_module,
        "ai4rag.utils.clients": mock.MagicMock(),
        "ai4rag.utils.clients.maas_client": maas_client_module,
        "ai4rag.utils.docling_io": docling_io_module,
        "ai4rag.utils.event_handler": event_handler_module,
        "pandas": pandas_mock,
    }

    return SimpleNamespace(
        modules=modules,
        AI4RAGExperiment=experiment_cls,
        UnitxtEvaluator=unitxt_evaluator_cls,
        RagasEvaluator=ragas_evaluator_cls,
        get_vector_store_config=get_vector_store_config,
        get_foundation_models=get_foundation_models,
        get_embedding_models=get_embedding_models,
        build_leaderboard_html=build_leaderboard_html,
        generate_notebook_from_template=generate_notebook_from_template,
        create_maas_client=create_maas_client,
        KFPEventHandler=event_handler_cls,
        pandas=pandas_mock,
    )


def _write_json(path: Path, data) -> None:
    """Write ``data`` as JSON to ``path``."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def _pattern_payload(name: str) -> dict:
    """Build a pattern payload with the ``settings`` shape ``_generate_output_artifacts`` expects."""
    return {
        "name": name,
        "settings": {
            "vector_store_binding": {"provider_type": "milvus", "collection_name": f"{name}-collection"},
            "embedding": {"model_id": "em-0", "embedding_params": {}},
            "chunking": {"method": "recursive", "chunk_size": 512, "chunk_overlap": 64},
        },
    }


def _artifacts(tmp_path: Path):
    """Build Output[Artifact]/HTML mocks with real, writable ``.path`` values."""
    rag_patterns = mock.MagicMock()
    rag_patterns.path = str(tmp_path / "rag_patterns")
    rag_patterns.uri = "gs://bucket/rag_patterns"
    rag_patterns.metadata = {}

    leaderboard_html = mock.MagicMock()
    leaderboard_html.path = str(tmp_path / "leaderboard.html")
    leaderboard_html.metadata = {}

    return rag_patterns, leaderboard_html


def _write_search_space_report(tmp_path: Path) -> str:
    """Write a minimal, valid search_space_mps_report fixture; return its path.

    ``test_data`` needs no fixture of its own: ``pandas`` is mocked, so
    ``pd.read_json`` never touches the filesystem.
    """
    search_space_path = tmp_path / "search_space_report.json"
    _write_json(
        search_space_path,
        {
            "foundation_model": [{"model_id": "fm-0"}],
            "embedding_model": [{"model_id": "em-0"}],
            "chunk_size": [256, 512],
        },
    )
    return str(search_space_path)


class TestRagTemplatesOptimizationInterface:
    """Static interface checks."""

    def test_component_function_exists(self):
        """Component factory exists and exposes python_func."""
        assert callable(rag_templates_optimization)
        assert hasattr(rag_templates_optimization, "python_func")

    def test_component_has_expected_interface(self):
        """Component has the expected parameters; evaluators is no longer configurable."""
        sig = inspect.signature(rag_templates_optimization.python_func)
        params = list(sig.parameters)
        for name in (
            "extracted_text",
            "test_data",
            "search_space_mps_report",
            "rag_patterns",
            "test_data_key",
            "maas_secret_name",
            "vector_db_secret_name",
            "input_data_secret_name",
            "input_data_bucket_name",
            "leaderboard",
            "embedded_artifact",
            "optimization_settings",
            "input_data_keys",
            "component_status",
            "preset",
        ):
            assert name in params
        assert "evaluators" not in params
        for name in ("maas_secret_name", "vector_db_secret_name", "input_data_secret_name", "input_data_bucket_name"):
            assert sig.parameters[name].default is inspect.Parameter.empty
        assert sig.parameters["preset"].default == "speed"
        assert sig.parameters["component_status"].default is None


class TestRagTemplatesOptimizationValidation:
    """Validation failures that must surface before any ai4rag work is done."""

    def test_preset_validation_rejects_invalid(self, tmp_path):
        """Invalid preset raises ValueError before touching env vars or ai4rag."""
        mocks = _make_ai4rag_mocks()
        rag_patterns, leaderboard_html = _artifacts(tmp_path)

        with mock.patch.dict("sys.modules", mocks.modules):
            with pytest.raises(ValueError, match="preset must be one of"):
                rag_templates_optimization.python_func(
                    extracted_text=str(tmp_path / "ext"),
                    test_data=str(tmp_path / "td.json"),
                    search_space_mps_report=str(tmp_path / "r.json"),
                    rag_patterns=rag_patterns,
                    test_data_key="key.json",
                    maas_secret_name="maas-secret",
                    vector_db_secret_name="vector-db-secret",
                    input_data_secret_name="s3-secret",
                    input_data_bucket_name="bucket",
                    leaderboard=leaderboard_html,
                    preset="invalid",
                )

    def test_missing_maas_env_raises_key_error(self, tmp_path):
        """Missing MaaS env vars raise KeyError."""
        mocks = _make_ai4rag_mocks()
        rag_patterns, leaderboard_html = _artifacts(tmp_path)

        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch.dict("sys.modules", mocks.modules):
                with pytest.raises(KeyError):
                    rag_templates_optimization.python_func(
                        extracted_text=str(tmp_path / "ext"),
                        test_data=str(tmp_path / "td.json"),
                        search_space_mps_report=str(tmp_path / "r.json"),
                        rag_patterns=rag_patterns,
                        test_data_key="key.json",
                        maas_secret_name="maas-secret",
                        vector_db_secret_name="vector-db-secret",
                        input_data_secret_name="s3-secret",
                        input_data_bucket_name="bucket",
                        leaderboard=leaderboard_html,
                    )

    @mock.patch.dict("os.environ", {"MAAS_BASE_URL": "https://maas.example.com/v1", "MAAS_API_KEY": "key"}, clear=True)
    def test_missing_vector_db_env_raises_value_error(self, tmp_path):
        """Absent MILVUS_*/PGVECTOR_* env vars raise a descriptive ValueError."""
        mocks = _make_ai4rag_mocks()
        rag_patterns, leaderboard_html = _artifacts(tmp_path)

        with mock.patch.dict("sys.modules", mocks.modules):
            with pytest.raises(ValueError, match="No vector database configuration found"):
                rag_templates_optimization.python_func(
                    extracted_text=str(tmp_path / "ext"),
                    test_data=str(tmp_path / "td.json"),
                    search_space_mps_report=str(tmp_path / "r.json"),
                    rag_patterns=rag_patterns,
                    test_data_key="key.json",
                    maas_secret_name="maas-secret",
                    vector_db_secret_name="vector-db-secret",
                    input_data_secret_name="s3-secret",
                    input_data_bucket_name="bucket",
                    leaderboard=leaderboard_html,
                )

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    @pytest.mark.parametrize("test_data_key", [None, "", "  ", "data/test.txt"])
    def test_invalid_test_data_key_raises_value_error(self, tmp_path, test_data_key):
        """test_data_key must be a non-empty string pointing at a .json file."""
        mocks = _make_ai4rag_mocks()
        rag_patterns, leaderboard_html = _artifacts(tmp_path)

        with mock.patch.dict("sys.modules", mocks.modules):
            with pytest.raises(ValueError, match="test_data_key must point to a JSON file"):
                rag_templates_optimization.python_func(
                    extracted_text=str(tmp_path / "ext"),
                    test_data=str(tmp_path / "td.json"),
                    search_space_mps_report=str(tmp_path / "r.json"),
                    rag_patterns=rag_patterns,
                    test_data_key=test_data_key,
                    maas_secret_name="maas-secret",
                    vector_db_secret_name="vector-db-secret",
                    input_data_secret_name="s3-secret",
                    input_data_bucket_name="bucket",
                    leaderboard=leaderboard_html,
                )

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_invalid_max_rag_patterns_raises_value_error(self, tmp_path):
        """Out-of-range max_number_of_rag_patterns raises before any ai4rag work."""
        mocks = _make_ai4rag_mocks()
        rag_patterns, leaderboard_html = _artifacts(tmp_path)

        with mock.patch.dict("sys.modules", mocks.modules):
            with pytest.raises(ValueError, match="max_number_of_rag_patterns must be in range"):
                rag_templates_optimization.python_func(
                    extracted_text=str(tmp_path / "ext"),
                    test_data=str(tmp_path / "td.json"),
                    search_space_mps_report=str(tmp_path / "r.json"),
                    rag_patterns=rag_patterns,
                    test_data_key="key.json",
                    maas_secret_name="maas-secret",
                    vector_db_secret_name="vector-db-secret",
                    input_data_secret_name="s3-secret",
                    input_data_bucket_name="bucket",
                    leaderboard=leaderboard_html,
                    optimization_settings={"max_number_of_rag_patterns": 1},
                )


class TestRagTemplatesOptimizationMetricResolution:
    """Metric resolution enforces the evaluator policy for each preset."""

    @pytest.mark.parametrize(
        ("preset", "metric_id", "expected_evaluator"),
        [
            ("speed", "unitxt:faithfulness", "unitxt"),
            ("speed", "faithfulness", "unitxt"),
            ("speed", "custom:overall_score", "custom"),
            ("balanced", "unitxt:faithfulness", "unitxt"),
            ("balanced", "ragas:faithfulness", "ragas"),
            ("balanced", "faithfulness", "ragas"),
            ("balanced", "ragas:context_precision", "ragas"),
            ("balanced", "custom:overall_score", "custom"),
        ],
    )
    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_resolves_to_expected_evaluator(self, tmp_path, preset, metric_id, expected_evaluator):
        """The resolved RAGMetric is the one this component actually optimizes for."""
        mocks = _make_ai4rag_mocks()
        search_space_path = _write_search_space_report(tmp_path)
        rag_patterns, leaderboard_html = _artifacts(tmp_path)

        with mock.patch.dict("sys.modules", mocks.modules):
            rag_templates_optimization.python_func(
                extracted_text=str(tmp_path / "ext"),
                test_data=str(tmp_path / "test_data.json"),
                search_space_mps_report=search_space_path,
                rag_patterns=rag_patterns,
                test_data_key="key.json",
                maas_secret_name="maas-secret",
                vector_db_secret_name="vector-db-secret",
                input_data_secret_name="s3-secret",
                input_data_bucket_name="bucket",
                leaderboard=leaderboard_html,
                optimization_settings={"metric": metric_id} if metric_id else None,
                preset=preset,
            )

        resolved_metric = mocks.AI4RAGExperiment.call_args.kwargs["optimization_metric"]
        assert resolved_metric.evaluator == expected_evaluator
        mocks.UnitxtEvaluator.assert_called_once_with()
        if preset == "speed":
            mocks.RagasEvaluator.assert_not_called()
        else:
            mocks.RagasEvaluator.assert_called_once()

        # The evaluator must travel with the name, or the leaderboard can't tell apart
        # metrics that collide across evaluators (e.g. unitxt vs RAGAS "faithfulness")
        # and ends up pinning/labeling the wrong evaluator's column.
        leaderboard_kwargs = mocks.build_leaderboard_html.call_args.kwargs
        assert leaderboard_kwargs["optimization_metric"] == resolved_metric.name
        assert leaderboard_kwargs["optimization_metric_evaluator"] == resolved_metric.evaluator

    @pytest.mark.parametrize(
        ("preset", "metric_id", "match"),
        [
            ("speed", "ragas:context_precision", "is unavailable for this preset"),
            ("balanced", "answer_relevance", "is unavailable for this preset"),
            ("speed", "nonexistent_metric", "is not supported"),
        ],
    )
    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_rejects_invalid_metric_for_preset(self, tmp_path, preset, metric_id, match):
        """Unknown, ambiguous, and preset-disabled metrics are rejected."""
        mocks = _make_ai4rag_mocks()
        rag_patterns, leaderboard_html = _artifacts(tmp_path)

        with mock.patch.dict("sys.modules", mocks.modules):
            with pytest.raises(ValueError, match=match):
                rag_templates_optimization.python_func(
                    extracted_text=str(tmp_path / "ext"),
                    test_data=str(tmp_path / "td.json"),
                    search_space_mps_report=str(tmp_path / "r.json"),
                    rag_patterns=rag_patterns,
                    test_data_key="key.json",
                    maas_secret_name="maas-secret",
                    vector_db_secret_name="vector-db-secret",
                    input_data_secret_name="s3-secret",
                    input_data_bucket_name="bucket",
                    leaderboard=leaderboard_html,
                    optimization_settings={"metric": metric_id},
                    preset=preset,
                )


class TestRagTemplatesOptimizationRun:
    """End-to-end orchestration: evaluator wiring, artifact generation, leaderboard."""

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_full_run_speed_wires_unitxt_and_writes_patterns(self, tmp_path):
        """The speed preset runs Unitxt only and persists pattern artifacts."""
        mocks = _make_ai4rag_mocks()
        mocks.get_vector_store_config.return_value = mock.MagicMock(name="vector_store_config")
        search_space_path = _write_search_space_report(tmp_path)
        mocks.KFPEventHandler.return_value.patterns = [
            {
                "payload": _pattern_payload("pattern_a"),
                "evaluation_results": [{"metric": "faithfulness", "score": 0.9}],
            },
        ]
        rag_patterns, leaderboard_html = _artifacts(tmp_path)

        with mock.patch.dict("sys.modules", mocks.modules):
            rag_templates_optimization.python_func(
                extracted_text=str(tmp_path / "extracted"),
                test_data=str(tmp_path / "test_data.json"),
                search_space_mps_report=search_space_path,
                rag_patterns=rag_patterns,
                test_data_key="data/test.json",
                maas_secret_name="maas-connection",
                vector_db_secret_name="vector-db-connection",
                input_data_secret_name="s3-input-connection",
                input_data_bucket_name="customer-docs",
                leaderboard=leaderboard_html,
                input_data_keys=["data/docs/"],
            )

        mocks.create_maas_client.assert_called_once_with(
            base_url="https://maas.example.com/v1",
            api_key="test-api-key",
        )
        mocks.get_vector_store_config.assert_called_once_with("milvus")

        mocks.UnitxtEvaluator.assert_called_once_with()
        mocks.RagasEvaluator.assert_not_called()
        exp_kwargs = mocks.AI4RAGExperiment.call_args.kwargs
        assert exp_kwargs["evaluators"] == [mocks.UnitxtEvaluator.return_value]
        assert exp_kwargs["optimization_metric"].name == "overall_score"
        mocks.AI4RAGExperiment.return_value.search.assert_called_once()

        pattern_dir = Path(rag_patterns.path) / "pattern_a"
        assert (pattern_dir / "pattern.json").exists()
        assert (pattern_dir / "evaluation_results.json").exists()

        # The whole list reaches the indexing blueprint, but the notebook takes the first key.
        pattern_json = json.loads((pattern_dir / "pattern.json").read_text(encoding="utf-8"))
        assert pattern_json["indexing"]["pipeline_spec"]["parameters"]["input_data_keys"] == ["data/docs/"]
        indexing_notebook_call = next(
            call for call in mocks.generate_notebook_from_template.call_args_list if call.args[0] == "maas_indexing"
        )
        assert indexing_notebook_call.kwargs["input_data_key"] == "data/docs/"

        assert rag_patterns.metadata["name"] == "rag_patterns_artifact"
        assert rag_patterns.metadata["uri"] == "gs://bucket/rag_patterns"
        assert rag_patterns.metadata["metadata"]["patterns"][0]["name"] == "pattern_a"

        mocks.build_leaderboard_html.assert_called_once()
        assert Path(leaderboard_html.path).read_text(encoding="utf-8") == "<html></html>"
        assert leaderboard_html.metadata["display_name"] == "autorag_leaderboard"

    @mock.patch.dict(
        "os.environ",
        {
            "MAAS_BASE_URL": "https://maas.example.com/v1",
            "MAAS_API_KEY": "test-api-key",
            "PGVECTOR_HOST": "pg.example.com",
        },
        clear=True,
    )
    def test_pgvector_provider_detected_from_env(self, tmp_path):
        """PGVECTOR_* env vars select the pgvector backend when no MILVUS_* keys are set."""
        mocks = _make_ai4rag_mocks()
        search_space_path = _write_search_space_report(tmp_path)
        rag_patterns, leaderboard_html = _artifacts(tmp_path)

        with mock.patch.dict("sys.modules", mocks.modules):
            rag_templates_optimization.python_func(
                extracted_text=str(tmp_path / "ext"),
                test_data=str(tmp_path / "test_data.json"),
                search_space_mps_report=search_space_path,
                rag_patterns=rag_patterns,
                test_data_key="key.json",
                maas_secret_name="maas-secret",
                vector_db_secret_name="vector-db-secret",
                input_data_secret_name="s3-secret",
                input_data_bucket_name="bucket",
                leaderboard=leaderboard_html,
            )

        mocks.get_vector_store_config.assert_called_once_with("pgvector")

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_none_input_data_keys_defaults_to_empty_list(self, tmp_path):
        """A None input_data_keys is normalized to [] in the indexing pipeline params."""
        mocks = _make_ai4rag_mocks()
        search_space_path = _write_search_space_report(tmp_path)
        mocks.KFPEventHandler.return_value.patterns = [
            {"payload": _pattern_payload("pattern_a"), "evaluation_results": []},
        ]
        rag_patterns, leaderboard_html = _artifacts(tmp_path)

        with mock.patch.dict("sys.modules", mocks.modules):
            rag_templates_optimization.python_func(
                extracted_text=str(tmp_path / "ext"),
                test_data=str(tmp_path / "test_data.json"),
                search_space_mps_report=search_space_path,
                rag_patterns=rag_patterns,
                test_data_key="key.json",
                maas_secret_name="maas-secret",
                vector_db_secret_name="vector-db-secret",
                input_data_secret_name="s3-secret",
                input_data_bucket_name="bucket",
                leaderboard=leaderboard_html,
                input_data_keys=None,
            )

        pattern_json = json.loads((Path(rag_patterns.path) / "pattern_a" / "pattern.json").read_text(encoding="utf-8"))
        indexing_params = pattern_json["indexing"]["pipeline_spec"]["parameters"]
        assert indexing_params["input_data_keys"] == []

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_propagates_ai4rag_exception(self, tmp_path):
        """Exceptions raised by the ai4rag search loop propagate to the caller."""
        mocks = _make_ai4rag_mocks()
        mocks.AI4RAGExperiment.return_value.search.side_effect = ValueError("search failed")
        search_space_path = _write_search_space_report(tmp_path)
        rag_patterns, leaderboard_html = _artifacts(tmp_path)

        with mock.patch.dict("sys.modules", mocks.modules):
            with pytest.raises(ValueError, match="search failed"):
                rag_templates_optimization.python_func(
                    extracted_text=str(tmp_path / "ext"),
                    test_data=str(tmp_path / "test_data.json"),
                    search_space_mps_report=search_space_path,
                    rag_patterns=rag_patterns,
                    test_data_key="key.json",
                    maas_secret_name="maas-secret",
                    vector_db_secret_name="vector-db-secret",
                    input_data_secret_name="s3-secret",
                    input_data_bucket_name="bucket",
                    leaderboard=leaderboard_html,
                )
