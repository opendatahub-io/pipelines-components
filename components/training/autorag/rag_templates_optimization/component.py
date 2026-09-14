from pathlib import Path
from typing import Any, Optional

from kfp import dsl
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]

_AUTORAG_SHARED = Path(__file__).parents[1] / "shared"


@dsl.component(
    base_image=AUTORAG_IMAGE,
    embedded_artifact_path=str(_AUTORAG_SHARED / "component_status.py"),
    install_kfp_package=False,
)
def rag_templates_optimization(
    extracted_text: dsl.InputPath(dsl.Artifact),
    test_data: dsl.InputPath(dsl.Artifact),
    search_space_mps_report: dsl.InputPath(dsl.Artifact),
    rag_patterns: dsl.Output[dsl.Artifact],
    test_data_key: str,
    maas_secret_name: str,
    vector_db_secret_name: str,
    input_data_secret_name: str,
    input_data_bucket_name: str,
    leaderboard: dsl.Output[dsl.HTML],
    embedded_artifact: dsl.EmbeddedInput[dsl.Dataset] = None,
    optimization_settings: Optional[dict] = None,
    input_data_keys: Optional[list[str]] = None,
    component_status: dsl.Output[dsl.Artifact] = None,
    preset: str = "speed",
):
    """RAG Templates Optimization component.

    Runs search-space construction, evaluator setup, and the optimization
    experiment directly against ``ai4rag`` primitives (``AI4RAGSearchSpace``,
    ``AI4RAGExperiment``, and the ``ragas``/``unitxt`` evaluators), since
    ``ai4rag`` no longer ships a single orchestration entry point for this flow.

    Args:
        extracted_text: Path to extracted text documents.
        test_data: Path to benchmark test data JSON.
        search_space_mps_report: Path to the JSON search space report.
        rag_patterns: Output artifact for generated RAG patterns.
        test_data_key: Path to benchmark JSON in object storage.
        maas_secret_name: Name of the K8s secret with MaaS inference credentials
            ("MAAS_BASE_URL", "MAAS_API_KEY"). Propagated into each generated
            ``pattern.json`` indexing spec for downstream deployment.
        vector_db_secret_name: Name of the K8s secret holding the vector database
            configuration. Its keys select the backend: ``MILVUS_*`` keys use
            Milvus, ``PGVECTOR_*`` keys use PGVector. Propagated into each
            generated ``pattern.json`` indexing spec.
        input_data_secret_name: Name of the K8s secret with S3 credentials for
            input data.
        input_data_bucket_name: S3 bucket containing input documents.
        leaderboard: Output HTML artifact; the leaderboard table is written to
            leaderboard_html.path (single file).
        component_status: Output artifact containing stage-level progress tracking.
        embedded_artifact: Embedded ``autorag.shared`` helpers injected by KFP at runtime.
        optimization_settings: Additional experiment settings.
        input_data_keys: Paths to documents dirs within bucket. Only the first entry is
            used for the generated indexing notebook; the full list is propagated to the
            indexing pipeline blueprint.
        preset: Pipeline quality tier. "speed" (default) uses 10 benchmark query
            threads. "balanced" uses 4 threads (reduced due to larger per-request
            context).

    Environment variables (required):
        MAAS_BASE_URL, MAAS_API_KEY for inference. Plus the vector database
        configuration injected from ``vector_db_secret_name``: ``MILVUS_*`` keys
        (at least ``MILVUS_URI``) select Milvus, ``PGVECTOR_*`` keys select
        PGVector.
    """
    import importlib.util
    import json
    import logging
    import os
    import sys
    from pathlib import Path

    import pandas as pd

    if getattr(sys.modules.get("sqlite3"), "__name__", None) == "pysqlite3":
        return
    try:
        import pysqlite3  # type: ignore[import-untyped]

        sys.modules["sqlite3"] = pysqlite3
    except ImportError:
        pass

    from ai4rag import handler
    from ai4rag.assets_generator import build_leaderboard_html, generate_notebook_from_template
    from ai4rag.core.experiment.experiment import AI4RAGExperiment
    from ai4rag.core.hpo.gam_opt import GAMOptSettings
    from ai4rag.evaluator import BaseEvaluator, RagasEvaluator, UnitxtEvaluator
    from ai4rag.evaluator.metric import Metrics, RAGMetric
    from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel
    from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
    from ai4rag.rag.vector_store import get_vector_store_config
    from ai4rag.search_space.prepare.models import get_embedding_models, get_foundation_models
    from ai4rag.search_space.src.parameter import Parameter
    from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
    from ai4rag.utils.clients.maas_client import create_maas_client
    from ai4rag.utils.docling_io import load_docling_documents
    from ai4rag.utils.event_handler import KFPEventHandler

    logging.basicConfig(level=logging.INFO)
    _logger = logging.getLogger("rag-templates-optimization")
    _logger.addHandler(handler)

    DEFAULT_METRIC = Metrics.OVERALL_SCORE.name

    # This component always runs both evaluators; "custom" (overall_score) is
    # aggregated from their scores, so it's available too without running on
    # its own. Metrics scored only by other evaluators (e.g. the LLM-as-judge
    # metric) are therefore never selectable as an optimization target here.
    ACTIVE_EVALUATORS = frozenset({"ragas", "unitxt", "custom"})

    DEFAULT_MAX_RAG_PATTERNS = 8
    MIN_MAX_RAG_PATTERNS_RANGE = (4, 20)

    VALID_PRESETS = {"speed", "balanced"}
    PRESET_INFERENCE_MAX_THREADS = {"speed": 10, "balanced": 4}

    def _build_evaluators(
        foundation_models: list[OpenAIFoundationModel],
        embedding_models: list[OpenAIEmbeddingModel],
    ) -> list[BaseEvaluator]:
        """Build the fixed evaluator pair used for every experiment.

        Args:
            foundation_models: Foundation models from the search space; the
                first is used as the RAGAS generation model.
            embedding_models: Embedding models from the search space; the
                first is used by RAGAS.

        Returns:
            ``[UnitxtEvaluator(), RagasEvaluator(...)]``.
        """
        ragas_model = foundation_models[0]
        _logger.info("RAGAS evaluator enabled with model: %s", ragas_model.model_id)
        return [
            UnitxtEvaluator(),
            RagasEvaluator(model=ragas_model, embedding_model=embedding_models[0]),
        ]

    def _generate_output_artifacts(
        patterns_raw: list[dict],
        output_dir: Path,
        input_data_key: str,
        test_data_key: str,
        indexing_pipeline_params: dict | None,
    ) -> list[dict]:
        """Write per-pattern artefacts (JSON, notebooks, evaluation results)."""
        patterns: list[dict] = []

        for pattern in patterns_raw:
            patt_dir = output_dir / pattern.get("payload").get("name")
            patt_dir.mkdir(parents=True, exist_ok=True)

            pattern_data = pattern.get("payload")
            if indexing_pipeline_params:
                settings = pattern_data["settings"]
                vector_store_binding = settings["vector_store_binding"]
                pattern_data["indexing"] = {
                    "pipeline_spec": {
                        "pipeline_name": indexing_pipeline_params.get("pipeline_name", "documents_indexing_pipeline"),
                        "parameters": {
                            "maas_secret_name": indexing_pipeline_params.get("maas_secret_name"),
                            "vector_db_secret_name": indexing_pipeline_params.get("vector_db_secret_name"),
                            "input_data_secret_name": indexing_pipeline_params.get("input_data_secret_name"),
                            "input_data_bucket_name": indexing_pipeline_params.get("input_data_bucket_name"),
                            "input_data_keys": indexing_pipeline_params.get("input_data_keys"),
                            "batch_size": indexing_pipeline_params.get("batch_size"),
                            "provider_type": vector_store_binding["provider_type"],
                            "collection_name": vector_store_binding["collection_name"],
                            "embedding_model_id": settings["embedding"]["model_id"],
                            "embedding_params": settings["embedding"]["embedding_params"],
                            "chunking_method": settings["chunking"]["method"],
                            "chunk_size": settings["chunking"]["chunk_size"],
                            "chunk_overlap": settings["chunking"]["chunk_overlap"],
                        },
                        "overrides_allowed": [
                            "input_data_secret_name",
                            "input_data_bucket_name",
                            "input_data_keys",
                            "collection_name",
                            "batch_size",
                        ],
                    }
                }

            generate_notebook_from_template(
                "maas_indexing",
                pattern_data,
                patt_dir / "indexing.ipynb",
                input_data_key=input_data_key,
            )
            generate_notebook_from_template(
                "maas_inference",
                pattern_data,
                patt_dir / "inference.ipynb",
                test_data_key=test_data_key,
            )

            with (patt_dir / "pattern.json").open("w", encoding="utf-8") as f:
                json.dump(pattern_data, f, indent=2, ensure_ascii=False)

            with (patt_dir / "evaluation_results.json").open("w", encoding="utf-8") as f:
                json.dump(pattern.get("evaluation_results", []), f, indent=2, ensure_ascii=False)

            patterns.append(pattern_data)

        return patterns

    def _validate_optimization_settings(optimization_settings: dict | None) -> dict:
        """Validate and normalize optimization settings.

        Returns:
            Validated settings dictionary (empty dict when input is ``None``).

        Raises:
            TypeError: If settings or ``max_number_of_rag_patterns`` have
                wrong types.
            ValueError: If ``max_number_of_rag_patterns`` is out of the
                allowed range or cannot be parsed as an integer.
        """
        if optimization_settings is None:
            return {}

        if not isinstance(optimization_settings, dict):
            raise TypeError("optimization_settings must be a dictionary.")

        max_rag_patterns = optimization_settings.get("max_number_of_rag_patterns", DEFAULT_MAX_RAG_PATTERNS)
        if isinstance(max_rag_patterns, str):
            try:
                max_rag_patterns = int(max_rag_patterns.strip())
            except ValueError as exc:
                raise ValueError(
                    "optimization_settings.max_number_of_rag_patterns must be a valid integer "
                    f"(e.g. from the pipeline UI); got {max_rag_patterns!r}."
                ) from exc

        if not isinstance(max_rag_patterns, int):
            raise TypeError("optimization_settings.max_number_of_rag_patterns must be an integer.")

        if not MIN_MAX_RAG_PATTERNS_RANGE[0] <= max_rag_patterns <= MIN_MAX_RAG_PATTERNS_RANGE[1]:
            raise ValueError(
                f"optimization_settings.max_number_of_rag_patterns must be in range "
                f"{MIN_MAX_RAG_PATTERNS_RANGE[0]} to {MIN_MAX_RAG_PATTERNS_RANGE[1]}."
            )

        return optimization_settings

    def _get_optimization_metric(metric_name: str | None) -> RAGMetric:
        """Resolve a metric name to the ``RAGMetric`` this component will optimize for.

        Some metric names are ambiguous across evaluators (e.g. ``"faithfulness"``
        is scored by both unitxt and RAGAS). Since both evaluators are always
        active here (see :data:`ACTIVE_EVALUATORS`), such names resolve to
        more than one candidate; RAGAS wins ties because it provides the more
        discriminative score for optimization.

        Args:
            metric_name: Metric requested via ``optimization_settings.metric``.
                Falls back to :data:`DEFAULT_METRIC` when falsy.

        Returns:
            The resolved metric, preferring the RAGAS variant on ties.

        Raises:
            ValueError: If ``metric_name`` doesn't match any metric known to
                ``Metrics``, or matches metrics only from evaluators this
                component doesn't run (e.g. the LLM-as-judge metric).
        """
        metric_name = metric_name or DEFAULT_METRIC

        candidates = [m for m in Metrics if m.name == metric_name]
        if not candidates:
            raise ValueError(
                f"Optimization metric {metric_name!r} is not supported. "
                f"Select one of {sorted({m.name for m in Metrics})}."
            )

        available = [m for m in candidates if m.evaluator in ACTIVE_EVALUATORS]
        if not available:
            raise ValueError(
                f"Optimization metric {metric_name!r} is only produced by evaluator(s) "
                f"{sorted({m.evaluator for m in candidates})}, but this component only "
                f"runs {sorted(ACTIVE_EVALUATORS - {'custom'})}."
            )

        return next((m for m in available if m.evaluator == "ragas"), available[0])

    # -------------------------------------------------------------------------
    # Component logic starts here
    # -------------------------------------------------------------------------

    if preset not in VALID_PRESETS:
        raise ValueError(f"preset must be one of {VALID_PRESETS}; got {preset!r}.")

    inference_max_threads = PRESET_INFERENCE_MAX_THREADS[preset]
    logging.info("Preset %r: inference_max_threads=%d", preset, inference_max_threads)

    if component_status is None:
        from kfp_components.components.training.autorag.shared.component_status import (  # pyright: ignore[reportMissingImports]
            null_component_status_tracker,
        )

        status = null_component_status_tracker()
    else:
        _embedded_path = Path(embedded_artifact.path)
        _module_path = _embedded_path if _embedded_path.is_file() else _embedded_path / "component_status.py"
        _spec = importlib.util.spec_from_file_location("_autorag_component_status", _module_path)
        if _spec is None or _spec.loader is None:
            raise ValueError(f"Cannot load embedded module from {_module_path}")
        _status_module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_status_module)
        status = _status_module.bootstrap_status_tracker(
            embedded_artifact, component_status, "rag_templates_optimization"
        )
    with status:
        if component_status is not None:
            status.set_metadata(display_name="RAG Templates Optimization Status")
            component_status.metadata["display_name"] = "RAG Templates Optimization Status"
        with status.stage("optimize_templates"):
            maas_client = create_maas_client(
                base_url=os.environ["MAAS_BASE_URL"],
                api_key=os.environ["MAAS_API_KEY"],
            )

            if any(k.startswith("MILVUS") for k in os.environ):
                provider = "milvus"
            elif any(k.startswith("PGVECTOR") for k in os.environ):
                provider = "pgvector"
            else:
                raise ValueError(
                    "No vector database configuration found. Expected MILVUS_* or PGVECTOR_* "
                    "environment variables injected from vector_db_secret_name."
                )
            vector_store_config = get_vector_store_config(provider)
            logging.info("Detected %s database provider from secret.", provider)

            output_dir = Path(rag_patterns.path)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Deployment blueprint stamped into every pattern.json so the indexing
            # pipeline can be reproduced. provider_type/collection_name are added
            # by ai4rag from each pattern's vector_store_binding.
            indexing_pipeline_params = {
                "pipeline_name": "documents-indexing-pipeline",
                "maas_secret_name": maas_secret_name,
                "vector_db_secret_name": vector_db_secret_name,
                "input_data_secret_name": input_data_secret_name,
                "input_data_bucket_name": input_data_bucket_name,
                "input_data_keys": input_data_keys or [],
                "batch_size": 20,
            }

            if (
                not isinstance(test_data_key, str)
                or not test_data_key.strip()
                or not test_data_key.lower().endswith(".json")
            ):
                raise ValueError("test_data_key must point to a JSON file.")

            settings = _validate_optimization_settings(optimization_settings)
            optimization_metric = _get_optimization_metric(settings.get("metric"))

            documents = load_docling_documents(extracted_text)
            benchmark_data = pd.read_json(Path(test_data))

            # --- Reconstruct search space from report ---
            with open(search_space_mps_report, "r", encoding="utf-8") as f:
                search_space_raw: dict[str, Any] = json.load(f)

            foundation_models: list[OpenAIFoundationModel] = get_foundation_models(
                maas_client, search_space_raw.get("foundation_model", []), validate=False
            )
            embedding_models: list[OpenAIEmbeddingModel] = get_embedding_models(
                maas_client, search_space_raw.get("embedding_model", []), validate=False
            )

            params: list[Parameter] = []
            for param_name, values in search_space_raw.items():
                if param_name == "foundation_model":
                    values = foundation_models
                elif param_name == "embedding_model":
                    values = embedding_models
                params.append(Parameter(param_name, "C", values=values))

            search_space = AI4RAGSearchSpace(params=params)

            evaluators = _build_evaluators(
                foundation_models=foundation_models,
                embedding_models=embedding_models,
            )

            # --- Configure experiment ---
            max_rag_patterns = settings.get("max_number_of_rag_patterns", DEFAULT_MAX_RAG_PATTERNS)
            if isinstance(max_rag_patterns, str):
                max_rag_patterns = int(max_rag_patterns.strip())
            optimizer_settings = GAMOptSettings(max_evals=max_rag_patterns)

            event_handler = KFPEventHandler()

            rag_exp = AI4RAGExperiment(
                event_handler=event_handler,
                optimizer_settings=optimizer_settings,
                search_space=search_space,
                benchmark_data=benchmark_data,
                vector_store_config=vector_store_config,
                documents=documents,
                optimization_metric=optimization_metric,
                inference_max_threads=inference_max_threads,
                evaluators=evaluators,
            )

            # --- Run the optimization loop ---
            rag_exp.search()

            # --- Generate output artefacts ---
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            patterns = _generate_output_artifacts(
                patterns_raw=event_handler.patterns,
                output_dir=output_dir,
                input_data_key=input_data_keys[0] if input_data_keys else "",
                test_data_key=test_data_key,
                indexing_pipeline_params=indexing_pipeline_params,
            )

            status.record(
                "optimize_templates",
                "completed",
                metrics={
                    "max_rag_patterns": len(patterns),
                    "selected_patterns": [p.get("name", "") for p in patterns],
                },
            )

            rag_patterns.metadata["name"] = "rag_patterns_artifact"
            rag_patterns.metadata["uri"] = rag_patterns.uri
            rag_patterns.metadata["metadata"] = {"patterns": patterns}

        with status.stage("build_leaderboard"):
            html_content = build_leaderboard_html(
                patterns_dir=output_dir,
                optimization_metric=optimization_metric.name,
                optimization_metric_evaluator=optimization_metric.evaluator,
            )

            Path(leaderboard.path).parent.mkdir(parents=True, exist_ok=True)
            with open(leaderboard.path, "w", encoding="utf-8") as f:
                f.write(html_content)
            leaderboard.metadata["display_name"] = "autorag_leaderboard"


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        rag_templates_optimization,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
