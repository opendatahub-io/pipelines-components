from pathlib import Path
from typing import Any, Optional

from kfp import dsl
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]

# Shared embed root (status + MLflow helpers). Committed under shared/ so DSPA can
# compile managed pipelines from a read-only site-packages install without
# import-time staging, and so there is a single source for those modules.
_AUTORAG_SHARED = Path(__file__).parents[1] / "shared"


@dsl.component(
    base_image=AUTORAG_IMAGE,
    embedded_artifact_path=str(_AUTORAG_SHARED / "runtime_embed"),
    install_kfp_package=False,
)
def rag_templates_optimization(
    extracted_text: dsl.InputPath(dsl.Artifact),
    test_data: dsl.InputPath(dsl.Artifact),
    search_space_mps_report: dsl.InputPath(dsl.Artifact),
    rag_patterns: dsl.Output[dsl.Artifact],
    test_data_key: str,
    maas_secret_name: str,
    db_secret_name: str,
    input_data_secret_name: str,
    input_data_bucket_name: str,
    leaderboard: dsl.Output[dsl.HTML],
    starter_kit: dsl.Output[dsl.Artifact],
    embedded_artifact: dsl.EmbeddedInput[dsl.Dataset] = None,
    optimization_settings: Optional[dict] = None,
    input_data_keys: Optional[list[str]] = None,
    component_status: dsl.Output[dsl.Artifact] = None,
    preset: str = "speed",
    pipeline_name: str = "",
    run_id: str = "",
    run_name: str = "",
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
        db_secret_name: Name of the K8s secret holding the database
            configuration. Its keys select the backend: ``MILVUS_*`` keys use
            Milvus, ``PGVECTOR_*`` keys use PGVector, ``NEO4J_*`` keys use Neo4j.
            Propagated into each generated ``pattern.json`` indexing spec.
        input_data_secret_name: Name of the K8s secret with S3 credentials for
            input data.
        input_data_bucket_name: S3 bucket containing input documents.
        leaderboard: Output HTML artifact; the leaderboard table is written to
            leaderboard_html.path (single file).
        starter_kit: Output ZIP artifact containing the generated starter kit for
            the best-performing RAG pattern.
        component_status: Output artifact containing stage-level progress tracking.
        embedded_artifact: Embedded ``autorag.shared`` helpers injected by KFP at runtime.
        optimization_settings: Additional experiment settings. The
            ``max_number_of_rag_patterns`` setting (4-10, default 5) limits
            optimization iterations and published patterns.
        input_data_keys: Paths to documents dirs within bucket, 1-10 of them. The full list
            is propagated both to the generated indexing notebook and to the indexing
            pipeline blueprint, so either route reingests the same corpus.
        preset: Pipeline quality tier. "speed" (default) uses 10 benchmark query
            threads. "balanced" uses 4 threads (reduced due to larger per-request
            context).
        pipeline_name: Pipeline identifier, logged to MLflow as a param and tag.
        run_id: KFP run ID (``dsl.PIPELINE_JOB_ID_PLACEHOLDER``), logged to MLflow.
        run_name: KFP run name (``dsl.PIPELINE_JOB_NAME_PLACEHOLDER``). Logged to
            MLflow, and used to name the fallback experiment/run when the platform
            supplies no parent run.
    MLflow logging:
        Tracking is server-configured, not parameter-driven: it activates only when the
        platform injects ``KFP_MLFLOW_CONFIG`` into the step (RHOAI supplies the
        endpoint, workspace, experiment, and parent run). When it is absent, or the
        ``mlflow`` package is missing from the image, optimization runs unchanged and
        nothing is logged. When it is present, the parent KFP run is resumed and each
        RAG pattern gets a nested child run written *as the optimizer evaluates it*,
        so the experiment fills in live rather than in one batch at the end.

    Environment variables (required):
        MAAS_BASE_URL, MAAS_API_KEY for inference. Plus the vector database
        configuration injected from ``db_secret_name``: ``MILVUS_*`` keys
        (at least ``MILVUS_URI``) select Milvus, ``PGVECTOR_*`` keys select
        PGVector, ``NEO4J_*`` keys (at least ``NEO4J_URI`` and ``NEO4J_PASSWORD``)
        select Neo4j.

    Environment variables (optional):
        KFP_MLFLOW_CONFIG, injected by the platform MLflow integration. See
        "MLflow logging" above.
    """
    import importlib.util
    import json
    import logging
    import os
    import shutil
    import sys
    import tempfile
    from pathlib import Path
    from zipfile import ZipFile

    import pandas as pd
    from ai4rag import handler
    from ai4rag.assets_generator import build_leaderboard_html, generate_notebook_from_template, generate_starter_kit
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

    DEFAULT_MAX_RAG_PATTERNS = 5
    MIN_MAX_RAG_PATTERNS_RANGE = (4, 10)

    # custom:overall_score aggregates the outputs of the evaluators enabled for the preset.
    PRESET_EVALUATORS = {
        "speed": frozenset({"unitxt", "custom"}),
        "balanced": frozenset({"unitxt", "ragas", "custom"}),
    }
    LEGACY_METRIC_PREFERENCES = {"faithfulness": "ragas"}
    PRESET_SETTINGS = {
        "speed": {"inference_max_threads": 10, "warm_start_strategy": "greedy"},
        "balanced": {
            "inference_max_threads": 4,
            "warm_start_strategy": "balanced",
            "fields_to_balance": ["foundation_model", "embedding_model", "chunking_method"],
        },
    }
    PRESET_KG_EXTRACTION_CONFIG = {
        "speed": {"mode": "constrained"},
        "balanced": {
            "mode": "free",
            "max_entities_per_chunk": 5,
            "max_relationships_per_chunk": 5,
        },
    }
    PRESET_GRAPH_RETRIEVAL_CONFIG = {
        # Use AI4RAG's Neo4j graph-retrieval defaults for the speed preset.
        "speed": {},
        "balanced": {
            "entity_pivot_limit": 3,
            "entity_relationship_hops": 2,
            "relationship_neighbor_limit": 5,
        },
    }

    def _build_evaluators(
        foundation_models: list[OpenAIFoundationModel],
        embedding_models: list[OpenAIEmbeddingModel],
        active_evaluators: frozenset[str],
    ) -> list[BaseEvaluator]:
        """Build the evaluators enabled by the selected preset.

        Args:
            foundation_models: Foundation models from the search space; the
                first is used as the RAGAS generation model.
            embedding_models: Embedding models from the search space; the
                first is used by RAGAS.
            active_evaluators: Evaluators enabled by the selected preset.

        Returns:
            Unitxt, plus RAGAS when it is enabled by the selected preset.
        """
        evaluators = [UnitxtEvaluator()]
        if "ragas" in active_evaluators:
            ragas_model = foundation_models[0]
            _logger.info("RAGAS evaluator enabled with model: %s", ragas_model.model_id)
            evaluators.append(RagasEvaluator(model=ragas_model, embedding_model=embedding_models[0]))
        return evaluators

    def _optimization_score(pattern_data: dict) -> float:
        """Return the mean score of the metric selected for optimization."""
        evaluation = pattern_data.get("evaluation") or {}
        metrics = evaluation.get("metrics", []) if isinstance(evaluation, dict) else []
        for metric in metrics:
            if not isinstance(metric, dict) or not metric.get("optimization_metric"):
                continue
            scores = metric.get("scores") or {}
            score = scores.get("mean") if isinstance(scores, dict) else None
            if score is not None:
                return float(score)
        return float("-inf")

    def _generate_output_artifacts(
        patterns_raw: list[dict],
        output_dir: Path,
        input_data_keys: list[str],
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
                store_binding = settings["store_binding"]
                pattern_data["indexing"] = {
                    "pipeline_spec": {
                        "pipeline_name": indexing_pipeline_params.get("pipeline_name", "documents-indexing-pipeline"),
                        "parameters": {
                            "maas_secret_name": indexing_pipeline_params.get("maas_secret_name"),
                            "db_secret_name": indexing_pipeline_params.get("db_secret_name"),
                            "input_data_secret_name": indexing_pipeline_params.get("input_data_secret_name"),
                            "input_data_bucket_name": indexing_pipeline_params.get("input_data_bucket_name"),
                            "input_data_keys": indexing_pipeline_params.get("input_data_keys"),
                            "batch_size": indexing_pipeline_params.get("batch_size"),
                            "provider_type": store_binding["provider_type"],
                            "collection_name": store_binding["collection_name"],
                            "embedding_model_id": settings["embedding"]["model_id"],
                            "embedding_params": settings["embedding"]["embedding_params"],
                            "foundation_model_id": settings["generation"]["model_id"],
                            "foundation_model_params": {
                                "temperature": settings["generation"]["temperature"],
                                "max_completion_tokens": settings["generation"]["max_completion_tokens"],
                            },
                            "chunking_method": settings["chunking"]["method"],
                            "chunk_size": settings["chunking"]["chunk_size"],
                            "chunk_overlap": settings["chunking"]["chunk_overlap"],
                            "kg_extraction_config": indexing_pipeline_params.get("kg_extraction_config"),
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

            # Neo4j patterns build and query a knowledge graph, rather than a
            # plain embedding index.  Their notebooks therefore need the graph
            # construction and graph-retrieval flows; other providers retain
            # the standard MaaS notebook pair.
            is_knowledge_graph_pattern = store_binding["provider_type"] == "neo4j"
            indexing_notebook_template = (
                "mass_creating_knowledge_graph" if is_knowledge_graph_pattern else "maas_indexing"
            )
            inference_notebook_template = (
                "mass_inference_knowledge_graph" if is_knowledge_graph_pattern else "maas_inference"
            )

            generate_notebook_from_template(
                indexing_notebook_template,
                pattern_data,
                patt_dir / "indexing.ipynb",
                input_data_keys=input_data_keys,
                test_data_key=test_data_key,
            )
            generate_notebook_from_template(
                inference_notebook_template,
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
            Validated settings dictionary, including the default evaluation limit
            when input is ``None``.

        Raises:
            TypeError: If settings or ``max_number_of_rag_patterns`` have
                wrong types.
            ValueError: If ``max_number_of_rag_patterns`` cannot be parsed as
                an integer or is outside its allowed range.
        """
        if optimization_settings is None:
            return {
                "max_number_of_rag_patterns": DEFAULT_MAX_RAG_PATTERNS,
            }

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

        return {
            **optimization_settings,
            "max_number_of_rag_patterns": max_rag_patterns,
        }

    def _get_optimization_metric(metric_id: str | None, *, active_evaluators: frozenset[str]) -> RAGMetric:
        """Resolve a preset-supported ``evaluator:metric`` ID to a ``RAGMetric``.

        Args:
            metric_id: Metric requested via ``optimization_settings.metric``.
                Qualified IDs (for example, ``"unitxt:faithfulness"``) avoid
                ambiguity between evaluator metric names. Unqualified IDs use
                the established RAGAS preference for ``"faithfulness"`` when
                RAGAS is enabled, and otherwise require one enabled metric.
            active_evaluators: Evaluators enabled by the selected preset.

        Returns:
            The resolved metric.

        Raises:
            ValueError: If the metric is unknown, unsupported by the selected
                preset, or ambiguous without an evaluator prefix.
        """
        if metric_id is not None and not isinstance(metric_id, str):
            raise TypeError("optimization_settings.metric must be a string.")
        metric_id = metric_id or DEFAULT_METRIC
        evaluator, separator, metric_name = metric_id.partition(":")
        if not separator:
            metric_name = evaluator
        candidates = [m for m in Metrics if m.name == metric_name and (not separator or m.evaluator == evaluator)]
        if not candidates:
            raise ValueError(
                f"Optimization metric {metric_id!r} is not supported. "
                f"Select one of {sorted(f'{m.evaluator}:{m.name}' for m in Metrics)}."
            )

        available = [m for m in candidates if m.evaluator in active_evaluators]
        if not available:
            raise ValueError(
                f"Optimization metric {metric_id!r} is unavailable for this preset. "
                f"It requires evaluator(s) {sorted({m.evaluator for m in candidates})}, "
                f"but this preset enables {sorted(active_evaluators)}."
            )
        if len(available) > 1:
            preferred_evaluator = LEGACY_METRIC_PREFERENCES.get(metric_name)
            preferred_metric = next((m for m in available if m.evaluator == preferred_evaluator), None)
            if preferred_metric is not None:
                return preferred_metric
            raise ValueError(
                f"Optimization metric {metric_id!r} is ambiguous. Select one of "
                f"{sorted(f'{m.evaluator}:{m.name}' for m in available)}."
            )

        return available[0]

    # -------------------------------------------------------------------------
    # Component logic starts here
    # -------------------------------------------------------------------------

    if preset not in PRESET_SETTINGS:
        raise ValueError(f"preset must be one of {set(PRESET_SETTINGS)}; got {preset!r}.")

    active_evaluators = PRESET_EVALUATORS[preset]
    preset_cfg = PRESET_SETTINGS[preset]
    inference_max_threads = preset_cfg["inference_max_threads"]
    kg_extraction_config = PRESET_KG_EXTRACTION_CONFIG[preset]
    graph_retrieval_config = PRESET_GRAPH_RETRIEVAL_CONFIG[preset]
    logging.info(
        "Preset %r: inference_max_threads=%d, KG extraction=%s",
        preset,
        inference_max_threads,
        kg_extraction_config,
    )

    def _load_embedded_module(module_filename: str, module_alias: str) -> Any:
        """Load one module from the embedded AutoRAG helpers.

        KFP mounts the staged embed as a directory. A single-file mount is also
        tolerated, for compatibility with embeds carrying only ``component_status.py``.
        """
        embedded_root = Path(embedded_artifact.path)
        if embedded_root.is_file():
            if embedded_root.name != module_filename:
                raise FileNotFoundError(f"Embedded artifact {embedded_root} does not provide {module_filename}.")
            module_path = embedded_root
        else:
            module_path = embedded_root / module_filename
        spec = importlib.util.spec_from_file_location(module_alias, module_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load embedded module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        # Register before exec_module: @dataclass resolves string annotations (these
        # modules use `from __future__ import annotations`) through sys.modules[__module__],
        # which raises AttributeError if the module is not there yet.
        sys.modules[module_alias] = module
        spec.loader.exec_module(module)
        return module

    if embedded_artifact is None:
        from kfp_components.components.training.autorag.shared import (  # pyright: ignore[reportMissingImports]
            mlflow_tracking as _mlflow_tracking,
        )
    else:
        _mlflow_tracking = _load_embedded_module("mlflow_tracking.py", "_autorag_mlflow_tracking")

    if component_status is None:
        from kfp_components.components.training.autorag.shared.component_status import (  # pyright: ignore[reportMissingImports]
            null_component_status_tracker,
        )

        status = null_component_status_tracker()
    else:
        _status_module = _load_embedded_module("component_status.py", "_autorag_component_status")
        status = _status_module.tracker_from_embedded(embedded_artifact, component_status, "rag_templates_optimization")

    # MLflow logging is best-effort and self-disabling: when the platform injects no
    # KFP_MLFLOW_CONFIG, run_logger is a no-op and optimization proceeds unchanged.
    with (
        status,
        _mlflow_tracking.experiment_run_logger(
            run_name=run_name or pipeline_name,
        ) as run_logger,
    ):
        if component_status is not None:
            status.set_metadata(display_name="RAG Templates Optimization Status")
            component_status.metadata["display_name"] = "RAG Templates Optimization Status"
        with status.stage("optimize_templates"):
            maas_client = create_maas_client(
                base_url=os.environ["MAAS_BASE_URL"],
                api_key=os.environ["MAAS_API_KEY"],
            )

            if "MILVUS_URI" in os.environ:
                provider = "milvus"
            elif "PGVECTOR_HOST" in os.environ:
                provider = "pgvector"
            elif "NEO4J_URI" in os.environ:
                provider = "neo4j"
            else:
                raise ValueError(
                    "No vector database configuration found. Expected MILVUS_*, PGVECTOR_*, or NEO4J_* "
                    "environment variables injected from db_secret_name."
                )
            vector_store_config = get_vector_store_config(provider)
            logging.info("Detected %s database provider from secret.", provider)

            output_dir = Path(rag_patterns.path)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Deployment blueprint stamped into every pattern.json so the indexing
            # pipeline can be reproduced. provider_type/collection_name are added
            # by ai4rag from each pattern's store_binding.
            indexing_pipeline_params = {
                "pipeline_name": "documents-indexing-pipeline",
                "maas_secret_name": maas_secret_name,
                "db_secret_name": db_secret_name,
                "input_data_secret_name": input_data_secret_name,
                "input_data_bucket_name": input_data_bucket_name,
                "input_data_keys": input_data_keys or [],
                "batch_size": 20,
                "kg_extraction_config": kg_extraction_config,
            }

            if (
                not isinstance(test_data_key, str)
                or not test_data_key.strip()
                or not test_data_key.lower().endswith(".json")
            ):
                raise ValueError("test_data_key must point to a JSON file.")

            settings = _validate_optimization_settings(optimization_settings)
            optimization_metric = _get_optimization_metric(settings.get("metric"), active_evaluators=active_evaluators)

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
                active_evaluators=active_evaluators,
            )

            # --- Configure experiment ---
            max_rag_patterns = settings["max_number_of_rag_patterns"]
            # In the worst balanced-preset case, 3 embedding models, 2 LLMs, and
            # 2 chunking methods require 12 warm-start evaluations. Reserve the
            # maximum allowed number of RAG patterns (10) beyond those evaluations.
            optimizer_settings = GAMOptSettings(
                max_evals=12 + MIN_MAX_RAG_PATTERNS_RANGE[1],
                max_iterations=max_rag_patterns,
                warm_start_strategy=preset_cfg["warm_start_strategy"],
                fields_to_balance=preset_cfg.get("fields_to_balance"),
            )

            run_logger.log_header(
                pipeline_name=pipeline_name,
                kfp_run_id=run_id,
                kfp_run_name=run_name,
                preset=preset,
                optimization_metric=f"{optimization_metric.evaluator}:{optimization_metric.name}",
                max_rag_patterns=max_rag_patterns,
                active_evaluators=active_evaluators,
                embedding_models=search_space_raw.get("embedding_model", []),
                generation_models=search_space_raw.get("foundation_model", []),
                test_data_key=test_data_key,
                input_data_bucket_name=input_data_bucket_name,
                input_data_keys=input_data_keys or [],
            )

            # Wrapped so every evaluated pattern is mirrored into a nested MLflow child run
            # the moment ai4rag emits it, rather than in one batch after search() returns.
            event_handler = _mlflow_tracking.MlflowPatternEventHandler(KFPEventHandler(), run_logger)

            rag_exp = AI4RAGExperiment(
                event_handler=event_handler,
                optimizer_settings=optimizer_settings,
                search_space=search_space,
                benchmark_data=benchmark_data,
                vector_store_config=vector_store_config,
                documents=documents,
                optimization_metric=optimization_metric,
                inference_max_threads=inference_max_threads,
                kg_extraction_config=kg_extraction_config,
                graph_retrieval_config=graph_retrieval_config,
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
                input_data_keys=input_data_keys or [],
                test_data_key=test_data_key,
                indexing_pipeline_params=indexing_pipeline_params,
            )
            run_logger.log_pattern_artifact_pointers(
                str(rag_patterns.uri),
                [str(pattern.get("name", "")) for pattern in patterns],
            )

            # Keep the ZIP in the task artifact directory, next to the
            # leaderboard and executor logs. ``rag_patterns`` is the
            # directory-shaped output in that directory, so its parent is
            # the stable task-artifact root. The default path assigned to
            # the starter-kit output may point to a separate output directory.
            rag_patterns_path = Path(rag_patterns.path)
            starter_kit_path = rag_patterns_path.parent / "starter_kit" / "starter_kit.zip"
            starter_kit_path.parent.mkdir(parents=True, exist_ok=True)

            rag_patterns_uri = str(rag_patterns.uri).rstrip("/")
            artifact_root_uri = rag_patterns_uri.rsplit("/", 1)[0] if "/" in rag_patterns_uri else rag_patterns_uri
            starter_kit.uri = f"{artifact_root_uri}/starter_kit/starter_kit.zip"
            starter_kit.set_path(str(starter_kit_path))

            if patterns:
                best_pattern = max(patterns, key=_optimization_score)
                with tempfile.TemporaryDirectory() as temp_dir:
                    generated_zip = generate_starter_kit(best_pattern, temp_dir)
                    shutil.copyfile(generated_zip, starter_kit_path)
            else:
                with ZipFile(starter_kit_path, "w"):
                    pass
            starter_kit.metadata["display_name"] = "starter_kit.zip"

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

        with status.stage("log_mlflow_results"):
            # Child runs were already written live during search(); this closes out the
            # parent with job-level aggregates. KFP owns output artifacts in S3, so
            # MLflow records metrics and metadata only, without copying those files.
            run_logger.finalize()
            mlflow_logged, mlflow_tracking_info = run_logger.result()
            for _key, _value in mlflow_tracking_info.items():
                rag_patterns.metadata[_key] = _value
            status.record(
                "log_mlflow_results",
                "completed",
                metrics={
                    "mlflow_tracking_enabled": run_logger.configured,
                    "mlflow_logged": mlflow_logged,
                    **mlflow_tracking_info,
                },
            )


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        rag_templates_optimization,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
