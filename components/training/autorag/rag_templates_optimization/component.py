from pathlib import Path
from typing import Optional

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
    test_data_key: Optional[str],
    maas_secret_name: str,
    vector_db_secret_name: str,
    input_data_secret_name: str,
    input_data_bucket_name: str,
    leaderboard: dsl.Output[dsl.HTML],
    embedded_artifact: dsl.EmbeddedInput[dsl.Dataset] = None,
    optimization_settings: Optional[dict] = None,
    input_data_key: Optional[str] = "",
    component_status: dsl.Output[dsl.Artifact] = None,
    preset: str = "speed",
    llm_judge_mode: str = "base",
):
    """RAG Templates Optimization component.

    Thin wrapper that delegates to
    ``ai4rag.components.optimization.rag_templates_optimization.run_rag_optimization``.

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
            leaderboard.path (single file).
        component_status: Output artifact containing stage-level progress tracking.
        embedded_artifact: Embedded ``autorag.shared`` helpers injected by KFP at runtime.
        optimization_settings: Additional experiment settings.
        input_data_key: Path to documents dir within bucket.
        preset: Pipeline quality tier. "speed" (default) uses 10 benchmark query
            threads. "balanced" uses 4 threads (reduced due to larger per-request
            context).
        llm_judge_mode: LLM-as-a-judge evaluation mode passed to ai4rag. One of
            "base" (default, in-house judge), "ragas" (RAGAS-based judge), "all"
            (both), or "none" (no LLM judge).

    Environment variables (required):
        MAAS_BASE_URL, MAAS_API_KEY for inference. Plus the vector database
        configuration injected from ``vector_db_secret_name``: ``MILVUS_*`` keys
        (at least ``MILVUS_URI``) select Milvus, ``PGVECTOR_*`` keys select
        PGVector.
    """
    import importlib.util
    import logging
    import os
    from pathlib import Path

    from ai4rag.utils.compat import ensure_sqlite3

    ensure_sqlite3()

    from ai4rag.components.assets_generator.leaderboard import build_leaderboard_html
    from ai4rag.components.optimization.rag_templates_optimization import DEFAULT_METRIC, run_rag_optimization
    from ai4rag.components.utils import create_maas_client
    from ai4rag.rag.vector_store import get_vector_store_config

    logging.basicConfig(level=logging.INFO)

    VALID_PRESETS = {"speed", "balanced"}
    PRESET_INFERENCE_MAX_THREADS = {"speed": 10, "balanced": 4}

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
    optimize_templates_steps = ["chunking", "embedding", "retrieval", "generation", "evaluation"]

    with status:
        if component_status is not None:
            status.set_metadata(display_name="RAG Templates Optimization Status")
            component_status.metadata["display_name"] = "RAG Templates Optimization Status"
        with status.stage("optimize_templates", steps=optimize_templates_steps):
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
                "input_data_key": input_data_key or "",
                "batch_size": 20,
            }

            result = run_rag_optimization(
                extracted_text_path=extracted_text,
                test_data_path=test_data,
                search_space_report_path=search_space_mps_report,
                output_dir=output_dir,
                maas_client=maas_client,
                vector_store_config=vector_store_config,
                test_data_key=test_data_key or "",
                input_data_key=input_data_key or "",
                optimization_settings=optimization_settings,
                inference_max_threads=inference_max_threads,
                indexing_pipeline_params=indexing_pipeline_params,
                llm_judge_mode=llm_judge_mode,
            )

            status.record(
                "optimize_templates",
                "completed",
                max_rag_patterns=len(result.patterns),
                selected_patterns=[p.get("name", "") for p in result.patterns],
                steps=optimize_templates_steps,
            )

            rag_patterns.metadata["name"] = "rag_patterns_artifact"
            rag_patterns.metadata["uri"] = rag_patterns.uri
            rag_patterns.metadata["metadata"] = {"patterns": result.patterns}

        with status.stage("build_leaderboard"):
            html_content = build_leaderboard_html(
                patterns_dir=output_dir,
                optimization_metric=(
                    optimization_settings.get("metric") or DEFAULT_METRIC
                    if isinstance(optimization_settings, dict)
                    else DEFAULT_METRIC
                ),
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
