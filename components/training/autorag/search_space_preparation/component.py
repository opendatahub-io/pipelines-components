from pathlib import Path
from typing import List, NamedTuple

from kfp import dsl
from kfp.compiler import Compiler
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]

_AUTORAG_SHARED = Path(__file__).parents[1] / "shared"


@dsl.component(
    base_image=AUTORAG_IMAGE,  # noqa: E501
    embedded_artifact_path=str(_AUTORAG_SHARED / "runtime_embed" / "component_status.py"),
    install_kfp_package=False,
)
def search_space_preparation(
    test_data: dsl.Input[dsl.Artifact],
    search_space_report: dsl.Output[dsl.Artifact],
    embedding_models: List[str],
    generation_models: List[str],
    embedded_artifact: dsl.EmbeddedInput[dsl.Dataset] = None,
    component_status: dsl.Output[dsl.Artifact] = None,
    preset: str = "speed",
) -> NamedTuple("SearchSpacePreparationOutputs", [("detected_ocr_lang", str)]):
    """Search space preparation and validation for AutoRAG experiments.

    Resolves and validates the requested MaaS models, builds the AutoRAG search
    space, and writes it as a JSON report. This step runs *before* text
    extraction so that unresponsive or misconfigured models fail the experiment
    fast, before any heavy document processing is performed.

    It also surfaces the language AutoRAG detects from the benchmark questions, so
    text extraction can pick the matching OCR model bundle.

    Args:
        test_data: Input artifact with benchmark questions and expected answers.
            Used for language detection during search-space preparation.
        search_space_report: Output artifact for the JSON search space report.
        embedding_models: List of embedding model identifiers to try.
        generation_models: List of generation model identifiers to try.
        embedded_artifact: Embedded ``autorag.shared`` helpers injected by KFP at runtime.
        component_status: Output artifact containing stage-level progress tracking.
        preset: Pipeline quality tier. "speed" (default) uses recursive chunking
            without contextual enrichment. "balanced" uses hybrid chunking with
            LLM contextual enrichment in the search space.

    Returns:
        detected_ocr_lang: ISO 639-1 code of the language AutoRAG detected from the
            benchmark questions, or an empty string when detection did not run or
            failed. Intended as the ``ocr_lang`` input of text extraction.

    Environment variables (required):
        MAAS_BASE_URL, MAAS_API_KEY.
    """
    import importlib.util
    import logging
    import os
    from collections import namedtuple
    from pathlib import Path

    import pandas as pd
    from ai4rag.search_space.prepare import build_search_space_report, prepare_search_space_with_maas
    from ai4rag.utils.clients import create_maas_client

    logging.basicConfig(level=logging.INFO)

    VALID_PRESETS = {"speed", "balanced"}
    PRESET_CHUNKING_METHODS = {"speed": ["recursive"], "balanced": ["recursive", "hybrid"]}
    PRESET_CHUNK_SIZES = {"speed": [128, 256, 512], "balanced": [512, 1024, 2048]}
    PRESET_CHUNK_OVERLAPS = {"speed": [32, 64], "balanced": [0, 128, 256]}

    if preset not in VALID_PRESETS:
        raise ValueError(f"preset must be one of {VALID_PRESETS}; got {preset!r}.")

    for name, models in (("generation_models", generation_models), ("embedding_models", embedding_models)):
        if not isinstance(models, list) or not models or any(not m for m in models):
            raise ValueError(f"{name} must be a non-empty list of non-empty model identifiers.")

    chunking_methods = PRESET_CHUNKING_METHODS[preset]

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
            embedded_artifact, component_status, "search_space_preparation"
        )
    with status:
        if component_status is not None:
            status.set_metadata(display_name="Search Space Preparation Status")
            component_status.metadata["display_name"] = "Search Space Preparation Status"
        with status.stage("prepare_search_space"):
            maas_client = create_maas_client(
                base_url=os.environ["MAAS_BASE_URL"],
                api_key=os.environ["MAAS_API_KEY"],
            )

            if "MILVUS_URI" in os.environ:
                vector_store_type = "milvus"
            elif "PGVECTOR_HOST" in os.environ:
                vector_store_type = "pgvector"
            elif "NEO4J_URI" in os.environ:
                vector_store_type = "neo4j"
            else:
                vector_store_type = "milvus"
                logging.warning(
                    "No MILVUS_URI, PGVECTOR_HOST, or NEO4J_URI environment variable found; defaulting to milvus."
                )
            logging.info("Detected vector store type: %s", vector_store_type)

            payload = {
                "foundation_models": [{"model_id": gm} for gm in generation_models],
                "embedding_models": [{"model_id": em} for em in embedding_models],
                "chunking_methods": chunking_methods,
            }
            if vector_store_type == "neo4j":
                # Neo4j owns its chunk geometry. Do not let a generic quality
                # preset override ai4rag's graph-safe defaults (1024 tokens and
                # its supported overlaps).
                logging.info(
                    "Preset %r: chunking_methods=%s; using ai4rag Neo4j chunk defaults.",
                    preset,
                    chunking_methods,
                )
            else:
                chunk_sizes = PRESET_CHUNK_SIZES[preset]
                chunk_overlaps = PRESET_CHUNK_OVERLAPS[preset]
                payload["chunk_sizes"] = chunk_sizes
                payload["chunk_overlaps"] = chunk_overlaps
                logging.info(
                    "Preset %r: chunking_methods=%s, chunk_sizes=%s, chunk_overlaps=%s",
                    preset,
                    chunking_methods,
                    chunk_sizes,
                    chunk_overlaps,
                )

            benchmark_df = pd.read_json(test_data.path)

            search_space = prepare_search_space_with_maas(
                payload,
                client=maas_client,
                benchmark_data=benchmark_df,
                vector_store_type=vector_store_type,
            )

            build_search_space_report(search_space).save_json(search_space_report.path)

            # ai4rag detects the language per foundation model from the same benchmark
            # questions, so the values normally agree; pick the first and warn otherwise.
            detected_codes = []
            for model in search_space["foundation_model"].values:
                code = getattr(getattr(model, "language", None), "code", "")
                # Normalize before the emptiness check: a whitespace-only code is
                # truthy but normalizes to "", which would sort ahead of a real
                # code below and silently downgrade detection to English.
                normalized = code.strip().lower() if code else ""
                if normalized:
                    detected_codes.append(normalized)

            distinct_codes = sorted(set(detected_codes))
            if len(distinct_codes) > 1:
                logging.warning(
                    "Foundation models disagree on the detected language (%s); using %r.",
                    ", ".join(distinct_codes),
                    distinct_codes[0],
                )
            detected_ocr_lang = distinct_codes[0] if distinct_codes else ""
            logging.info("Detected language for OCR: %r", detected_ocr_lang)

    outputs = namedtuple("SearchSpacePreparationOutputs", ["detected_ocr_lang"])
    return outputs(detected_ocr_lang)


if __name__ == "__main__":
    Compiler().compile(
        search_space_preparation,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
