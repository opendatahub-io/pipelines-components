from pathlib import Path
from typing import List, Optional

from kfp import dsl
from kfp.compiler import Compiler
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]

_AUTORAG_SHARED = Path(__file__).parents[1] / "shared"


@dsl.component(
    base_image=AUTORAG_IMAGE,  # noqa: E501
    embedded_artifact_path=str(_AUTORAG_SHARED / "component_status.py"),
    install_kfp_package=False,
)
def search_space_preparation(
    test_data: dsl.Input[dsl.Artifact],
    extracted_text: dsl.Input[dsl.Artifact],
    search_space_prep_report: dsl.Output[dsl.Artifact],
    embedded_artifact: dsl.EmbeddedInput[dsl.Dataset] = None,
    embedding_models: Optional[List] = None,
    generation_models: Optional[List] = None,
    metric: str = None,
    component_status: dsl.Output[dsl.Artifact] = None,
    preset: str = "speed",
):
    """Search space preparation for AutoRAG experiments.

    Thin wrapper that delegates to
    ``ai4rag.components.optimization.search_space_preparation.prepare_search_space_report``.

    Args:
        test_data: Input artifact with benchmark questions and expected answers.
        extracted_text: Input artifact with extracted text documents.
        search_space_prep_report: Output artifact for the JSON search space report.
        component_status: Output artifact containing stage-level progress tracking.
        embedded_artifact: Embedded ``autorag.shared`` helpers injected by KFP at runtime.
        embedding_models: List of embedding model identifiers to try.
        generation_models: List of generation model identifiers to try.
        metric: Quality metric for evaluation (e.g. "faithfulness").
        preset: Pipeline quality tier. "speed" (default) uses recursive chunking
            without contextual enrichment. "balanced" uses hybrid chunking with
            LLM contextual enrichment in the search space.

    Environment variables (required):
        OGX_CLIENT_BASE_URL, OGX_CLIENT_API_KEY.
    """
    import importlib.util
    import logging
    import os
    from pathlib import Path

    from ai4rag.utils.compat import ensure_sqlite3

    ensure_sqlite3()

    from ai4rag.components.optimization.search_space_preparation import prepare_search_space_report
    from ai4rag.components.utils.ogx_client import create_ogx_client

    logging.basicConfig(level=logging.INFO)

    VALID_PRESETS = {"speed", "balanced"}
    PRESET_CHUNKING_METHODS = {"speed": ["recursive"], "balanced": ["recursive", "hybrid"]}
    PRESET_CHUNK_SIZES = {"speed": [128, 256, 512], "balanced": [512, 1024, 2048]}
    PRESET_CHUNK_OVERLAPS = {"speed": [32, 64], "balanced": [0, 128, 256]}
    PRESET_INFERENCE_MAX_THREADS = {"speed": 10, "balanced": 4}

    if preset not in VALID_PRESETS:
        raise ValueError(f"preset must be one of {VALID_PRESETS}; got {preset!r}.")

    chunking_methods = PRESET_CHUNKING_METHODS[preset]
    chunk_sizes = PRESET_CHUNK_SIZES[preset]
    chunk_overlaps = PRESET_CHUNK_OVERLAPS[preset]
    inference_max_threads = PRESET_INFERENCE_MAX_THREADS[preset]

    logging.info(
        "Preset %r: chunking_methods=%s, chunk_sizes=%s, chunks_overlaps=%s, inference_max_threads=%s",
        preset,
        chunking_methods,
        chunk_sizes,
        chunk_overlaps,
        inference_max_threads,
    )

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
            ogx_client = create_ogx_client(
                base_url=os.environ["OGX_CLIENT_BASE_URL"],
                api_key=os.environ["OGX_CLIENT_API_KEY"],
            )

            report = prepare_search_space_report(
                test_data_path=test_data.path,
                extracted_text_path=extracted_text.path,
                ogx_client=ogx_client,
                embedding_models=embedding_models,
                generation_models=generation_models,
                metric=metric if metric is not None else "faithfulness",
                chunking_methods=chunking_methods,
                chunk_sizes=chunk_sizes,
                chunk_overlaps=chunk_overlaps,
                inference_max_threads=inference_max_threads,
            )

            report.save_json(search_space_prep_report.path)


if __name__ == "__main__":
    Compiler().compile(
        search_space_preparation,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
