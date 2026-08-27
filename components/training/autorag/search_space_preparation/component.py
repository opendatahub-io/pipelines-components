from pathlib import Path
from typing import List

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
    search_space_report: dsl.Output[dsl.Artifact],
    embedding_models: List[str],
    generation_models: List[str],
    embedded_artifact: dsl.EmbeddedInput[dsl.Dataset] = None,
    component_status: dsl.Output[dsl.Artifact] = None,
    preset: str = "speed",
):
    """Search space preparation and validation for AutoRAG experiments.

    Resolves and validates the requested MaaS models, builds the AutoRAG search
    space, and writes it as a JSON report. This step runs *before* text
    extraction so that unresponsive or misconfigured models fail the experiment
    fast, before any heavy document processing is performed.

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

    Environment variables (required):
        MAAS_BASE_URL, MAAS_API_KEY.
    """
    import importlib.util
    import logging
    import os
    from pathlib import Path

    from ai4rag.utils.compat import ensure_sqlite3

    ensure_sqlite3()

    import pandas as pd
    from ai4rag.components.utils import create_maas_client
    from ai4rag.search_space.prepare import build_search_space_report, prepare_search_space_with_maas

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
    chunk_sizes = PRESET_CHUNK_SIZES[preset]
    chunk_overlaps = PRESET_CHUNK_OVERLAPS[preset]

    logging.info(
        "Preset %r: chunking_methods=%s, chunk_sizes=%s, chunk_overlaps=%s",
        preset,
        chunking_methods,
        chunk_sizes,
        chunk_overlaps,
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
            maas_client = create_maas_client(
                base_url=os.environ["MAAS_BASE_URL"],
                api_key=os.environ["MAAS_API_KEY"],
            )

            payload = {
                "foundation_models": [{"model_id": gm} for gm in generation_models],
                "embedding_models": [{"model_id": em} for em in embedding_models],
                "chunking_methods": chunking_methods,
                "chunk_sizes": chunk_sizes,
                "chunk_overlaps": chunk_overlaps,
            }

            benchmark_df = pd.read_json(test_data.path)

            search_space = prepare_search_space_with_maas(
                payload,
                client=maas_client,
                benchmark_data=benchmark_df,
            )

            build_search_space_report(search_space).save_json(search_space_report.path)


if __name__ == "__main__":
    Compiler().compile(
        search_space_preparation,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
