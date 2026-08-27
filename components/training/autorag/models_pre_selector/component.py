from pathlib import Path

from kfp import dsl
from kfp.compiler import Compiler
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]

_AUTORAG_SHARED = Path(__file__).parents[1] / "shared"


@dsl.component(
    base_image=AUTORAG_IMAGE,  # noqa: E501
    embedded_artifact_path=str(_AUTORAG_SHARED / "component_status.py"),
    install_kfp_package=False,
)
def models_pre_selector(
    search_space_report: dsl.Input[dsl.Artifact],
    extracted_text: dsl.Input[dsl.Artifact],
    test_data: dsl.Input[dsl.Artifact],
    search_space_mps_report: dsl.Output[dsl.Artifact],
    embedded_artifact: dsl.EmbeddedInput[dsl.Dataset] = None,
    component_status: dsl.Output[dsl.Artifact] = None,
    preset: str = "speed",
):
    """Model pre-selection for AutoRAG experiments.

    Trims the candidate model set when the search space carries more models than
    the optimizer should explore. This component **always runs** and decides
    internally: when the report holds more than ``DEFAULT_N_FOUNDATION_MODELS``
    foundation models or more than ``DEFAULT_N_EMBEDDING_MODELS`` embedding
    models, it runs ``ai4rag.core.experiment.mps.ModelsPreSelector`` over a
    benchmark sample and writes a *limited* report; otherwise it passes the full
    report through unchanged (a cheap copy — no MaaS calls, no document loading).

    Running as a distinct step keeps the (potentially expensive) pre-selection
    evaluation observable in the pipeline and produces the reduced search-space
    report consumed by ``rag_templates_optimization``.

    Args:
        search_space_report: Input artifact with the full search space report
            produced by ``search_space_preparation``.
        extracted_text: Input artifact with extracted text documents. Only read
            when pre-selection actually runs.
        test_data: Input artifact with benchmark questions and expected answers.
            Only read when pre-selection actually runs.
        search_space_mps_report: Output artifact for the (possibly reduced) JSON
            search space report.
        embedded_artifact: Embedded ``autorag.shared`` helpers injected by KFP at runtime.
        component_status: Output artifact containing stage-level progress tracking.
        preset: Pipeline quality tier. "speed" (default) uses 10 benchmark query
            threads during pre-selection; "balanced" uses 4 (reduced due to
            larger per-request context).

    Environment variables (required only when pre-selection runs):
        MAAS_BASE_URL, MAAS_API_KEY.
    """
    import importlib.util
    import json
    import logging
    import os
    from pathlib import Path

    from ai4rag.utils.compat import ensure_sqlite3

    ensure_sqlite3()

    from ai4rag.core.experiment.mps import ModelsPreSelector
    from ai4rag.search_space.prepare import SearchSpaceReport

    logging.basicConfig(level=logging.INFO)

    VALID_PRESETS = {"speed", "balanced"}
    PRESET_INFERENCE_MAX_THREADS = {"speed": 10, "balanced": 4}
    SAMPLE_SIZE = 5
    RANDOM_SEED = 17

    if preset not in VALID_PRESETS:
        raise ValueError(f"preset must be one of {VALID_PRESETS}; got {preset!r}.")

    inference_max_threads = PRESET_INFERENCE_MAX_THREADS[preset]

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
        status = _status_module.bootstrap_status_tracker(embedded_artifact, component_status, "models_pre_selector")
    with status:
        if component_status is not None:
            status.set_metadata(display_name="Model Pre-Selection Status")
            component_status.metadata["display_name"] = "Model Pre-Selection Status"
        with status.stage("model_pre_selection"):
            with open(search_space_report.path, "r", encoding="utf-8") as f:
                report = json.load(f)

            foundation_specs = report.get("foundation_model", [])
            embedding_specs = report.get("embedding_model", [])

            n_foundation = ModelsPreSelector.DEFAULT_N_FOUNDATION_MODELS
            n_embedding = ModelsPreSelector.DEFAULT_N_EMBEDDING_MODELS

            needs_pre_selection = len(foundation_specs) > n_foundation or len(embedding_specs) > n_embedding

            if needs_pre_selection:
                logging.info(
                    "Running model pre-selection: %d foundation / %d embedding models exceed caps (%d / %d).",
                    len(foundation_specs),
                    len(embedding_specs),
                    n_foundation,
                    n_embedding,
                )

                import pandas as pd
                from ai4rag.components.utils import create_maas_client
                from ai4rag.components.utils.docling_io import load_docling_documents
                from ai4rag.core.experiment.benchmark_data import BenchmarkData
                from ai4rag.search_space.prepare import (
                    get_embedding_models,
                    get_foundation_models,
                    serialize_model,
                )

                maas_client = create_maas_client(
                    base_url=os.environ["MAAS_BASE_URL"],
                    api_key=os.environ["MAAS_API_KEY"],
                )

                foundation_models = get_foundation_models(maas_client, foundation_specs, validate=False)
                embedding_models = get_embedding_models(maas_client, embedding_specs, validate=False)

                documents = load_docling_documents(extracted_text.path)
                benchmark = BenchmarkData(pd.read_json(test_data.path))

                mps = ModelsPreSelector(
                    benchmark_data=benchmark.get_random_sample(n_records=SAMPLE_SIZE, random_seed=RANDOM_SEED),
                    documents=documents,
                    foundation_models=foundation_models,
                    embedding_models=embedding_models,
                    max_threads=inference_max_threads,
                )
                mps.evaluate_patterns()
                selected = mps.select_models(
                    n_foundation_models=n_foundation,
                    n_embedding_models=n_embedding,
                )

                report["foundation_model"] = [serialize_model(m) for m in selected["foundation_models"]]
                report["embedding_model"] = [serialize_model(m) for m in selected["embedding_models"]]

                logging.info(
                    "Model pre-selection selected %d foundation / %d embedding models.",
                    len(report["foundation_model"]),
                    len(report["embedding_model"]),
                )
            else:
                logging.info(
                    "Model counts (%d foundation / %d embedding) within caps (%d / %d); "
                    "passing the search space report through unchanged.",
                    len(foundation_specs),
                    len(embedding_specs),
                    n_foundation,
                    n_embedding,
                )

            SearchSpaceReport(search_space=report).save_json(search_space_mps_report.path)


if __name__ == "__main__":
    Compiler().compile(
        models_pre_selector,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
