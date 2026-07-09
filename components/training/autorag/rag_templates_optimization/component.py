from pathlib import Path
from typing import Optional

from kfp import dsl
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]

_AUTORAG_SHARED = Path(__file__).parents[1] / "shared"


@dsl.component(
    base_image=AUTORAG_IMAGE,  # noqa: E501
    embedded_artifact_path=str(_AUTORAG_SHARED / "component_status.py"),
    install_kfp_package=False,
)
def rag_templates_optimization(
    extracted_text: dsl.InputPath(dsl.Artifact),
    test_data: dsl.InputPath(dsl.Artifact),
    search_space_prep_report: dsl.InputPath(dsl.Artifact),
    rag_patterns: dsl.Output[dsl.Artifact],
    test_data_key: Optional[str],
    vector_io_provider_id: str,
    html_artifact: dsl.Output[dsl.HTML],
    embedded_artifact: dsl.EmbeddedInput[dsl.Dataset] = None,
    optimization_settings: Optional[dict] = None,
    input_data_key: Optional[str] = "",
    component_status: dsl.Output[dsl.Artifact] = None,
):
    """RAG Templates Optimization component.

    Thin wrapper that delegates to
    ``ai4rag.components.optimization.rag_templates_optimization.run_rag_optimization``.

    Args:
        extracted_text: Path to extracted text documents.
        test_data: Path to benchmark test data JSON.
        search_space_prep_report: Path to the YAML search space report.
        rag_patterns: Output artifact for generated RAG patterns.
        test_data_key: Path to benchmark JSON in object storage.
        vector_io_provider_id: Vector I/O provider identifier in OGX.
        html_artifact: Output HTML artifact; the leaderboard table is written to
            html_artifact.path (single file).
        component_status: Output artifact containing stage-level progress tracking.
        embedded_artifact: Embedded ``autorag.shared`` helpers injected by KFP at runtime.
        optimization_settings: Additional experiment settings.
        input_data_key: Path to documents dir within bucket.

    Environment variables (required):
        OGX_CLIENT_BASE_URL, OGX_CLIENT_API_KEY.
    """
    import importlib.util
    import logging
    import os
    from pathlib import Path

    from ai4rag.utils.compat import ensure_sqlite3

    ensure_sqlite3()

    from ai4rag.components.assets_generator.leaderboard import build_leaderboard_html
    from ai4rag.components.optimization.rag_templates_optimization import DEFAULT_METRIC, run_rag_optimization
    from ai4rag.components.utils.ogx_client import create_ogx_client

    logging.basicConfig(level=logging.INFO)

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
            ogx_client = create_ogx_client(
                base_url=os.environ["OGX_CLIENT_BASE_URL"],
                api_key=os.environ["OGX_CLIENT_API_KEY"],
            )

            output_dir = Path(rag_patterns.path)
            output_dir.mkdir(parents=True, exist_ok=True)

            result = run_rag_optimization(
                extracted_text_path=extracted_text,
                test_data_path=test_data,
                search_space_report_path=search_space_prep_report,
                output_dir=output_dir,
                ogx_client=ogx_client,
                vector_io_provider_id=vector_io_provider_id,
                test_data_key=test_data_key or "",
                input_data_key=input_data_key or "",
                optimization_settings=optimization_settings,
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

            Path(html_artifact.path).parent.mkdir(parents=True, exist_ok=True)
            with open(html_artifact.path, "w", encoding="utf-8") as f:
                f.write(html_content)
            html_artifact.metadata["display_name"] = "autorag_leaderboard"


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        rag_templates_optimization,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
