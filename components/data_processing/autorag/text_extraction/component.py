from pathlib import Path
from typing import Optional

from kfp import dsl
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]

_AUTORAG_SHARED = Path(__file__).parents[3] / "training" / "autorag" / "shared"


@dsl.component(
    base_image=AUTORAG_IMAGE,  # noqa: E501
    embedded_artifact_path=str(_AUTORAG_SHARED / "component_status.py"),
    install_kfp_package=False,
)
def text_extraction(
    documents_descriptor: dsl.Input[dsl.Artifact],
    extracted_text: dsl.Output[dsl.Artifact],
    component_status: dsl.Output[dsl.Artifact] = None,
    embedded_artifact: dsl.EmbeddedInput[dsl.Dataset] = None,
    error_tolerance: Optional[float] = None,
    max_extraction_workers: Optional[int] = None,
    preset: str = "speed",
):
    """Text Extraction component.

    Thin wrapper that delegates to ``ai4rag.components.data.text_extraction.extract_text``.

    Args:
        documents_descriptor: Input artifact containing
            documents_descriptor.json with bucket, prefix, and documents list.
        extracted_text: Output artifact directory where DoclingDocument JSON files
            will be written.
        component_status: Output artifact containing stage-level progress tracking.
        embedded_artifact: Embedded ``autorag.shared`` helpers injected by KFP at runtime.
        error_tolerance: Fraction of documents (0.0-1.0) allowed to fail without
            raising an error. None (the default) means zero tolerance.
        max_extraction_workers: Number of parallel worker processes used for text
            extraction. Defaults to 4. Set to None to use all available CPU cores.
        preset: Pipeline quality tier. "speed" (default) disables Docling table
            structure parsing. "balanced" enables TableFormer table reconstruction.
    """
    import importlib.util
    import json
    import logging
    import os
    from pathlib import Path

    from ai4rag.components.data.text_extraction import extract_text

    logging.basicConfig(level=logging.INFO)

    VALID_PRESETS = {"speed", "balanced"}
    PRESET_DO_TABLE_STRUCTURE = {"speed": False, "balanced": True}

    if preset not in VALID_PRESETS:
        raise ValueError(f"preset must be one of {VALID_PRESETS}; got {preset!r}.")

    do_table_structure = PRESET_DO_TABLE_STRUCTURE[preset]
    logging.info("Preset %r: do_table_structure=%s", preset, do_table_structure)

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
        status = _status_module.bootstrap_status_tracker(embedded_artifact, component_status, "text_extraction")
    with status:
        if component_status is not None:
            status.set_metadata(display_name="Text Extraction Status")
            component_status.metadata["display_name"] = "Text Extraction Status"
        with status.stage("extract_documents"):
            descriptor_path = Path(documents_descriptor.path) / "documents_descriptor.json"
            with open(descriptor_path, "r", encoding="utf-8") as f:
                descriptor = json.load(f)

            output_dir = Path(extracted_text.path)
            output_dir.mkdir(parents=True, exist_ok=True)

            extract_text(
                documents=descriptor["documents"],
                bucket=descriptor["bucket"],
                output_dir=output_dir,
                s3_endpoint=os.environ.get("AWS_S3_ENDPOINT"),
                s3_access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
                s3_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                s3_region=os.environ.get("AWS_DEFAULT_REGION"),
                error_tolerance=error_tolerance,
                max_extraction_workers=max_extraction_workers,
                docling_artifacts_path=os.environ.get("DOCLING_ARTIFACTS_PATH"),
                do_table_structure=do_table_structure,
            )


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        text_extraction,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
