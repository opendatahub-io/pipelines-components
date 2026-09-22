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
    ocr_lang: Optional[str] = None,
):
    """Text Extraction component.

    Thin wrapper that delegates to ``ai4rag.utils.data.text_extraction.extract_text``.

    OCR is always enabled. Docling runs RapidOCR only on pages it flags as needing it,
    so pages carrying a text layer are read directly and scanned or image-only pages are
    OCR'd.

    The four RapidOCR model paths are pinned explicitly from ``$DOCLING_ARTIFACTS_PATH``
    rather than left to Docling. Docling resolves an unpinned language to PP-OCRv6 and
    looks for flat filenames directly under ``RapidOcr/``, but the AutoRAG image ships the
    PP-OCRv4 bundle in its nested ``RapidOcr/onnx/PP-OCRv4/...`` layout, so leaving the
    paths unset fails with ``FileNotFoundError`` at conversion time. Pinning them makes
    Docling skip resolution and use the models that are actually present.

    Args:
        documents_descriptor: Input artifact containing
            documents_descriptor.json with bucket, prefix, and documents list.
            Each document entry's ``key`` also names the extracted document,
            so the prefix is not passed on separately.
        extracted_text: Output artifact directory where DoclingDocument JSON files
            will be written.
        component_status: Output artifact containing stage-level progress tracking,
            extraction outcomes, and configured-engine candidate counts.
        embedded_artifact: Embedded ``autorag.shared`` helpers injected by KFP at runtime.
        error_tolerance: Fraction of documents (0.0-1.0) allowed to fail without
            raising an error. None (the default) means zero tolerance.
        max_extraction_workers: Number of parallel worker processes used for text
            extraction. Defaults to 4. Set to None to use all available CPU cores.
        preset: Pipeline quality tier. "speed" (default) disables Docling table
            structure parsing. "balanced" enables TableFormer table reconstruction.
        ocr_lang: Language of the document text, used only to pick the RapidOCR model
            bundle. Accepts a language name or ISO 639-1 code. Chinese ("chinese", "zh",
            "ch") selects the Chinese bundle; everything else, including None (the
            default), selects the English bundle, which covers all Latin-script
            languages. In the optimization pipeline this is filled from the language
            AutoRAG detects; for the indexing pipeline pass ``pattern.json``
            ``settings.generation.language.code``.
    """
    import importlib.util
    import json
    import logging
    import os
    from pathlib import Path

    from ai4rag.utils.data.text_extraction import DoclingExtractionConfig, extract_text

    logging.basicConfig(level=logging.INFO)

    VALID_PRESETS = {"speed", "balanced"}
    PRESET_DO_TABLE_STRUCTURE = {"speed": False, "balanced": True}
    LAYOUT_OCR_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    ASR_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"}

    if preset not in VALID_PRESETS:
        raise ValueError(f"preset must be one of {VALID_PRESETS}; got {preset!r}.")

    do_table_structure = PRESET_DO_TABLE_STRUCTURE[preset]
    logging.info("Preset %r: do_table_structure=%s", preset, do_table_structure)

    # Paths are relative to $DOCLING_ARTIFACTS_PATH/RapidOcr/ and mirror the on-disk
    # layout of the RHAI OGX modelcar baked into the AutoRAG image. The classifier is
    # script-agnostic, so both bundles share it.
    RAPIDOCR_BUNDLES = {
        "english": {
            "ocr_det_model_path": "onnx/PP-OCRv4/det/en_PP-OCRv3_det_mobile.onnx",
            "ocr_cls_model_path": "onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
            "ocr_rec_model_path": "onnx/PP-OCRv4/rec/en_PP-OCRv4_rec_mobile.onnx",
            "ocr_rec_keys_path": "paddle/PP-OCRv4/rec/en_PP-OCRv4_rec_mobile/en_dict.txt",
        },
        "chinese": {
            "ocr_det_model_path": "onnx/PP-OCRv4/det/ch_PP-OCRv4_det_mobile.onnx",
            "ocr_cls_model_path": "onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
            "ocr_rec_model_path": "onnx/PP-OCRv4/rec/ch_PP-OCRv4_rec_mobile.onnx",
            "ocr_rec_keys_path": "paddle/PP-OCRv4/rec/ch_PP-OCRv4_rec_mobile/ppocr_keys_v1.txt",
        },
    }
    CHINESE_ALIASES = {"chinese", "ch", "zh", "zho", "chi", "zh-cn", "zh-tw"}

    # Only Chinese has a dedicated bundle; every other language, detected or not, is
    # served by the English models, which cover all Latin scripts. Anything unrecognised
    # therefore degrades to English rather than failing the run.
    bundle_name = "chinese" if (ocr_lang or "").strip().lower() in CHINESE_ALIASES else "english"
    logging.info("OCR language %r resolved to the %s RapidOCR bundle", ocr_lang, bundle_name)

    ocr_model_paths = {}
    docling_artifacts_path = os.environ.get("DOCLING_ARTIFACTS_PATH")
    if docling_artifacts_path:
        ocr_root = Path(docling_artifacts_path) / "RapidOcr"
        resolved = {key: ocr_root / rel for key, rel in RAPIDOCR_BUNDLES[bundle_name].items()}
        missing = sorted(str(path) for path in resolved.values() if not path.is_file())
        if missing:
            raise FileNotFoundError(
                f"RapidOCR {bundle_name} models are missing under {ocr_root}:\n  - " + "\n  - ".join(missing)
            )
        ocr_model_paths = {key: str(path) for key, path in resolved.items()}
    else:
        # No baked-in artifacts (e.g. a dev image): let ai4rag and Docling resolve and
        # download the models themselves.
        logging.warning("DOCLING_ARTIFACTS_PATH is unset; RapidOCR models will be resolved at runtime.")

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
            documents = descriptor["documents"]
            suffixes = [Path(document["key"]).suffix.lower() for document in documents]
            candidate_metrics = {
                "documents_total": len(documents),
                "layout_candidate_documents": sum(suffix in LAYOUT_OCR_EXTENSIONS for suffix in suffixes),
                "layout_model": "Docling Layout Heron",
                "ocr_candidate_documents": sum(suffix in LAYOUT_OCR_EXTENSIONS for suffix in suffixes),
                "ocr_engine": "RapidOCR",
                "ocr_language": bundle_name,
                "asr_candidate_documents": sum(suffix in ASR_EXTENSIONS for suffix in suffixes),
                "asr_model": "Whisper Tiny",
            }
            status.record("extract_documents", "running", metrics=candidate_metrics)

            output_dir = Path(extracted_text.path)
            output_dir.mkdir(parents=True, exist_ok=True)

            docling_config = DoclingExtractionConfig(
                do_table_structure=do_table_structure,
                do_ocr=True,
                ocr_lang=bundle_name,
                **ocr_model_paths,
            )

            extraction_result = extract_text(
                documents=documents,
                bucket=descriptor["bucket"],
                output_dir=output_dir,
                s3_endpoint=os.environ.get("AWS_S3_ENDPOINT"),
                s3_access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
                s3_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                s3_region=os.environ.get("AWS_DEFAULT_REGION"),
                error_tolerance=error_tolerance,
                max_extraction_workers=max_extraction_workers,
                docling_artifacts_path=os.environ.get("DOCLING_ARTIFACTS_PATH"),
                docling_config=docling_config,
            )
            status.record(
                "extract_documents",
                "completed",
                metrics={
                    "documents_total": extraction_result.total_documents,
                    "documents_processed": extraction_result.processed_count,
                    "documents_failed": extraction_result.error_count,
                },
            )


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        text_extraction,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
