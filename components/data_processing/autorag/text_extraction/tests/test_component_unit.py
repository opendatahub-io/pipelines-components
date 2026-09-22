"""Tests for the text_extraction thin wrapper component."""

import inspect
import json
from types import MappingProxyType, SimpleNamespace
from unittest import mock

import pytest

from ..component import _AUTORAG_SHARED, text_extraction

MOCKED_ENV_VARIABLES = {
    "AWS_ACCESS_KEY_ID": "test_key",
    "AWS_SECRET_ACCESS_KEY": "test_secret",
    "AWS_S3_ENDPOINT": "https://s3.example.com",
    "AWS_DEFAULT_REGION": "us-east-1",
}


def _make_ai4rag_mocks():
    """Build mock modules matching the component's ``ai4rag.utils.data.text_extraction`` import.

    Both ``extract_text`` and ``DoclingExtractionConfig`` are mocked; the config
    class is mocked (rather than using the real frozen dataclass) since ``ai4rag``
    is not an installed dependency in the unit test environment.
    """
    mock_extract_text = mock.MagicMock(name="extract_text")
    mock_docling_config_cls = mock.MagicMock(name="DoclingExtractionConfig")

    mock_text_extraction_module = mock.MagicMock()
    mock_text_extraction_module.extract_text = mock_extract_text
    mock_text_extraction_module.DoclingExtractionConfig = mock_docling_config_cls

    modules = {
        "ai4rag": mock.MagicMock(),
        "ai4rag.utils": mock.MagicMock(),
        "ai4rag.utils.data": mock.MagicMock(),
        "ai4rag.utils.data.text_extraction": mock_text_extraction_module,
    }
    return modules, mock_extract_text, mock_docling_config_cls


ENGLISH_BUNDLE = MappingProxyType(
    {
        "ocr_det_model_path": "onnx/PP-OCRv4/det/en_PP-OCRv3_det_mobile.onnx",
        "ocr_cls_model_path": "onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        "ocr_rec_model_path": "onnx/PP-OCRv4/rec/en_PP-OCRv4_rec_mobile.onnx",
        "ocr_rec_keys_path": "paddle/PP-OCRv4/rec/en_PP-OCRv4_rec_mobile/en_dict.txt",
    }
)
CHINESE_BUNDLE = MappingProxyType(
    {
        "ocr_det_model_path": "onnx/PP-OCRv4/det/ch_PP-OCRv4_det_mobile.onnx",
        "ocr_cls_model_path": "onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        "ocr_rec_model_path": "onnx/PP-OCRv4/rec/ch_PP-OCRv4_rec_mobile.onnx",
        "ocr_rec_keys_path": "paddle/PP-OCRv4/rec/ch_PP-OCRv4_rec_mobile/ppocr_keys_v1.txt",
    }
)


def _make_docling_artifacts(root, bundles=(ENGLISH_BUNDLE, CHINESE_BUNDLE)):
    """Lay out an empty stand-in for the RapidOCR tree the AutoRAG image ships.

    The component only checks that each file exists, so empty files suffice.
    """
    for bundle in bundles:
        for rel in bundle.values():
            path = root / "RapidOcr" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    return root


def _write_descriptor(tmp_path, descriptor=None):
    """Write a minimal documents_descriptor.json and return its directory artifact."""
    descriptor_dir = tmp_path / "descriptor"
    descriptor_dir.mkdir(exist_ok=True)
    (descriptor_dir / "documents_descriptor.json").write_text(
        json.dumps(descriptor or {"bucket": "b", "documents": []}), encoding="utf-8"
    )
    artifact = mock.MagicMock()
    artifact.path = str(descriptor_dir)
    return artifact


class TestTextExtractionUnitTests:
    """Unit tests for the text_extraction thin wrapper."""

    def test_component_function_exists(self):
        """Component factory exists and exposes python_func."""
        assert callable(text_extraction)
        assert hasattr(text_extraction, "python_func")

    def test_component_has_expected_interface(self):
        """Component has expected parameters."""
        sig = inspect.signature(text_extraction.python_func)
        params = list(sig.parameters)
        assert "documents_descriptor" in params
        assert "extracted_text" in params
        assert "error_tolerance" in params
        assert "max_extraction_workers" in params
        assert "preset" in params
        assert sig.parameters["error_tolerance"].default is None
        assert sig.parameters["max_extraction_workers"].default is None
        assert sig.parameters["preset"].default == "speed"

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_delegates_to_ai4rag_extract_text(self, tmp_path):
        """Wrapper reads descriptor and calls extract_text with correct args."""
        modules, mock_extract, mock_docling_config_cls = _make_ai4rag_mocks()

        descriptor_dir = tmp_path / "descriptor"
        descriptor_dir.mkdir()
        descriptor = {
            "bucket": "my-bucket",
            "prefix": "docs/",
            "documents": [{"key": "docs/a.pdf", "size_bytes": 1000}],
        }
        (descriptor_dir / "documents_descriptor.json").write_text(json.dumps(descriptor), encoding="utf-8")

        descriptor_artifact = mock.MagicMock()
        descriptor_artifact.path = str(descriptor_dir)

        output_dir = tmp_path / "output"
        output_artifact = mock.MagicMock()
        output_artifact.path = str(output_dir)

        with mock.patch.dict("sys.modules", modules):
            text_extraction.python_func(
                documents_descriptor=descriptor_artifact,
                extracted_text=output_artifact,
                error_tolerance=0.1,
                max_extraction_workers=4,
            )

        assert output_dir.exists()
        mock_docling_config_cls.assert_called_once_with(
            do_table_structure=False,
            do_ocr=True,
            ocr_lang="english",
        )
        mock_extract.assert_called_once_with(
            documents=[{"key": "docs/a.pdf", "size_bytes": 1000}],
            bucket="my-bucket",
            output_dir=output_dir,
            s3_endpoint="https://s3.example.com",
            s3_access_key="test_key",
            s3_secret_key="test_secret",
            s3_region="us-east-1",
            error_tolerance=0.1,
            max_extraction_workers=4,
            docling_artifacts_path=None,
            docling_config=mock_docling_config_cls.return_value,
        )

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_forwards_document_keys_verbatim(self, tmp_path):
        """Documents are forwarded untouched; the prefix is not passed separately.

        A document's ``key`` is what names it downstream, so the component must
        not rewrite or strip it on the way through.
        """
        modules, mock_extract, _ = _make_ai4rag_mocks()

        descriptor_dir = tmp_path / "descriptor"
        descriptor_dir.mkdir()
        documents = [
            {"key": "docs/a/setup.txt", "size_bytes": 10},
            {"key": "docs/b/setup.txt", "size_bytes": 20},
        ]
        descriptor = {"bucket": "b", "prefix": "docs/", "documents": documents}
        (descriptor_dir / "documents_descriptor.json").write_text(json.dumps(descriptor), encoding="utf-8")

        descriptor_artifact = mock.MagicMock()
        descriptor_artifact.path = str(descriptor_dir)
        output_artifact = mock.MagicMock()
        output_artifact.path = str(tmp_path / "output")

        with mock.patch.dict("sys.modules", modules):
            text_extraction.python_func(
                documents_descriptor=descriptor_artifact,
                extracted_text=output_artifact,
            )

        call_kwargs = mock_extract.call_args.kwargs
        assert call_kwargs["documents"] == documents
        assert "input_data_key" not in call_kwargs

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_records_engine_candidates_and_extraction_outcomes(self, tmp_path):
        """Status reports the configured engines, candidate inputs, and aggregate outcomes."""
        modules, mock_extract, _ = _make_ai4rag_mocks()
        mock_extract.return_value = SimpleNamespace(total_documents=5, processed_count=4, error_count=1)
        descriptor_artifact = _write_descriptor(
            tmp_path,
            {
                "bucket": "b",
                "documents": [
                    {"key": "report.pdf"},
                    {"key": "scan.PNG"},
                    {"key": "recording.mp3"},
                    {"key": "meeting.wav"},
                    {"key": "notes.txt"},
                ],
            },
        )
        output_artifact = SimpleNamespace(path=str(tmp_path / "output"))
        component_status = SimpleNamespace(path=str(tmp_path / "status"), metadata={})
        embedded_artifact = SimpleNamespace(path=str(_AUTORAG_SHARED))

        with mock.patch.dict("sys.modules", modules):
            text_extraction.python_func(
                documents_descriptor=descriptor_artifact,
                extracted_text=output_artifact,
                component_status=component_status,
                embedded_artifact=embedded_artifact,
            )

        status = json.loads((tmp_path / "status" / "component_status.json").read_text(encoding="utf-8"))
        metrics = status["stages"][0]["metrics"]
        assert metrics == {
            "layout_candidate_documents": 2,
            "layout_model": "Docling Layout Heron",
            "ocr_candidate_documents": 2,
            "ocr_engine": "RapidOCR",
            "ocr_language": "english",
            "asr_candidate_documents": 2,
            "asr_model": "Whisper Tiny",
            "documents_total": 5,
            "documents_processed": 4,
            "documents_failed": 1,
        }

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_failed_extraction_preserves_candidate_metrics(self, tmp_path):
        """Failure status retains inputs known before ai4rag begins extraction."""
        modules, mock_extract, _ = _make_ai4rag_mocks()
        mock_extract.side_effect = RuntimeError("Text extraction failed")
        descriptor_artifact = _write_descriptor(
            tmp_path,
            {
                "bucket": "b",
                "documents": [{"key": "report.pdf"}, {"key": "recording.mp3"}],
            },
        )
        output_artifact = SimpleNamespace(path=str(tmp_path / "output"))
        component_status = SimpleNamespace(path=str(tmp_path / "status"), metadata={})
        embedded_artifact = SimpleNamespace(path=str(_AUTORAG_SHARED))

        with mock.patch.dict("sys.modules", modules):
            with pytest.raises(RuntimeError, match="Text extraction failed"):
                text_extraction.python_func(
                    documents_descriptor=descriptor_artifact,
                    extracted_text=output_artifact,
                    component_status=component_status,
                    embedded_artifact=embedded_artifact,
                )

        status = json.loads((tmp_path / "status" / "component_status.json").read_text(encoding="utf-8"))
        stage = status["stages"][0]
        assert stage["status"]["state"] == "failed"
        assert stage["metrics"]["documents_total"] == 2
        assert stage["metrics"]["layout_candidate_documents"] == 1
        assert stage["metrics"]["asr_candidate_documents"] == 1
        assert "documents_processed" not in stage["metrics"]
        assert "documents_failed" not in stage["metrics"]

    def test_passes_docling_artifacts_path(self, tmp_path):
        """DOCLING_ARTIFACTS_PATH env var is forwarded to extract_text."""
        modules, mock_extract, _ = _make_ai4rag_mocks()

        artifacts = _make_docling_artifacts(tmp_path / "artifacts")
        descriptor_artifact = _write_descriptor(tmp_path, {"bucket": "b", "documents": []})
        output_artifact = mock.MagicMock()
        output_artifact.path = str(tmp_path / "output")

        env = {**MOCKED_ENV_VARIABLES, "DOCLING_ARTIFACTS_PATH": str(artifacts)}
        with mock.patch.dict("os.environ", env, clear=True), mock.patch.dict("sys.modules", modules):
            text_extraction.python_func(
                documents_descriptor=descriptor_artifact,
                extracted_text=output_artifact,
            )

        assert mock_extract.call_args.kwargs["docling_artifacts_path"] == str(artifacts)

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_default_params_passed_as_none(self, tmp_path):
        """Default error_tolerance and max_extraction_workers are None."""
        modules, mock_extract, _ = _make_ai4rag_mocks()

        descriptor_dir = tmp_path / "descriptor"
        descriptor_dir.mkdir()
        descriptor = {"bucket": "b", "documents": []}
        (descriptor_dir / "documents_descriptor.json").write_text(json.dumps(descriptor), encoding="utf-8")

        descriptor_artifact = mock.MagicMock()
        descriptor_artifact.path = str(descriptor_dir)
        output_artifact = mock.MagicMock()
        output_artifact.path = str(tmp_path / "output")

        with mock.patch.dict("sys.modules", modules):
            text_extraction.python_func(
                documents_descriptor=descriptor_artifact,
                extracted_text=output_artifact,
            )

        call_kwargs = mock_extract.call_args.kwargs
        assert call_kwargs["error_tolerance"] is None
        assert call_kwargs["max_extraction_workers"] is None

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_propagates_ai4rag_exception(self, tmp_path):
        """Exceptions from ai4rag are propagated to the caller."""
        modules, mock_extract, _ = _make_ai4rag_mocks()
        mock_extract.side_effect = RuntimeError("Text extraction failed: 5/10 document(s) failed")

        descriptor_dir = tmp_path / "descriptor"
        descriptor_dir.mkdir()
        descriptor = {"bucket": "b", "documents": [{"key": "a.pdf", "size_bytes": 100}]}
        (descriptor_dir / "documents_descriptor.json").write_text(json.dumps(descriptor), encoding="utf-8")

        descriptor_artifact = mock.MagicMock()
        descriptor_artifact.path = str(descriptor_dir)
        output_artifact = mock.MagicMock()
        output_artifact.path = str(tmp_path / "output")

        with mock.patch.dict("sys.modules", modules):
            with pytest.raises(RuntimeError, match="Text extraction failed"):
                text_extraction.python_func(
                    documents_descriptor=descriptor_artifact,
                    extracted_text=output_artifact,
                )

    def test_missing_descriptor_file_raises(self, tmp_path):
        """Missing documents_descriptor.json raises FileNotFoundError."""
        modules, _, _ = _make_ai4rag_mocks()

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        descriptor_artifact = mock.MagicMock()
        descriptor_artifact.path = str(empty_dir)
        output_artifact = mock.MagicMock()
        output_artifact.path = str(tmp_path / "output")

        with mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True):
            with mock.patch.dict("sys.modules", modules):
                with pytest.raises(FileNotFoundError):
                    text_extraction.python_func(
                        documents_descriptor=descriptor_artifact,
                        extracted_text=output_artifact,
                    )

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_preset_validation_rejects_invalid(self, tmp_path):
        """Invalid preset raises ValueError."""
        modules, _, _ = _make_ai4rag_mocks()

        descriptor_dir = tmp_path / "descriptor"
        descriptor_dir.mkdir()
        descriptor = {"bucket": "b", "documents": []}
        (descriptor_dir / "documents_descriptor.json").write_text(json.dumps(descriptor), encoding="utf-8")

        descriptor_artifact = mock.MagicMock()
        descriptor_artifact.path = str(descriptor_dir)
        output_artifact = mock.MagicMock()
        output_artifact.path = str(tmp_path / "output")

        with mock.patch.dict("sys.modules", modules):
            with pytest.raises(ValueError, match="preset must be one of"):
                text_extraction.python_func(
                    documents_descriptor=descriptor_artifact,
                    extracted_text=output_artifact,
                    preset="invalid",
                )

    @pytest.mark.parametrize("preset_value", ["speed", "balanced"])
    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_valid_presets_accepted(self, tmp_path, preset_value):
        """Both 'speed' and 'balanced' presets are accepted without error."""
        modules, mock_extract, _ = _make_ai4rag_mocks()

        descriptor_dir = tmp_path / "descriptor"
        descriptor_dir.mkdir()
        descriptor = {"bucket": "b", "documents": []}
        (descriptor_dir / "documents_descriptor.json").write_text(json.dumps(descriptor), encoding="utf-8")

        descriptor_artifact = mock.MagicMock()
        descriptor_artifact.path = str(descriptor_dir)
        output_artifact = mock.MagicMock()
        output_artifact.path = str(tmp_path / "output")

        with mock.patch.dict("sys.modules", modules):
            text_extraction.python_func(
                documents_descriptor=descriptor_artifact,
                extracted_text=output_artifact,
                preset=preset_value,
            )

        mock_extract.assert_called_once()

    @pytest.mark.parametrize(
        ("preset_value", "expected_do_table_structure"),
        [("speed", False), ("balanced", True)],
    )
    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_preset_sets_do_table_structure(self, tmp_path, preset_value, expected_do_table_structure):
        """Preset controls do_table_structure passed to DoclingExtractionConfig."""
        modules, mock_extract, mock_docling_config_cls = _make_ai4rag_mocks()

        descriptor_dir = tmp_path / "descriptor"
        descriptor_dir.mkdir()
        descriptor = {"bucket": "b", "documents": []}
        (descriptor_dir / "documents_descriptor.json").write_text(json.dumps(descriptor), encoding="utf-8")

        descriptor_artifact = mock.MagicMock()
        descriptor_artifact.path = str(descriptor_dir)
        output_artifact = mock.MagicMock()
        output_artifact.path = str(tmp_path / "output")

        with mock.patch.dict("sys.modules", modules):
            text_extraction.python_func(
                documents_descriptor=descriptor_artifact,
                extracted_text=output_artifact,
                preset=preset_value,
            )

        mock_docling_config_cls.assert_called_once_with(
            do_table_structure=expected_do_table_structure,
            do_ocr=True,
            ocr_lang="english",
        )
        assert mock_extract.call_args.kwargs["docling_config"] == mock_docling_config_cls.return_value

    def _run_with_artifacts(self, tmp_path, modules, ocr_lang=None, artifacts=True):
        """Invoke the component against a stand-in artifacts tree and return its env path."""
        root = _make_docling_artifacts(tmp_path / "artifacts") if artifacts else None
        descriptor_artifact = _write_descriptor(tmp_path)
        output_artifact = mock.MagicMock()
        output_artifact.path = str(tmp_path / "output")

        env = dict(MOCKED_ENV_VARIABLES)
        if root is not None:
            env["DOCLING_ARTIFACTS_PATH"] = str(root)

        with mock.patch.dict("os.environ", env, clear=True), mock.patch.dict("sys.modules", modules):
            text_extraction.python_func(
                documents_descriptor=descriptor_artifact,
                extracted_text=output_artifact,
                ocr_lang=ocr_lang,
            )
        return root

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_ocr_is_always_enabled(self, tmp_path):
        """do_ocr is always True and defaults to the English bundle."""
        modules, _, mock_docling_config_cls = _make_ai4rag_mocks()

        output_artifact = mock.MagicMock()
        output_artifact.path = str(tmp_path / "output")

        with mock.patch.dict("sys.modules", modules):
            text_extraction.python_func(
                documents_descriptor=_write_descriptor(tmp_path),
                extracted_text=output_artifact,
            )

        kwargs = mock_docling_config_cls.call_args.kwargs
        assert kwargs["do_ocr"] is True
        assert kwargs["ocr_lang"] == "english"

    @pytest.mark.parametrize("ocr_lang", [None, "", "english", "en", "french", "pl", "unknown"])
    def test_non_chinese_languages_use_the_english_bundle(self, tmp_path, ocr_lang):
        """Latin scripts and unrecognised values degrade to English rather than failing."""
        modules, _, mock_docling_config_cls = _make_ai4rag_mocks()
        root = self._run_with_artifacts(tmp_path, modules, ocr_lang=ocr_lang)

        kwargs = mock_docling_config_cls.call_args.kwargs
        assert kwargs["ocr_lang"] == "english"
        for key, rel in ENGLISH_BUNDLE.items():
            assert kwargs[key] == str(root / "RapidOcr" / rel)

    @pytest.mark.parametrize("ocr_lang", ["chinese", "zh", "ch", "ZH", " Chinese ", "zh-cn"])
    def test_chinese_aliases_select_the_chinese_bundle(self, tmp_path, ocr_lang):
        """Chinese is the only language with a dedicated bundle; aliases all reach it."""
        modules, _, mock_docling_config_cls = _make_ai4rag_mocks()
        root = self._run_with_artifacts(tmp_path, modules, ocr_lang=ocr_lang)

        kwargs = mock_docling_config_cls.call_args.kwargs
        assert kwargs["ocr_lang"] == "chinese"
        for key, rel in CHINESE_BUNDLE.items():
            assert kwargs[key] == str(root / "RapidOcr" / rel)

    def test_model_paths_are_pinned_not_left_to_docling(self, tmp_path):
        """All four RapidOCR paths are pinned so Docling skips its own resolution.

        Unpinned, Docling resolves PP-OCRv6 and expects flat filenames under
        ``RapidOcr/``, which the AutoRAG image's nested PP-OCRv4 layout does not
        provide, and conversion dies with FileNotFoundError.
        """
        modules, _, mock_docling_config_cls = _make_ai4rag_mocks()
        self._run_with_artifacts(tmp_path, modules)

        kwargs = mock_docling_config_cls.call_args.kwargs
        assert set(ENGLISH_BUNDLE) <= set(kwargs)
        assert all(kwargs[key] for key in ENGLISH_BUNDLE)

    def test_missing_ocr_models_raise(self, tmp_path):
        """An artifacts path without the RapidOCR bundle fails fast and names the files."""
        modules, _, _ = _make_ai4rag_mocks()

        empty_artifacts = tmp_path / "artifacts"
        empty_artifacts.mkdir()
        output_artifact = mock.MagicMock()
        output_artifact.path = str(tmp_path / "output")

        env = {**MOCKED_ENV_VARIABLES, "DOCLING_ARTIFACTS_PATH": str(empty_artifacts)}
        with mock.patch.dict("os.environ", env, clear=True), mock.patch.dict("sys.modules", modules):
            with pytest.raises(FileNotFoundError, match="RapidOCR english models are missing"):
                text_extraction.python_func(
                    documents_descriptor=_write_descriptor(tmp_path),
                    extracted_text=output_artifact,
                )

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_no_artifacts_path_leaves_models_unpinned(self, tmp_path):
        """Without DOCLING_ARTIFACTS_PATH the paths are omitted so ai4rag can resolve them."""
        modules, _, mock_docling_config_cls = _make_ai4rag_mocks()

        output_artifact = mock.MagicMock()
        output_artifact.path = str(tmp_path / "output")

        with mock.patch.dict("sys.modules", modules):
            text_extraction.python_func(
                documents_descriptor=_write_descriptor(tmp_path),
                extracted_text=output_artifact,
            )

        kwargs = mock_docling_config_cls.call_args.kwargs
        assert not set(ENGLISH_BUNDLE) & set(kwargs)

    @pytest.mark.parametrize("preset_value", ["speed", "balanced"])
    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_ocr_is_independent_of_preset(self, tmp_path, preset_value):
        """OCR stays on for every preset; the preset only drives table structure."""
        modules, _, mock_docling_config_cls = _make_ai4rag_mocks()

        descriptor_dir = tmp_path / "descriptor"
        descriptor_dir.mkdir()
        (descriptor_dir / "documents_descriptor.json").write_text(
            json.dumps({"bucket": "b", "documents": []}), encoding="utf-8"
        )

        descriptor_artifact = mock.MagicMock()
        descriptor_artifact.path = str(descriptor_dir)
        output_artifact = mock.MagicMock()
        output_artifact.path = str(tmp_path / "output")

        with mock.patch.dict("sys.modules", modules):
            text_extraction.python_func(
                documents_descriptor=descriptor_artifact,
                extracted_text=output_artifact,
                preset=preset_value,
            )

        kwargs = mock_docling_config_cls.call_args.kwargs
        assert kwargs["do_ocr"] is True
        assert kwargs["do_table_structure"] is (preset_value == "balanced")
