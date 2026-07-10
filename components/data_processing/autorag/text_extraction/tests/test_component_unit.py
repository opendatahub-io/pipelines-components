"""Tests for the text_extraction thin wrapper component."""

import inspect
import json
from unittest import mock

import pytest

from ..component import text_extraction

MOCKED_ENV_VARIABLES = {
    "AWS_ACCESS_KEY_ID": "test_key",
    "AWS_SECRET_ACCESS_KEY": "test_secret",
    "AWS_S3_ENDPOINT": "https://s3.example.com",
    "AWS_DEFAULT_REGION": "us-east-1",
}


def _make_ai4rag_mocks():
    """Build mock modules for ai4rag.components.data.text_extraction."""
    mock_extract_text = mock.MagicMock(name="extract_text")

    mock_text_extraction_module = mock.MagicMock()
    mock_text_extraction_module.extract_text = mock_extract_text

    modules = {
        "ai4rag": mock.MagicMock(),
        "ai4rag.components": mock.MagicMock(),
        "ai4rag.components.data": mock.MagicMock(),
        "ai4rag.components.data.text_extraction": mock_text_extraction_module,
    }
    return modules, mock_extract_text


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
        modules, mock_extract = _make_ai4rag_mocks()

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
            do_table_structure=False,
        )

    @mock.patch.dict(
        "os.environ",
        {**MOCKED_ENV_VARIABLES, "DOCLING_ARTIFACTS_PATH": "/opt/docling/models"},
        clear=True,
    )
    def test_passes_docling_artifacts_path(self, tmp_path):
        """DOCLING_ARTIFACTS_PATH env var is forwarded to extract_text."""
        modules, mock_extract = _make_ai4rag_mocks()

        descriptor_dir = tmp_path / "descriptor"
        descriptor_dir.mkdir()
        descriptor = {"bucket": "b", "documents": [{"key": "a.pdf", "size_bytes": 100}]}
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

        assert mock_extract.call_args.kwargs["docling_artifacts_path"] == "/opt/docling/models"

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_default_params_passed_as_none(self, tmp_path):
        """Default error_tolerance and max_extraction_workers are None."""
        modules, mock_extract = _make_ai4rag_mocks()

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
        modules, mock_extract = _make_ai4rag_mocks()
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
        modules, mock_extract = _make_ai4rag_mocks()

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
        modules, _ = _make_ai4rag_mocks()

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
        modules, mock_extract = _make_ai4rag_mocks()

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
        """Preset controls do_table_structure passed to extract_text."""
        modules, mock_extract = _make_ai4rag_mocks()

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

        assert mock_extract.call_args.kwargs["do_table_structure"] == expected_do_table_structure
