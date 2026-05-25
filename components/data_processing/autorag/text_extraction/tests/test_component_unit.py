"""Tests for the text_extraction component."""

import inspect
import json
import shutil
import sys
from pathlib import Path
from unittest import mock

import pytest

from ..component import text_extraction

MOCKED_ENV_VARIABLES = {
    "AWS_ACCESS_KEY_ID": "test_key",
    "AWS_SECRET_ACCESS_KEY": "test_secret",
    "AWS_S3_ENDPOINT": "https://s3.example.com",
    "AWS_DEFAULT_REGION": "us-east-1",
}


class _MockSSLError(Exception):
    """Stand-in for botocore.exceptions.SSLError used in unit tests."""


def _mock_boto_modules():
    """Return mocked boto3 and botocore modules used in component imports."""
    mock_boto3 = mock.MagicMock()
    mock_botocore = mock.MagicMock()
    mock_botocore_exceptions = mock.MagicMock()
    mock_botocore_exceptions.SSLError = _MockSSLError
    mock_botocore.exceptions = mock_botocore_exceptions
    return {
        "boto3": mock_boto3,
        "botocore": mock_botocore,
        "botocore.exceptions": mock_botocore_exceptions,
    }


def _docling_modules():
    """Return mock modules for docling imports used in component.

    The mocked DocumentConverter.convert() returns a result whose
    export_to_markdown() produces a plain string so that output_file.write_text()
    succeeds inside worker_process_document.
    """
    mock_converter_instance = mock.MagicMock()
    mock_converter_instance.convert.return_value = mock.MagicMock(
        document=mock.MagicMock(export_to_markdown=mock.MagicMock(return_value="# text"))
    )
    mock_converter_class = mock.MagicMock(return_value=mock_converter_instance)
    return {
        "docling": mock.MagicMock(),
        "docling.datamodel": mock.MagicMock(),
        "docling.datamodel.accelerator_options": mock.MagicMock(AcceleratorOptions=mock.MagicMock()),
        "docling.datamodel.base_models": mock.MagicMock(InputFormat=mock.MagicMock(PDF="PDF")),
        "docling.datamodel.pipeline_options": mock.MagicMock(
            PdfPipelineOptions=mock.MagicMock(),
            PaginatedPipelineOptions=mock.MagicMock(),
        ),
        "docling.document_converter": mock.MagicMock(
            DocumentConverter=mock_converter_class,
            PdfFormatOption=mock.MagicMock(),
            WordFormatOption=mock.MagicMock(),
            PowerpointFormatOption=mock.MagicMock(),
            HTMLFormatOption=mock.MagicMock(),
            MarkdownFormatOption=mock.MagicMock(),
        ),
    }


def _sync_multiprocess_modules():
    """Return a mocked multiprocess module that executes tasks synchronously.

    The pool runs worker_initializer and all apply_async tasks in the calling
    process so that sys.modules patches made by tests are visible inside workers.
    This avoids spawning real child processes that would bypass mock.patch.dict.
    """

    class _SyncAsyncResult:
        def __init__(self, func, args):
            try:
                self._value = func(*args)
                self._exc = None
            except Exception as exc:
                self._value = None
                self._exc = exc

        def ready(self):
            return True

        def get(self):
            if self._exc is not None:
                raise self._exc
            return self._value

    class _SyncPool:
        def __init__(self, *args, processes=None, initializer=None, initargs=(), **kwargs):
            if initializer is not None:
                initializer(*initargs)

        def apply_async(self, func, args=()):
            return _SyncAsyncResult(func, args)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class _SyncContext:
        def Pool(self, *args, **kwargs):
            return _SyncPool(*args, **kwargs)

    mock_mp = mock.MagicMock()
    mock_mp.get_context.return_value = _SyncContext()
    return {"multiprocess": mock_mp}


class TestTextExtractionUnitTests:
    """Unit tests for text_extraction success and failure paths."""

    def test_component_function_exists(self):
        """Component factory exists and exposes python_func."""
        assert callable(text_extraction)
        assert hasattr(text_extraction, "python_func")

    def test_component_with_default_parameters(self):
        """Component has expected interface (required args)."""
        sig = inspect.signature(text_extraction.python_func)
        params = list(sig.parameters)
        assert "documents_descriptor" in params
        assert "extracted_text" in params

    def test_missing_descriptor_file_raises_file_not_found(self, tmp_path):
        """Missing documents_descriptor.json raises FileNotFoundError."""
        documents_descriptor_artifact = mock.MagicMock()
        documents_descriptor_artifact.path = str(tmp_path / "missing_descriptor_dir")
        extracted_text_artifact = mock.MagicMock()
        extracted_text_artifact.path = str(tmp_path / "output")

        with mock.patch.dict(
            sys.modules, {**_mock_boto_modules(), **_docling_modules(), **_sync_multiprocess_modules()}
        ):
            with pytest.raises(FileNotFoundError):
                text_extraction.python_func(
                    documents_descriptor=documents_descriptor_artifact,
                    extracted_text=extracted_text_artifact,
                )

    @mock.patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "x"}, clear=True)
    def test_missing_required_env_raises_value_error(self, tmp_path):
        """Missing required S3 env vars raises ValueError with variable name."""
        descriptor_dir = tmp_path / "descriptor"
        descriptor_dir.mkdir()
        (descriptor_dir / "documents_descriptor.json").write_text(
            json.dumps({"bucket": "my-bucket", "documents": []}),
            encoding="utf-8",
        )
        documents_descriptor_artifact = mock.MagicMock(path=str(descriptor_dir))
        extracted_text_artifact = mock.MagicMock(path=str(tmp_path / "output"))

        with mock.patch.dict(
            sys.modules, {**_mock_boto_modules(), **_docling_modules(), **_sync_multiprocess_modules()}
        ):
            with pytest.raises(ValueError, match="AWS_SECRET_ACCESS_KEY environment variable not set"):
                text_extraction.python_func(
                    documents_descriptor=documents_descriptor_artifact,
                    extracted_text=extracted_text_artifact,
                )

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_ssl_error_retries_with_verify_false(self, tmp_path):
        """SSLError on download_file triggers a retry with verify=False."""
        descriptor_dir = tmp_path / "descriptor"
        descriptor_dir.mkdir()
        descriptor = {
            "bucket": "my-bucket",
            "prefix": "docs/",
            "documents": [{"key": "docs/file1.pdf", "size_bytes": 1000}],
            "total_size_bytes": 1000,
            "count": 1,
        }
        (descriptor_dir / "documents_descriptor.json").write_text(json.dumps(descriptor), encoding="utf-8")

        mock_boto3 = mock.MagicMock()
        mock_s3_fail = mock.MagicMock()
        mock_s3_ok = mock.MagicMock()
        mock_s3_fail.download_file.side_effect = _MockSSLError("SSL validation failed")

        def ok_download(bucket, key, dest):
            Path(dest).write_bytes(b"fake pdf content")

        mock_s3_ok.download_file.side_effect = ok_download

        session_call_count = 0

        def fake_session_client(*_args, **_kwargs):
            nonlocal session_call_count
            session_call_count += 1
            return mock_s3_fail if session_call_count == 1 else mock_s3_ok

        mock_session = mock.MagicMock()
        mock_session.client.side_effect = fake_session_client
        mock_boto3.session.Session.return_value = mock_session

        mock_botocore = mock.MagicMock()
        mock_botocore_exceptions = mock.MagicMock()
        mock_botocore_exceptions.SSLError = _MockSSLError
        mock_botocore.exceptions = mock_botocore_exceptions

        documents_descriptor_artifact = mock.MagicMock(path=str(descriptor_dir))
        extracted_text_artifact = mock.MagicMock(path=str(tmp_path / "output"))

        with mock.patch.dict(
            sys.modules,
            {
                "boto3": mock_boto3,
                "botocore": mock_botocore,
                "botocore.exceptions": mock_botocore_exceptions,
                **_docling_modules(),
                **_sync_multiprocess_modules(),
            },
        ):
            text_extraction.python_func(
                documents_descriptor=documents_descriptor_artifact,
                extracted_text=extracted_text_artifact,
            )

        assert session_call_count == 2
        second_call_kwargs = mock_session.client.call_args_list[1][1]
        assert second_call_kwargs["verify"] is False


class TestTextExtractionMultiFormatUnitTests:
    """Enhanced unit tests verifying multi-format support."""

    def test_component_has_expected_parameters(self):
        """Test component has expected interface (required args)."""
        sig = inspect.signature(text_extraction.python_func)
        params = list(sig.parameters)
        assert "documents_descriptor" in params
        assert "extracted_text" in params

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES)
    def test_multi_format_converter_configuration(self, tmp_path):
        """Test that DocumentConverter is configured with all format options."""
        descriptor_dir = tmp_path / "descriptor"
        descriptor_dir.mkdir()
        descriptor = {
            "bucket": "my-bucket",
            "prefix": "docs/",
            "documents": [
                {"key": "docs/file1.pdf", "size_bytes": 1000},
                {"key": "docs/file2.docx", "size_bytes": 2000},
                {"key": "docs/file3.txt", "size_bytes": 500},
            ],
            "total_size_bytes": 3500,
            "count": 3,
        }
        (descriptor_dir / "documents_descriptor.json").write_text(json.dumps(descriptor))

        mock_boto3 = mock.MagicMock()
        mock_s3_client = mock.MagicMock()
        mock_session = mock.MagicMock()
        mock_session.client.return_value = mock_s3_client
        mock_boto3.session.Session.return_value = mock_session

        converter_init_args = {}

        def capture_converter_init(*args, **kwargs):
            converter_init_args.update(kwargs)
            mock_converter_instance = mock.MagicMock()
            mock_converter_instance.convert.return_value = mock.MagicMock(
                document=mock.MagicMock(export_to_markdown=mock.MagicMock(return_value="# Test"))
            )
            return mock_converter_instance

        mock_docling_converter_class = mock.MagicMock(side_effect=capture_converter_init)
        mock_input_format = mock.MagicMock()
        mock_input_format.PDF = "PDF"
        mock_input_format.DOCX = "DOCX"
        mock_input_format.PPTX = "PPTX"
        mock_input_format.HTML = "HTML"
        mock_input_format.MD = "MD"

        documents_descriptor_artifact = mock.MagicMock()
        documents_descriptor_artifact.path = str(descriptor_dir)
        extracted_text_artifact = mock.MagicMock()
        extracted_text_artifact.path = str(tmp_path / "output")

        def fake_download_file(bucket, key, dest):
            Path(dest).write_text("fake content")

        mock_s3_client.download_file.side_effect = fake_download_file

        with mock.patch.dict(
            sys.modules,
            {
                "boto3": mock_boto3,
                "botocore.exceptions": mock.MagicMock(SSLError=_MockSSLError),
                "docling.datamodel.accelerator_options": mock.MagicMock(AcceleratorOptions=mock.MagicMock()),
                "docling.datamodel.base_models": mock.MagicMock(InputFormat=mock_input_format),
                "docling.datamodel.pipeline_options": mock.MagicMock(
                    PdfPipelineOptions=mock.MagicMock(),
                    PaginatedPipelineOptions=mock.MagicMock(),
                ),
                "docling.document_converter": mock.MagicMock(
                    DocumentConverter=mock_docling_converter_class,
                    PdfFormatOption=mock.MagicMock(),
                    WordFormatOption=mock.MagicMock(),
                    PowerpointFormatOption=mock.MagicMock(),
                    HTMLFormatOption=mock.MagicMock(),
                    MarkdownFormatOption=mock.MagicMock(),
                ),
                **_sync_multiprocess_modules(),
            },
        ):
            text_extraction.python_func(
                documents_descriptor=documents_descriptor_artifact,
                extracted_text=extracted_text_artifact,
            )

        assert "format_options" in converter_init_args
        format_options = converter_init_args["format_options"]
        assert "PDF" in format_options
        assert "DOCX" in format_options
        assert "PPTX" in format_options
        assert "HTML" in format_options
        assert "MD" in format_options

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES)
    def test_pdf_processing_with_optimized_settings(self, tmp_path):
        """Test that PDF format uses optimized pipeline settings (no OCR, no table structure)."""
        descriptor_dir = tmp_path / "descriptor"
        descriptor_dir.mkdir()
        descriptor = {
            "bucket": "my-bucket",
            "prefix": "docs/",
            "documents": [{"key": "docs/file1.pdf", "size_bytes": 1000}],
            "total_size_bytes": 1000,
            "count": 1,
        }
        (descriptor_dir / "documents_descriptor.json").write_text(json.dumps(descriptor))

        mock_boto3 = mock.MagicMock()
        mock_s3_client = mock.MagicMock()
        mock_session = mock.MagicMock()
        mock_session.client.return_value = mock_s3_client
        mock_boto3.session.Session.return_value = mock_session

        pdf_pipeline_options_instance = mock.MagicMock()
        mock_pdf_pipeline_options = mock.MagicMock(return_value=pdf_pipeline_options_instance)

        mock_converter_class = mock.MagicMock()
        mock_converter_instance = mock.MagicMock()
        mock_converter_instance.convert.return_value = mock.MagicMock(
            document=mock.MagicMock(export_to_markdown=mock.MagicMock(return_value="# Test"))
        )
        mock_converter_class.return_value = mock_converter_instance

        documents_descriptor_artifact = mock.MagicMock()
        documents_descriptor_artifact.path = str(descriptor_dir)
        extracted_text_artifact = mock.MagicMock()
        extracted_text_artifact.path = str(tmp_path / "output")

        def fake_download_file(bucket, key, dest):
            Path(dest).write_text("fake pdf content")

        mock_s3_client.download_file.side_effect = fake_download_file

        with mock.patch.dict(
            sys.modules,
            {
                "boto3": mock_boto3,
                "botocore.exceptions": mock.MagicMock(SSLError=_MockSSLError),
                "docling.datamodel.accelerator_options": mock.MagicMock(AcceleratorOptions=mock.MagicMock()),
                "docling.datamodel.base_models": mock.MagicMock(InputFormat=mock.MagicMock(PDF="PDF")),
                "docling.datamodel.pipeline_options": mock.MagicMock(
                    PdfPipelineOptions=mock_pdf_pipeline_options,
                    ThreadedPdfPipelineOptions=mock_pdf_pipeline_options,
                    PaginatedPipelineOptions=mock.MagicMock(),
                ),
                "docling.document_converter": mock.MagicMock(
                    DocumentConverter=mock_converter_class,
                    PdfFormatOption=mock.MagicMock(),
                    WordFormatOption=mock.MagicMock(),
                    PowerpointFormatOption=mock.MagicMock(),
                    HTMLFormatOption=mock.MagicMock(),
                    MarkdownFormatOption=mock.MagicMock(),
                ),
                **_sync_multiprocess_modules(),
            },
        ):
            text_extraction.python_func(
                documents_descriptor=documents_descriptor_artifact,
                extracted_text=extracted_text_artifact,
            )

        mock_pdf_pipeline_options.assert_called_once()
        call_kwargs = mock_pdf_pipeline_options.call_args[1]
        assert call_kwargs["do_ocr"] is False
        assert call_kwargs["do_table_structure"] is False

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES)
    def test_txt_file_handling(self, tmp_path):
        """Test that TXT files are copied directly without invoking docling converter."""
        descriptor_dir = tmp_path / "descriptor"
        descriptor_dir.mkdir()
        descriptor = {
            "bucket": "my-bucket",
            "prefix": "docs/",
            "documents": [{"key": "docs/file1.txt", "size_bytes": 500}],
            "total_size_bytes": 500,
            "count": 1,
        }
        (descriptor_dir / "documents_descriptor.json").write_text(json.dumps(descriptor))

        mock_boto3 = mock.MagicMock()
        mock_s3_client = mock.MagicMock()
        mock_session = mock.MagicMock()
        mock_session.client.return_value = mock_s3_client
        mock_boto3.session.Session.return_value = mock_session

        converter_calls = []

        def capture_convert_call(*args, **kwargs):
            converter_calls.append(args)
            return mock.MagicMock(document=mock.MagicMock(export_to_markdown=mock.MagicMock(return_value="# Test")))

        mock_converter_instance = mock.MagicMock()
        mock_converter_instance.convert.side_effect = capture_convert_call
        mock_converter_class = mock.MagicMock(return_value=mock_converter_instance)

        documents_descriptor_artifact = mock.MagicMock()
        documents_descriptor_artifact.path = str(descriptor_dir)
        extracted_text_artifact = mock.MagicMock()
        extracted_text_artifact.path = str(tmp_path / "output")

        def fake_download_file(bucket, key, dest):
            Path(dest).write_text("This is a text file content")

        mock_s3_client.download_file.side_effect = fake_download_file

        with mock.patch.dict(
            sys.modules,
            {
                "boto3": mock_boto3,
                "botocore.exceptions": mock.MagicMock(SSLError=_MockSSLError),
                "docling.datamodel.accelerator_options": mock.MagicMock(AcceleratorOptions=mock.MagicMock()),
                "docling.datamodel.base_models": mock.MagicMock(InputFormat=mock.MagicMock()),
                "docling.datamodel.pipeline_options": mock.MagicMock(
                    PdfPipelineOptions=mock.MagicMock(),
                    PaginatedPipelineOptions=mock.MagicMock(),
                ),
                "docling.document_converter": mock.MagicMock(
                    DocumentConverter=mock_converter_class,
                    PdfFormatOption=mock.MagicMock(),
                    WordFormatOption=mock.MagicMock(),
                    PowerpointFormatOption=mock.MagicMock(),
                    HTMLFormatOption=mock.MagicMock(),
                    MarkdownFormatOption=mock.MagicMock(),
                ),
                **_sync_multiprocess_modules(),
            },
        ):
            text_extraction.python_func(
                documents_descriptor=documents_descriptor_artifact,
                extracted_text=extracted_text_artifact,
            )

        assert len(converter_calls) == 0

        output_dir = Path(extracted_text_artifact.path)
        output_files = list(output_dir.glob("*.md"))
        assert len(output_files) == 1
        assert output_files[0].name == "file1.txt.md"
        assert output_files[0].read_text() == "This is a text file content"


class TestMultiFormatProcessing:
    """Test processing of multiple document formats using fixtures."""

    @pytest.fixture
    def fixtures_dir(self):
        """Return the path to the fixtures directory."""
        return Path(__file__).parent / "fixtures"

    @pytest.fixture
    def mock_docling_components(self):
        """Create standard docling mocks with a converter that returns markdown strings."""
        mock_converter_instance = mock.MagicMock()
        mock_converter_instance.convert.return_value = mock.MagicMock(
            document=mock.MagicMock(export_to_markdown=mock.MagicMock(return_value="# Extracted Content"))
        )
        mock_converter_class = mock.MagicMock(return_value=mock_converter_instance)

        mock_input_format = mock.MagicMock()
        mock_input_format.PDF = "PDF"
        mock_input_format.DOCX = "DOCX"
        mock_input_format.PPTX = "PPTX"
        mock_input_format.HTML = "HTML"
        mock_input_format.MD = "MD"

        return {
            "converter_class": mock_converter_class,
            "converter_instance": mock_converter_instance,
            "input_format": mock_input_format,
        }

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES)
    @pytest.mark.parametrize(
        "file_extension,file_name",
        [
            (".pdf", "sample.pdf"),
            (".docx", "sample.docx"),
            (".pptx", "sample.pptx"),
            (".html", "sample.html"),
            (".md", "sample.md"),
            (".txt", "sample.txt"),
        ],
    )
    def test_format_processing(self, tmp_path, fixtures_dir, mock_docling_components, file_extension, file_name):
        """Test that each supported format is processed correctly."""
        descriptor_dir = tmp_path / "descriptor"
        descriptor_dir.mkdir()
        descriptor = {
            "bucket": "my-bucket",
            "prefix": "docs/",
            "documents": [{"key": f"docs/{file_name}", "size_bytes": 1000}],
            "total_size_bytes": 1000,
            "count": 1,
        }
        (descriptor_dir / "documents_descriptor.json").write_text(json.dumps(descriptor))

        mock_boto3 = mock.MagicMock()
        mock_s3_client = mock.MagicMock()
        mock_session = mock.MagicMock()
        mock_session.client.return_value = mock_s3_client
        mock_boto3.session.Session.return_value = mock_session

        def fake_download_file(bucket, key, dest):
            fixture_file = fixtures_dir / file_name
            if fixture_file.exists():
                shutil.copy(fixture_file, dest)
            else:
                Path(dest).write_text(f"Sample content for {file_name}")

        mock_s3_client.download_file.side_effect = fake_download_file

        documents_descriptor_artifact = mock.MagicMock()
        documents_descriptor_artifact.path = str(descriptor_dir)
        extracted_text_artifact = mock.MagicMock()
        extracted_text_artifact.path = str(tmp_path / "output")

        with mock.patch.dict(
            sys.modules,
            {
                "boto3": mock_boto3,
                "botocore.exceptions": mock.MagicMock(SSLError=_MockSSLError),
                "docling.datamodel.accelerator_options": mock.MagicMock(AcceleratorOptions=mock.MagicMock()),
                "docling.datamodel.base_models": mock.MagicMock(InputFormat=mock_docling_components["input_format"]),
                "docling.datamodel.pipeline_options": mock.MagicMock(
                    PdfPipelineOptions=mock.MagicMock(),
                    PaginatedPipelineOptions=mock.MagicMock(),
                ),
                "docling.document_converter": mock.MagicMock(
                    DocumentConverter=mock_docling_components["converter_class"],
                    PdfFormatOption=mock.MagicMock(),
                    WordFormatOption=mock.MagicMock(),
                    PowerpointFormatOption=mock.MagicMock(),
                    HTMLFormatOption=mock.MagicMock(),
                    MarkdownFormatOption=mock.MagicMock(),
                ),
                **_sync_multiprocess_modules(),
            },
        ):
            text_extraction.python_func(
                documents_descriptor=documents_descriptor_artifact,
                extracted_text=extracted_text_artifact,
            )

        output_dir = Path(extracted_text_artifact.path)
        output_files = list(output_dir.glob("*.md"))
        assert len(output_files) == 1, f"Expected 1 output file for {file_name}, found {len(output_files)}"
        assert output_files[0].name == f"{file_name}.md"
        if file_extension == ".txt":
            mock_docling_components["converter_instance"].convert.assert_not_called()
        else:
            mock_docling_components["converter_instance"].convert.assert_called_once()

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES)
    def test_multiple_formats_in_batch(self, tmp_path, fixtures_dir, mock_docling_components):
        """Test processing a batch with multiple different formats."""
        descriptor_dir = tmp_path / "descriptor"
        descriptor_dir.mkdir()
        documents = [
            {"key": "docs/sample.pdf", "size_bytes": 1000},
            {"key": "docs/sample.docx", "size_bytes": 2000},
            {"key": "docs/sample.txt", "size_bytes": 500},
            {"key": "docs/sample.html", "size_bytes": 1500},
            {"key": "docs/sample.md", "size_bytes": 800},
            {"key": "docs/sample.pptx", "size_bytes": 3000},
        ]
        descriptor = {
            "bucket": "my-bucket",
            "prefix": "docs/",
            "documents": documents,
            "total_size_bytes": sum(d["size_bytes"] for d in documents),
            "count": len(documents),
        }
        (descriptor_dir / "documents_descriptor.json").write_text(json.dumps(descriptor))

        mock_boto3 = mock.MagicMock()
        mock_s3_client = mock.MagicMock()
        mock_session = mock.MagicMock()
        mock_session.client.return_value = mock_s3_client
        mock_boto3.session.Session.return_value = mock_session

        def fake_download_file(bucket, key, dest):
            file_name = key.split("/")[-1]
            fixture_file = fixtures_dir / file_name
            if fixture_file.exists():
                shutil.copy(fixture_file, dest)
            else:
                Path(dest).write_text(f"Sample content for {file_name}")

        mock_s3_client.download_file.side_effect = fake_download_file

        documents_descriptor_artifact = mock.MagicMock()
        documents_descriptor_artifact.path = str(descriptor_dir)
        extracted_text_artifact = mock.MagicMock()
        extracted_text_artifact.path = str(tmp_path / "output")

        with mock.patch.dict(
            sys.modules,
            {
                "boto3": mock_boto3,
                "botocore.exceptions": mock.MagicMock(SSLError=_MockSSLError),
                "docling.datamodel.accelerator_options": mock.MagicMock(AcceleratorOptions=mock.MagicMock()),
                "docling.datamodel.base_models": mock.MagicMock(InputFormat=mock_docling_components["input_format"]),
                "docling.datamodel.pipeline_options": mock.MagicMock(
                    PdfPipelineOptions=mock.MagicMock(),
                    PaginatedPipelineOptions=mock.MagicMock(),
                ),
                "docling.document_converter": mock.MagicMock(
                    DocumentConverter=mock_docling_components["converter_class"],
                    PdfFormatOption=mock.MagicMock(),
                    WordFormatOption=mock.MagicMock(),
                    PowerpointFormatOption=mock.MagicMock(),
                    HTMLFormatOption=mock.MagicMock(),
                    MarkdownFormatOption=mock.MagicMock(),
                ),
                **_sync_multiprocess_modules(),
            },
        ):
            text_extraction.python_func(
                documents_descriptor=documents_descriptor_artifact,
                extracted_text=extracted_text_artifact,
            )

        output_dir = Path(extracted_text_artifact.path)
        all_output_files = list(output_dir.glob("*.md"))
        assert len(all_output_files) == 6, f"Expected 6 output files, found {len(all_output_files)}"

        expected_files = {
            "sample.pdf.md",
            "sample.docx.md",
            "sample.txt.md",
            "sample.html.md",
            "sample.md.md",
            "sample.pptx.md",
        }
        actual_files = {f.name for f in all_output_files}
        assert actual_files == expected_files

        assert mock_docling_components["converter_instance"].convert.call_count == 5

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES)
    def test_unsupported_file_extension_ignored(self, tmp_path, mock_docling_components):
        """Test that files with unsupported extensions are not downloaded or processed."""
        descriptor_dir = tmp_path / "descriptor"
        descriptor_dir.mkdir()
        descriptor = {
            "bucket": "my-bucket",
            "prefix": "docs/",
            "documents": [
                {"key": "docs/file.zip", "size_bytes": 1000},
                {"key": "docs/file.exe", "size_bytes": 2000},
            ],
            "total_size_bytes": 3000,
            "count": 2,
        }
        (descriptor_dir / "documents_descriptor.json").write_text(json.dumps(descriptor))

        mock_boto3 = mock.MagicMock()
        mock_s3_client = mock.MagicMock()
        mock_session = mock.MagicMock()
        mock_session.client.return_value = mock_s3_client
        mock_boto3.session.Session.return_value = mock_session

        def fake_download_file(bucket, key, dest):
            Path(dest).write_text("fake content")

        mock_s3_client.download_file.side_effect = fake_download_file

        documents_descriptor_artifact = mock.MagicMock()
        documents_descriptor_artifact.path = str(descriptor_dir)
        extracted_text_artifact = mock.MagicMock()
        extracted_text_artifact.path = str(tmp_path / "output")

        with mock.patch.dict(
            sys.modules,
            {
                "boto3": mock_boto3,
                "botocore.exceptions": mock.MagicMock(SSLError=_MockSSLError),
                "docling.datamodel.accelerator_options": mock.MagicMock(AcceleratorOptions=mock.MagicMock()),
                "docling.datamodel.base_models": mock.MagicMock(InputFormat=mock_docling_components["input_format"]),
                "docling.datamodel.pipeline_options": mock.MagicMock(
                    PdfPipelineOptions=mock.MagicMock(),
                    PaginatedPipelineOptions=mock.MagicMock(),
                ),
                "docling.document_converter": mock.MagicMock(
                    DocumentConverter=mock_docling_components["converter_class"],
                    PdfFormatOption=mock.MagicMock(),
                    WordFormatOption=mock.MagicMock(),
                    PowerpointFormatOption=mock.MagicMock(),
                    HTMLFormatOption=mock.MagicMock(),
                    MarkdownFormatOption=mock.MagicMock(),
                ),
                **_sync_multiprocess_modules(),
            },
        ):
            text_extraction.python_func(
                documents_descriptor=documents_descriptor_artifact,
                extracted_text=extracted_text_artifact,
            )

        output_dir = Path(extracted_text_artifact.path)
        output_files = list(output_dir.glob("*.md"))
        assert len(output_files) == 0, "Unsupported file types should not produce output"
        mock_docling_components["converter_instance"].convert.assert_not_called()
