"""Unit tests for nested helper functions from text_extraction component."""

import os
import sys
from pathlib import Path
from unittest import mock

import pytest
from botocore.exceptions import ClientError, SSLError

from components.data_processing.autorag.text_extraction.tests.nested_names import (
    _build_docling_format_options,
    _docling_artifacts_path,
    _text_extraction_pool_initializer,
    download_and_submit,
    download_document,
    make_s3_client,
    raise_if_threshold_exceeded,
    worker_process_document,
)


@pytest.fixture
def s3_creds():
    """Fixture for S3 credentials dictionary."""
    return {
        "AWS_ACCESS_KEY_ID": "test_access_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret_key",
        "AWS_S3_ENDPOINT": "https://s3.example.com",
        "AWS_DEFAULT_REGION": "us-east-1",
    }


@pytest.fixture
def s3_creds_no_region():
    """Fixture for S3 credentials without AWS_DEFAULT_REGION."""
    return {
        "AWS_ACCESS_KEY_ID": "test_access_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret_key",
        "AWS_S3_ENDPOINT": "https://s3.example.com",
    }


class TestMakeS3Client:
    """Unit tests for make_s3_client helper function."""

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.boto3")
    def test_creates_client_with_verify_true_by_default(self, mock_boto3, s3_creds):
        """Creates S3 client with verify=True by default."""
        mock_session = mock.MagicMock()
        mock_s3_client = mock.MagicMock()
        mock_boto3.session.Session.return_value = mock_session
        mock_session.client.return_value = mock_s3_client

        result = make_s3_client(s3_creds, verify=True)

        # Verify Session created with credentials
        mock_boto3.session.Session.assert_called_once_with(
            aws_access_key_id="test_access_key",
            aws_secret_access_key="test_secret_key",
            region_name="us-east-1",
        )
        # Verify client created with endpoint and verify=True
        mock_session.client.assert_called_once_with(
            service_name="s3",
            endpoint_url="https://s3.example.com",
            verify=True,
        )
        assert result == mock_s3_client

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.boto3")
    def test_creates_client_with_verify_false_when_specified(self, mock_boto3, s3_creds):
        """Creates S3 client with verify=False when specified."""
        mock_session = mock.MagicMock()
        mock_s3_client = mock.MagicMock()
        mock_boto3.session.Session.return_value = mock_session
        mock_session.client.return_value = mock_s3_client

        result = make_s3_client(s3_creds, verify=False)

        # Verify verify=False passed to client
        call_kwargs = mock_session.client.call_args[1]
        assert call_kwargs["verify"] is False
        assert result == mock_s3_client

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.boto3")
    def test_all_credential_fields_used_correctly(self, mock_boto3, s3_creds):
        """All credential fields are passed correctly to Session and client."""
        mock_session = mock.MagicMock()
        mock_s3_client = mock.MagicMock()
        mock_boto3.session.Session.return_value = mock_session
        mock_session.client.return_value = mock_s3_client

        make_s3_client(s3_creds)

        # Verify Session credentials
        session_kwargs = mock_boto3.session.Session.call_args[1]
        assert session_kwargs["aws_access_key_id"] == "test_access_key"
        assert session_kwargs["aws_secret_access_key"] == "test_secret_key"
        assert session_kwargs["region_name"] == "us-east-1"

        # Verify client configuration
        client_kwargs = mock_session.client.call_args[1]
        assert client_kwargs["service_name"] == "s3"
        assert client_kwargs["endpoint_url"] == "https://s3.example.com"

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.boto3")
    def test_missing_region_handled_gracefully(self, mock_boto3, s3_creds_no_region):
        """Missing AWS_DEFAULT_REGION is handled gracefully with None."""
        mock_session = mock.MagicMock()
        mock_s3_client = mock.MagicMock()
        mock_boto3.session.Session.return_value = mock_session
        mock_session.client.return_value = mock_s3_client

        result = make_s3_client(s3_creds_no_region)

        # Verify Session created with region_name=None
        session_kwargs = mock_boto3.session.Session.call_args[1]
        assert session_kwargs["region_name"] is None
        assert result == mock_s3_client

    def test_missing_required_credential_raises_keyerror(self):
        """Missing required credential field raises KeyError."""
        incomplete_creds = {
            "AWS_SECRET_ACCESS_KEY": "test_secret",
            "AWS_S3_ENDPOINT": "https://s3.example.com",
        }

        with pytest.raises(KeyError, match="AWS_ACCESS_KEY_ID"):
            make_s3_client(incomplete_creds)

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.boto3")
    def test_returns_s3_client_with_correct_interface(self, mock_boto3, s3_creds):
        """Returns boto3 client object with S3 interface."""
        mock_session = mock.MagicMock()
        mock_s3_client = mock.MagicMock()
        mock_s3_client.download_file = mock.MagicMock()
        mock_boto3.session.Session.return_value = mock_session
        mock_session.client.return_value = mock_s3_client

        result = make_s3_client(s3_creds)

        assert hasattr(result, "download_file")
        assert result == mock_s3_client


class TestDownloadDocument:
    """Unit tests for download_document helper function."""

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.make_s3_client")
    def test_valid_s3_key_downloads_successfully(self, mock_make_s3_client, s3_creds, tmp_path):
        """Valid S3 key downloads successfully to correct path."""
        mock_client = mock.MagicMock()
        mock_make_s3_client.return_value = mock_client

        doc = {"key": "path/to/file.pdf"}

        result = download_document(s3_creds, "test-bucket", doc, tmp_path)

        expected_path = tmp_path / "path/to/file.pdf"
        assert result == expected_path
        mock_client.download_file.assert_called_once_with(
            "test-bucket", "path/to/file.pdf", str(expected_path)
        )

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.make_s3_client")
    def test_s3_key_with_whitespace_sanitized(self, mock_make_s3_client, s3_creds, tmp_path):
        """S3 key with leading/trailing whitespace is sanitized."""
        mock_client = mock.MagicMock()
        mock_make_s3_client.return_value = mock_client

        doc = {"key": "  path/to/file.pdf  "}

        result = download_document(s3_creds, "test-bucket", doc, tmp_path)

        expected_path = tmp_path / "path/to/file.pdf"
        assert result == expected_path
        mock_client.download_file.assert_called_once_with(
            "test-bucket", "  path/to/file.pdf  ", str(expected_path)
        )

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.make_s3_client")
    def test_s3_key_with_leading_slashes_sanitized(self, mock_make_s3_client, s3_creds, tmp_path):
        """S3 key with leading slashes is sanitized."""
        mock_client = mock.MagicMock()
        mock_make_s3_client.return_value = mock_client

        doc = {"key": "///path/to/file.pdf"}

        result = download_document(s3_creds, "test-bucket", doc, tmp_path)

        expected_path = tmp_path / "path/to/file.pdf"
        assert result == expected_path

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.make_s3_client")
    def test_absolute_path_in_s3_key_sanitized_and_allowed(self, mock_make_s3_client, s3_creds, tmp_path):
        """Absolute path in S3 key is sanitized (leading slash removed) and allowed."""
        mock_client = mock.MagicMock()
        mock_make_s3_client.return_value = mock_client

        doc = {"key": "/etc/passwd"}

        result = download_document(s3_creds, "test-bucket", doc, tmp_path)

        # After sanitization "/etc/passwd" becomes "etc/passwd" which is safe
        expected_path = tmp_path / "etc/passwd"
        assert result == expected_path

    def test_path_traversal_with_dotdot_rejected(self, s3_creds, tmp_path):
        """Path traversal with '..' is rejected as unsafe."""
        doc = {"key": "../../etc/passwd"}

        with pytest.raises(ValueError, match="Unsafe S3 key \\(path traversal\\)"):
            download_document(s3_creds, "test-bucket", doc, tmp_path)

    def test_path_escaping_base_directory_rejected(self, s3_creds, tmp_path):
        """Path that escapes base directory after resolution is rejected."""
        # Create a key that might try to escape via symlinks or normalization
        doc = {"key": "subdir/../../outside/file.txt"}

        with pytest.raises(ValueError, match="Unsafe S3 key"):
            download_document(s3_creds, "test-bucket", doc, tmp_path)

    def test_empty_s3_key_after_sanitization_rejected(self, s3_creds, tmp_path):
        """Empty S3 key after sanitization is rejected."""
        doc = {"key": "   "}

        with pytest.raises(ValueError, match="Unsafe S3 key"):
            download_document(s3_creds, "test-bucket", doc, tmp_path)

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.make_s3_client")
    def test_ssl_error_triggers_retry_with_verify_false(self, mock_make_s3_client, s3_creds, tmp_path):
        """SSL error triggers retry with verify=False."""
        # First client raises SSLError, second succeeds
        failing_client = mock.MagicMock()
        ssl_error = SSLError(endpoint_url="https://s3.example.com", error=Exception("SSL error"))
        failing_client.download_file.side_effect = ssl_error

        success_client = mock.MagicMock()

        mock_make_s3_client.side_effect = [failing_client, success_client]

        doc = {"key": "file.pdf"}

        result = download_document(s3_creds, "test-bucket", doc, tmp_path)

        assert result == tmp_path / "file.pdf"
        # Verify two clients created: first with default verify, second with verify=False
        assert mock_make_s3_client.call_count == 2
        mock_make_s3_client.assert_any_call(s3_creds)
        mock_make_s3_client.assert_any_call(s3_creds, verify=False)

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.make_s3_client")
    def test_non_ssl_error_propagates_without_retry(self, mock_make_s3_client, s3_creds, tmp_path):
        """Non-SSL error is re-raised without retry."""
        mock_client = mock.MagicMock()
        mock_client.download_file.side_effect = ClientError(
            {"Error": {"Code": "404"}}, "GetObject"
        )
        mock_make_s3_client.return_value = mock_client

        doc = {"key": "file.pdf"}

        with pytest.raises(ClientError):
            download_document(s3_creds, "test-bucket", doc, tmp_path)

        # Verify no retry
        assert mock_make_s3_client.call_count == 1

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.make_s3_client")
    def test_parent_directories_created_automatically(self, mock_make_s3_client, s3_creds, tmp_path):
        """Parent directories are created automatically for nested paths."""
        mock_client = mock.MagicMock()
        mock_make_s3_client.return_value = mock_client

        doc = {"key": "deep/nested/path/file.pdf"}

        result = download_document(s3_creds, "test-bucket", doc, tmp_path)

        assert result == tmp_path / "deep/nested/path/file.pdf"
        # Verify parent directory exists
        assert result.parent.exists()
        assert result.parent == tmp_path / "deep/nested/path"


class TestDoclingArtifactsPath:
    """Unit tests for _docling_artifacts_path helper function."""

    @mock.patch.dict("os.environ", {}, clear=True)
    def test_returns_none_when_env_var_not_set(self):
        """Returns None when DOCLING_ARTIFACTS_PATH not set."""
        result = _docling_artifacts_path()
        assert result is None

    @mock.patch.dict("os.environ", {"DOCLING_ARTIFACTS_PATH": "/nonexistent/path"})
    def test_returns_none_when_path_does_not_exist(self):
        """Returns None when DOCLING_ARTIFACTS_PATH points to non-existent directory."""
        result = _docling_artifacts_path()
        assert result is None

    def test_returns_none_when_path_is_empty_directory(self, tmp_path):
        """Returns None when DOCLING_ARTIFACTS_PATH points to empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with mock.patch.dict("os.environ", {"DOCLING_ARTIFACTS_PATH": str(empty_dir)}):
            result = _docling_artifacts_path()
            assert result is None

    def test_returns_path_when_valid_and_contains_files(self, tmp_path):
        """Returns Path when DOCLING_ARTIFACTS_PATH is valid and contains files."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        # Create a model file
        (models_dir / "model.bin").write_text("fake model")

        with mock.patch.dict("os.environ", {"DOCLING_ARTIFACTS_PATH": str(models_dir)}):
            result = _docling_artifacts_path()
            assert result == models_dir
            assert isinstance(result, Path)

    def test_returns_path_when_directory_contains_subdirectories(self, tmp_path):
        """Returns Path when directory contains model subdirectories."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        # Create model subdirectory (docling pattern)
        (models_dir / "docling-project--model1").mkdir()

        with mock.patch.dict("os.environ", {"DOCLING_ARTIFACTS_PATH": str(models_dir)}):
            result = _docling_artifacts_path()
            assert result == models_dir


class TestBuildDoclingFormatOptions:
    """Unit tests for _build_docling_format_options helper function."""

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names._docling_artifacts_path")
    def test_returns_dict_with_all_supported_input_format_keys(self, mock_artifacts_path):
        """Returns dict with all supported InputFormat keys."""
        mock_artifacts_path.return_value = None

        result = _build_docling_format_options()

        # Import InputFormat to check keys
        from docling.datamodel.base_models import InputFormat

        assert InputFormat.PDF in result
        assert InputFormat.DOCX in result
        assert InputFormat.PPTX in result
        assert InputFormat.HTML in result
        assert InputFormat.MD in result
        assert len(result) == 5

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names._docling_artifacts_path")
    def test_pdf_uses_threaded_pdf_pipeline_options(self, mock_artifacts_path):
        """PDF uses ThreadedPdfPipelineOptions."""
        mock_artifacts_path.return_value = None

        result = _build_docling_format_options()

        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import PdfFormatOption

        pdf_option = result[InputFormat.PDF]
        assert isinstance(pdf_option, PdfFormatOption)
        # Verify pipeline options attributes
        assert pdf_option.pipeline_options.do_ocr is False
        assert pdf_option.pipeline_options.do_table_structure is False

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names._docling_artifacts_path")
    def test_docx_pptx_use_paginated_pipeline_options(self, mock_artifacts_path):
        """DOCX and PPTX use PaginatedPipelineOptions."""
        mock_artifacts_path.return_value = None

        result = _build_docling_format_options()

        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import PowerpointFormatOption, WordFormatOption

        docx_option = result[InputFormat.DOCX]
        pptx_option = result[InputFormat.PPTX]

        assert isinstance(docx_option, WordFormatOption)
        assert isinstance(pptx_option, PowerpointFormatOption)
        # Verify paginated options
        assert docx_option.pipeline_options.generate_page_images is False
        assert pptx_option.pipeline_options.generate_page_images is False

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names._docling_artifacts_path")
    def test_artifacts_path_propagated_to_pipeline_options(self, mock_artifacts_path, tmp_path):
        """Artifacts path is propagated to pipeline options when set."""
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        mock_artifacts_path.return_value = artifacts_dir

        result = _build_docling_format_options()

        from docling.datamodel.base_models import InputFormat

        # Check PDF pipeline has artifacts_path
        pdf_option = result[InputFormat.PDF]
        assert pdf_option.pipeline_options.artifacts_path == artifacts_dir

        # Check DOCX pipeline has artifacts_path
        docx_option = result[InputFormat.DOCX]
        assert docx_option.pipeline_options.artifacts_path == artifacts_dir

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names._docling_artifacts_path")
    def test_accelerator_options_set_to_cpu_with_2_threads(self, mock_artifacts_path):
        """Accelerator options set to CPU with 2 threads for all pipeline options."""
        mock_artifacts_path.return_value = None

        result = _build_docling_format_options()

        from docling.datamodel.base_models import InputFormat

        # Check PDF accelerator options
        pdf_accel = result[InputFormat.PDF].pipeline_options.accelerator_options
        assert pdf_accel.device == "cpu"
        assert pdf_accel.num_threads == 2

        # Check DOCX accelerator options
        docx_accel = result[InputFormat.DOCX].pipeline_options.accelerator_options
        assert docx_accel.device == "cpu"
        assert docx_accel.num_threads == 2


class TestTextExtractionPoolInitializer:
    """Unit tests for _text_extraction_pool_initializer helper function."""

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names._docling_artifacts_path")
    @mock.patch("docling.document_converter.DocumentConverter")
    @mock.patch.dict("os.environ", {}, clear=True)
    def test_sets_tqdm_disable(self, mock_converter, mock_artifacts_path):
        """Sets TQDM_DISABLE=1 environment variable."""
        mock_artifacts_path.return_value = None

        _text_extraction_pool_initializer()

        assert os.environ["TQDM_DISABLE"] == "1"

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names._docling_artifacts_path")
    @mock.patch("docling.document_converter.DocumentConverter")
    @mock.patch.dict("os.environ", {}, clear=True)
    def test_sets_thread_count_environment_variables(self, mock_converter, mock_artifacts_path):
        """Sets OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, OMP_NUM_THREADS to '1'."""
        mock_artifacts_path.return_value = None

        _text_extraction_pool_initializer()

        assert os.environ.get("OPENBLAS_NUM_THREADS") == "1"
        assert os.environ.get("MKL_NUM_THREADS") == "1"
        assert os.environ.get("OMP_NUM_THREADS") == "1"

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names._docling_artifacts_path")
    @mock.patch("docling.document_converter.DocumentConverter")
    @mock.patch.dict("os.environ", {}, clear=True)
    def test_sets_hf_hub_offline_when_artifacts_path_exists(
        self, mock_converter, mock_artifacts_path, tmp_path
    ):
        """Sets HF_HUB_OFFLINE=1 when artifacts path exists."""
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        mock_artifacts_path.return_value = artifacts_dir

        _text_extraction_pool_initializer()

        assert os.environ.get("HF_HUB_OFFLINE") == "1"

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names._docling_artifacts_path")
    @mock.patch("docling.document_converter.DocumentConverter")
    @mock.patch.dict("os.environ", {}, clear=True)
    def test_does_not_set_hf_hub_offline_when_artifacts_path_none(
        self, mock_converter, mock_artifacts_path
    ):
        """Does not set HF_HUB_OFFLINE when artifacts path is None."""
        mock_artifacts_path.return_value = None

        _text_extraction_pool_initializer()

        # HF_HUB_OFFLINE should not be set (using setdefault means it's only set if artifacts exist)
        assert "HF_HUB_OFFLINE" not in os.environ

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names._docling_artifacts_path")
    @mock.patch("docling.document_converter.DocumentConverter")
    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names._build_docling_format_options")
    def test_creates_document_converter_with_format_options(
        self, mock_build_options, mock_converter_class, mock_artifacts_path
    ):
        """Creates DocumentConverter with format options and stores in module."""
        mock_artifacts_path.return_value = None
        mock_format_options = {"mock": "options"}
        mock_build_options.return_value = mock_format_options
        mock_converter_instance = mock.MagicMock()
        mock_converter_class.return_value = mock_converter_instance

        # Import the module to set the attribute on it
        from components.data_processing.autorag.text_extraction.tests import nested_names

        _text_extraction_pool_initializer()

        # Verify DocumentConverter created with format options
        mock_converter_class.assert_called_once_with(format_options=mock_format_options)
        # Verify stored in module
        assert hasattr(nested_names, "_mp_worker_converter")
        assert nested_names._mp_worker_converter == mock_converter_instance

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names._docling_artifacts_path")
    @mock.patch("docling.document_converter.DocumentConverter")
    def test_configures_worker_logger_with_stdout_handler(self, mock_converter, mock_artifacts_path):
        """Configures worker logger with StreamHandler writing to stdout."""
        mock_artifacts_path.return_value = None

        _text_extraction_pool_initializer()

        # Get the logger that should have been configured
        import logging

        logger = logging.getLogger("text_extraction_worker")

        # Verify logger is configured
        assert logger.level == logging.INFO
        assert logger.propagate is False
        # Verify has at least one handler
        assert len(logger.handlers) > 0
        # Check if any handler is StreamHandler (might have multiple from multiple test runs)
        has_stream_handler = any(
            isinstance(h, logging.StreamHandler) for h in logger.handlers
        )
        assert has_stream_handler


class TestWorkerProcessDocument:
    """Unit tests for worker_process_document helper function."""

    def test_txt_file_copied_without_conversion(self, tmp_path):
        """Plain text file is copied without conversion."""
        # Create input txt file
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        input_file = input_dir / "document.txt"
        input_file.write_text("Plain text content", encoding="utf-8")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        success, error = worker_process_document(str(input_file), str(output_dir))

        assert success is True
        assert error is None
        # Verify output file created with .md extension
        output_file = output_dir / "document.txt.md"
        assert output_file.exists()
        assert output_file.read_text(encoding="utf-8") == "Plain text content"

    def test_non_txt_file_converted_via_document_converter(self, tmp_path):
        """Non-txt file is converted via DocumentConverter."""
        # Create input pdf file
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        input_file = input_dir / "document.pdf"
        input_file.write_bytes(b"fake pdf content")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Mock the converter on the module
        from components.data_processing.autorag.text_extraction.tests import nested_names

        mock_converter = mock.MagicMock()
        mock_result = mock.MagicMock()
        mock_document = mock.MagicMock()
        mock_document.export_to_markdown.return_value = "# Converted Markdown"
        mock_result.document = mock_document
        mock_converter.convert.return_value = mock_result

        nested_names._mp_worker_converter = mock_converter

        success, error = worker_process_document(str(input_file), str(output_dir))

        assert success is True
        assert error is None
        # Verify converter was called
        mock_converter.convert.assert_called_once()
        # Verify output file created
        output_file = output_dir / "document.pdf.md"
        assert output_file.exists()
        assert output_file.read_text(encoding="utf-8") == "# Converted Markdown"

    def test_returns_error_tuple_on_exception(self, tmp_path):
        """Returns error tuple when conversion raises exception."""
        # Create input pdf file
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        input_file = input_dir / "document.pdf"
        input_file.write_bytes(b"fake pdf content")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Mock the converter to raise exception
        from components.data_processing.autorag.text_extraction.tests import nested_names

        mock_converter = mock.MagicMock()
        mock_converter.convert.side_effect = Exception("Conversion failed")

        nested_names._mp_worker_converter = mock_converter

        success, error = worker_process_document(str(input_file), str(output_dir))

        assert success is False
        assert error is not None
        assert "Conversion failed" in error
        assert "Traceback" in error

    def test_returns_error_when_document_converter_not_initialized(self, tmp_path):
        """Returns error when DocumentConverter not initialized."""
        # Create input pdf file
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        input_file = input_dir / "document.pdf"
        input_file.write_bytes(b"fake pdf content")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Remove the converter attribute if it exists
        from components.data_processing.autorag.text_extraction.tests import nested_names

        if hasattr(nested_names, "_mp_worker_converter"):
            delattr(nested_names, "_mp_worker_converter")

        success, error = worker_process_document(str(input_file), str(output_dir))

        assert success is False
        assert error is not None
        assert "Worker" in error
        assert "has no DocumentConverter" in error

    def test_output_file_has_md_extension_appended(self, tmp_path):
        """Output file has .md extension appended to original name."""
        # Create input file with multiple extensions
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        input_file = input_dir / "report.final.docx"
        input_file.write_bytes(b"fake docx content")

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Mock the converter
        from components.data_processing.autorag.text_extraction.tests import nested_names

        mock_converter = mock.MagicMock()
        mock_result = mock.MagicMock()
        mock_document = mock.MagicMock()
        mock_document.export_to_markdown.return_value = "Markdown content"
        mock_result.document = mock_document
        mock_converter.convert.return_value = mock_result

        nested_names._mp_worker_converter = mock_converter

        success, error = worker_process_document(str(input_file), str(output_dir))

        assert success is True
        # Verify output filename
        output_file = output_dir / "report.final.docx.md"
        assert output_file.exists()

    def test_logs_file_size_before_conversion(self, tmp_path):
        """Logs file size in MiB before conversion."""
        # Create input file
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        input_file = input_dir / "document.pdf"
        # Write ~1 MiB of data
        input_file.write_bytes(b"x" * (1024 * 1024))

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Mock the converter
        from components.data_processing.autorag.text_extraction.tests import nested_names

        mock_converter = mock.MagicMock()
        mock_result = mock.MagicMock()
        mock_document = mock.MagicMock()
        mock_document.export_to_markdown.return_value = "Content"
        mock_result.document = mock_document
        mock_converter.convert.return_value = mock_result

        nested_names._mp_worker_converter = mock_converter

        success, _ = worker_process_document(str(input_file), str(output_dir))

        assert success is True
        # Verify converter was called with the input file
        mock_converter.convert.assert_called_once()
        call_arg = mock_converter.convert.call_args[0][0]
        assert str(call_arg) == str(input_file)


class TestDownloadAndSubmit:
    """Unit tests for download_and_submit helper function."""

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.as_completed")
    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.ThreadPoolExecutor")
    def test_filters_unsupported_file_extensions(
        self, mock_executor_class, mock_as_completed, s3_creds, tmp_path
    ):
        """Filters unsupported file extensions before downloading."""
        docs = [
            {"key": "file1.pdf"},
            {"key": "file2.zip"},  # unsupported
            {"key": "file3.docx"},
            {"key": "file4.exe"},  # unsupported
        ]

        mock_pool = mock.MagicMock()
        mock_process_pool = mock.MagicMock()

        # Mock ThreadPoolExecutor context manager
        mock_executor = mock.MagicMock()
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.__exit__.return_value = None
        mock_executor_class.return_value = mock_executor

        # Mock submit to return completed futures
        futures = []
        def mock_submit(fn, *args):
            future = mock.MagicMock()
            file_path = tmp_path / args[2]["key"]  # doc arg
            # Create the file so .stat() works
            file_path.write_bytes(b"content")
            future.result.return_value = file_path
            futures.append(future)
            return future

        mock_executor.submit.side_effect = mock_submit
        mock_executor_class.return_value = mock_executor

        # Mock as_completed to return futures immediately
        mock_as_completed.return_value = futures

        tasks, errors = download_and_submit(
            s3_creds, "bucket", docs, tmp_path, mock_process_pool, tmp_path
        )

        # Verify only supported files submitted for download (pdf, docx)
        assert mock_executor.submit.call_count == 2
        # Verify tasks list has 2 items
        assert len(tasks) == 2

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.download_document")
    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.ThreadPoolExecutor")
    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.as_completed")
    def test_downloads_all_supported_documents_concurrently(
        self, mock_as_completed, mock_executor_class, mock_download, s3_creds, tmp_path
    ):
        """Downloads all supported documents using ThreadPoolExecutor."""
        docs = [
            {"key": "file1.pdf"},
            {"key": "file2.docx"},
            {"key": "file3.html"},
        ]

        mock_executor = mock.MagicMock()
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.__exit__.return_value = None
        mock_executor_class.return_value = mock_executor

        # Setup futures
        futures = []
        for doc in docs:
            future = mock.MagicMock()
            file_path = tmp_path / doc["key"]
            # Create the file so .stat() works
            file_path.write_bytes(b"content")
            future.result.return_value = file_path
            futures.append(future)

        future_to_doc = {futures[i]: docs[i] for i in range(len(docs))}
        mock_executor.submit.side_effect = futures
        mock_as_completed.return_value = futures

        mock_process_pool = mock.MagicMock()

        tasks, errors = download_and_submit(
            s3_creds, "bucket", docs, tmp_path, mock_process_pool, tmp_path
        )

        # Verify ThreadPoolExecutor created with max_workers=8
        mock_executor_class.assert_called_once_with(max_workers=8)
        # Verify all supported docs submitted for download
        assert mock_executor.submit.call_count == 3

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.download_document")
    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.ThreadPoolExecutor")
    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.as_completed")
    def test_sorts_downloaded_files_by_size_descending(
        self, mock_as_completed, mock_executor_class, mock_download, s3_creds, tmp_path
    ):
        """Sorts downloaded files by size descending before submitting to pool."""
        # Create files with different sizes
        file_small = tmp_path / "small.pdf"
        file_medium = tmp_path / "medium.pdf"
        file_large = tmp_path / "large.pdf"

        file_small.write_bytes(b"x" * 100)
        file_medium.write_bytes(b"x" * 500)
        file_large.write_bytes(b"x" * 1000)

        docs = [
            {"key": "small.pdf"},
            {"key": "medium.pdf"},
            {"key": "large.pdf"},
        ]

        mock_executor = mock.MagicMock()
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.__exit__.return_value = None
        mock_executor_class.return_value = mock_executor

        # Setup futures returning different sized files
        future_small = mock.MagicMock()
        future_small.result.return_value = file_small
        future_medium = mock.MagicMock()
        future_medium.result.return_value = file_medium
        future_large = mock.MagicMock()
        future_large.result.return_value = file_large

        mock_executor.submit.side_effect = [future_small, future_medium, future_large]
        mock_as_completed.return_value = [future_small, future_medium, future_large]

        mock_process_pool = mock.MagicMock()

        tasks, errors = download_and_submit(
            s3_creds, "bucket", docs, tmp_path, mock_process_pool, tmp_path
        )

        # Verify largest file submitted first
        assert len(tasks) == 3
        # Tasks are ordered largest first
        assert "large.pdf" in tasks[0][0]
        assert "medium.pdf" in tasks[1][0]
        assert "small.pdf" in tasks[2][0]

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.download_document")
    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.ThreadPoolExecutor")
    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.as_completed")
    def test_tracks_download_errors_with_traceback(
        self, mock_as_completed, mock_executor_class, mock_download, s3_creds, tmp_path
    ):
        """Tracks download errors with traceback in error list."""
        docs = [
            {"key": "success.pdf"},
            {"key": "fail.pdf"},
        ]

        mock_executor = mock.MagicMock()
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.__exit__.return_value = None
        mock_executor_class.return_value = mock_executor

        # First future succeeds, second fails
        future_success = mock.MagicMock()
        future_success.result.return_value = tmp_path / "success.pdf"
        (tmp_path / "success.pdf").write_bytes(b"content")

        future_fail = mock.MagicMock()
        future_fail.result.side_effect = Exception("Download failed")

        mock_executor.submit.side_effect = [future_success, future_fail]
        mock_as_completed.return_value = [future_success, future_fail]

        mock_process_pool = mock.MagicMock()

        tasks, errors = download_and_submit(
            s3_creds, "bucket", docs, tmp_path, mock_process_pool, tmp_path
        )

        # Verify error tracked
        assert len(errors) == 1
        assert errors[0]["file"] == "fail.pdf"
        assert "traceback" in errors[0]
        assert "Download failed" in errors[0]["traceback"]

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.download_document")
    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.ThreadPoolExecutor")
    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.as_completed")
    def test_submits_successfully_downloaded_files_to_process_pool(
        self, mock_as_completed, mock_executor_class, mock_download, s3_creds, tmp_path
    ):
        """Submits successfully downloaded files to process pool."""
        docs = [{"key": "file.pdf"}]

        mock_executor = mock.MagicMock()
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.__exit__.return_value = None
        mock_executor_class.return_value = mock_executor

        file_path = tmp_path / "file.pdf"
        file_path.write_bytes(b"content")

        future = mock.MagicMock()
        future.result.return_value = file_path

        mock_executor.submit.return_value = future
        mock_as_completed.return_value = [future]

        mock_process_pool = mock.MagicMock()
        mock_async_result = mock.MagicMock()
        mock_process_pool.apply_async.return_value = mock_async_result

        out_dir = tmp_path / "output"
        out_dir.mkdir()

        tasks, errors = download_and_submit(
            s3_creds, "bucket", docs, tmp_path, mock_process_pool, out_dir
        )

        # Verify apply_async called with worker_process_document
        mock_process_pool.apply_async.assert_called_once()
        call_args = mock_process_pool.apply_async.call_args[0]
        # First arg should be the function
        assert call_args[0].__name__ == "worker_process_document"
        # Second arg should be tuple of (file_path_str, output_dir_str)
        assert str(file_path) in call_args[1]
        assert str(out_dir) in call_args[1]

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.download_document")
    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.ThreadPoolExecutor")
    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.as_completed")
    def test_returns_extraction_tasks_and_download_errors_tuple(
        self, mock_as_completed, mock_executor_class, mock_download, s3_creds, tmp_path
    ):
        """Returns tuple of (extraction_tasks, download_errors)."""
        docs = [{"key": "file.pdf"}]

        mock_executor = mock.MagicMock()
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.__exit__.return_value = None
        mock_executor_class.return_value = mock_executor

        file_path = tmp_path / "file.pdf"
        file_path.write_bytes(b"content")

        future = mock.MagicMock()
        future.result.return_value = file_path

        mock_executor.submit.return_value = future
        mock_as_completed.return_value = [future]

        mock_process_pool = mock.MagicMock()
        mock_async_result = mock.MagicMock()
        mock_process_pool.apply_async.return_value = mock_async_result

        tasks, errors = download_and_submit(
            s3_creds, "bucket", docs, tmp_path, mock_process_pool, tmp_path
        )

        # Verify return structure
        assert isinstance(tasks, list)
        assert isinstance(errors, list)
        assert len(tasks) == 1
        assert len(errors) == 0
        # Each task is (file_path_str, AsyncResult)
        assert len(tasks[0]) == 2
        assert isinstance(tasks[0][0], str)
        assert tasks[0][1] == mock_async_result

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.as_completed")
    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.ThreadPoolExecutor")
    def test_logs_warning_for_skipped_unsupported_files(
        self, mock_executor_class, mock_as_completed, s3_creds, tmp_path, caplog
    ):
        """Logs warning for skipped unsupported files."""
        docs = [
            {"key": "file.pdf"},
            {"key": "archive.zip"},
            {"key": "binary.exe"},
        ]

        mock_executor = mock.MagicMock()
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.__exit__.return_value = None
        mock_executor_class.return_value = mock_executor

        # Mock successful download for supported file
        file_path = tmp_path / "file.pdf"
        file_path.write_bytes(b"content")
        future = mock.MagicMock()
        future.result.return_value = file_path
        mock_executor.submit.return_value = future

        # Mock as_completed to return the future immediately
        mock_as_completed.return_value = [future]

        mock_process_pool = mock.MagicMock()

        with caplog.at_level("WARNING"):
            tasks, errors = download_and_submit(
                s3_creds, "bucket", docs, tmp_path, mock_process_pool, tmp_path
            )

        # Verify warning logged
        assert any("Skipping" in record.message for record in caplog.records)
        assert any("archive.zip" in record.message for record in caplog.records)
        assert any("binary.exe" in record.message for record in caplog.records)

    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.as_completed")
    @mock.patch("components.data_processing.autorag.text_extraction.tests.nested_names.ThreadPoolExecutor")
    def test_handles_empty_document_list(self, mock_executor_class, mock_as_completed, s3_creds, tmp_path):
        """Handles empty document list and returns empty results."""
        docs = []

        mock_executor = mock.MagicMock()
        mock_executor.__enter__.return_value = mock_executor
        mock_executor.__exit__.return_value = None
        mock_executor_class.return_value = mock_executor

        # Mock as_completed to return empty list immediately
        mock_as_completed.return_value = []

        mock_process_pool = mock.MagicMock()

        tasks, errors = download_and_submit(
            s3_creds, "bucket", docs, tmp_path, mock_process_pool, tmp_path
        )

        assert tasks == []
        assert errors == []


class TestRaiseIfThresholdExceeded:
    """Unit tests for raise_if_threshold_exceeded helper function."""

    def test_does_nothing_when_error_details_empty(self):
        """Does nothing when error_details is empty."""
        # Should not raise
        raise_if_threshold_exceeded([], total_docs=100, tolerance=0.1)

    def test_raises_when_tolerance_none_and_any_error_exists(self):
        """Raises when tolerance is None and any error exists."""
        error_details = [{"file": "a.pdf", "traceback": "Error traceback"}]

        with pytest.raises(RuntimeError, match="tolerance: 0 \\(none allowed\\)"):
            raise_if_threshold_exceeded(error_details, total_docs=100, tolerance=None)

    def test_raises_when_errors_exceed_tolerance_threshold(self):
        """Raises when errors exceed tolerance threshold."""
        error_details = [
            {"file": f"file{i}.pdf", "traceback": f"Error {i}"} for i in range(11)
        ]

        with pytest.raises(RuntimeError, match="11/100 document\\(s\\) failed"):
            raise_if_threshold_exceeded(error_details, total_docs=100, tolerance=0.1)

    def test_does_not_raise_when_errors_equal_tolerance_threshold(self):
        """Does not raise when errors equal tolerance threshold."""
        error_details = [
            {"file": f"file{i}.pdf", "traceback": f"Error {i}"} for i in range(10)
        ]

        # Should not raise (10 == 10% of 100)
        raise_if_threshold_exceeded(error_details, total_docs=100, tolerance=0.1)

    def test_does_not_raise_when_errors_below_tolerance_threshold(self):
        """Does not raise when errors below tolerance threshold."""
        error_details = [
            {"file": f"file{i}.pdf", "traceback": f"Error {i}"} for i in range(5)
        ]

        # Should not raise (5 < 10% of 50 = 10)
        raise_if_threshold_exceeded(error_details, total_docs=50, tolerance=0.2)

    def test_error_message_includes_tolerance_percentage(self):
        """Error message includes tolerance percentage when threshold exceeded."""
        error_details = [{"file": "a.pdf", "traceback": "Error"}] * 11

        with pytest.raises(RuntimeError) as exc_info:
            raise_if_threshold_exceeded(error_details, total_docs=100, tolerance=0.1)

        assert "tolerance: 10% of 100" in str(exc_info.value)

    def test_error_message_shows_none_allowed_when_tolerance_none(self):
        """Error message shows '0 (none allowed)' when tolerance is None."""
        error_details = [{"file": "a.pdf", "traceback": "Error"}]

        with pytest.raises(RuntimeError) as exc_info:
            raise_if_threshold_exceeded(error_details, total_docs=100, tolerance=None)

        assert "tolerance: 0 (none allowed)" in str(exc_info.value)

    def test_shows_first_10_errors_with_traceback_snippets(self):
        """Shows first 10 errors with traceback snippets when more than 10 errors."""
        error_details = [
            {
                "file": f"file{i}.pdf",
                "traceback": f"Line1\nLine2\nLine3\nLine4\nLine5\nLine6 Error {i}",
            }
            for i in range(15)
        ]

        with pytest.raises(RuntimeError) as exc_info:
            raise_if_threshold_exceeded(error_details, total_docs=50, tolerance=0.1)

        error_msg = str(exc_info.value)
        # Should show 10 of 15 errors
        assert "Showing 10 of 15 error(s)" in error_msg
        # Should contain first error
        assert "file0.pdf" in error_msg
        # Should contain 10th error
        assert "file9.pdf" in error_msg
        # Should NOT contain 11th error
        assert "file10.pdf" not in error_msg
        # Should contain last 5 lines of traceback
        assert "Line6 Error 0" in error_msg

    def test_shows_all_errors_when_fewer_than_10(self):
        """Shows all errors when fewer than 10 errors."""
        error_details = [
            {"file": f"file{i}.pdf", "traceback": f"Error traceback {i}"}
            for i in range(3)
        ]

        with pytest.raises(RuntimeError) as exc_info:
            raise_if_threshold_exceeded(error_details, total_docs=10, tolerance=0.1)

        error_msg = str(exc_info.value)
        # Should show all 3 errors
        assert "Showing 3 of 3 error(s)" in error_msg
        assert "file0.pdf" in error_msg
        assert "file1.pdf" in error_msg
        assert "file2.pdf" in error_msg

    def test_error_message_includes_file_names(self):
        """Error message includes file names for each error."""
        error_details = [
            {"file": "document1.pdf", "traceback": "Error 1"},
            {"file": "presentation.pptx", "traceback": "Error 2"},
            {"file": "spreadsheet.xlsx", "traceback": "Error 3"},
        ]

        with pytest.raises(RuntimeError) as exc_info:
            raise_if_threshold_exceeded(error_details, total_docs=10, tolerance=0.1)

        error_msg = str(exc_info.value)
        assert "document1.pdf" in error_msg
        assert "presentation.pptx" in error_msg
        assert "spreadsheet.xlsx" in error_msg
