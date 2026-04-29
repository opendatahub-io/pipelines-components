"""Unit tests for nested helper functions from documents_discovery component."""

import json
from pathlib import Path
from unittest import mock

import pytest

from components.data_processing.autorag.documents_discovery.tests.nested_names import (
    _make_s3_client,
    get_test_data_docs_names,
)


class TestGetTestDataDocsNames:
    """Unit tests for get_test_data_docs_names helper function."""

    @pytest.fixture
    def fixtures_dir(self):
        """Return the path to the fixtures directory."""
        return Path(__file__).parent / "fixtures"

    def test_none_input_returns_empty_list(self):
        """None input returns empty list."""
        result = get_test_data_docs_names(None)
        assert result == []

    def test_valid_benchmark_single_question(self, fixtures_dir):
        """Valid benchmark with single question returns document IDs."""
        test_data = mock.MagicMock()
        test_data.path = str(fixtures_dir / "single_question.json")

        result = get_test_data_docs_names(test_data)

        assert result == ["doc1.pdf", "doc2.txt"]

    def test_valid_benchmark_multiple_questions(self, fixtures_dir):
        """Valid benchmark with multiple questions returns all document IDs."""
        test_data = mock.MagicMock()
        test_data.path = str(fixtures_dir / "multiple_questions.json")

        result = get_test_data_docs_names(test_data)

        assert result == ["doc1.pdf", "doc2.txt", "doc3.pdf", "doc4.docx", "doc5.md", "doc6.html"]

    def test_empty_benchmark_array(self, fixtures_dir):
        """Empty benchmark array returns empty list."""
        test_data = mock.MagicMock()
        test_data.path = str(fixtures_dir / "empty_benchmark.json")

        result = get_test_data_docs_names(test_data)

        assert result == []

    def test_question_with_no_document_ids(self, fixtures_dir):
        """Question with empty correct_answer_document_ids array returns empty list."""
        test_data = mock.MagicMock()
        test_data.path = str(fixtures_dir / "no_document_ids.json")

        result = get_test_data_docs_names(test_data)

        assert result == []

    def test_duplicate_document_ids_preserved(self, fixtures_dir):
        """Duplicate document IDs across questions are preserved in result."""
        test_data = mock.MagicMock()
        test_data.path = str(fixtures_dir / "duplicates.json")

        result = get_test_data_docs_names(test_data)

        assert result == ["doc1.pdf", "doc2.txt", "doc1.pdf", "doc3.pdf"]
        assert result.count("doc1.pdf") == 2

    def test_file_not_found_error(self):
        """Invalid file path raises FileNotFoundError."""
        test_data = mock.MagicMock()
        test_data.path = "/nonexistent/path/to/file.json"

        with pytest.raises(FileNotFoundError):
            get_test_data_docs_names(test_data)

    def test_invalid_json_format(self, fixtures_dir):
        """Malformed JSON file raises JSONDecodeError."""
        test_data = mock.MagicMock()
        test_data.path = str(fixtures_dir / "malformed.json")

        with pytest.raises(json.JSONDecodeError):
            get_test_data_docs_names(test_data)

    def test_missing_required_field_in_json(self, fixtures_dir):
        """Missing correct_answer_document_ids field raises KeyError."""
        test_data = mock.MagicMock()
        test_data.path = str(fixtures_dir / "missing_field.json")

        with pytest.raises(KeyError):
            get_test_data_docs_names(test_data)


class TestMakeS3Client:
    """Unit tests for _make_s3_client helper function."""

    @mock.patch("components.data_processing.autorag.documents_discovery.tests.nested_names.boto3")
    def test_creates_client_with_verify_true(self, mock_boto3):
        """Creates S3 client with verify=True by default."""
        mock_s3_client = mock.MagicMock()
        mock_boto3.client.return_value = mock_s3_client

        s3_creds = {
            "AWS_ACCESS_KEY_ID": "test_key",
            "AWS_SECRET_ACCESS_KEY": "test_secret",
            "AWS_S3_ENDPOINT": "https://s3.example.com",
            "AWS_DEFAULT_REGION": "us-east-1",
        }

        result = _make_s3_client(s3_creds, verify=True)

        mock_boto3.client.assert_called_once_with(
            "s3",
            endpoint_url="https://s3.example.com",
            region_name="us-east-1",
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
            verify=True,
        )
        assert result == mock_s3_client

    @mock.patch("components.data_processing.autorag.documents_discovery.tests.nested_names.boto3")
    def test_creates_client_with_verify_false(self, mock_boto3):
        """Creates S3 client with verify=False when specified."""
        mock_s3_client = mock.MagicMock()
        mock_boto3.client.return_value = mock_s3_client

        s3_creds = {
            "AWS_ACCESS_KEY_ID": "test_key",
            "AWS_SECRET_ACCESS_KEY": "test_secret",
            "AWS_S3_ENDPOINT": "https://s3.example.com",
            "AWS_DEFAULT_REGION": "us-east-1",
        }

        result = _make_s3_client(s3_creds, verify=False)

        call_kwargs = mock_boto3.client.call_args[1]
        assert call_kwargs["verify"] is False
        assert result == mock_s3_client

    @mock.patch("components.data_processing.autorag.documents_discovery.tests.nested_names.boto3")
    def test_all_credential_fields_used(self, mock_boto3):
        """All credential fields are passed to boto3.client."""
        mock_s3_client = mock.MagicMock()
        mock_boto3.client.return_value = mock_s3_client

        s3_creds = {
            "AWS_ACCESS_KEY_ID": "my_access_key",
            "AWS_SECRET_ACCESS_KEY": "my_secret_key",
            "AWS_S3_ENDPOINT": "https://custom.s3.endpoint.com",
            "AWS_DEFAULT_REGION": "eu-west-1",
        }

        _make_s3_client(s3_creds)

        call_kwargs = mock_boto3.client.call_args[1]
        assert call_kwargs["aws_access_key_id"] == "my_access_key"
        assert call_kwargs["aws_secret_access_key"] == "my_secret_key"
        assert call_kwargs["endpoint_url"] == "https://custom.s3.endpoint.com"
        assert call_kwargs["region_name"] == "eu-west-1"

    @mock.patch("components.data_processing.autorag.documents_discovery.tests.nested_names.boto3")
    def test_missing_region_handled_gracefully(self, mock_boto3):
        """Missing AWS_DEFAULT_REGION is handled gracefully."""
        mock_s3_client = mock.MagicMock()
        mock_boto3.client.return_value = mock_s3_client

        s3_creds = {
            "AWS_ACCESS_KEY_ID": "test_key",
            "AWS_SECRET_ACCESS_KEY": "test_secret",
            "AWS_S3_ENDPOINT": "https://s3.example.com",
        }

        result = _make_s3_client(s3_creds)

        call_kwargs = mock_boto3.client.call_args[1]
        assert call_kwargs["region_name"] is None
        assert result == mock_s3_client

    def test_missing_required_credential_raises_error(self):
        """Missing required credential field raises KeyError."""
        s3_creds = {
            "AWS_SECRET_ACCESS_KEY": "test_secret",
            "AWS_S3_ENDPOINT": "https://s3.example.com",
        }

        with pytest.raises(KeyError, match="AWS_ACCESS_KEY_ID"):
            _make_s3_client(s3_creds)

    @mock.patch("components.data_processing.autorag.documents_discovery.tests.nested_names.boto3")
    def test_empty_endpoint_url(self, mock_boto3):
        """Empty endpoint URL is passed through to boto3."""
        mock_s3_client = mock.MagicMock()
        mock_boto3.client.return_value = mock_s3_client

        s3_creds = {
            "AWS_ACCESS_KEY_ID": "test_key",
            "AWS_SECRET_ACCESS_KEY": "test_secret",
            "AWS_S3_ENDPOINT": "",
            "AWS_DEFAULT_REGION": "us-east-1",
        }

        _make_s3_client(s3_creds)

        call_kwargs = mock_boto3.client.call_args[1]
        assert call_kwargs["endpoint_url"] == ""

    @mock.patch("components.data_processing.autorag.documents_discovery.tests.nested_names.boto3")
    def test_returns_s3_client_object(self, mock_boto3):
        """Returns boto3 client object with S3 interface."""
        mock_s3_client = mock.MagicMock()
        mock_s3_client.list_objects_v2 = mock.MagicMock()
        mock_boto3.client.return_value = mock_s3_client

        s3_creds = {
            "AWS_ACCESS_KEY_ID": "test_key",
            "AWS_SECRET_ACCESS_KEY": "test_secret",
            "AWS_S3_ENDPOINT": "https://s3.example.com",
            "AWS_DEFAULT_REGION": "us-east-1",
        }

        result = _make_s3_client(s3_creds)

        assert hasattr(result, "list_objects_v2")
        assert result == mock_s3_client
