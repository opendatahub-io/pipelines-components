"""Tests for the documents_discovery component."""

import inspect
import json
import sys
from unittest import mock

import pytest

from ..component import documents_discovery

MOCKED_ENV_VARIABLES = {
    "AWS_ACCESS_KEY_ID": "test_key",
    "AWS_SECRET_ACCESS_KEY": "test_secret",
    "AWS_S3_ENDPOINT": "https://s3.example.com",
    "AWS_DEFAULT_REGION": "us-east-1",
}
MOCKED_ENV_VARIABLES_NO_REGION = {
    "AWS_ACCESS_KEY_ID": "test_key",
    "AWS_SECRET_ACCESS_KEY": "test_secret",
    "AWS_S3_ENDPOINT": "https://s3.example.com",
}


class _MockSSLError(Exception):
    """Stand-in for botocore.exceptions.SSLError used in unit tests."""


def _make_botocore_modules():
    mock_botocore = mock.MagicMock()
    mock_botocore_exceptions = mock.MagicMock()
    mock_botocore_exceptions.SSLError = _MockSSLError
    mock_botocore.exceptions = mock_botocore_exceptions
    return mock_botocore, mock_botocore_exceptions


class TestDocumentsDiscoveryUnitTests:
    """Unit tests for documents_discovery success and error handling."""

    def test_component_function_exists(self):
        """Component factory exists and exposes python_func."""
        assert callable(documents_discovery)
        assert hasattr(documents_discovery, "python_func")

    def test_component_with_default_parameters(self):
        """Component has expected required interface."""
        sig = inspect.signature(documents_discovery.python_func)
        params = list(sig.parameters)
        assert "input_data_bucket_name" in params
        assert "input_data_path" in params

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_success_writes_descriptor(self, tmp_path):
        """Supported S3 objects produce a descriptor JSON with expected keys."""
        mock_boto3 = mock.MagicMock()
        mock_s3 = mock.MagicMock()
        mock_s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "docs/a.pdf", "Size": 1000},
                {"Key": "docs/b.txt", "Size": 2000},
                {"Key": "docs/ignore.csv", "Size": 3000},
            ]
        }
        mock_boto3.client.return_value = mock_s3
        mock_botocore, mock_botocore_exceptions = _make_botocore_modules()

        discovered = mock.MagicMock()
        discovered.path = str(tmp_path / "descriptor")

        with mock.patch.dict(
            sys.modules,
            {
                "boto3": mock_boto3,
                "botocore": mock_botocore,
                "botocore.exceptions": mock_botocore_exceptions,
            },
        ):
            documents_discovery.python_func(
                input_data_bucket_name="my-bucket",
                input_data_path="docs/",
                discovered_documents=discovered,
                sampling_enabled=False,
            )

        descriptor_file = tmp_path / "descriptor" / "documents_descriptor.json"
        assert descriptor_file.exists()
        payload = json.loads(descriptor_file.read_text(encoding="utf-8"))
        assert payload["bucket"] == "my-bucket"
        assert payload["count"] == 2
        assert payload["total_size_bytes"] == 3000

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES_NO_REGION, clear=True)
    def test_missing_region_is_allowed(self, tmp_path):
        """Component works when AWS_DEFAULT_REGION is not present."""
        mock_boto3 = mock.MagicMock()
        mock_s3 = mock.MagicMock()
        mock_s3.list_objects_v2.return_value = {"Contents": [{"Key": "docs/a.pdf", "Size": 1000}]}
        mock_boto3.client.return_value = mock_s3
        mock_botocore, mock_botocore_exceptions = _make_botocore_modules()

        discovered = mock.MagicMock()
        discovered.path = str(tmp_path / "descriptor")

        with mock.patch.dict(
            sys.modules,
            {
                "boto3": mock_boto3,
                "botocore": mock_botocore,
                "botocore.exceptions": mock_botocore_exceptions,
            },
        ):
            documents_discovery.python_func(
                input_data_bucket_name="my-bucket",
                input_data_path="docs/",
                discovered_documents=discovered,
                sampling_enabled=False,
            )

        client_kwargs = mock_boto3.client.call_args.kwargs
        assert client_kwargs["region_name"] is None
        descriptor_file = tmp_path / "descriptor" / "documents_descriptor.json"
        assert descriptor_file.exists()

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_ssl_error_retries_with_verify_false(self, tmp_path):
        """SSLError on list_objects_v2 triggers a retry with verify=False."""
        mock_boto3 = mock.MagicMock()
        mock_s3_ok = mock.MagicMock()
        mock_s3_fail = mock.MagicMock()

        mock_s3_fail.list_objects_v2.side_effect = _MockSSLError("SSL validation failed")
        mock_s3_ok.list_objects_v2.return_value = {"Contents": [{"Key": "docs/file1.pdf", "Size": 1000}]}

        call_count = 0

        def fake_client(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_s3_fail
            return mock_s3_ok

        mock_boto3.client.side_effect = fake_client
        mock_botocore, mock_botocore_exceptions = _make_botocore_modules()

        discovered = mock.MagicMock()
        discovered.path = str(tmp_path / "output")

        with mock.patch.dict(
            sys.modules,
            {
                "boto3": mock_boto3,
                "botocore": mock_botocore,
                "botocore.exceptions": mock_botocore_exceptions,
            },
        ):
            documents_discovery.python_func(
                input_data_bucket_name="my-bucket",
                input_data_path="docs/",
                discovered_documents=discovered,
            )

        assert call_count == 2
        second_call_kwargs = mock_boto3.client.call_args_list[1][1]
        assert second_call_kwargs["verify"] is False

    @mock.patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "x"}, clear=True)
    def test_missing_env_raises_value_error(self, tmp_path):
        """Missing required S3 env variable raises ValueError with variable name."""
        mock_boto3 = mock.MagicMock()
        mock_botocore, mock_botocore_exceptions = _make_botocore_modules()
        discovered = mock.MagicMock()
        discovered.path = str(tmp_path / "output")

        with mock.patch.dict(
            sys.modules,
            {
                "boto3": mock_boto3,
                "botocore": mock_botocore,
                "botocore.exceptions": mock_botocore_exceptions,
            },
        ):
            with pytest.raises(ValueError, match="AWS_SECRET_ACCESS_KEY environment variable not set"):
                documents_discovery.python_func(
                    input_data_bucket_name="my-bucket",
                    input_data_path="docs/",
                    discovered_documents=discovered,
                )

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_no_supported_documents_raises_exception(self, tmp_path):
        """No supported extension in S3 listing raises a component exception."""
        mock_boto3 = mock.MagicMock()
        mock_s3 = mock.MagicMock()
        mock_s3.list_objects_v2.return_value = {"Contents": [{"Key": "docs/file.csv", "Size": 10}]}
        mock_boto3.client.return_value = mock_s3
        mock_botocore, mock_botocore_exceptions = _make_botocore_modules()
        discovered = mock.MagicMock()
        discovered.path = str(tmp_path / "output")

        with mock.patch.dict(
            sys.modules,
            {
                "boto3": mock_boto3,
                "botocore": mock_botocore,
                "botocore.exceptions": mock_botocore_exceptions,
            },
        ):
            with pytest.raises(Exception, match="No supported documents found"):
                documents_discovery.python_func(
                    input_data_bucket_name="my-bucket",
                    input_data_path="docs/",
                    discovered_documents=discovered,
                )
