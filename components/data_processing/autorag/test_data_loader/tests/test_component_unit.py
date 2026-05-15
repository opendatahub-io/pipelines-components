"""Tests for the test_data_loader component."""

import inspect
import json
import sys
from unittest import mock

import pytest

from ..component import test_data_loader

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


class _MockClientError(Exception):
    """Stand-in for botocore.exceptions.ClientError used in unit tests."""

    def __init__(self, message="", response=None):
        super().__init__(message)
        self.response = response or {}


def _mock_botocore_modules():
    """Return mocked botocore and botocore.exceptions modules."""
    mock_botocore = mock.MagicMock()
    mock_botocore_exceptions = mock.MagicMock()
    mock_botocore_exceptions.SSLError = _MockSSLError
    mock_botocore_exceptions.ClientError = _MockClientError
    mock_botocore.exceptions = mock_botocore_exceptions
    return mock_botocore, mock_botocore_exceptions


class TestTestDataLoaderUnitTests:
    """Unit tests for test_data_loader success and error handling."""

    def test_component_function_exists(self):
        """Component factory exists and exposes python_func."""
        assert callable(test_data_loader)
        assert hasattr(test_data_loader, "python_func")

    def test_component_with_default_parameters(self):
        """Component has expected required parameters in interface."""
        sig = inspect.signature(test_data_loader.python_func)
        params = list(sig.parameters)
        assert "test_data_bucket_name" in params
        assert "test_data_path" in params

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_successful_download_and_json_validation(self, tmp_path):
        """Successful download with valid JSON returns without raising."""
        mock_boto3 = mock.MagicMock()
        mock_s3 = mock.MagicMock()

        out_path = tmp_path / "test_data.json"

        def _write_valid_json(_bucket, _key, destination):
            with open(destination, "w", encoding="utf-8") as f:
                json.dump({"dataset": "ok"}, f)

        mock_s3.download_file.side_effect = _write_valid_json
        mock_boto3.client.return_value = mock_s3

        mock_botocore, mock_botocore_exceptions = _mock_botocore_modules()
        test_data_artifact = mock.MagicMock()
        test_data_artifact.path = str(out_path)

        with mock.patch.dict(
            sys.modules,
            {
                "boto3": mock_boto3,
                "botocore": mock_botocore,
                "botocore.exceptions": mock_botocore_exceptions,
            },
        ):
            test_data_loader.python_func(
                test_data_bucket_name="my-bucket",
                test_data_path="data/test.json",
                test_data=test_data_artifact,
            )

        assert out_path.exists()
        assert json.loads(out_path.read_text(encoding="utf-8"))["dataset"] == "ok"
        mock_s3.download_file.assert_called_once()

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES_NO_REGION, clear=True)
    def test_missing_region_is_allowed(self, tmp_path):
        """Component works when AWS_DEFAULT_REGION is not present."""
        mock_boto3 = mock.MagicMock()
        mock_s3 = mock.MagicMock()

        out_path = tmp_path / "test_data.json"

        def _write_valid_json(_bucket, _key, destination):
            with open(destination, "w", encoding="utf-8") as f:
                json.dump({"dataset": "ok"}, f)

        mock_s3.download_file.side_effect = _write_valid_json
        mock_boto3.client.return_value = mock_s3

        mock_botocore, mock_botocore_exceptions = _mock_botocore_modules()
        test_data_artifact = mock.MagicMock()
        test_data_artifact.path = str(out_path)

        with mock.patch.dict(
            sys.modules,
            {
                "boto3": mock_boto3,
                "botocore": mock_botocore,
                "botocore.exceptions": mock_botocore_exceptions,
            },
        ):
            test_data_loader.python_func(
                test_data_bucket_name="my-bucket",
                test_data_path="data/test.json",
                test_data=test_data_artifact,
            )

        client_kwargs = mock_boto3.client.call_args.kwargs
        assert client_kwargs["region_name"] is None
        assert out_path.exists()

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_ssl_error_retries_with_verify_false(self, tmp_path):
        """SSLError on first download triggers retry with verify=False."""
        mock_boto3 = mock.MagicMock()
        mock_s3_fail = mock.MagicMock()
        mock_s3_ok = mock.MagicMock()

        out_path = tmp_path / "test_data.json"

        mock_s3_fail.download_file.side_effect = _MockSSLError("SSL validation failed")

        def _write_valid_json(_bucket, _key, destination):
            with open(destination, "w", encoding="utf-8") as f:
                json.dump({"retried": True}, f)

        mock_s3_ok.download_file.side_effect = _write_valid_json

        call_count = 0

        def fake_client(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_s3_fail
            return mock_s3_ok

        mock_boto3.client.side_effect = fake_client
        mock_botocore, mock_botocore_exceptions = _mock_botocore_modules()

        test_data_artifact = mock.MagicMock()
        test_data_artifact.path = str(out_path)

        with mock.patch.dict(
            sys.modules,
            {
                "boto3": mock_boto3,
                "botocore": mock_botocore,
                "botocore.exceptions": mock_botocore_exceptions,
            },
        ):
            test_data_loader.python_func(
                test_data_bucket_name="my-bucket",
                test_data_path="data/test.json",
                test_data=test_data_artifact,
            )

        assert call_count == 2
        second_call_kwargs = mock_boto3.client.call_args_list[1][1]
        assert second_call_kwargs["verify"] is False
        assert json.loads(out_path.read_text(encoding="utf-8"))["retried"] is True

    def test_empty_bucket_name_raises_type_error(self, tmp_path):
        """Empty test_data_bucket_name raises TypeError."""
        mock_boto3 = mock.MagicMock()
        mock_botocore, mock_botocore_exceptions = _mock_botocore_modules()
        artifact = mock.MagicMock()
        artifact.path = str(tmp_path / "test_data.json")

        with mock.patch.dict(
            sys.modules,
            {
                "boto3": mock_boto3,
                "botocore": mock_botocore,
                "botocore.exceptions": mock_botocore_exceptions,
            },
        ):
            with pytest.raises(TypeError, match="test_data_bucket_name must be a non-empty string"):
                test_data_loader.python_func(
                    test_data_bucket_name="",
                    test_data_path="data/test.json",
                    test_data=artifact,
                )

    @mock.patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "x"}, clear=True)
    def test_missing_required_env_variable_raises_value_error(self, tmp_path):
        """Missing S3 env var raises ValueError with variable name."""
        mock_boto3 = mock.MagicMock()
        mock_botocore, mock_botocore_exceptions = _mock_botocore_modules()
        artifact = mock.MagicMock()
        artifact.path = str(tmp_path / "test_data.json")

        with mock.patch.dict(
            sys.modules,
            {
                "boto3": mock_boto3,
                "botocore": mock_botocore,
                "botocore.exceptions": mock_botocore_exceptions,
            },
        ):
            with pytest.raises(ValueError, match="AWS_SECRET_ACCESS_KEY environment variable not set"):
                test_data_loader.python_func(
                    test_data_bucket_name="my-bucket",
                    test_data_path="data/test.json",
                    test_data=artifact,
                )

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_missing_s3_object_raises_file_not_found_error(self, tmp_path):
        """S3 404/NoSuchKey maps to FileNotFoundError with key/bucket context."""
        mock_boto3 = mock.MagicMock()
        mock_s3 = mock.MagicMock()
        mock_s3.download_file.side_effect = _MockClientError(
            "not found",
            response={"Error": {"Code": "NoSuchKey"}},
        )
        mock_boto3.client.return_value = mock_s3
        mock_botocore, mock_botocore_exceptions = _mock_botocore_modules()

        artifact = mock.MagicMock()
        artifact.path = str(tmp_path / "test_data.json")

        with mock.patch.dict(
            sys.modules,
            {
                "boto3": mock_boto3,
                "botocore": mock_botocore,
                "botocore.exceptions": mock_botocore_exceptions,
            },
        ):
            with pytest.raises(FileNotFoundError, match="Test data object not found in S3"):
                test_data_loader.python_func(
                    test_data_bucket_name="my-bucket",
                    test_data_path="missing/test.json",
                    test_data=artifact,
                )

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_invalid_downloaded_json_raises_exception(self, tmp_path):
        """Downloaded non-JSON content raises component exception."""
        mock_boto3 = mock.MagicMock()
        mock_s3 = mock.MagicMock()

        def _write_invalid_json(_bucket, _key, destination):
            with open(destination, "w", encoding="utf-8") as f:
                f.write("not-json")

        mock_s3.download_file.side_effect = _write_invalid_json
        mock_boto3.client.return_value = mock_s3
        mock_botocore, mock_botocore_exceptions = _mock_botocore_modules()

        artifact = mock.MagicMock()
        artifact.path = str(tmp_path / "test_data.json")

        with mock.patch.dict(
            sys.modules,
            {
                "boto3": mock_boto3,
                "botocore": mock_botocore,
                "botocore.exceptions": mock_botocore_exceptions,
            },
        ):
            with pytest.raises(Exception, match="test_data_path must point to a valid JSON file"):
                test_data_loader.python_func(
                    test_data_bucket_name="my-bucket",
                    test_data_path="data/not_json.txt",
                    test_data=artifact,
                )

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_unexpected_download_error_is_wrapped(self, tmp_path):
        """Unexpected download exceptions are wrapped by component exception."""
        mock_boto3 = mock.MagicMock()
        mock_s3 = mock.MagicMock()
        mock_s3.download_file.side_effect = RuntimeError("network issue")
        mock_boto3.client.return_value = mock_s3
        mock_botocore, mock_botocore_exceptions = _mock_botocore_modules()

        artifact = mock.MagicMock()
        artifact.path = str(tmp_path / "test_data.json")

        with mock.patch.dict(
            sys.modules,
            {
                "boto3": mock_boto3,
                "botocore": mock_botocore,
                "botocore.exceptions": mock_botocore_exceptions,
            },
        ):
            with pytest.raises(Exception, match="Failed to fetch"):
                test_data_loader.python_func(
                    test_data_bucket_name="my-bucket",
                    test_data_path="data/test.json",
                    test_data=artifact,
                )
