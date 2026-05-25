"""Tests for the documents_discovery component."""

import inspect
import json
import sys
from unittest import mock

import pytest

from ..component import documents_discovery

_GiB = 1024**3
_MiB = 1024**2

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


class TestDocumentsDiscoverySampling:
    """Unit tests for the sampling and test-data-priority logic in documents_discovery."""

    def _run(
        self,
        tmp_path,
        contents,
        test_data_json=None,
        sampling_enabled=True,
        sampling_max_size=1.0,
    ):
        """Run the component with mocked S3 and return the parsed descriptor dict."""
        mock_boto3 = mock.MagicMock()
        mock_s3 = mock.MagicMock()
        mock_s3.list_objects_v2.return_value = {"Contents": contents}
        mock_boto3.client.return_value = mock_s3
        mock_botocore, mock_botocore_exceptions = _make_botocore_modules()

        discovered = mock.MagicMock()
        discovered.path = str(tmp_path / "descriptor")

        test_data_artifact = None
        if test_data_json is not None:
            td_file = tmp_path / "test_data.json"
            td_file.write_text(json.dumps(test_data_json), encoding="utf-8")
            test_data_artifact = mock.MagicMock()
            test_data_artifact.path = str(td_file)

        with mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True):
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
                    test_data=test_data_artifact,
                    sampling_enabled=sampling_enabled,
                    sampling_max_size=sampling_max_size,
                    discovered_documents=discovered,
                )

        descriptor_file = tmp_path / "descriptor" / "documents_descriptor.json"
        return json.loads(descriptor_file.read_text(encoding="utf-8"))

    def test_sampling_disabled_includes_all_supported_docs(self, tmp_path):
        """With sampling disabled every supported doc is selected regardless of total size."""
        contents = [
            {"Key": "docs/a.pdf", "Size": 5 * _GiB},
            {"Key": "docs/b.txt", "Size": 5 * _GiB},
            {"Key": "docs/c.docx", "Size": 5 * _GiB},
        ]
        payload = self._run(tmp_path, contents, sampling_enabled=False)
        assert payload["count"] == 3

    def test_sampling_limits_total_size(self, tmp_path):
        """Docs are accumulated until adding the next one would exceed the size cap."""
        contents = [
            {"Key": "docs/a.pdf", "Size": 400 * _MiB},
            {"Key": "docs/b.pdf", "Size": 400 * _MiB},
            {"Key": "docs/c.pdf", "Size": 400 * _MiB},
        ]
        payload = self._run(tmp_path, contents, sampling_max_size=1.0)
        assert payload["count"] == 2
        assert payload["total_size_bytes"] == 800 * _MiB

    def test_doc_exactly_at_size_limit_is_included(self, tmp_path):
        """A document whose size exactly equals the cap is selected."""
        contents = [{"Key": "docs/a.pdf", "Size": _GiB}]
        payload = self._run(tmp_path, contents, sampling_max_size=1.0)
        assert payload["count"] == 1

    def test_all_docs_exceed_size_limit_raises_value_error(self, tmp_path):
        """ValueError is raised when every individual doc is larger than the cap."""
        contents = [
            {"Key": "docs/a.pdf", "Size": 2 * _GiB},
            {"Key": "docs/b.pdf", "Size": 2 * _GiB},
        ]
        with pytest.raises(ValueError, match="No documents to process"):
            self._run(tmp_path, contents, sampling_max_size=1.0)

    def test_test_data_priority_flat_keys(self, tmp_path):
        """Docs named in test data are selected first when S3 keys have no prefix."""
        contents = [
            {"Key": "irrelevant.pdf", "Size": 600 * _MiB},
            {"Key": "important.txt", "Size": 600 * _MiB},
        ]
        test_data = [{"question": "q", "correct_answer_document_ids": ["important.txt"]}]
        payload = self._run(tmp_path, contents, test_data_json=test_data, sampling_max_size=1.0)
        assert payload["count"] == 1
        assert payload["documents"][0]["key"] == "important.txt"

    def test_test_data_priority_prefixed_keys(self, tmp_path):
        """Docs named in test data are selected first when S3 keys include a folder prefix."""
        contents = [
            {"Key": "docs/irrelevant.pdf", "Size": 600 * _MiB},
            {"Key": "docs/important.txt", "Size": 600 * _MiB},
        ]
        test_data = [{"question": "q", "correct_answer_document_ids": ["important.txt"]}]
        payload = self._run(tmp_path, contents, test_data_json=test_data, sampling_max_size=1.0)
        assert payload["count"] == 1
        assert payload["documents"][0]["key"] == "docs/important.txt"

    def test_test_data_priority_deeply_nested_prefix(self, tmp_path):
        """Priority matching works for multi-level S3 key prefixes."""
        contents = [
            {"Key": "a/b/c/irrelevant.pdf", "Size": 600 * _MiB},
            {"Key": "a/b/c/important.txt", "Size": 600 * _MiB},
        ]
        test_data = [{"question": "q", "correct_answer_document_ids": ["important.txt"]}]
        payload = self._run(tmp_path, contents, test_data_json=test_data, sampling_max_size=1.0)
        assert payload["count"] == 1
        assert payload["documents"][0]["key"] == "a/b/c/important.txt"

    def test_no_test_data_selects_all_within_limit(self, tmp_path):
        """When test_data is None no priority sort is applied; docs within limit are selected."""
        contents = [
            {"Key": "docs/a.pdf", "Size": 400 * _MiB},
            {"Key": "docs/b.pdf", "Size": 400 * _MiB},
        ]
        payload = self._run(tmp_path, contents, test_data_json=None, sampling_max_size=1.0)
        assert payload["count"] == 2

    def test_empty_benchmark_no_priority_sorting(self, tmp_path):
        """An empty benchmark list is treated the same as no test data."""
        contents = [
            {"Key": "docs/a.pdf", "Size": 400 * _MiB},
            {"Key": "docs/b.pdf", "Size": 400 * _MiB},
        ]
        payload = self._run(tmp_path, contents, test_data_json=[], sampling_max_size=1.0)
        assert payload["count"] == 2

    def test_referenced_doc_absent_from_s3_does_not_raise(self, tmp_path):
        """Test data referencing a doc not present in S3 is silently ignored."""
        contents = [{"Key": "docs/a.pdf", "Size": 100}]
        test_data = [{"question": "q", "correct_answer_document_ids": ["ghost.txt"]}]
        payload = self._run(tmp_path, contents, test_data_json=test_data)
        assert payload["count"] == 1
        assert payload["documents"][0]["key"] == "docs/a.pdf"

    def test_doc_referenced_by_multiple_questions_selected_once(self, tmp_path):
        """A doc cited in several questions appears exactly once in the output."""
        contents = [
            {"Key": "docs/irrelevant.pdf", "Size": 600 * _MiB},
            {"Key": "docs/shared.txt", "Size": 600 * _MiB},
        ]
        test_data = [
            {"question": "q1", "correct_answer_document_ids": ["shared.txt"]},
            {"question": "q2", "correct_answer_document_ids": ["shared.txt"]},
        ]
        payload = self._run(tmp_path, contents, test_data_json=test_data, sampling_max_size=1.0)
        assert payload["count"] == 1
        assert payload["documents"][0]["key"] == "docs/shared.txt"

    def test_multiple_referenced_docs_all_prioritised(self, tmp_path):
        """All docs named across test data questions are sorted ahead of non-referenced docs."""
        contents = [
            {"Key": "docs/filler1.pdf", "Size": 100 * _MiB},
            {"Key": "docs/filler2.pdf", "Size": 100 * _MiB},
            {"Key": "docs/ref_a.txt", "Size": 100 * _MiB},
            {"Key": "docs/ref_b.docx", "Size": 100 * _MiB},
        ]
        test_data = [
            {"question": "q1", "correct_answer_document_ids": ["ref_a.txt"]},
            {"question": "q2", "correct_answer_document_ids": ["ref_b.docx"]},
        ]
        # Cap allows 3 docs; referenced docs should occupy the first 2 slots
        payload = self._run(tmp_path, contents, test_data_json=test_data, sampling_max_size=300 / 1024)
        selected_keys = {d["key"] for d in payload["documents"]}
        assert "docs/ref_a.txt" in selected_keys
        assert "docs/ref_b.docx" in selected_keys
