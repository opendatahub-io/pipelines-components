"""Tests for the documents_discovery thin wrapper component."""

import inspect
import json
from types import SimpleNamespace
from unittest import mock

import pytest

from ..component import documents_discovery

VALID_BENCHMARK_RECORDS = [
    {"question": "What is X?", "correct_answers": ["Answer X"], "correct_answer_document_keys": ["doc_a.pdf"]},
    {
        "question": "What is Y?",
        "correct_answers": ["Answer Y"],
        "correct_answer_document_keys": ["doc_a.pdf", "doc_b.txt"],
    },
]


def _make_ai4rag_mocks(include_test_data_loader=False):
    """Build mock modules matching the component's imports.

    ``discover_documents`` and ``load_test_data`` are both imported from the
    ``ai4rag.utils.data`` package, while ``create_s3_client`` comes from
    ``ai4rag.utils.clients.s3``.
    """
    mock_create_s3_client = mock.MagicMock(name="create_s3_client")
    mock_discover_documents = mock.MagicMock(name="discover_documents")

    mock_s3_module = mock.MagicMock()
    mock_s3_module.create_s3_client = mock_create_s3_client

    mock_data_module = mock.MagicMock()
    mock_data_module.discover_documents = mock_discover_documents

    modules = {
        "ai4rag": mock.MagicMock(),
        "ai4rag.utils": mock.MagicMock(),
        "ai4rag.utils.data": mock_data_module,
        "ai4rag.utils.clients": mock.MagicMock(),
        "ai4rag.utils.clients.s3": mock_s3_module,
    }

    mock_load_test_data = None
    if include_test_data_loader:
        mock_load_test_data = mock.MagicMock(name="load_test_data")
        mock_data_module.load_test_data = mock_load_test_data

    return modules, mock_create_s3_client, mock_discover_documents, mock_load_test_data


class TestDocumentsDiscoveryUnitTests:
    """Unit tests for the documents_discovery wrapper."""

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
        assert "test_data_bucket_name" in params
        assert "test_data_path_key" in params
        assert "benchmark_sample_size" in params
        assert sig.parameters["test_data_bucket_name"].default == ""
        assert sig.parameters["test_data_path_key"].default == ""
        assert sig.parameters["benchmark_sample_size"].default == 25

    def test_delegates_to_ai4rag_discover_documents(self, tmp_path):
        """Wrapper calls create_s3_client and discover_documents with correct args."""
        modules, mock_create_s3, mock_discover, _ = _make_ai4rag_mocks()
        mock_s3_client = mock.MagicMock(name="s3_client_instance")
        mock_create_s3.return_value = mock_s3_client
        mock_result = mock.MagicMock()
        mock_discover.return_value = mock_result

        discovered = mock.MagicMock()
        discovered.path = str(tmp_path / "descriptor")

        with mock.patch.dict("sys.modules", modules):
            documents_discovery.python_func(
                input_data_bucket_name="my-bucket",
                input_data_path="docs/",
                sampling_enabled=True,
                sampling_max_size=2.5,
                discovered_documents=discovered,
            )

        mock_create_s3.assert_called_once()
        mock_discover.assert_called_once_with(
            bucket_name="my-bucket",
            prefix="docs/",
            test_data_doc_names=None,
            sampling_enabled=True,
            sampling_max_size_gb=2.5,
            s3_client=mock_s3_client,
        )

    def test_saves_result_to_artifact_path(self, tmp_path):
        """DiscoveryResult.save is called with the correct output path."""
        modules, mock_create_s3, mock_discover, _ = _make_ai4rag_mocks()
        mock_create_s3.return_value = mock.MagicMock()
        mock_result = mock.MagicMock()
        mock_discover.return_value = mock_result

        discovered = mock.MagicMock()
        discovered.path = str(tmp_path / "descriptor")

        with mock.patch.dict("sys.modules", modules):
            documents_discovery.python_func(
                input_data_bucket_name="my-bucket",
                input_data_path="docs/",
                discovered_documents=discovered,
            )

        expected_dir = tmp_path / "descriptor"
        assert expected_dir.exists()
        mock_result.save.assert_called_once_with(
            path=expected_dir,
            filename="documents_descriptor.json",
        )

    def test_no_test_data_params_skips_benchmark_loading(self, tmp_path):
        """When test_data_bucket_name is empty, load_benchmark stage is skipped."""
        modules, mock_create_s3, mock_discover, _ = _make_ai4rag_mocks()
        mock_create_s3.return_value = mock.MagicMock()
        mock_discover.return_value = mock.MagicMock()

        discovered = mock.MagicMock()
        discovered.path = str(tmp_path / "descriptor")

        with mock.patch.dict("sys.modules", modules):
            documents_discovery.python_func(
                input_data_bucket_name="my-bucket",
                input_data_path="docs/",
                test_data_bucket_name="",
                discovered_documents=discovered,
            )

        assert mock_discover.call_args.kwargs["test_data_doc_names"] is None
        mock_create_s3.assert_called_once()

    def test_propagates_ai4rag_exception(self, tmp_path):
        """Exceptions from ai4rag are propagated to the caller."""
        modules, mock_create_s3, mock_discover, _ = _make_ai4rag_mocks()
        mock_create_s3.return_value = mock.MagicMock()
        mock_discover.side_effect = ValueError("No documents to process")

        discovered = mock.MagicMock()
        discovered.path = str(tmp_path / "descriptor")

        with mock.patch.dict("sys.modules", modules):
            with pytest.raises(ValueError, match="No documents to process"):
                documents_discovery.python_func(
                    input_data_bucket_name="my-bucket",
                    input_data_path="docs/",
                    discovered_documents=discovered,
                )


class TestDocumentsDiscoveryWithTestDataUnitTests:
    """Unit tests for documents_discovery with benchmark test data loading enabled."""

    def test_calls_load_test_data_then_discover_documents(self, tmp_path):
        """Wrapper calls load_test_data first, then discover_documents with doc names."""
        modules, mock_create_s3, mock_discover, mock_load = _make_ai4rag_mocks(include_test_data_loader=True)

        test_s3 = mock.MagicMock(name="test_s3_client")
        input_s3 = mock.MagicMock(name="input_s3_client")
        mock_create_s3.side_effect = [test_s3, input_s3]

        mock_load.return_value = SimpleNamespace(data=VALID_BENCHMARK_RECORDS)
        mock_discover.return_value = mock.MagicMock()

        test_data_artifact = mock.MagicMock()
        test_data_artifact.path = str(tmp_path / "test_data.json")
        discovered = mock.MagicMock()
        discovered.path = str(tmp_path / "descriptor")

        with mock.patch.dict("sys.modules", modules):
            documents_discovery.python_func(
                input_data_bucket_name="input-bucket",
                test_data_bucket_name="test-bucket",
                test_data_path_key="data/test.json",
                input_data_path="docs/",
                test_data=test_data_artifact,
                discovered_documents=discovered,
            )

        mock_load.assert_called_once_with(
            bucket_name="test-bucket",
            key="data/test.json",
            benchmark_sample_size=25,
            s3_client=test_s3,
        )
        mock_discover.assert_called_once()
        call_kwargs = mock_discover.call_args.kwargs
        assert call_kwargs["bucket_name"] == "input-bucket"
        assert call_kwargs["prefix"] == "docs/"
        assert set(call_kwargs["test_data_doc_names"]) == {"doc_a.pdf", "doc_b.txt"}
        assert call_kwargs["s3_client"] == input_s3

    def test_writes_test_data_to_artifact(self, tmp_path):
        """Test data JSON is written to the test_data artifact path."""
        modules, mock_create_s3, mock_discover, mock_load = _make_ai4rag_mocks(include_test_data_loader=True)
        mock_create_s3.return_value = mock.MagicMock()
        mock_load.return_value = SimpleNamespace(data=VALID_BENCHMARK_RECORDS)
        mock_discover.return_value = mock.MagicMock()

        test_data_artifact = mock.MagicMock()
        out_path = tmp_path / "test_data.json"
        test_data_artifact.path = str(out_path)
        discovered = mock.MagicMock()
        discovered.path = str(tmp_path / "descriptor")

        with mock.patch.dict("sys.modules", modules):
            documents_discovery.python_func(
                input_data_bucket_name="bucket",
                test_data_bucket_name="test-bucket",
                test_data_path_key="test.json",
                test_data=test_data_artifact,
                discovered_documents=discovered,
            )

        assert out_path.exists()
        result = json.loads(out_path.read_text(encoding="utf-8"))
        assert result == VALID_BENCHMARK_RECORDS

    def test_creates_s3_client_twice(self, tmp_path):
        """Both load_benchmark and discover_documents create their own S3 client."""
        modules, mock_create_s3, mock_discover, mock_load = _make_ai4rag_mocks(include_test_data_loader=True)
        mock_create_s3.return_value = mock.MagicMock()
        mock_load.return_value = SimpleNamespace(data=[])
        mock_discover.return_value = mock.MagicMock()

        test_data_artifact = mock.MagicMock()
        test_data_artifact.path = str(tmp_path / "td.json")
        discovered = mock.MagicMock()
        discovered.path = str(tmp_path / "desc")

        with mock.patch.dict("sys.modules", modules):
            documents_discovery.python_func(
                input_data_bucket_name="bucket",
                test_data_bucket_name="test-bucket",
                test_data_path_key="t.json",
                test_data=test_data_artifact,
                discovered_documents=discovered,
            )

        assert mock_create_s3.call_count == 2

    def test_propagates_load_test_data_exception(self, tmp_path):
        """Exceptions from load_test_data are propagated."""
        modules, mock_create_s3, mock_discover, mock_load = _make_ai4rag_mocks(include_test_data_loader=True)
        mock_create_s3.return_value = mock.MagicMock()
        mock_load.side_effect = FileNotFoundError("Test data not found")

        test_data_artifact = mock.MagicMock()
        test_data_artifact.path = str(tmp_path / "td.json")
        discovered = mock.MagicMock()
        discovered.path = str(tmp_path / "desc")

        with mock.patch.dict("sys.modules", modules):
            with pytest.raises(FileNotFoundError, match="Test data not found"):
                documents_discovery.python_func(
                    input_data_bucket_name="bucket",
                    test_data_bucket_name="test-bucket",
                    test_data_path_key="missing.json",
                    test_data=test_data_artifact,
                    discovered_documents=discovered,
                )

    def test_propagates_discover_documents_exception(self, tmp_path):
        """Exceptions from discover_documents are propagated."""
        modules, mock_create_s3, mock_discover, mock_load = _make_ai4rag_mocks(include_test_data_loader=True)
        mock_create_s3.return_value = mock.MagicMock()
        mock_load.return_value = SimpleNamespace(data=[])
        mock_discover.side_effect = RuntimeError("No supported documents found.")

        test_data_artifact = mock.MagicMock()
        test_data_artifact.path = str(tmp_path / "td.json")
        discovered = mock.MagicMock()
        discovered.path = str(tmp_path / "desc")

        with mock.patch.dict("sys.modules", modules):
            with pytest.raises(RuntimeError, match="No supported documents found"):
                documents_discovery.python_func(
                    input_data_bucket_name="bucket",
                    test_data_bucket_name="test-bucket",
                    test_data_path_key="t.json",
                    test_data=test_data_artifact,
                    discovered_documents=discovered,
                )
