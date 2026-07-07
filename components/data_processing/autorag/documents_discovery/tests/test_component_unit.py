"""Tests for the documents_discovery thin wrapper component."""

import inspect
import json
from types import SimpleNamespace
from unittest import mock

import pytest

from ..component import documents_discovery

VALID_BENCHMARK_RECORDS = [
    {"question": "What is X?", "correct_answers": ["Answer X"], "correct_answer_document_ids": ["doc_1"]},
    {"question": "What is Y?", "correct_answers": ["Answer Y"], "correct_answer_document_ids": ["doc_2"]},
]


def _make_ai4rag_mocks():
    """Build mock modules for ai4rag.components and ai4rag.components.data."""
    mock_create_s3_client = mock.MagicMock(name="create_s3_client")
    mock_discover_documents = mock.MagicMock(name="discover_documents")
    mock_load_test_data = mock.MagicMock(name="load_test_data")

    mock_s3_module = mock.MagicMock()
    mock_s3_module.create_s3_client = mock_create_s3_client

    mock_discovery_module = mock.MagicMock()
    mock_discovery_module.discover_documents = mock_discover_documents

    mock_loader_module = mock.MagicMock()
    mock_loader_module.load_test_data = mock_load_test_data

    modules = {
        "ai4rag": mock.MagicMock(),
        "ai4rag.components": mock.MagicMock(),
        "ai4rag.components.data": mock.MagicMock(),
        "ai4rag.components.data.documents_discovery": mock_discovery_module,
        "ai4rag.components.data.test_data_loader": mock_loader_module,
        "ai4rag.components.utils": mock.MagicMock(),
        "ai4rag.components.utils.s3": mock_s3_module,
    }
    return modules, mock_create_s3_client, mock_discover_documents, mock_load_test_data


class TestDocumentsDiscoveryUnitTests:
    """Unit tests for the documents_discovery thin wrapper."""

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
        assert "test_data_path" in params
        assert "benchmark_sample_size" in params
        assert sig.parameters["benchmark_sample_size"].default == 25

    def test_delegates_to_ai4rag_discover_documents(self, tmp_path):
        """Wrapper calls create_s3_client and discover_documents with correct args."""
        modules, mock_create_s3, mock_discover, _mock_load = _make_ai4rag_mocks()
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
        modules, mock_create_s3, mock_discover, _mock_load = _make_ai4rag_mocks()
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

    def test_loads_test_data_from_s3_and_extracts_doc_names(self, tmp_path):
        """Benchmark test data is loaded from S3 and doc names passed to discover_documents."""
        modules, mock_create_s3, mock_discover, mock_load = _make_ai4rag_mocks()
        mock_s3_client = mock.MagicMock()
        mock_create_s3.return_value = mock_s3_client
        mock_discover.return_value = mock.MagicMock()
        mock_load.return_value = SimpleNamespace(
            data=[
                {"question": "q1", "correct_answer_document_ids": ["doc_a.pdf", "doc_b.pdf"]},
                {"question": "q2", "correct_answer_document_ids": ["doc_a.pdf", "doc_c.txt"]},
            ]
        )

        discovered = mock.MagicMock()
        discovered.path = str(tmp_path / "descriptor")
        test_data_out = mock.MagicMock()
        test_data_out.path = str(tmp_path / "test_data.json")

        with mock.patch.dict("sys.modules", modules):
            documents_discovery.python_func(
                input_data_bucket_name="my-bucket",
                input_data_path="docs/",
                test_data_bucket_name="test-bucket",
                test_data_path="benchmark.json",
                benchmark_sample_size=10,
                discovered_documents=discovered,
                test_data=test_data_out,
            )

        mock_load.assert_called_once_with(
            bucket_name="test-bucket",
            key="benchmark.json",
            benchmark_sample_size=10,
            s3_client=mock_s3_client,
        )
        call_kwargs = mock_discover.call_args.kwargs
        passed_names = set(call_kwargs["test_data_doc_names"])
        assert passed_names == {"doc_a.pdf", "doc_b.pdf", "doc_c.txt"}

    def test_writes_test_data_to_artifact_path(self, tmp_path):
        """Loaded benchmark JSON is written to the test_data artifact path."""
        modules, mock_create_s3, mock_discover, mock_load = _make_ai4rag_mocks()
        mock_create_s3.return_value = mock.MagicMock()
        mock_discover.return_value = mock.MagicMock()
        mock_load.return_value = SimpleNamespace(data=VALID_BENCHMARK_RECORDS)

        discovered = mock.MagicMock()
        discovered.path = str(tmp_path / "descriptor")
        out_path = tmp_path / "test_data.json"
        test_data_out = mock.MagicMock()
        test_data_out.path = str(out_path)

        with mock.patch.dict("sys.modules", modules):
            documents_discovery.python_func(
                input_data_bucket_name="my-bucket",
                input_data_path="docs/",
                test_data_bucket_name="test-bucket",
                test_data_path="data/test.json",
                discovered_documents=discovered,
                test_data=test_data_out,
            )

        assert out_path.exists()
        result = json.loads(out_path.read_text(encoding="utf-8"))
        assert result == VALID_BENCHMARK_RECORDS

    def test_default_benchmark_sample_size(self, tmp_path):
        """Default benchmark_sample_size=25 is passed to load_test_data."""
        modules, mock_create_s3, mock_discover, mock_load = _make_ai4rag_mocks()
        mock_create_s3.return_value = mock.MagicMock()
        mock_discover.return_value = mock.MagicMock()
        mock_load.return_value = SimpleNamespace(data=[])

        discovered = mock.MagicMock()
        discovered.path = str(tmp_path / "descriptor")
        test_data_out = mock.MagicMock()
        test_data_out.path = str(tmp_path / "test_data.json")

        with mock.patch.dict("sys.modules", modules):
            documents_discovery.python_func(
                input_data_bucket_name="my-bucket",
                input_data_path="docs/",
                test_data_bucket_name="bucket",
                test_data_path="key.json",
                discovered_documents=discovered,
                test_data=test_data_out,
            )

        assert mock_load.call_args.kwargs["benchmark_sample_size"] == 25

    def test_no_test_data_skips_load_benchmark(self, tmp_path):
        """When test data S3 params are omitted, load_test_data is not called."""
        modules, mock_create_s3, mock_discover, mock_load = _make_ai4rag_mocks()
        mock_create_s3.return_value = mock.MagicMock()
        mock_discover.return_value = mock.MagicMock()

        discovered = mock.MagicMock()
        discovered.path = str(tmp_path / "descriptor")

        with mock.patch.dict("sys.modules", modules):
            documents_discovery.python_func(
                input_data_bucket_name="my-bucket",
                input_data_path="docs/",
                discovered_documents=discovered,
            )

        mock_load.assert_not_called()
        assert mock_discover.call_args.kwargs["test_data_doc_names"] is None

    def test_propagates_test_data_load_exception(self, tmp_path):
        """Exceptions from load_test_data are propagated to the caller."""
        modules, mock_create_s3, _mock_discover, mock_load = _make_ai4rag_mocks()
        mock_create_s3.return_value = mock.MagicMock()
        mock_load.side_effect = FileNotFoundError("Test data object not found in S3")

        discovered = mock.MagicMock()
        discovered.path = str(tmp_path / "descriptor")
        test_data_out = mock.MagicMock()
        test_data_out.path = str(tmp_path / "test_data.json")

        with mock.patch.dict("sys.modules", modules):
            with pytest.raises(FileNotFoundError, match="Test data object not found"):
                documents_discovery.python_func(
                    input_data_bucket_name="my-bucket",
                    input_data_path="docs/",
                    test_data_bucket_name="my-bucket",
                    test_data_path="missing/test.json",
                    discovered_documents=discovered,
                    test_data=test_data_out,
                )

    def test_propagates_discover_documents_exception(self, tmp_path):
        """Exceptions from discover_documents are propagated to the caller."""
        modules, mock_create_s3, mock_discover, _mock_load = _make_ai4rag_mocks()
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
