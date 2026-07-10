"""Tests for the documents_indexing thin wrapper component."""

import inspect
from unittest import mock

import pytest

from ..component import documents_indexing

MOCKED_ENV_VARIABLES = {
    "OGX_CLIENT_BASE_URL": "https://ogx.example.com",
    "OGX_CLIENT_API_KEY": "test-api-key",
}


def _make_ai4rag_mocks():
    """Build mock modules for ai4rag submodule imports."""
    mock_create_ogx_client = mock.MagicMock(name="create_ogx_client")
    mock_index_documents = mock.MagicMock(name="index_documents")

    mock_ogx_client_module = mock.MagicMock()
    mock_ogx_client_module.create_ogx_client = mock_create_ogx_client

    mock_indexing_module = mock.MagicMock()
    mock_indexing_module.index_documents = mock_index_documents

    modules = {
        "ai4rag": mock.MagicMock(),
        "ai4rag.components": mock.MagicMock(),
        "ai4rag.components.utils": mock.MagicMock(),
        "ai4rag.components.utils.ogx_client": mock_ogx_client_module,
        "ai4rag.components.data": mock.MagicMock(),
        "ai4rag.components.data.indexing": mock_indexing_module,
    }
    return modules, mock_create_ogx_client, mock_index_documents


class TestDocumentsIndexingUnitTests:
    """Unit tests for the documents_indexing thin wrapper."""

    def test_component_function_exists(self):
        """Component factory exists and exposes python_func."""
        assert callable(documents_indexing)
        assert hasattr(documents_indexing, "python_func")

    def test_component_has_expected_interface(self):
        """Component has expected required parameters."""
        sig = inspect.signature(documents_indexing.python_func)
        params = list(sig.parameters)
        assert "embedding_model_id" in params
        assert "extracted_text" in params
        assert "vector_io_provider_id" in params
        assert "vector_store_id" in params
        assert sig.parameters["distance_metric"].default == "cosine"
        assert sig.parameters["chunk_size"].default == 1024
        assert sig.parameters["batch_size"].default == 20

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_delegates_to_ai4rag_index_documents(self, tmp_path):
        """Wrapper calls create_ogx_client and index_documents with correct args."""
        modules, mock_create_ogx, mock_index = _make_ai4rag_mocks()
        mock_ogx_client = mock.MagicMock(name="ogx_client_instance")
        mock_create_ogx.return_value = mock_ogx_client

        extracted = mock.MagicMock()
        extracted.path = str(tmp_path / "extracted")

        with mock.patch.dict("sys.modules", modules):
            documents_indexing.python_func(
                embedding_model_id="embed-v1",
                extracted_text=extracted,
                vector_io_provider_id="provider-1",
                embedding_params={"embedding_dimension": 768},
                distance_metric="cosine",
                chunking_method="hybrid",
                chunk_size=512,
                chunk_overlap=64,
                batch_size=10,
                vector_store_id="my-vector-store",
            )

        mock_create_ogx.assert_called_once_with(
            base_url="https://ogx.example.com",
            api_key="test-api-key",
        )
        mock_index.assert_called_once_with(
            extracted_text_dir=str(tmp_path / "extracted"),
            embedding_model_id="embed-v1",
            vector_io_provider_id="provider-1",
            ogx_client=mock_ogx_client,
            embedding_params={"embedding_dimension": 768},
            distance_metric="cosine",
            chunking_method="hybrid",
            chunk_size=512,
            chunk_overlap=64,
            batch_size=10,
            collection_name="my-vector-store",
        )

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_default_parameters_passed_through(self, tmp_path):
        """Default parameter values are forwarded to index_documents."""
        modules, mock_create_ogx, mock_index = _make_ai4rag_mocks()
        mock_create_ogx.return_value = mock.MagicMock()

        extracted = mock.MagicMock()
        extracted.path = str(tmp_path)

        with mock.patch.dict("sys.modules", modules):
            documents_indexing.python_func(
                embedding_model_id="embed-v1",
                extracted_text=extracted,
                vector_io_provider_id="provider-1",
            )

        call_kwargs = mock_index.call_args.kwargs
        assert call_kwargs["distance_metric"] == "cosine"
        assert call_kwargs["chunking_method"] == "recursive"
        assert call_kwargs["chunk_size"] == 1024
        assert call_kwargs["chunk_overlap"] == 0
        assert call_kwargs["batch_size"] == 20
        assert call_kwargs["collection_name"] is None
        assert call_kwargs["embedding_params"] is None

    def test_missing_ogx_env_raises_key_error(self, tmp_path):
        """Missing OGX env vars raise KeyError."""
        modules, mock_create_ogx, mock_index = _make_ai4rag_mocks()

        extracted = mock.MagicMock()
        extracted.path = str(tmp_path)

        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch.dict("sys.modules", modules):
                with pytest.raises(KeyError):
                    documents_indexing.python_func(
                        embedding_model_id="embed-v1",
                        extracted_text=extracted,
                        vector_io_provider_id="provider-1",
                    )

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_propagates_ai4rag_exception(self, tmp_path):
        """Exceptions from ai4rag are propagated to the caller."""
        modules, mock_create_ogx, mock_index = _make_ai4rag_mocks()
        mock_create_ogx.return_value = mock.MagicMock()
        mock_index.side_effect = ValueError("embedding_model_id must be a non-empty string.")

        extracted = mock.MagicMock()
        extracted.path = str(tmp_path)

        with mock.patch.dict("sys.modules", modules):
            with pytest.raises(ValueError, match="embedding_model_id must be a non-empty string"):
                documents_indexing.python_func(
                    embedding_model_id="",
                    extracted_text=extracted,
                    vector_io_provider_id="provider-1",
                )
