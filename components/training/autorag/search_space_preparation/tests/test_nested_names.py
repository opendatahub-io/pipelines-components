"""Unit tests for nested_names.py functions."""

import logging
import ssl
from dataclasses import dataclass
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest
from kfp_components.components.training.autorag.search_space_preparation.tests.nested_names import (
    _create_llama_stack_client,
    _create_openai_client,
    _get_model_metadata_from,
    _is_ssl_error,
    load_as_langchain_doc,
    prepare_ai4rag_search_space,
    represent_model_instance,
)


class TestIsSSLError:
    """Tests for _is_ssl_error function."""

    def test_ssl_cert_verification_error(self):
        """Test SSL certificate verification error detection."""
        exc = ssl.SSLCertVerificationError("CERTIFICATE_VERIFY_FAILED")
        assert _is_ssl_error(exc) is True

    def test_ssl_in_message(self):
        """Test SSL keyword in exception message."""
        exc = Exception("SSL handshake failed")
        assert _is_ssl_error(exc) is True

    def test_ssl_error_in_cause_chain(self):
        """Test SSL error detection from exception cause."""
        ssl_exc = ssl.SSLCertVerificationError("CERTIFICATE_VERIFY_FAILED")
        wrapper = Exception("Connection failed")
        wrapper.__cause__ = ssl_exc
        assert _is_ssl_error(wrapper) is True

    def test_ssl_error_in_context_chain(self):
        """Test SSL error detection from exception context."""
        ssl_exc = Exception("SSL verification failed")
        wrapper = Exception("Request failed")
        wrapper.__context__ = ssl_exc
        assert _is_ssl_error(wrapper) is True

    def test_non_ssl_error(self):
        """Test non-SSL errors return False."""
        exc = ValueError("Invalid value")
        assert _is_ssl_error(exc) is False

    def test_no_cause_or_context(self):
        """Test exception with no cause or context."""
        exc = Exception("No chain")
        assert _is_ssl_error(exc) is False


class TestCreateOpenAIClient:
    """Tests for _create_openai_client function."""

    @patch("kfp_components.components.training.autorag.search_space_preparation.tests.nested_names.OpenAI")
    def test_successful_connection(self, mock_openai_class):
        """Test successful client creation without SSL issues."""
        mock_client = MagicMock()
        mock_client.models.list.return_value = []
        mock_openai_class.return_value = mock_client

        result = _create_openai_client("key", "https://api.example.com")

        assert result == mock_client
        mock_openai_class.assert_called_once_with(api_key="key", base_url="https://api.example.com")

    @patch("kfp_components.components.training.autorag.search_space_preparation.tests.nested_names.OpenAI")
    @patch("kfp_components.components.training.autorag.search_space_preparation.tests.nested_names.httpx.Client")
    def test_ssl_verification_error_fallback(self, mock_httpx_client, mock_openai_class):
        """Test fallback to unverified client on SSLCertVerificationError."""
        from kfp_components.components.training.autorag.search_space_preparation.tests import nested_names

        nested_names._ssl_logger = logging.getLogger("test")

        failing_client = MagicMock()
        failing_client.models.list.side_effect = ssl.SSLCertVerificationError("CERTIFICATE_VERIFY_FAILED")

        success_client = MagicMock()
        success_client.models.list.return_value = []

        mock_openai_class.side_effect = [failing_client, success_client]

        result = _create_openai_client("key", "https://api.example.com")

        assert result == success_client
        mock_httpx_client.assert_called_once_with(verify=False)

    @patch("kfp_components.components.training.autorag.search_space_preparation.tests.nested_names.OpenAI")
    @patch("kfp_components.components.training.autorag.search_space_preparation.tests.nested_names.httpx.Client")
    def test_httpx_connect_error_fallback(self, mock_httpx_client, mock_openai_class):
        """Test fallback to unverified client on httpx.ConnectError with SSL."""
        from kfp_components.components.training.autorag.search_space_preparation.tests import nested_names

        nested_names._ssl_logger = logging.getLogger("test")

        failing_client = MagicMock()
        failing_client.models.list.side_effect = httpx.ConnectError("SSL verification failed")

        success_client = MagicMock()
        success_client.models.list.return_value = []

        mock_openai_class.side_effect = [failing_client, success_client]

        result = _create_openai_client("key", "https://api.example.com")

        assert result == success_client
        mock_httpx_client.assert_called_once_with(verify=False)

    @patch("kfp_components.components.training.autorag.search_space_preparation.tests.nested_names.OpenAI")
    @patch("kfp_components.components.training.autorag.search_space_preparation.tests.nested_names.httpx.Client")
    def test_oai_api_connection_error_fallback(self, mock_httpx_client, mock_openai_class):
        """Test fallback to unverified client on httpx.ConnectError with SSL in message."""
        from kfp_components.components.training.autorag.search_space_preparation.tests import nested_names

        nested_names._ssl_logger = logging.getLogger("test")

        failing_client = MagicMock()
        # Use httpx.ConnectError with SSL keyword which will be caught and checked
        failing_client.models.list.side_effect = httpx.ConnectError("SSL: CERTIFICATE_VERIFY_FAILED")

        success_client = MagicMock()
        success_client.models.list.return_value = []

        mock_openai_class.side_effect = [failing_client, success_client]

        result = _create_openai_client("key", "https://api.example.com")

        assert result == success_client
        mock_httpx_client.assert_called_once_with(verify=False)

    @patch("kfp_components.components.training.autorag.search_space_preparation.tests.nested_names.OpenAI")
    def test_non_ssl_error_propagation(self, mock_openai_class):
        """Test that non-SSL exceptions are re-raised."""
        mock_client = MagicMock()
        mock_client.models.list.side_effect = ValueError("Some other error")
        mock_openai_class.return_value = mock_client

        with pytest.raises(ValueError, match="Some other error"):
            _create_openai_client("key", "https://api.example.com")


class TestCreateLlamaStackClient:
    """Tests for _create_llama_stack_client function."""

    @patch("kfp_components.components.training.autorag.search_space_preparation.tests.nested_names.LlamaStackClient")
    def test_successful_connection(self, mock_client_class):
        """Test successful client creation without SSL issues."""
        mock_client = MagicMock()
        mock_client.models.list.return_value = []
        mock_client_class.return_value = mock_client

        result = _create_llama_stack_client(base_url="https://api.example.com")

        assert result == mock_client
        mock_client_class.assert_called_once_with(base_url="https://api.example.com")

    @patch("kfp_components.components.training.autorag.search_space_preparation.tests.nested_names.LlamaStackClient")
    @patch("kfp_components.components.training.autorag.search_space_preparation.tests.nested_names.httpx.Client")
    def test_ssl_error_fallback(self, mock_httpx_client, mock_client_class):
        """Test fallback to unverified client on SSL error."""
        from kfp_components.components.training.autorag.search_space_preparation.tests import nested_names

        nested_names._ssl_logger = logging.getLogger("test")

        failing_client = MagicMock()
        failing_client.models.list.side_effect = ssl.SSLCertVerificationError("SSL error")

        success_client = MagicMock()
        success_client.models.list.return_value = []

        mock_client_class.side_effect = [failing_client, success_client]

        result = _create_llama_stack_client(base_url="https://api.example.com")

        assert result == success_client
        mock_httpx_client.assert_called_once_with(verify=False)

    @patch("kfp_components.components.training.autorag.search_space_preparation.tests.nested_names.LlamaStackClient")
    @patch("kfp_components.components.training.autorag.search_space_preparation.tests.nested_names.httpx.Client")
    def test_httpx_connect_error_fallback(self, mock_httpx_client, mock_client_class):
        """Test fallback on httpx.ConnectError with SSL."""
        from kfp_components.components.training.autorag.search_space_preparation.tests import nested_names

        nested_names._ssl_logger = logging.getLogger("test")

        failing_client = MagicMock()
        failing_client.models.list.side_effect = httpx.ConnectError("SSL verification failed")

        success_client = MagicMock()
        success_client.models.list.return_value = []

        mock_client_class.side_effect = [failing_client, success_client]

        result = _create_llama_stack_client(base_url="https://api.example.com")

        assert result == success_client
        mock_httpx_client.assert_called_once_with(verify=False)

    @patch("kfp_components.components.training.autorag.search_space_preparation.tests.nested_names.LlamaStackClient")
    @patch("kfp_components.components.training.autorag.search_space_preparation.tests.nested_names.httpx.Client")
    def test_ls_api_connection_error_fallback(self, mock_httpx_client, mock_client_class):
        """Test fallback on httpx.ConnectError with SSL in message."""
        from kfp_components.components.training.autorag.search_space_preparation.tests import nested_names

        nested_names._ssl_logger = logging.getLogger("test")

        failing_client = MagicMock()
        # Use httpx.ConnectError with SSL keyword which will be caught and checked
        failing_client.models.list.side_effect = httpx.ConnectError("SSL: CERTIFICATE_VERIFY_FAILED")

        success_client = MagicMock()
        success_client.models.list.return_value = []

        mock_client_class.side_effect = [failing_client, success_client]

        result = _create_llama_stack_client(base_url="https://api.example.com")

        assert result == success_client
        mock_httpx_client.assert_called_once_with(verify=False)

    @patch("kfp_components.components.training.autorag.search_space_preparation.tests.nested_names.LlamaStackClient")
    def test_kwargs_propagation(self, mock_client_class):
        """Test that all kwargs are passed to client constructor."""
        mock_client = MagicMock()
        mock_client.models.list.return_value = []
        mock_client_class.return_value = mock_client

        _create_llama_stack_client(base_url="https://api.example.com", api_key="test", timeout=30)

        mock_client_class.assert_called_once_with(base_url="https://api.example.com", api_key="test", timeout=30)

    @patch("kfp_components.components.training.autorag.search_space_preparation.tests.nested_names.LlamaStackClient")
    def test_non_ssl_error_propagation(self, mock_client_class):
        """Test that non-SSL exceptions are re-raised."""
        mock_client = MagicMock()
        mock_client.models.list.side_effect = ValueError("Some other error")
        mock_client_class.return_value = mock_client

        with pytest.raises(ValueError, match="Some other error"):
            _create_llama_stack_client(base_url="https://api.example.com")


class TestGetModelMetadataFrom:
    """Tests for _get_model_metadata_from function."""

    @patch(
        "kfp_components.components.training.autorag.search_space_preparation.tests.nested_names._create_openai_client"
    )
    def test_successful_metadata_retrieval_with_complete_data(self, mock_create_client):
        """Test successful metadata retrieval with id and max_model_len."""
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.id = "gpt-4"
        mock_model.max_model_len = 8192
        mock_client.models.list.return_value.data = [mock_model]
        mock_create_client.return_value = mock_client

        result = _get_model_metadata_from("https://api.example.com", "token123")

        assert result["id"] == "gpt-4"
        assert result["max_model_len"] == 8192
        mock_create_client.assert_called_once_with(api_key="token123", base_url="https://api.example.com")

    @patch(
        "kfp_components.components.training.autorag.search_space_preparation.tests.nested_names._create_openai_client"
    )
    def test_successful_metadata_retrieval_without_max_model_len(self, mock_create_client):
        """Test successful metadata retrieval with only id."""
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.id = "gpt-3.5-turbo"
        # Simulate missing max_model_len attribute
        del mock_model.max_model_len
        mock_client.models.list.return_value.data = [mock_model]
        mock_create_client.return_value = mock_client

        result = _get_model_metadata_from("https://api.example.com", "token123")

        assert result["id"] == "gpt-3.5-turbo"
        assert result["max_model_len"] == 0

    @patch(
        "kfp_components.components.training.autorag.search_space_preparation.tests.nested_names._create_openai_client"
    )
    def test_empty_model_list_raises_value_error(self, mock_create_client):
        """Test empty model list raises ValueError."""
        mock_client = MagicMock()
        mock_client.models.list.return_value.data = []
        mock_create_client.return_value = mock_client

        with pytest.raises(ValueError, match="Could not retrieve all the required model metadata"):
            _get_model_metadata_from("https://api.example.com", "token123")

    @patch(
        "kfp_components.components.training.autorag.search_space_preparation.tests.nested_names._create_openai_client"
    )
    def test_model_with_empty_id_raises_value_error(self, mock_create_client):
        """Test model with empty string id raises ValueError."""
        mock_client = MagicMock()
        mock_model = MagicMock()
        mock_model.id = ""
        mock_model.max_model_len = 8192
        mock_client.models.list.return_value.data = [mock_model]
        mock_create_client.return_value = mock_client

        with pytest.raises(ValueError, match="Could not retrieve all the required model metadata"):
            _get_model_metadata_from("https://api.example.com", "token123")

    @patch(
        "kfp_components.components.training.autorag.search_space_preparation.tests.nested_names._create_openai_client"
    )
    def test_error_message_includes_url(self, mock_create_client):
        """Test that ValueError message includes the URL."""
        mock_client = MagicMock()
        mock_client.models.list.return_value.data = []
        mock_create_client.return_value = mock_client

        with pytest.raises(ValueError, match="https://api.example.com"):
            _get_model_metadata_from("https://api.example.com", "token123")


class TestLoadAsLangchainDoc:
    """Tests for load_as_langchain_doc function."""

    def test_load_single_file_as_string_path(self, tmp_path):
        """Test loading a single file with string path."""
        test_file = tmp_path / "document.txt"
        test_file.write_text("Test content", encoding="utf-8")

        result = load_as_langchain_doc(str(test_file))

        assert len(result) == 1
        assert result[0].page_content == "Test content"
        assert result[0].metadata["document_id"] == "document"

    def test_load_single_file_as_path(self, tmp_path):
        """Test loading a single file with Path object."""
        test_file = tmp_path / "document.txt"
        test_file.write_text("Test content", encoding="utf-8")

        result = load_as_langchain_doc(test_file)

        assert len(result) == 1
        assert result[0].page_content == "Test content"
        assert result[0].metadata["document_id"] == "document"

    def test_load_directory(self, tmp_path):
        """Test loading all files from a directory."""
        doc1 = tmp_path / "doc1.txt"
        doc2 = tmp_path / "doc2.txt"
        doc1.write_text("Content 1", encoding="utf-8")
        doc2.write_text("Content 2", encoding="utf-8")

        result = load_as_langchain_doc(tmp_path)

        assert len(result) == 2
        contents = {doc.page_content for doc in result}
        assert "Content 1" in contents
        assert "Content 2" in contents

    def test_document_id_from_filename_stem(self, tmp_path):
        """Test that document_id is extracted from filename stem."""
        test_file = tmp_path / "my_document.txt"
        test_file.write_text("Content", encoding="utf-8")

        result = load_as_langchain_doc(test_file)

        assert result[0].metadata["document_id"] == "my_document"


class TestPrepareAi4ragSearchSpace:
    """Tests for prepare_ai4rag_search_space function."""

    @patch(
        "kfp_components.components.training.autorag.search_space_preparation.tests.nested_names._get_model_metadata_from"
    )
    def test_in_memory_scenario_with_complete_metadata(self, mock_get_metadata):
        """Test in-memory scenario with complete model metadata."""
        from kfp_components.components.training.autorag.search_space_preparation.tests import nested_names

        # Setup module globals
        nested_names.in_memory_vector_store_scenario = True
        nested_names.chat_model_url = "https://chat.example.com"
        nested_names.chat_model_token = "chat_token"
        nested_names.embedding_model_url = "https://embed.example.com"
        nested_names.embedding_model_token = "embed_token"

        mock_client = Mock()
        mock_client.generation_model = Mock()
        mock_client.embedding_model = Mock()
        nested_names.client = mock_client

        # Mock model classes
        mock_gen_model = Mock()
        mock_em_model = Mock()
        nested_names.OpenAIFoundationModel = Mock(return_value=mock_gen_model)
        nested_names.OpenAIEmbeddingModel = Mock(return_value=mock_em_model)
        nested_names.Parameter = Mock()
        nested_names.AI4RAGSearchSpace = Mock()

        # Setup metadata responses
        mock_get_metadata.side_effect = [
            {"id": "gpt-4", "max_model_len": 8192},
            {"id": "text-embedding-ada-002", "max_model_len": 8191},
        ]

        _ = prepare_ai4rag_search_space()

        # Verify _get_model_metadata_from called correctly
        assert mock_get_metadata.call_count == 2
        mock_get_metadata.assert_any_call("https://chat.example.com", "chat_token")
        mock_get_metadata.assert_any_call("https://embed.example.com", "embed_token")

        # Verify OpenAIFoundationModel created correctly
        nested_names.OpenAIFoundationModel.assert_called_once_with(
            client=mock_client.generation_model,
            model_id="gpt-4",
            params={"max_completion_tokens": 2048, "temperature": 0.2},
        )

        # Verify OpenAIEmbeddingModel created correctly
        nested_names.OpenAIEmbeddingModel.assert_called_once_with(
            client=mock_client.embedding_model,
            model_id="text-embedding-ada-002",
            params={"context_length": 8191},
        )

        # Verify AI4RAGSearchSpace called with vector_store_type="chroma"
        nested_names.AI4RAGSearchSpace.assert_called_once()
        call_kwargs = nested_names.AI4RAGSearchSpace.call_args[1]
        assert call_kwargs["vector_store_type"] == "chroma"

    @patch(
        "kfp_components.components.training.autorag.search_space_preparation.tests.nested_names._get_model_metadata_from"
    )
    def test_in_memory_scenario_without_max_model_len(self, mock_get_metadata):
        """Test in-memory scenario when max_model_len is 0."""
        from kfp_components.components.training.autorag.search_space_preparation.tests import nested_names

        # Setup module globals
        nested_names.in_memory_vector_store_scenario = True
        nested_names.chat_model_url = "https://chat.example.com"
        nested_names.chat_model_token = "chat_token"
        nested_names.embedding_model_url = "https://embed.example.com"
        nested_names.embedding_model_token = "embed_token"

        mock_client = Mock()
        mock_client.generation_model = Mock()
        mock_client.embedding_model = Mock()
        nested_names.client = mock_client

        # Mock model classes
        nested_names.OpenAIFoundationModel = Mock()
        nested_names.OpenAIEmbeddingModel = Mock()
        nested_names.Parameter = Mock()
        nested_names.AI4RAGSearchSpace = Mock()

        # Setup metadata responses with 0 max_model_len
        mock_get_metadata.side_effect = [
            {"id": "gpt-4", "max_model_len": 8192},
            {"id": "text-embedding-ada-002", "max_model_len": 0},
        ]

        prepare_ai4rag_search_space()

        # Verify OpenAIEmbeddingModel called with empty params dict
        nested_names.OpenAIEmbeddingModel.assert_called_once_with(
            client=mock_client.embedding_model,
            model_id="text-embedding-ada-002",
            params={},
        )

    @patch(
        "kfp_components.components.training.autorag.search_space_preparation.tests.nested_names.prepare_search_space_with_llama_stack"
    )
    def test_llama_stack_scenario_with_both_models(self, mock_prepare_ss):
        """Test LlamaStack scenario with both generation and embedding models."""
        from kfp_components.components.training.autorag.search_space_preparation.tests import nested_names

        # Setup module globals
        nested_names.in_memory_vector_store_scenario = False
        nested_names.generation_models = ["llama-3-70b", "llama-3-8b"]
        nested_names.embeddings_models = ["llama-embed-v1"]

        mock_client = Mock()
        mock_client.llama_stack = Mock()
        nested_names.client = mock_client

        mock_result = Mock()
        mock_prepare_ss.return_value = mock_result

        result = prepare_ai4rag_search_space()

        # Verify prepare_search_space_with_llama_stack called with correct payload
        expected_payload = {
            "foundation_models": [{"model_id": "llama-3-70b"}, {"model_id": "llama-3-8b"}],
            "embedding_models": [{"model_id": "llama-embed-v1"}],
        }
        mock_prepare_ss.assert_called_once_with(expected_payload, client=mock_client.llama_stack)
        assert result == mock_result

    @patch(
        "kfp_components.components.training.autorag.search_space_preparation.tests.nested_names.prepare_search_space_with_llama_stack"
    )
    def test_llama_stack_scenario_only_generation_models(self, mock_prepare_ss):
        """Test LlamaStack scenario with only generation models."""
        from kfp_components.components.training.autorag.search_space_preparation.tests import nested_names

        # Setup module globals
        nested_names.in_memory_vector_store_scenario = False
        nested_names.generation_models = ["llama-3-70b"]
        nested_names.embeddings_models = []

        mock_client = Mock()
        mock_client.llama_stack = Mock()
        nested_names.client = mock_client

        prepare_ai4rag_search_space()

        # Verify prepare_search_space_with_llama_stack called with only foundation_models
        expected_payload = {
            "foundation_models": [{"model_id": "llama-3-70b"}],
        }
        mock_prepare_ss.assert_called_once_with(expected_payload, client=mock_client.llama_stack)

    @patch(
        "kfp_components.components.training.autorag.search_space_preparation.tests.nested_names.prepare_search_space_with_llama_stack"
    )
    def test_llama_stack_scenario_only_embedding_models(self, mock_prepare_ss):
        """Test LlamaStack scenario with only embedding models."""
        from kfp_components.components.training.autorag.search_space_preparation.tests import nested_names

        # Setup module globals
        nested_names.in_memory_vector_store_scenario = False
        nested_names.generation_models = []
        nested_names.embeddings_models = ["llama-embed-v1"]

        mock_client = Mock()
        mock_client.llama_stack = Mock()
        nested_names.client = mock_client

        prepare_ai4rag_search_space()

        # Verify prepare_search_space_with_llama_stack called with only embedding_models
        expected_payload = {
            "embedding_models": [{"model_id": "llama-embed-v1"}],
        }
        mock_prepare_ss.assert_called_once_with(expected_payload, client=mock_client.llama_stack)

    @patch(
        "kfp_components.components.training.autorag.search_space_preparation.tests.nested_names.prepare_search_space_with_llama_stack"
    )
    def test_llama_stack_scenario_no_models(self, mock_prepare_ss):
        """Test LlamaStack scenario with no models."""
        from kfp_components.components.training.autorag.search_space_preparation.tests import nested_names

        # Setup module globals
        nested_names.in_memory_vector_store_scenario = False
        nested_names.generation_models = []
        nested_names.embeddings_models = []

        mock_client = Mock()
        mock_client.llama_stack = Mock()
        nested_names.client = mock_client

        prepare_ai4rag_search_space()

        # Verify prepare_search_space_with_llama_stack called with empty payload
        mock_prepare_ss.assert_called_once_with({}, client=mock_client.llama_stack)


class TestRepresentModelInstance:
    """Tests for represent_model_instance function."""

    def test_embedding_model_with_dataclass_params(self):
        """Test embedding model with dataclass params."""

        @dataclass
        class EmbeddingParams:
            embedding_dimension: int = 768
            context_length: int | None = None

        mock_model = Mock()
        mock_model.model_id = "text-embedding-ada-002"
        mock_model.params = EmbeddingParams(embedding_dimension=768, context_length=8191)

        # Import BaseEmbeddingModel for isinstance check
        from ai4rag.rag.embedding.base_model import BaseEmbeddingModel

        # Make mock_model an instance of BaseEmbeddingModel
        mock_model.__class__ = BaseEmbeddingModel

        mock_dumper = Mock()
        mock_dumper.represent_mapping.return_value = "yaml_node"

        _ = represent_model_instance(mock_dumper, mock_model)

        # Verify represent_mapping called with correct structure
        mock_dumper.represent_mapping.assert_called_once()
        call_args = mock_dumper.represent_mapping.call_args[0]
        assert call_args[0] == "!Model"
        mapping = call_args[1]
        assert mapping["type_"] == "embedding"
        assert mapping["text-embedding-ada-002"]["embedding_dimension"] == 768
        assert mapping["text-embedding-ada-002"]["context_length"] == 8191

    def test_embedding_model_with_pydantic_v2_params(self):
        """Test embedding model with Pydantic v2 params."""
        mock_params = Mock()
        mock_params.model_dump.return_value = {"embedding_dimension": 1024}

        mock_model = Mock()
        mock_model.model_id = "custom-embed"
        mock_model.params = mock_params

        # Import BaseEmbeddingModel
        from ai4rag.rag.embedding.base_model import BaseEmbeddingModel

        mock_model.__class__ = BaseEmbeddingModel

        mock_dumper = Mock()

        represent_model_instance(mock_dumper, mock_model)

        # Verify model_dump was called with exclude_unset=True
        mock_params.model_dump.assert_called_once_with(exclude_unset=True)

    def test_embedding_model_with_pydantic_v1_params(self):
        """Test embedding model with Pydantic v1 params."""
        mock_params = Mock()
        # Pydantic v1 has .dict() but not .model_dump()
        del mock_params.model_dump
        mock_params.dict.return_value = {"embedding_dimension": 512}

        mock_model = Mock()
        mock_model.model_id = "legacy-embed"
        mock_model.params = mock_params

        from ai4rag.rag.embedding.base_model import BaseEmbeddingModel

        mock_model.__class__ = BaseEmbeddingModel

        mock_dumper = Mock()

        represent_model_instance(mock_dumper, mock_model)

        # Verify dict was called with exclude_unset=True
        mock_params.dict.assert_called_once_with(exclude_unset=True)

    def test_embedding_model_with_dict_params(self):
        """Test embedding model with regular dict params."""
        mock_model = Mock()
        mock_model.model_id = "dict-embed"
        mock_model.params = {"embedding_dimension": 256}

        from ai4rag.rag.embedding.base_model import BaseEmbeddingModel

        mock_model.__class__ = BaseEmbeddingModel

        mock_dumper = Mock()

        represent_model_instance(mock_dumper, mock_model)

        # Verify dict passed through
        call_args = mock_dumper.represent_mapping.call_args[0]
        mapping = call_args[1]
        assert mapping["dict-embed"] == {"embedding_dimension": 256}

    def test_embedding_model_with_empty_dict_params(self):
        """Test embedding model with empty dict params."""
        mock_model = Mock()
        mock_model.model_id = "no-params-embed"
        mock_model.params = {}

        from ai4rag.rag.embedding.base_model import BaseEmbeddingModel

        mock_model.__class__ = BaseEmbeddingModel

        mock_dumper = Mock()

        represent_model_instance(mock_dumper, mock_model)

        call_args = mock_dumper.represent_mapping.call_args[0]
        mapping = call_args[1]
        assert mapping["no-params-embed"] == {}

    def test_foundation_model_with_dataclass_params(self):
        """Test foundation model with dataclass params."""

        @dataclass
        class GenerationParams:
            temperature: float = 0.7
            max_tokens: int | None = None

        mock_model = Mock()
        mock_model.model_id = "gpt-4"
        mock_model.params = GenerationParams(temperature=0.5, max_tokens=100)

        from ai4rag.rag.foundation_models.base_model import BaseFoundationModel

        mock_model.__class__ = BaseFoundationModel

        mock_dumper = Mock()

        represent_model_instance(mock_dumper, mock_model)

        call_args = mock_dumper.represent_mapping.call_args[0]
        mapping = call_args[1]
        assert mapping["type_"] == "generation"
        assert mapping["gpt-4"]["temperature"] == 0.5
        assert mapping["gpt-4"]["max_tokens"] == 100

    def test_foundation_model_with_pydantic_v2_params(self):
        """Test foundation model with Pydantic v2 params."""
        mock_params = Mock()
        mock_params.model_dump.return_value = {"temperature": 0.8}

        mock_model = Mock()
        mock_model.model_id = "custom-gen"
        mock_model.params = mock_params

        from ai4rag.rag.foundation_models.base_model import BaseFoundationModel

        mock_model.__class__ = BaseFoundationModel

        mock_dumper = Mock()

        represent_model_instance(mock_dumper, mock_model)

        mock_params.model_dump.assert_called_once_with(exclude_unset=True)

    def test_foundation_model_with_pydantic_v1_params(self):
        """Test foundation model with Pydantic v1 params."""
        mock_params = Mock()
        del mock_params.model_dump
        mock_params.dict.return_value = {"temperature": 0.3}

        mock_model = Mock()
        mock_model.model_id = "legacy-gen"
        mock_model.params = mock_params

        from ai4rag.rag.foundation_models.base_model import BaseFoundationModel

        mock_model.__class__ = BaseFoundationModel

        mock_dumper = Mock()

        represent_model_instance(mock_dumper, mock_model)

        mock_params.dict.assert_called_once_with(exclude_unset=True)

    def test_foundation_model_with_dict_params(self):
        """Test foundation model with regular dict params."""
        mock_model = Mock()
        mock_model.model_id = "dict-gen"
        mock_model.params = {"temperature": 0.9, "max_tokens": 500}

        from ai4rag.rag.foundation_models.base_model import BaseFoundationModel

        mock_model.__class__ = BaseFoundationModel

        mock_dumper = Mock()

        represent_model_instance(mock_dumper, mock_model)

        call_args = mock_dumper.represent_mapping.call_args[0]
        mapping = call_args[1]
        assert mapping["dict-gen"] == {"temperature": 0.9, "max_tokens": 500}

    def test_dataclass_params_filters_none_values(self):
        """Test that dataclass params filters out None values."""

        @dataclass
        class ParamsWithNone:
            value1: int = 100
            value2: int | None = None
            value3: str | None = None

        mock_model = Mock()
        mock_model.model_id = "filter-test"
        mock_model.params = ParamsWithNone(value1=100, value2=None, value3=None)

        from ai4rag.rag.embedding.base_model import BaseEmbeddingModel

        mock_model.__class__ = BaseEmbeddingModel

        mock_dumper = Mock()

        represent_model_instance(mock_dumper, mock_model)

        call_args = mock_dumper.represent_mapping.call_args[0]
        mapping = call_args[1]
        # Only value1 should be included (value2 and value3 are None)
        assert "value1" in mapping["filter-test"]
        assert mapping["filter-test"]["value1"] == 100
        # None values should be filtered out
        assert "value2" not in mapping["filter-test"]
        assert "value3" not in mapping["filter-test"]

    def test_yaml_model_tag_used(self):
        """Test that !Model tag is used in YAML representation."""
        mock_model = Mock()
        mock_model.model_id = "test-model"
        mock_model.params = {}

        from ai4rag.rag.embedding.base_model import BaseEmbeddingModel

        mock_model.__class__ = BaseEmbeddingModel

        mock_dumper = Mock()

        represent_model_instance(mock_dumper, mock_model)

        # Verify first argument is "!Model"
        call_args = mock_dumper.represent_mapping.call_args[0]
        assert call_args[0] == "!Model"
