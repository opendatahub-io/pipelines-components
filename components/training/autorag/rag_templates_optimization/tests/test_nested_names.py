"""Module unit testing the functions nested within the kfp.component-decorated functions."""

import ssl
from json import dump as json_dump
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest
from kfp_components.components.training.autorag.rag_templates_optimization.tests.nested_names import (
    Notebook,
    NotebookCell,
    TmpEventHandler,
    _build_pattern_json,
    _create_llama_stack_client,
    _create_openai_client,
    _evaluation_result_fallback,
    _is_ssl_error,
    create_placeholder_mapping,
    generate_notebook_from_templates,
    load_as_langchain_doc,
)


class TestIsSSLError:
    """Tests for _is_ssl_error function."""

    @pytest.mark.parametrize(
        "message,expected",
        [
            ("CERTIFICATE_VERIFY_FAILED", True),
            ("certificate_verify_failed", True),  # case insensitive
            ("SSL error occurred", True),
            ("ssl verification", True),
            ("Connection timeout", False),
            ("Generic error", False),
            ("", False),
        ],
    )
    def test_ssl_error_detection_in_message(self, message, expected):
        """Test SSL error detection from exception message."""
        exc = Exception(message)
        assert _is_ssl_error(exc) == expected

    def test_ssl_error_in_cause_chain(self):
        """Test SSL error detection in __cause__ chain."""
        inner = Exception("CERTIFICATE_VERIFY_FAILED")
        outer = Exception("Connection failed")
        outer.__cause__ = inner
        assert _is_ssl_error(outer) is True

    def test_ssl_error_in_context_chain(self):
        """Test SSL error detection in __context__ chain."""
        inner = Exception("SSL verification failed")
        outer = Exception("Connection failed")
        outer.__context__ = inner
        assert _is_ssl_error(outer) is True

    def test_circular_reference_handling(self):
        """Test that circular references don't cause infinite loop."""
        exc1 = Exception("error1")
        exc2 = Exception("error2")
        exc1.__cause__ = exc2
        exc2.__cause__ = exc1
        assert _is_ssl_error(exc1) is False

    def test_empty_exception_chain(self):
        """Test exception with no cause or context."""
        exc = Exception("No chain")
        assert _is_ssl_error(exc) is False


class TestCreateOpenAIClient:
    """Tests for _create_openai_client function."""

    @patch("kfp_components.components.training.autorag.rag_templates_optimization.tests.nested_names.OpenAI")
    def test_successful_connection(self, mock_openai_class):
        """Test successful client creation without SSL issues."""
        mock_client = MagicMock()
        mock_client.models.list.return_value = []
        mock_openai_class.return_value = mock_client

        ssl_verify = [True]
        result = _create_openai_client("key", "https://api.example.com", ssl_verify)

        assert result == mock_client
        assert ssl_verify[0] is True
        mock_openai_class.assert_called_once_with(api_key="key", base_url="https://api.example.com")

    @patch("kfp_components.components.training.autorag.rag_templates_optimization.tests.nested_names.OpenAI")
    @patch("kfp_components.components.training.autorag.rag_templates_optimization.tests.nested_names.httpx.Client")
    def test_ssl_verification_error_fallback(self, mock_httpx_client, mock_openai_class):
        """Test fallback to unverified client on SSLCertVerificationError."""
        # Add missing module variables
        import logging

        from kfp_components.components.training.autorag.rag_templates_optimization.tests import nested_names

        nested_names._ssl_logger = logging.getLogger("test")

        failing_client = MagicMock()
        failing_client.models.list.side_effect = ssl.SSLCertVerificationError("CERTIFICATE_VERIFY_FAILED")

        success_client = MagicMock()
        success_client.models.list.return_value = []

        mock_openai_class.side_effect = [failing_client, success_client]

        ssl_verify = [True]
        result = _create_openai_client("key", "https://api.example.com", ssl_verify)

        assert result == success_client
        assert ssl_verify[0] is False
        mock_httpx_client.assert_called_once_with(verify=False)

    @patch("kfp_components.components.training.autorag.rag_templates_optimization.tests.nested_names.OpenAI")
    @patch("kfp_components.components.training.autorag.rag_templates_optimization.tests.nested_names.httpx.Client")
    def test_httpx_connect_error_fallback(self, mock_httpx_client, mock_openai_class):
        """Test fallback to unverified client on httpx.ConnectError with SSL."""
        # Add missing module variables
        import logging

        from kfp_components.components.training.autorag.rag_templates_optimization.tests import nested_names

        nested_names._ssl_logger = logging.getLogger("test")

        failing_client = MagicMock()
        failing_client.models.list.side_effect = httpx.ConnectError("SSL verification failed")

        success_client = MagicMock()
        success_client.models.list.return_value = []

        mock_openai_class.side_effect = [failing_client, success_client]

        ssl_verify = [True]
        result = _create_openai_client("key", "https://api.example.com", ssl_verify)

        assert result == success_client
        assert ssl_verify[0] is False

    @patch("kfp_components.components.training.autorag.rag_templates_optimization.tests.nested_names.OpenAI")
    def test_non_ssl_error_propagation(self, mock_openai_class):
        """Test that non-SSL exceptions are re-raised."""
        mock_client = MagicMock()
        mock_client.models.list.side_effect = ValueError("Some other error")
        mock_openai_class.return_value = mock_client

        ssl_verify = [True]
        with pytest.raises(ValueError, match="Some other error"):
            _create_openai_client("key", "https://api.example.com", ssl_verify)


class TestCreateLlamaStackClient:
    """Tests for _create_llama_stack_client function."""

    @patch("kfp_components.components.training.autorag.rag_templates_optimization.tests.nested_names.LlamaStackClient")
    def test_successful_connection(self, mock_client_class):
        """Test successful client creation without SSL issues."""
        mock_client = MagicMock()
        mock_client.models.list.return_value = []
        mock_client_class.return_value = mock_client

        result = _create_llama_stack_client(base_url="https://api.example.com")

        assert result == mock_client
        mock_client_class.assert_called_once_with(base_url="https://api.example.com")

    @patch("kfp_components.components.training.autorag.rag_templates_optimization.tests.nested_names.LlamaStackClient")
    @patch("kfp_components.components.training.autorag.rag_templates_optimization.tests.nested_names.httpx.Client")
    def test_ssl_error_fallback(self, mock_httpx_client, mock_client_class):
        """Test fallback to unverified client on SSL error."""
        # Add missing module variables
        import logging

        from kfp_components.components.training.autorag.rag_templates_optimization.tests import nested_names

        nested_names._ssl_logger = logging.getLogger("test")
        nested_names._ls_ssl_verify = [True]

        failing_client = MagicMock()
        failing_client.models.list.side_effect = ssl.SSLCertVerificationError("SSL error")

        success_client = MagicMock()
        success_client.models.list.return_value = []

        mock_client_class.side_effect = [failing_client, success_client]

        result = _create_llama_stack_client(base_url="https://api.example.com")

        assert result == success_client
        mock_httpx_client.assert_called_once_with(verify=False)

    @patch("kfp_components.components.training.autorag.rag_templates_optimization.tests.nested_names.LlamaStackClient")
    def test_kwargs_propagation(self, mock_client_class):
        """Test that all kwargs are passed to client constructor."""
        mock_client = MagicMock()
        mock_client.models.list.return_value = []
        mock_client_class.return_value = mock_client

        _create_llama_stack_client(base_url="https://api.example.com", api_key="test", timeout=30)

        mock_client_class.assert_called_once_with(base_url="https://api.example.com", api_key="test", timeout=30)


class TestNotebookCell:
    """Tests for NotebookCell class."""

    def test_code_cell_initialization(self):
        """Test code cell creates execution fields."""
        cell = NotebookCell(cell_type="code", source="x=1")

        assert cell.cell_type == "code"
        assert cell.source == "x=1"
        assert cell.execution_count is None
        assert cell.outputs == []

    def test_markdown_cell_initialization(self):
        """Test markdown cell doesn't create execution fields."""
        cell = NotebookCell(cell_type="markdown", source="# Title")

        assert cell.cell_type == "markdown"
        assert cell.source == "# Title"
        assert not hasattr(cell, "execution_count")
        assert not hasattr(cell, "outputs")

    def test_list_source_initialization(self):
        """Test cell with list source."""
        cell = NotebookCell(cell_type="code", source=["line1\n", "line2\n"])

        assert cell.source == ["line1\n", "line2\n"]

    def test_metadata_provided(self):
        """Test cell with custom metadata."""
        metadata = {"tags": ["test"], "custom": "value"}
        cell = NotebookCell(cell_type="code", source="x=1", metadata=metadata)

        assert cell.metadata == metadata

    def test_metadata_none_defaults_to_empty_dict(self):
        """Test that None metadata becomes empty dict."""
        cell = NotebookCell(cell_type="code", source="x=1", metadata=None)

        assert cell.metadata == {}

    def test_to_dict_code_cell(self):
        """Test code cell serialization."""
        cell = NotebookCell(cell_type="code", source="x=1")
        cell.execution_count = 1
        cell.outputs = [{"output_type": "stream"}]

        result = cell.to_dict()

        assert result["cell_type"] == "code"
        assert result["source"] == "x=1"
        assert result["execution_count"] == 1
        assert result["outputs"] == [{"output_type": "stream"}]

    def test_to_dict_markdown_cell(self):
        """Test markdown cell serialization."""
        cell = NotebookCell(cell_type="markdown", source="# Title")

        result = cell.to_dict()

        assert result["cell_type"] == "markdown"
        assert result["source"] == "# Title"
        assert "execution_count" not in result
        assert "outputs" not in result

    @pytest.mark.parametrize(
        "source,mapping,expected",
        [
            ("Hello {name}", {"name": "World"}, "Hello World"),
            ("{a} {b}", {"a": "X", "b": "Y"}, "X Y"),
            ("No placeholders", {"a": "X"}, "No placeholders"),
            ("Missing {placeholder}", {"placeholder": ""}, "Missing "),
            ("", {}, ""),
        ],
    )
    def test_format_source_string(self, source, mapping, expected):
        """Test string source formatting."""
        cell = NotebookCell(cell_type="code", source=source)
        result = cell.format_source(mapping)

        assert result is cell
        assert cell.source == expected

    def test_format_source_list(self):
        """Test list source formatting."""
        cell = NotebookCell(cell_type="code", source=["Hello {name}\n", "Value: {val}\n"])
        result = cell.format_source({"name": "World", "val": "42"})

        assert result is cell
        assert cell.source == ["Hello World\n", "Value: 42\n"]

    def test_format_source_list_missing_placeholder(self):
        """Test list source with missing placeholders."""
        cell = NotebookCell(cell_type="code", source=["Hello {name}\n"])
        _ = cell.format_source({})

        assert cell.source == ["Hello \n"]

    def test_format_source_empty_list(self):
        """Test formatting empty list source."""
        cell = NotebookCell(cell_type="code", source=[])
        result = cell.format_source({"name": "value"})

        assert cell.source == []
        assert result is cell


class TestNotebook:
    """Tests for Notebook class."""

    def test_initialization_defaults(self):
        """Test notebook creation with default parameters."""
        notebook = Notebook()

        assert notebook.cells == []
        assert notebook.metadata["kernelspec"]["name"] == "python3"
        assert notebook.metadata["kernelspec"]["display_name"] == "Python 3"
        assert notebook.metadata["language_info"]["name"] == "python"
        assert notebook.nbformat == 4
        assert notebook.nbformat_minor == 4

    def test_initialization_custom_parameters(self):
        """Test notebook creation with custom parameters."""
        notebook = Notebook(
            kernel_name="custom_kernel",
            kernel_display_name="Custom Kernel",
            language="julia",
            language_version="1.8.0",
        )

        assert notebook.metadata["kernelspec"]["name"] == "custom_kernel"
        assert notebook.metadata["kernelspec"]["display_name"] == "Custom Kernel"
        assert notebook.metadata["language_info"]["name"] == "julia"
        assert notebook.metadata["language_info"]["version"] == "1.8.0"

    def test_initialization_with_cells(self):
        """Test notebook creation with pre-existing cells."""
        cells = [NotebookCell("code", "x=1"), NotebookCell("markdown", "# Title")]
        notebook = Notebook(cells=cells)

        assert len(notebook.cells) == 2
        assert notebook.cells[0].source == "x=1"

    def test_to_dict_empty_notebook(self):
        """Test serialization of empty notebook."""
        notebook = Notebook()
        result = notebook.to_dict()

        assert result["cells"] == []
        assert "metadata" in result
        assert result["nbformat"] == 4
        assert result["nbformat_minor"] == 4

    def test_to_dict_with_cells(self):
        """Test serialization of notebook with cells."""
        cells = [NotebookCell("code", "x=1"), NotebookCell("markdown", "# Title")]
        notebook = Notebook(cells=cells)
        result = notebook.to_dict()

        assert len(result["cells"]) == 2
        assert result["cells"][0]["cell_type"] == "code"
        assert result["cells"][1]["cell_type"] == "markdown"

    def test_save_creates_file(self, tmp_path):
        """Test that save creates a file at specified path."""
        notebook = Notebook()
        output_file = tmp_path / "output.ipynb"

        result = notebook.save(output_file)

        assert output_file.exists()
        assert result is notebook

    def test_save_creates_parent_directories(self, tmp_path):
        """Test that save creates missing parent directories."""
        notebook = Notebook()
        output_file = tmp_path / "nested" / "dir" / "output.ipynb"

        notebook.save(output_file)

        assert output_file.exists()
        assert output_file.parent.exists()

    def test_save_with_string_path(self, tmp_path):
        """Test save with string path."""
        notebook = Notebook()
        output_file = str(tmp_path / "output.ipynb")

        notebook.save(output_file)

        assert Path(output_file).exists()

    def test_save_with_custom_indent(self, tmp_path):
        """Test JSON indentation."""
        notebook = Notebook(cells=[NotebookCell("code", "x=1")])
        output_file = tmp_path / "output.ipynb"

        notebook.save(output_file, indent=4)

        content = output_file.read_text()
        assert "    " in content

    def test_save_overwrites_existing_file(self, tmp_path):
        """Test that save overwrites existing file."""
        output_file = tmp_path / "output.ipynb"
        output_file.write_text("old content")

        notebook = Notebook()
        notebook.save(output_file)

        # File should contain notebook JSON, not "old content"
        content = output_file.read_text()
        assert "old content" not in content
        assert "cells" in content

    def test_load_template_notebook(self, tmp_path, sample_notebook_dict):
        """Test loading a template notebook."""
        # Create a mock embedded_artifact
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        template_file = template_dir / "ls_indexing_template.ipynb"

        with template_file.open("w") as f:
            json_dump(sample_notebook_dict, f)

        # Add embedded_artifact to module namespace
        from kfp_components.components.training.autorag.rag_templates_optimization.tests import nested_names

        mock_artifact = Mock()
        mock_artifact.path = template_dir
        nested_names.embedded_artifact = mock_artifact

        notebook = Notebook.load("ls_indexing_template.ipynb")

        assert len(notebook.cells) == 2
        assert notebook.cells[0].cell_type == "markdown"
        assert notebook.cells[1].cell_type == "code"

    def test_load_preserves_metadata(self, tmp_path, sample_notebook_dict):
        """Test that load preserves original metadata."""
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        template_file = template_dir / "chroma_template.ipynb"

        with template_file.open("w") as f:
            json_dump(sample_notebook_dict, f)

        # Add embedded_artifact to module namespace
        from kfp_components.components.training.autorag.rag_templates_optimization.tests import nested_names

        mock_artifact = Mock()
        mock_artifact.path = template_dir
        nested_names.embedded_artifact = mock_artifact

        notebook = Notebook.load("chroma_template.ipynb")

        assert notebook.metadata == sample_notebook_dict["metadata"]
        assert notebook.nbformat == 4
        assert notebook.nbformat_minor == 4


class TestCreatePlaceholderMapping:
    """Tests for create_placeholder_mapping function."""

    def test_complete_data_extraction(self, sample_output_data):
        """Test all fields extracted from complete data."""
        result = create_placeholder_mapping(
            sample_output_data,
            test_data_key="test.csv",
            input_data_key="docs/",
            chat_model_url="https://chat.example.com",
            embedding_model_url="https://embed.example.com",
            ls_ssl_verify=True,
            chat_ssl_verify=False,
            embedding_ssl_verify=True,
        )

        assert result["PATTERN_NAME"] == "test_pattern"
        assert result["FM_MODEL_ID"] == "gpt-4"
        assert result["SYSTEM_MESSAGE"] == "You are helpful"
        assert result["EMBEDDING_MODEL_ID"] == "text-embedding-ada-002"
        assert result["DISTANCE_METRIC"] == "cosine"
        assert result["PROVIDER_ID"] == "chroma"
        assert result["COLLECTION_NAME"] == "test_collection"
        assert result["RETRIEVAL_METHOD"] == "simple"
        assert result["NUMBER_OF_CHUNKS"] == 5
        assert result["CHUNKING_METHOD"] == "recursive"
        assert result["CHUNK_SIZE"] == 512
        assert result["CHUNK_OVERLAP"] == 50
        assert result["TEST_DATA_KEY"] == "test.csv"
        assert result["INPUT_DATA_KEY"] == "docs/"
        assert result["CHAT_MODEL_URL"] == "https://chat.example.com"
        assert result["EMBEDDING_MODEL_URL"] == "https://embed.example.com"
        assert result["LS_SSL_VERIFY"] is True
        assert result["CHAT_SSL_VERIFY"] is False
        assert result["EMBEDDING_SSL_VERIFY"] is True

    def test_missing_settings(self, minimal_output_data):
        """Test handling of missing settings sections."""
        result = create_placeholder_mapping(minimal_output_data)

        assert result["PATTERN_NAME"] == "minimal_pattern"
        assert result["FM_MODEL_ID"] == ""
        assert result["EMBEDDING_MODEL_ID"] == ""
        assert result["CHUNK_SIZE"] == 512
        assert result["NUMBER_OF_CHUNKS"] == 5

    def test_default_values(self):
        """Test default values for missing fields."""
        output_data = {"name": "test"}
        result = create_placeholder_mapping(output_data)

        assert result["EMBEDDING_PARAMS"] == {"embedding_dimension": 768}
        assert result["NUMBER_OF_CHUNKS"] == 5
        assert result["CHUNK_SIZE"] == 512
        assert result["CHUNK_OVERLAP"] == 50

    def test_default_parameters(self, sample_output_data):
        """Test default function parameters."""
        result = create_placeholder_mapping(sample_output_data)

        assert result["TEST_DATA_KEY"] == ""
        assert result["INPUT_DATA_KEY"] == ""
        assert result["CHAT_MODEL_URL"] == ""
        assert result["EMBEDDING_MODEL_URL"] == ""
        assert result["LS_SSL_VERIFY"] is True
        assert result["CHAT_SSL_VERIFY"] is True
        assert result["EMBEDDING_SSL_VERIFY"] is True


class TestGenerateNotebookFromTemplates:
    """Tests for generate_notebook_from_templates function."""

    def test_template_generation(self, tmp_path, sample_output_data, sample_notebook_dict):
        """Test notebook generation from template."""
        # Setup template directory
        template_dir = tmp_path / "templates"
        template_dir.mkdir()
        template_file = template_dir / "ls_inference_template.ipynb"

        # Create template with placeholders
        template_dict = sample_notebook_dict.copy()
        template_dict["cells"][0]["source"] = "Pattern: {PATTERN_NAME}"

        with template_file.open("w") as f:
            json_dump(template_dict, f)

        # Add variables to module namespace
        from kfp_components.components.training.autorag.rag_templates_optimization.tests import nested_names

        mock_artifact = Mock()
        mock_artifact.path = template_dir
        nested_names.embedded_artifact = mock_artifact
        nested_names.chat_model_url = "https://chat.example.com"
        nested_names.embedding_model_url = "https://embed.example.com"

        output_path = tmp_path / "output.ipynb"

        generate_notebook_from_templates(
            notebook_template="ls_inference",
            output_data=sample_output_data,
            output_notebook_path=output_path,
            test_data_key="test.csv",
        )

        assert output_path.exists()

        # Verify content
        with output_path.open("r") as f:
            from json import load as json_load

            result = json_load(f)

        assert result["cells"][0]["source"] == "Pattern: test_pattern"


class TestTmpEventHandler:
    """Tests for TmpEventHandler class."""

    def test_on_status_change_with_step(self):
        """Test on_status_change with step parameter."""
        from ai4rag.utils.event_handler.event_handler import LogLevel

        handler = TmpEventHandler()

        # Should not raise
        handler.on_status_change(LogLevel.INFO, "Test message", step="step1")

    def test_on_status_change_without_step(self):
        """Test on_status_change with default step=None."""
        from ai4rag.utils.event_handler.event_handler import LogLevel

        handler = TmpEventHandler()

        # Should not raise
        handler.on_status_change(LogLevel.INFO, "Test message")

    def test_on_pattern_creation(self):
        """Test on_pattern_creation method."""
        handler = TmpEventHandler()

        # Should not raise
        handler.on_pattern_creation(payload={"key": "value"}, evaluation_results=[])

    def test_on_pattern_creation_with_kwargs(self):
        """Test on_pattern_creation with additional kwargs."""
        handler = TmpEventHandler()

        # Should not raise
        handler.on_pattern_creation(
            payload={"key": "value"}, evaluation_results=[], extra_param="value", another="param"
        )


class TestLoadAsLangchainDoc:
    """Tests for load_as_langchain_doc function."""

    def test_single_file(self, tmp_path):
        """Test loading single text file."""
        test_file = tmp_path / "doc.txt"
        test_file.write_text("Sample content", encoding="utf-8")

        result = load_as_langchain_doc(test_file)

        assert len(result) == 1
        assert result[0].page_content == "Sample content"
        assert result[0].metadata["document_id"] == "doc"

    def test_directory_with_multiple_files(self, tmp_path):
        """Test loading directory with multiple files."""
        (tmp_path / "doc1.txt").write_text("Content 1", encoding="utf-8")
        (tmp_path / "doc2.txt").write_text("Content 2", encoding="utf-8")

        result = load_as_langchain_doc(tmp_path)

        assert len(result) == 2
        contents = [doc.page_content for doc in result]
        assert "Content 1" in contents
        assert "Content 2" in contents

        doc_ids = [doc.metadata["document_id"] for doc in result]
        assert "doc1" in doc_ids
        assert "doc2" in doc_ids

    def test_string_path_conversion(self, tmp_path):
        """Test that string paths are converted to Path."""
        test_file = tmp_path / "doc.txt"
        test_file.write_text("Content", encoding="utf-8")

        result = load_as_langchain_doc(str(test_file))

        assert len(result) == 1
        assert result[0].page_content == "Content"

    def test_empty_directory(self, tmp_path):
        """Test loading empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        result = load_as_langchain_doc(empty_dir)

        assert result == []

    def test_document_id_extraction(self, tmp_path):
        """Test that file stem is used as document_id."""
        test_file = tmp_path / "my_document.txt"
        test_file.write_text("Content", encoding="utf-8")

        result = load_as_langchain_doc(test_file)

        assert result[0].metadata["document_id"] == "my_document"


class TestConstructModelInstance:
    """Tests for construct_model_instance function."""

    def test_embedding_model_in_memory_scenario(self):
        """Test embedding model construction with in-memory vector store."""
        from kfp_components.components.training.autorag.rag_templates_optimization.tests import nested_names

        # Setup mock loader and node
        mock_loader = Mock()
        mock_node = Mock()

        # Mock the mapping that would be constructed from YAML
        mapping = {"type_": "embedding", "text-embedding-ada-002": {"embedding_dimension": 768}}
        mock_loader.construct_mapping.return_value = mapping

        # Setup module globals
        nested_names.in_memory_vector_store_scenario = True
        mock_client = Mock()
        mock_client.embedding_model = Mock()
        nested_names.client = mock_client

        # Mock the model class
        mock_model = Mock()
        nested_names.OpenAIEmbeddingModel = Mock(return_value=mock_model)

        result = nested_names.construct_model_instance(mock_loader, mock_node)

        assert result == mock_model
        nested_names.OpenAIEmbeddingModel.assert_called_once_with(
            client=mock_client.embedding_model, model_id="text-embedding-ada-002", params={"embedding_dimension": 768}
        )

    def test_embedding_model_llama_stack_scenario(self):
        """Test embedding model construction with LlamaStack."""
        from kfp_components.components.training.autorag.rag_templates_optimization.tests import nested_names

        # Setup mock loader and node
        mock_loader = Mock()
        mock_node = Mock()

        mapping = {"type_": "embedding", "llama-embed-v1": {"embedding_dimension": 1024}}
        mock_loader.construct_mapping.return_value = mapping

        # Setup module globals
        nested_names.in_memory_vector_store_scenario = False
        mock_client = Mock()
        mock_client.llama_stack = Mock()
        nested_names.client = mock_client

        # Mock the model class
        mock_model = Mock()
        nested_names.LSEmbeddingModel = Mock(return_value=mock_model)

        result = nested_names.construct_model_instance(mock_loader, mock_node)

        assert result == mock_model
        nested_names.LSEmbeddingModel.assert_called_once_with(
            client=mock_client.llama_stack, model_id="llama-embed-v1", params={"embedding_dimension": 1024}
        )

    def test_generation_model_in_memory_scenario(self):
        """Test generation model construction with in-memory vector store."""
        from kfp_components.components.training.autorag.rag_templates_optimization.tests import nested_names

        # Setup mock loader and node
        mock_loader = Mock()
        mock_node = Mock()

        mapping = {"type_": "generation", "gpt-4": {"temperature": 0.7, "max_tokens": 100}}
        mock_loader.construct_mapping.return_value = mapping

        # Setup module globals
        nested_names.in_memory_vector_store_scenario = True
        mock_client = Mock()
        mock_client.generation_model = Mock()
        nested_names.client = mock_client

        # Mock the model class
        mock_model = Mock()
        nested_names.OpenAIFoundationModel = Mock(return_value=mock_model)

        result = nested_names.construct_model_instance(mock_loader, mock_node)

        assert result == mock_model
        nested_names.OpenAIFoundationModel.assert_called_once_with(
            client=mock_client.generation_model, model_id="gpt-4", params={"temperature": 0.7, "max_tokens": 100}
        )

    def test_generation_model_llama_stack_scenario(self):
        """Test generation model construction with LlamaStack."""
        from kfp_components.components.training.autorag.rag_templates_optimization.tests import nested_names

        # Setup mock loader and node
        mock_loader = Mock()
        mock_node = Mock()

        mapping = {"type_": "generation", "llama-3-70b": {"temperature": 0.5}}
        mock_loader.construct_mapping.return_value = mapping

        # Setup module globals
        nested_names.in_memory_vector_store_scenario = False
        mock_client = Mock()
        mock_client.llama_stack = Mock()
        nested_names.client = mock_client

        # Mock the model class
        mock_model = Mock()
        nested_names.LSFoundationModel = Mock(return_value=mock_model)

        result = nested_names.construct_model_instance(mock_loader, mock_node)

        assert result == mock_model
        nested_names.LSFoundationModel.assert_called_once_with(
            client=mock_client.llama_stack, model_id="llama-3-70b", params={"temperature": 0.5}
        )

    def test_invalid_mapping_raises_value_error(self):
        """Test that invalid mapping raises ValueError."""
        from kfp_components.components.training.autorag.rag_templates_optimization.tests import nested_names

        # Setup mock loader and node
        mock_loader = Mock()
        mock_node = Mock()

        # Invalid mapping without type_
        mapping = {"model_id": "some-model", "params": {}}
        mock_loader.construct_mapping.return_value = mapping

        with pytest.raises(ValueError, match="Cannot load the yml-serialized !Model tag"):
            nested_names.construct_model_instance(mock_loader, mock_node)

    def test_loader_construct_mapping_called_with_deep_true(self):
        """Test that loader.construct_mapping is called with deep=True."""
        from kfp_components.components.training.autorag.rag_templates_optimization.tests import nested_names

        # Setup mock loader and node
        mock_loader = Mock()
        mock_node = Mock()

        mapping = {"type_": "embedding", "test-model": {"param": "value"}}
        mock_loader.construct_mapping.return_value = mapping

        # Setup module globals
        nested_names.in_memory_vector_store_scenario = True
        mock_client = Mock()
        mock_client.embedding_model = Mock()
        nested_names.client = mock_client
        nested_names.OpenAIEmbeddingModel = Mock(return_value=Mock())

        nested_names.construct_model_instance(mock_loader, mock_node)

        mock_loader.construct_mapping.assert_called_once_with(mock_node, deep=True)


class TestEvaluationResultFallback:
    """Tests for _evaluation_result_fallback function."""

    def test_complete_data(self, mock_eval_data):
        """Test with complete evaluation data."""
        eval_result = Mock()
        eval_result.scores = {"question_scores": {"metric1": {"q1": 0.95}}}

        result = _evaluation_result_fallback([mock_eval_data], eval_result)

        assert len(result) == 1
        assert result[0]["question"] == "What is AI?"
        assert result[0]["correct_answers"] == ["Artificial Intelligence"]
        assert result[0]["answer"] == "AI is Artificial Intelligence"
        assert len(result[0]["answer_contexts"]) == 1
        assert result[0]["answer_contexts"][0]["text"] == "AI stands for Artificial Intelligence"
        assert result[0]["answer_contexts"][0]["document_id"] == "doc1"
        assert result[0]["scores"]["metric1"] == 0.95

    def test_missing_contexts(self, mock_eval_data):
        """Test with missing contexts attribute."""
        mock_eval_data.contexts = None

        eval_result = Mock()
        eval_result.scores = {}

        result = _evaluation_result_fallback([mock_eval_data], eval_result)

        assert result[0]["answer_contexts"] == []

    def test_missing_context_ids(self, mock_eval_data):
        """Test with missing context_ids attribute."""
        mock_eval_data.context_ids = None

        eval_result = Mock()
        eval_result.scores = {}

        result = _evaluation_result_fallback([mock_eval_data], eval_result)

        assert result[0]["answer_contexts"] == []

    def test_evaluation_result_scores_none(self, mock_eval_data):
        """Test with evaluation_result.scores being None."""
        eval_result = Mock()
        eval_result.scores = None

        result = _evaluation_result_fallback([mock_eval_data], eval_result)

        assert result[0]["scores"] == {}

    def test_missing_question_scores(self, mock_eval_data):
        """Test with missing question_scores in scores."""
        eval_result = Mock()
        eval_result.scores = {"other_key": {}}

        result = _evaluation_result_fallback([mock_eval_data], eval_result)

        assert result[0]["scores"] == {}

    def test_empty_eval_data_list(self):
        """Test with empty eval_data_list."""
        eval_result = Mock()
        eval_result.scores = {}

        result = _evaluation_result_fallback([], eval_result)

        assert result == []

    def test_partial_scores(self, mock_eval_data):
        """Test with partial scores matching question_id."""
        eval_result = Mock()
        eval_result.scores = {
            "question_scores": {
                "metric1": {"q1": 0.95, "q2": 0.85},
                "metric2": {"q1": 0.90},
                "metric3": {"q2": 0.80},  # q1 not in this metric
            }
        }

        result = _evaluation_result_fallback([mock_eval_data], eval_result)

        assert result[0]["scores"]["metric1"] == 0.95
        assert result[0]["scores"]["metric2"] == 0.90
        assert "metric3" not in result[0]["scores"]


class TestBuildPatternJson:
    """Tests for _build_pattern_json function."""

    def test_complete_result(self, mock_evaluation_result):
        """Test building pattern JSON from complete result."""
        result = _build_pattern_json(mock_evaluation_result, iteration=1, max_combinations=10)

        assert result["name"] == "test_pattern"
        assert result["iteration"] == 1
        assert result["max_combinations"] == 10
        assert result["duration_seconds"] == 123.45
        assert result["settings"]["chunking"]["method"] == "recursive"
        assert result["settings"]["chunking"]["chunk_size"] == 512
        assert result["settings"]["embedding"]["model_id"] == "embed-model"
        assert result["settings"]["generation"]["model_id"] == "gen-model"

    def test_missing_indexing_params(self):
        """Test with missing indexing_params."""
        result = Mock()
        result.pattern_name = "test"
        result.execution_time = 100
        result.collection = "col"
        result.indexing_params = None
        result.rag_params = {"vector_store": {"datasource_type": "chroma"}}

        json_result = _build_pattern_json(result, 1, 10)

        assert json_result["settings"]["chunking"]["method"] == "recursive"
        assert json_result["settings"]["chunking"]["chunk_size"] == 2048

    def test_ranker_k_zero_vs_none(self, mock_evaluation_result):
        """Test that 0 is included but None is excluded for ranker_k."""
        # Test with ranker_k = 0
        mock_evaluation_result.rag_params["retrieval"]["ranker_k"] = 0
        json_zero = _build_pattern_json(mock_evaluation_result, 1, 10)

        assert "ranker_k" in json_zero["settings"]["retrieval"]
        assert json_zero["settings"]["retrieval"]["ranker_k"] == 0

        # Test with ranker_k missing (None)
        del mock_evaluation_result.rag_params["retrieval"]["ranker_k"]
        json_none = _build_pattern_json(mock_evaluation_result, 1, 10)

        assert "ranker_k" not in json_none["settings"]["retrieval"]

    def test_ranker_alpha_zero_vs_none(self, mock_evaluation_result):
        """Test that 0 is included but None is excluded for ranker_alpha."""
        # Test with ranker_alpha = 0
        mock_evaluation_result.rag_params["retrieval"]["ranker_alpha"] = 0
        json_zero = _build_pattern_json(mock_evaluation_result, 1, 10)

        assert "ranker_alpha" in json_zero["settings"]["retrieval"]
        assert json_zero["settings"]["retrieval"]["ranker_alpha"] == 0

        # Test with ranker_alpha missing (None)
        del mock_evaluation_result.rag_params["retrieval"]["ranker_alpha"]
        json_none = _build_pattern_json(mock_evaluation_result, 1, 10)

        assert "ranker_alpha" not in json_none["settings"]["retrieval"]

    def test_optional_retrieval_fields(self, mock_evaluation_result):
        """Test optional retrieval fields are conditionally included."""
        # Add optional fields
        mock_evaluation_result.rag_params["retrieval"]["search_mode"] = "hybrid"
        mock_evaluation_result.rag_params["retrieval"]["ranker_strategy"] = "rerank"

        result = _build_pattern_json(mock_evaluation_result, 1, 10)

        assert result["settings"]["retrieval"]["search_mode"] == "hybrid"
        assert result["settings"]["retrieval"]["ranker_strategy"] == "rerank"

    def test_embedding_model_id_priority(self):
        """Test embedding model_id extraction priority."""
        result = Mock()
        result.pattern_name = "test"
        result.execution_time = 100
        result.collection = "col"

        # Test priority: idx.embedding
        result.indexing_params = {
            "embedding": {"model_id": "from-idx-embedding"},
            "vector_store": {"datasource_type": "chroma"},
        }
        result.rag_params = {"embeddings": {"model_id": "from-rp-embeddings"}}

        json_result = _build_pattern_json(result, 1, 10)
        assert json_result["settings"]["embedding"]["model_id"] == "from-idx-embedding"

    def test_default_template_texts(self):
        """Test default template texts when missing."""
        result = Mock()
        result.pattern_name = "test"
        result.execution_time = 100
        result.collection = "col"
        result.indexing_params = {"vector_store": {"datasource_type": "chroma"}}
        result.rag_params = {"generation": {}}

        json_result = _build_pattern_json(result, 1, 10)

        assert json_result["settings"]["generation"]["context_template_text"] == "{document}"
        assert "Question: {question}" in json_result["settings"]["generation"]["user_message_text"]
        assert "Please answer the question" in json_result["settings"]["generation"]["system_message_text"]

    def test_embedding_model_id_from_string(self):
        """Test embedding_model_id extraction from string embedding_model."""
        result = Mock()
        result.pattern_name = "test"
        result.execution_time = 100
        result.collection = "col"
        result.indexing_params = {"vector_store": {"datasource_type": "chroma"}}
        result.rag_params = {"embedding_model": "string-embedding-model-id"}

        json_result = _build_pattern_json(result, 1, 10)

        assert json_result["settings"]["embedding"]["model_id"] == "string-embedding-model-id"

    def test_embedding_model_id_from_object(self):
        """Test embedding_model_id extraction from object with model_id attribute."""
        result = Mock()
        result.pattern_name = "test"
        result.execution_time = 100
        result.collection = "col"
        result.indexing_params = {"vector_store": {"datasource_type": "chroma"}}

        # Create a mock object with model_id attribute
        embedding_model_obj = Mock()
        embedding_model_obj.model_id = "object-embedding-model-id"
        result.rag_params = {"embedding_model": embedding_model_obj}

        json_result = _build_pattern_json(result, 1, 10)

        assert json_result["settings"]["embedding"]["model_id"] == "object-embedding-model-id"

    def test_generation_model_id_from_string(self):
        """Test generation_model_id extraction from string foundation_model."""
        result = Mock()
        result.pattern_name = "test"
        result.execution_time = 100
        result.collection = "col"
        result.indexing_params = {"vector_store": {"datasource_type": "chroma"}}
        result.rag_params = {"foundation_model": "string-foundation-model-id"}

        json_result = _build_pattern_json(result, 1, 10)

        assert json_result["settings"]["generation"]["model_id"] == "string-foundation-model-id"

    def test_generation_model_id_from_object(self):
        """Test generation_model_id extraction from object with model_id attribute."""
        result = Mock()
        result.pattern_name = "test"
        result.execution_time = 100
        result.collection = "col"
        result.indexing_params = {"vector_store": {"datasource_type": "chroma"}}

        # Create a mock object with model_id attribute
        foundation_model_obj = Mock()
        foundation_model_obj.model_id = "object-foundation-model-id"
        result.rag_params = {"foundation_model": foundation_model_obj}

        json_result = _build_pattern_json(result, 1, 10)

        assert json_result["settings"]["generation"]["model_id"] == "object-foundation-model-id"
