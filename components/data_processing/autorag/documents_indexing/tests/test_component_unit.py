"""Tests for the documents_indexing component."""

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest import mock

import pytest

from ..component import documents_indexing


@dataclass
class _FakeEmbeddingParams:
    """Dataclass stand-in for OpenAIEmbeddingParams (supports dataclasses.asdict)."""

    embedding_dimension: Optional[int] = None
    context_length: Optional[int] = None


MOCKED_ENV_VARIABLES = {
    "MAAS_BASE_URL": "https://maas.example.com/v1",
    "MAAS_API_KEY": "test-api-key",
    "MILVUS_URI": "https://milvus.example.com:19530",
}


def _make_module_mock(**attrs):
    """Create a MagicMock with explicit attributes (avoids auto-attribute creation)."""
    m = mock.MagicMock()
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def _make_ai4rag_mocks():
    """Build mock modules matching the component's direct imports."""
    mock_create_maas_client = mock.MagicMock(name="create_maas_client")
    mock_DoclingChunker = mock.MagicMock(name="DoclingChunker")
    mock_LangChainChunker = mock.MagicMock(name="LangChainChunker")
    mock_OpenAIEmbeddingModel = mock.MagicMock(name="OpenAIEmbeddingModel")
    mock_OpenAIEmbeddingModel.return_value.params = _FakeEmbeddingParams()
    mock_OpenAIEmbeddingParams = mock.MagicMock(name="OpenAIEmbeddingParams")

    # get_vector_store(...) returns a store used as a context manager; wire
    # __enter__ back to the store so `with get_vector_store(...) as vs` yields it.
    mock_get_vector_store = mock.MagicMock(name="get_vector_store")
    mock_store = mock_get_vector_store.return_value
    mock_store.collection_name = "test-collection"
    mock_store.__enter__.return_value = mock_store
    mock_store.__exit__.return_value = False

    mock_get_vector_store_config = mock.MagicMock(name="get_vector_store_config")
    mock_DoclingDocument = mock.MagicMock(name="DoclingDocument")

    mock_ChunkingConstraints = mock.MagicMock()
    mock_ChunkingConstraints.METHODS = ("recursive", "hybrid")
    mock_ChunkingConstraints.MIN_CHUNK_SIZE = 128
    mock_ChunkingConstraints.MAX_CHUNK_SIZE = 2048

    modules = {
        "ai4rag": mock.MagicMock(),
        "ai4rag.components": mock.MagicMock(),
        "ai4rag.components.utils": _make_module_mock(create_maas_client=mock_create_maas_client),
        "ai4rag.rag": mock.MagicMock(),
        "ai4rag.rag.chunking": _make_module_mock(
            DoclingChunker=mock_DoclingChunker, LangChainChunker=mock_LangChainChunker
        ),
        "ai4rag.rag.embedding": mock.MagicMock(),
        "ai4rag.rag.embedding.openai_model": _make_module_mock(
            OpenAIEmbeddingModel=mock_OpenAIEmbeddingModel, OpenAIEmbeddingParams=mock_OpenAIEmbeddingParams
        ),
        "ai4rag.rag.vector_store": _make_module_mock(
            get_vector_store=mock_get_vector_store, get_vector_store_config=mock_get_vector_store_config
        ),
        "ai4rag.utils": mock.MagicMock(),
        "ai4rag.utils.compat": _make_module_mock(ensure_sqlite3=mock.MagicMock(name="ensure_sqlite3")),
        "ai4rag.utils.constants": _make_module_mock(ChunkingConstraints=mock_ChunkingConstraints),
        "docling_core": mock.MagicMock(),
        "docling_core.types": mock.MagicMock(),
        "docling_core.types.doc": mock.MagicMock(),
        "docling_core.types.doc.document": _make_module_mock(DoclingDocument=mock_DoclingDocument),
    }
    mocks = {
        "create_maas_client": mock_create_maas_client,
        "DoclingChunker": mock_DoclingChunker,
        "LangChainChunker": mock_LangChainChunker,
        "OpenAIEmbeddingModel": mock_OpenAIEmbeddingModel,
        "OpenAIEmbeddingParams": mock_OpenAIEmbeddingParams,
        "get_vector_store": mock_get_vector_store,
        "get_vector_store_config": mock_get_vector_store_config,
        "vector_store": mock_store,
        "DoclingDocument": mock_DoclingDocument,
        "ChunkingConstraints": mock_ChunkingConstraints,
    }
    return modules, mocks


def _make_extracted_text(tmp_path, filenames=None):
    """Create a mock extracted_text artifact with optional JSON files."""
    extracted_dir = tmp_path / "extracted"
    extracted_dir.mkdir()
    for name in filenames or []:
        (extracted_dir / name).write_text("{}")
    artifact = mock.MagicMock()
    artifact.path = str(extracted_dir)
    return artifact


def _make_report_artifact(tmp_path):
    """Create a mock indexing_report output artifact."""
    report = mock.MagicMock()
    report.path = str(tmp_path / "indexing_report.json")
    report.metadata = {}
    return report


def _make_indexing_report_html_artifact(tmp_path):
    """Create a mock indexing_report_html output artifact."""
    art = mock.MagicMock()
    art.path = str(tmp_path / "report.html")
    art.metadata = {}
    return art


def _make_embedded_artifact():
    """Create a mock embedded_artifact pointing to the real HTML template on disk."""
    art = mock.MagicMock()
    art.path = str(Path(__file__).resolve().parents[1] / "indexing_report_template.html")
    return art


def _call_component(tmp_path, modules, mocks, *, filenames=None, **overrides):
    """Invoke the component with sensible defaults and return (report, html) artifacts."""
    extracted = _make_extracted_text(tmp_path, filenames=filenames)
    report = _make_report_artifact(tmp_path)
    html = _make_indexing_report_html_artifact(tmp_path)

    defaults = {
        "embedding_model_id": "embed-v1",
        "extracted_text": extracted,
        "indexing_report": report,
        "indexing_report_html": html,
        "embedded_artifact": _make_embedded_artifact(),
    }
    defaults.update(overrides)

    with mock.patch.dict("sys.modules", modules):
        documents_indexing.python_func(**defaults)

    return report, html


class TestDocumentsIndexingInterface:
    """Tests for component existence and KFP interface."""

    def test_component_function_exists(self):
        """Component factory exists and exposes python_func."""
        assert callable(documents_indexing)
        assert hasattr(documents_indexing, "python_func")

    def test_component_has_expected_interface(self):
        """Component declares expected parameters."""
        sig = inspect.signature(documents_indexing.python_func)
        params = list(sig.parameters)
        assert "embedding_model_id" in params
        assert "extracted_text" in params
        assert "indexing_report" in params
        assert "indexing_report_html" in params
        assert "embedded_artifact" in params
        assert "collection_name" in params
        assert "vector_io_provider_id" not in params
        assert "vector_store_id" not in params
        assert sig.parameters["chunk_size"].default == 1024
        assert sig.parameters["batch_size"].default == 20


class TestDocumentsIndexingValidation:
    """Tests for input validation — runs before any API calls."""

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_empty_embedding_model_id_raises(self, tmp_path):
        """Empty embedding_model_id raises ValueError."""
        modules, mocks = _make_ai4rag_mocks()
        with mock.patch.dict("sys.modules", modules):
            with pytest.raises(ValueError, match="embedding_model_id must be a non-empty string"):
                _call_component(tmp_path, modules, mocks, embedding_model_id="")

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_validates_before_maas_connection(self, tmp_path):
        """Validation errors fire before create_maas_client is called."""
        modules, mocks = _make_ai4rag_mocks()
        with mock.patch.dict("sys.modules", modules):
            with pytest.raises(ValueError):
                _call_component(tmp_path, modules, mocks, embedding_model_id="")
        mocks["create_maas_client"].assert_not_called()

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_whitespace_embedding_model_id_raises(self, tmp_path):
        """Whitespace-only embedding_model_id raises ValueError."""
        modules, mocks = _make_ai4rag_mocks()
        with mock.patch.dict("sys.modules", modules):
            with pytest.raises(ValueError, match="embedding_model_id must be a non-empty string"):
                _call_component(tmp_path, modules, mocks, embedding_model_id="   ")

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_negative_chunk_overlap_raises(self, tmp_path):
        """Negative chunk_overlap raises ValueError."""
        modules, mocks = _make_ai4rag_mocks()
        with mock.patch.dict("sys.modules", modules):
            with pytest.raises(ValueError, match="chunk_overlap must be a non-negative integer"):
                _call_component(tmp_path, modules, mocks, chunk_overlap=-1)

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_invalid_chunking_method_raises(self, tmp_path):
        """Unsupported chunking_method raises ValueError."""
        modules, mocks = _make_ai4rag_mocks()
        with mock.patch.dict("sys.modules", modules):
            with pytest.raises(ValueError, match="chunking_method.*is not supported"):
                _call_component(tmp_path, modules, mocks, chunking_method="semantic")

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_chunk_size_out_of_range_raises(self, tmp_path):
        """Out-of-range chunk_size raises ValueError."""
        modules, mocks = _make_ai4rag_mocks()
        with mock.patch.dict("sys.modules", modules):
            with pytest.raises(ValueError, match="chunk_size must be an integer in the range"):
                _call_component(tmp_path, modules, mocks, chunk_size=64)

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_negative_batch_size_raises(self, tmp_path):
        """Negative batch_size raises ValueError."""
        modules, mocks = _make_ai4rag_mocks()
        with mock.patch.dict("sys.modules", modules):
            with pytest.raises(ValueError, match="batch_size must be a non-negative integer"):
                _call_component(tmp_path, modules, mocks, batch_size=-1)

    def test_missing_maas_env_raises_key_error(self, tmp_path):
        """Missing MaaS env vars raise KeyError."""
        modules, mocks = _make_ai4rag_mocks()
        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch.dict("sys.modules", modules):
                with pytest.raises(KeyError):
                    _call_component(tmp_path, modules, mocks, filenames=["doc.json"])

    def test_missing_vector_db_env_raises_value_error(self, tmp_path):
        """Absent MILVUS_*/PGVECTOR_* env vars raise a descriptive ValueError."""
        modules, mocks = _make_ai4rag_mocks()
        maas_only = {"MAAS_BASE_URL": "https://maas.example.com/v1", "MAAS_API_KEY": "test-api-key"}
        with mock.patch.dict("os.environ", maas_only, clear=True):
            with mock.patch.dict("sys.modules", modules):
                with pytest.raises(ValueError, match="No vector database configuration found"):
                    _call_component(tmp_path, modules, mocks, filenames=["doc.json"])


class TestDocumentsIndexingProcessing:
    """Tests for the document processing pipeline."""

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_creates_maas_client_from_env(self, tmp_path):
        """MaaS client is created with correct env var values."""
        modules, mocks = _make_ai4rag_mocks()
        _call_component(tmp_path, modules, mocks)
        mocks["create_maas_client"].assert_called_once_with(
            base_url="https://maas.example.com/v1",
            api_key="test-api-key",
        )

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_milvus_provider_detected_from_env(self, tmp_path):
        """MILVUS_* env vars select the milvus backend."""
        modules, mocks = _make_ai4rag_mocks()
        _call_component(tmp_path, modules, mocks, filenames=["a.json"])
        mocks["get_vector_store_config"].assert_called_once_with("milvus")

    @mock.patch.dict(
        "os.environ",
        {
            "MAAS_BASE_URL": "https://maas.example.com/v1",
            "MAAS_API_KEY": "test-api-key",
            "PGVECTOR_HOST": "pg.example.com",
        },
        clear=True,
    )
    def test_pgvector_provider_detected_from_env(self, tmp_path):
        """PGVECTOR_* env vars select the pgvector backend when no MILVUS_* keys are set."""
        modules, mocks = _make_ai4rag_mocks()
        mocks["LangChainChunker"].return_value.split_documents.return_value = []
        _call_component(tmp_path, modules, mocks, filenames=["a.json"])
        mocks["get_vector_store_config"].assert_called_once_with("pgvector")

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_recursive_chunker_selected(self, tmp_path):
        """chunking_method='recursive' instantiates LangChainChunker."""
        modules, mocks = _make_ai4rag_mocks()
        mocks["LangChainChunker"].return_value.split_documents.return_value = []
        _call_component(
            tmp_path,
            modules,
            mocks,
            filenames=["a.json"],
            chunking_method="recursive",
            chunk_size=512,
            chunk_overlap=64,
        )
        mocks["LangChainChunker"].assert_called_once_with(method="recursive", chunk_size=512, chunk_overlap=64)
        mocks["DoclingChunker"].assert_not_called()

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_hybrid_chunker_selected(self, tmp_path):
        """chunking_method='hybrid' instantiates DoclingChunker."""
        modules, mocks = _make_ai4rag_mocks()
        mocks["DoclingChunker"].return_value.split_documents.return_value = []
        _call_component(tmp_path, modules, mocks, filenames=["a.json"], chunking_method="hybrid", chunk_size=512)
        mocks["DoclingChunker"].assert_called_once_with(max_tokens=512)
        mocks["LangChainChunker"].assert_not_called()

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_documents_loaded_and_indexed(self, tmp_path):
        """Documents are loaded, chunked, and passed to the vector store."""
        modules, mocks = _make_ai4rag_mocks()
        mock_doc = mock.MagicMock(name="loaded_doc")
        mocks["DoclingDocument"].load_from_json.return_value = mock_doc

        fake_chunks = [mock.MagicMock(name="chunk_1"), mock.MagicMock(name="chunk_2")]
        mocks["LangChainChunker"].return_value.split_documents.return_value = fake_chunks

        _call_component(tmp_path, modules, mocks, filenames=["doc1.json", "doc2.json"])

        assert mocks["DoclingDocument"].load_from_json.call_count == 2
        vector_store = mocks["vector_store"]
        vector_store.add_documents.assert_called_once()
        added_chunks = vector_store.add_documents.call_args[0][0]
        assert len(added_chunks) == 4  # 2 chunks per doc × 2 docs

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_empty_directory_no_vector_store_calls(self, tmp_path):
        """Empty directory produces no vector store calls and an empty report."""
        modules, mocks = _make_ai4rag_mocks()
        _call_component(tmp_path, modules, mocks, filenames=[])

        mocks["get_vector_store"].assert_not_called()
        mocks["DoclingDocument"].load_from_json.assert_not_called()

        report_path = tmp_path / "indexing_report.json"
        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert data["total_documents"] == 0
        assert data["completed"] == 0
        assert data["documents"] == []
        assert "settings" in data

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_get_vector_store_receives_collection_name(self, tmp_path):
        """collection_name is forwarded to get_vector_store."""
        modules, mocks = _make_ai4rag_mocks()
        _call_component(tmp_path, modules, mocks, filenames=["a.json"], collection_name="my-collection")
        call_kwargs = mocks["get_vector_store"].call_args.kwargs
        assert call_kwargs["collection_name"] == "my-collection"

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_get_vector_store_receives_none_collection_name(self, tmp_path):
        """collection_name defaults to None and is forwarded as None."""
        modules, mocks = _make_ai4rag_mocks()
        _call_component(tmp_path, modules, mocks, filenames=["a.json"], collection_name=None)
        call_kwargs = mocks["get_vector_store"].call_args.kwargs
        assert call_kwargs["collection_name"] is None

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_default_parameters(self, tmp_path):
        """Default values for optional parameters."""
        sig = inspect.signature(documents_indexing.python_func)
        assert sig.parameters["chunking_method"].default == "recursive"
        assert sig.parameters["chunk_size"].default == 1024
        assert sig.parameters["chunk_overlap"].default == 0
        assert sig.parameters["batch_size"].default == 20
        assert sig.parameters["collection_name"].default is None
        assert sig.parameters["embedding_params"].default is None


class TestDocumentsIndexingErrorHandling:
    """Tests for per-document error resilience."""

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_failed_document_skipped_and_reported(self, tmp_path):
        """A failing document is skipped; others are still indexed."""
        modules, mocks = _make_ai4rag_mocks()

        good_doc = mock.MagicMock(name="good_doc")
        good_chunks = [mock.MagicMock(name="chunk")]

        def load_side_effect(path):
            if "bad" in str(path):
                raise ValueError("Corrupt JSON")
            return good_doc

        mocks["DoclingDocument"].load_from_json.side_effect = load_side_effect
        mocks["LangChainChunker"].return_value.split_documents.return_value = good_chunks

        _call_component(
            tmp_path,
            modules,
            mocks,
            filenames=["good1.json", "bad.json", "good2.json"],
        )

        vector_store = mocks["vector_store"]
        vector_store.add_documents.assert_called_once()
        added = vector_store.add_documents.call_args[0][0]
        assert len(added) == 2  # only the 2 good docs' chunks

        report_path = tmp_path / "indexing_report.json"
        data = json.loads(report_path.read_text())
        assert data["total_documents"] == 3
        assert data["completed"] == 2
        assert data["failed"] == 1

        failed_entry = next(e for e in data["documents"] if e["status"] == "failed")
        assert failed_entry["file"] == "bad.json"
        assert "Corrupt JSON" in failed_entry["error"]

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_all_documents_fail_produces_report_no_vector_insert(self, tmp_path):
        """When every document fails, no vector store insert occurs."""
        modules, mocks = _make_ai4rag_mocks()
        mocks["DoclingDocument"].load_from_json.side_effect = ValueError("bad")

        _call_component(tmp_path, modules, mocks, filenames=["a.json", "b.json"])

        vector_store = mocks["vector_store"]
        vector_store.add_documents.assert_not_called()

        report_path = tmp_path / "indexing_report.json"
        data = json.loads(report_path.read_text())
        assert data["completed"] == 0
        assert data["failed"] == 2
        assert data["total_chunks"] == 0


class TestDocumentsIndexingReport:
    """Tests for the indexing report artifact."""

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_report_json_structure(self, tmp_path):
        """Report contains expected top-level keys and per-document entries."""
        modules, mocks = _make_ai4rag_mocks()
        mocks["DoclingDocument"].load_from_json.return_value = mock.MagicMock()
        fake_chunks = [mock.MagicMock(), mock.MagicMock(), mock.MagicMock()]
        mocks["LangChainChunker"].return_value.split_documents.return_value = fake_chunks

        _call_component(tmp_path, modules, mocks, filenames=["d1.json", "d2.json"])

        report_path = tmp_path / "indexing_report.json"
        assert report_path.exists()
        data = json.loads(report_path.read_text())

        assert data["total_documents"] == 2
        assert data["completed"] == 2
        assert data["failed"] == 0
        assert data["total_chunks"] == 6  # 3 chunks × 2 docs
        assert len(data["documents"]) == 2
        for entry in data["documents"]:
            assert entry["status"] == "completed"
            assert entry["chunks"] == 3

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_report_metadata_set(self, tmp_path):
        """Report artifact metadata is populated."""
        modules, mocks = _make_ai4rag_mocks()
        report, _ = _call_component(tmp_path, modules, mocks, filenames=[])

        assert report.metadata["display_name"] == "Documents Indexing Report"
        assert report.metadata["total_documents"] == 0
        assert report.metadata["completed"] == 0
        assert report.metadata["failed"] == 0
        assert report.metadata["total_chunks"] == 0

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_report_contains_settings_section(self, tmp_path):
        """Report includes settings with vector store, chunking, and embedding config."""
        modules, mocks = _make_ai4rag_mocks()
        mocks["DoclingDocument"].load_from_json.return_value = mock.MagicMock()
        mocks["LangChainChunker"].return_value.split_documents.return_value = [mock.MagicMock()]
        mocks["vector_store"].collection_name = "vs_auto_generated"
        mocks["OpenAIEmbeddingModel"].return_value.params = _FakeEmbeddingParams(
            embedding_dimension=1024,
            context_length=8192,
        )

        _call_component(
            tmp_path,
            modules,
            mocks,
            filenames=["doc.json"],
            embedding_model_id="bge-m3",
            embedding_params={"embedding_dimension": 1024},
            chunking_method="recursive",
            chunk_size=256,
            chunk_overlap=64,
        )

        report_path = tmp_path / "indexing_report.json"
        data = json.loads(report_path.read_text())

        assert "settings" in data
        settings = data["settings"]
        assert settings["vector_store_binding"]["provider_type"] == "milvus"
        assert settings["vector_store_binding"]["collection_name"] == "vs_auto_generated"
        assert settings["chunking"]["method"] == "recursive"
        assert settings["chunking"]["chunk_size"] == 256
        assert settings["chunking"]["chunk_overlap"] == 64
        assert settings["embedding"]["model_id"] == "bge-m3"
        assert settings["embedding"]["embedding_params"]["embedding_dimension"] == 1024
        assert settings["embedding"]["embedding_params"]["context_length"] == 8192

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_report_settings_in_empty_directory(self, tmp_path):
        """Empty directory report includes settings with the detected provider and null collection."""
        modules, mocks = _make_ai4rag_mocks()

        _call_component(tmp_path, modules, mocks, filenames=[], collection_name=None)

        report_path = tmp_path / "indexing_report.json"
        data = json.loads(report_path.read_text())
        assert "settings" in data
        assert data["settings"]["vector_store_binding"]["provider_type"] == "milvus"
        assert data["settings"]["vector_store_binding"]["collection_name"] is None
        assert data["settings"]["chunking"]["method"] == "recursive"

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_report_settings_resolved_embedding_params(self, tmp_path):
        """Embedding params in settings reflect resolved values from the model."""
        modules, mocks = _make_ai4rag_mocks()
        mocks["DoclingDocument"].load_from_json.return_value = mock.MagicMock()
        mocks["LangChainChunker"].return_value.split_documents.return_value = [mock.MagicMock()]

        _call_component(tmp_path, modules, mocks, filenames=["doc.json"], embedding_params=None)

        report_path = tmp_path / "indexing_report.json"
        data = json.loads(report_path.read_text())
        emb_params = data["settings"]["embedding"]["embedding_params"]
        assert "embedding_dimension" in emb_params
        assert "context_length" in emb_params


class TestDocumentsIndexingHtmlReport:
    """Tests for the HTML report artifact."""

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_indexing_report_html_written(self, tmp_path):
        """HTML report is written with expected structure and content."""
        modules, mocks = _make_ai4rag_mocks()
        mocks["DoclingDocument"].load_from_json.return_value = mock.MagicMock()
        fake_chunks = [mock.MagicMock(), mock.MagicMock()]
        mocks["LangChainChunker"].return_value.split_documents.return_value = fake_chunks
        mocks["vector_store"].collection_name = "vs_abc"

        _, html = _call_component(
            tmp_path,
            modules,
            mocks,
            filenames=["doc1.json", "doc2.json"],
            embedding_model_id="bge-m3",
            chunking_method="recursive",
            chunk_size=512,
        )

        html_path = Path(html.path)
        assert html_path.exists()

        html_text = html_path.read_text(encoding="utf-8")
        assert "Documents Indexing Report" in html_text
        assert "doc1.json" in html_text
        assert "doc2.json" in html_text
        assert "completed" in html_text.lower()
        assert "milvus" in html_text
        assert "bge-m3" in html_text
        assert "vs_abc" in html_text
        assert html.metadata["display_name"] == "Documents Indexing Report"

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_indexing_report_html_shows_failed_documents(self, tmp_path):
        """HTML report shows failed status badge for failed documents."""
        modules, mocks = _make_ai4rag_mocks()
        mocks["DoclingDocument"].load_from_json.side_effect = ValueError("Bad format")

        _, html = _call_component(tmp_path, modules, mocks, filenames=["bad.json"])

        html_text = Path(html.path).read_text(encoding="utf-8")
        assert "failed" in html_text.lower()
        assert "bad.json" in html_text

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_indexing_report_html_written_for_empty_directory(self, tmp_path):
        """HTML report is generated even when no documents are found."""
        modules, mocks = _make_ai4rag_mocks()

        _, html = _call_component(tmp_path, modules, mocks, filenames=[])

        html_path = Path(html.path)
        assert html_path.exists()
        html_text = html_path.read_text(encoding="utf-8")
        assert "Documents Indexing Report" in html_text
        assert "No documents were found" in html_text
        assert html.metadata["display_name"] == "Documents Indexing Report"
