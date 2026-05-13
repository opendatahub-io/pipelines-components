"""Tests for the rag_templates_optimization component."""

import os
import sys
import types
from unittest import mock

import pytest

from ..component import rag_templates_optimization


class _SentinelAbort(Exception):
    """Raised by mocks to abort the component after client creation."""


def _make_httpx_module():
    """Return a minimal fake httpx module with a trackable Client class."""
    mod = types.ModuleType("httpx")

    class ConnectError(Exception):
        pass

    class Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    mod.ConnectError = ConnectError
    mod.Client = Client
    return mod


def _make_minimal_httpx_module():
    """Return a minimal httpx stub for validation-only test paths."""
    mod = types.ModuleType("httpx")

    class ConnectError(Exception):
        pass

    class Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    mod.ConnectError = ConnectError
    mod.Client = Client
    return mod


def _make_llama_stack_client_module():
    """Stub llama_stack_client with a real APIConnectionError (MagicMock breaks except clauses)."""
    mod = types.ModuleType("llama_stack_client")

    class APIConnectionError(Exception):
        pass

    mod.APIConnectionError = APIConnectionError
    mod.LlamaStackClient = mock.MagicMock()
    return mod


def _make_openai_module():
    """Stub openai with a real APIConnectionError (MagicMock breaks except clauses)."""
    mod = types.ModuleType("openai")

    class APIConnectionError(Exception):
        pass

    mod.APIConnectionError = APIConnectionError
    mod.OpenAI = mock.MagicMock()
    return mod


def _make_all_mocks():
    """Build sys.modules patch dict for all heavy dependencies."""
    mocks = {}
    for name in [
        "pysqlite3",
        "ai4rag",
        "ai4rag.core",
        "ai4rag.core.experiment",
        "ai4rag.core.experiment.experiment",
        "ai4rag.core.experiment.results",
        "ai4rag.core.hpo",
        "ai4rag.core.hpo.gam_opt",
        "ai4rag.rag",
        "ai4rag.rag.embedding",
        "ai4rag.rag.embedding.base_model",
        "ai4rag.rag.embedding.llama_stack",
        "ai4rag.rag.embedding.openai_model",
        "ai4rag.rag.foundation_models",
        "ai4rag.rag.foundation_models.base_model",
        "ai4rag.rag.foundation_models.llama_stack",
        "ai4rag.rag.foundation_models.openai_model",
        "ai4rag.search_space",
        "ai4rag.search_space.src",
        "ai4rag.search_space.src.parameter",
        "ai4rag.search_space.src.search_space",
        "ai4rag.utils",
        "ai4rag.utils.event_handler",
        "ai4rag.utils.event_handler.event_handler",
        "langchain_core",
        "langchain_core.documents",
        "pandas",
    ]:
        mocks[name] = mock.MagicMock()

    httpx_mod = _make_httpx_module()
    mocks["httpx"] = httpx_mod

    # yaml needs safe_load to return a dict with .items()
    mock_yaml = mock.MagicMock()
    mock_yaml.safe_load.return_value = {}
    mocks["yaml"] = mock_yaml

    return mocks


def _minimal_dependency_modules():
    """Mock imported heavy third-party modules for validation-path tests."""
    return {
        "pandas": mock.MagicMock(),
        "yaml": mock.MagicMock(),
        "ai4rag": mock.MagicMock(),
        "ai4rag.core": mock.MagicMock(),
        "ai4rag.core.experiment": mock.MagicMock(),
        "ai4rag.core.experiment.experiment": mock.MagicMock(AI4RAGExperiment=mock.MagicMock()),
        "ai4rag.core.experiment.results": mock.MagicMock(ExperimentResults=mock.MagicMock()),
        "ai4rag.core.hpo": mock.MagicMock(),
        "ai4rag.core.hpo.gam_opt": mock.MagicMock(GAMOptSettings=mock.MagicMock()),
        "ai4rag.rag": mock.MagicMock(),
        "ai4rag.rag.embedding": mock.MagicMock(),
        "ai4rag.rag.embedding.base_model": mock.MagicMock(BaseEmbeddingModel=mock.MagicMock()),
        "ai4rag.rag.embedding.llama_stack": mock.MagicMock(LSEmbeddingModel=mock.MagicMock()),
        "ai4rag.rag.embedding.openai_model": mock.MagicMock(OpenAIEmbeddingModel=mock.MagicMock()),
        "ai4rag.rag.foundation_models": mock.MagicMock(),
        "ai4rag.rag.foundation_models.base_model": mock.MagicMock(BaseFoundationModel=mock.MagicMock()),
        "ai4rag.rag.foundation_models.llama_stack": mock.MagicMock(LSFoundationModel=mock.MagicMock()),
        "ai4rag.rag.foundation_models.openai_model": mock.MagicMock(OpenAIFoundationModel=mock.MagicMock()),
        "ai4rag.search_space": mock.MagicMock(),
        "ai4rag.search_space.src": mock.MagicMock(),
        "ai4rag.search_space.src.parameter": mock.MagicMock(Parameter=mock.MagicMock()),
        "ai4rag.search_space.src.search_space": mock.MagicMock(AI4RAGSearchSpace=mock.MagicMock()),
        "ai4rag.utils": mock.MagicMock(),
        "ai4rag.utils.event_handler": mock.MagicMock(),
        "ai4rag.utils.event_handler.event_handler": mock.MagicMock(
            BaseEventHandler=type("BaseEventHandler", (), {}),
            LogLevel=mock.MagicMock(),
        ),
        "langchain_core": mock.MagicMock(),
        "langchain_core.documents": mock.MagicMock(Document=mock.MagicMock()),
        "llama_stack_client": mock.MagicMock(LlamaStackClient=mock.MagicMock()),
        "openai": mock.MagicMock(OpenAI=mock.MagicMock()),
        "httpx": _make_minimal_httpx_module(),
    }


class TestRagTemplatesOptimizationUnitTests:
    """Unit tests for component logic."""

    def test_component_function_exists(self):
        """Test that the component function is properly imported."""
        assert callable(rag_templates_optimization)
        assert hasattr(rag_templates_optimization, "python_func")

    def test_component_with_default_parameters(self):
        """Test component has expected interface (required args)."""
        import inspect

        sig = inspect.signature(rag_templates_optimization.python_func)
        params = list(sig.parameters)
        assert "extracted_text" in params
        assert "test_data" in params
        assert "search_space_prep_report" in params
        assert "rag_patterns" in params

    def test_missing_chat_model_url_raises_type_error(self):
        """Missing required model endpoint args raises TypeError early."""
        with mock.patch.dict(sys.modules, _minimal_dependency_modules()):
            with pytest.raises(TypeError, match="chat_model_url must be a non-empty string"):
                rag_templates_optimization.python_func(
                    extracted_text="/tmp/extracted",
                    test_data="/tmp/test_data.json",
                    search_space_prep_report="/tmp/report.yml",
                    rag_patterns=mock.MagicMock(path="/tmp/rag_patterns", metadata={}, uri=""),
                    embedded_artifact=mock.MagicMock(path="/tmp/embedded"),
                    test_data_key="small-dataset/benchmark.json",
                    chat_model_url="",
                    chat_model_token="token",
                    embedding_model_url="https://emb",
                    embedding_model_token="token",
                    optimization_settings={"metric": "faithfulness", "max_number_of_rag_patterns": 8},
                )

    def _setup_llama_stack_mocks(self, tmp_path, abort_at_experiment=True):
        """Set up mocks and temp files for llama-stack (non-in-memory) vector store tests.

        Returns (mocks, extracted_text, test_data_path, search_space_report).
        """
        mocks = _make_all_mocks()
        llama_mod = _make_llama_stack_client_module()
        mock_ls = mock.MagicMock()
        mock_ls.models.list.return_value = []
        llama_mod.LlamaStackClient.return_value = mock_ls
        mocks["llama_stack_client"] = llama_mod
        mocks["openai"] = _make_openai_module()
        if abort_at_experiment:
            mocks["ai4rag.core.experiment.experiment"].AI4RAGExperiment.side_effect = _SentinelAbort

        search_space_report = tmp_path / "report.yml"
        search_space_report.write_text("{}")
        test_data_path = tmp_path / "test_data.json"
        test_data_path.write_text("[]")
        extracted_text = str(tmp_path / "extracted_text")

        return mocks, extracted_text, str(test_data_path), str(search_space_report)

    def _run_with_llama_stack(self, mocks, extracted_text, test_data, search_space_report, **kwargs):
        """Run the component with llama-stack env vars and the given mocks."""
        with (
            mock.patch.dict(sys.modules, mocks),
            mock.patch.dict(
                os.environ,
                {
                    "LLAMA_STACK_CLIENT_BASE_URL": "https://llama-stack.example.com",
                    "LLAMA_STACK_CLIENT_API_KEY": "test-api-key",
                },
            ),
        ):
            defaults = {
                "extracted_text": extracted_text,
                "test_data": test_data,
                "search_space_prep_report": search_space_report,
                "rag_patterns": mock.MagicMock(path="/tmp/rag_patterns", metadata={}, uri=""),
                "embedded_artifact": mock.MagicMock(path="/tmp/embedded"),
                "test_data_key": "small-dataset/benchmark.json",
                "optimization_settings": {"metric": "faithfulness", "max_number_of_rag_patterns": 8},
            }
            defaults.update(kwargs)
            rag_templates_optimization.python_func(**defaults)

    def test_any_vector_store_id_is_accepted(self, tmp_path):
        """Any non-empty llama_stack_vector_io_provider_id string is accepted (no allowlist)."""
        mocks, extracted_text, test_data, report = self._setup_llama_stack_mocks(tmp_path)
        with pytest.raises(_SentinelAbort):
            self._run_with_llama_stack(
                mocks, extracted_text, test_data, report, llama_stack_vector_io_provider_id="my_custom_milvus"
            )

    def test_ls_prefix_added_for_non_in_memory_scenario(self, tmp_path):
        """AI4RAGExperiment receives vector_store_type with 'ls_' prefix when not in-memory."""
        mocks, extracted_text, test_data, report = self._setup_llama_stack_mocks(tmp_path)
        with pytest.raises(_SentinelAbort):
            self._run_with_llama_stack(
                mocks, extracted_text, test_data, report, llama_stack_vector_io_provider_id="milvus"
            )

        ai4rag_exp = mocks["ai4rag.core.experiment.experiment"].AI4RAGExperiment
        ai4rag_exp.assert_called_once()
        assert ai4rag_exp.call_args.kwargs["vector_store_type"] == "ls_milvus"

    def test_ls_prefix_not_doubled(self, tmp_path):
        """When user provides a value already prefixed with 'ls_', it is not doubled."""
        mocks, extracted_text, test_data, report = self._setup_llama_stack_mocks(tmp_path)
        with pytest.raises(_SentinelAbort):
            self._run_with_llama_stack(
                mocks, extracted_text, test_data, report, llama_stack_vector_io_provider_id="ls_milvus"
            )

        ai4rag_exp = mocks["ai4rag.core.experiment.experiment"].AI4RAGExperiment
        ai4rag_exp.assert_called_once()
        assert ai4rag_exp.call_args.kwargs["vector_store_type"] == "ls_milvus"

    def test_missing_provider_id_non_in_memory_raises_value_error(self, tmp_path):
        """None provider_id in non-in-memory (llama-stack) mode raises ValueError."""
        mocks, extracted_text, test_data, report = self._setup_llama_stack_mocks(tmp_path, abort_at_experiment=False)
        with pytest.raises(ValueError, match="llama_stack_vector_io_provider_id must be provided"):
            self._run_with_llama_stack(mocks, extracted_text, test_data, report, llama_stack_vector_io_provider_id=None)

    def test_whitespace_provider_id_non_in_memory_raises_value_error(self, tmp_path):
        """Whitespace-only provider_id in non-in-memory (llama-stack) mode raises ValueError."""
        mocks, extracted_text, test_data, report = self._setup_llama_stack_mocks(tmp_path, abort_at_experiment=False)
        with pytest.raises(ValueError, match="llama_stack_vector_io_provider_id must be provided"):
            self._run_with_llama_stack(
                mocks, extracted_text, test_data, report, llama_stack_vector_io_provider_id="   "
            )

    def test_max_number_of_rag_patterns_non_numeric_string_raises_value_error(self):
        """UI may pass string parameters; non-numeric strings are rejected with a clear error."""
        with mock.patch.dict(sys.modules, _minimal_dependency_modules()):
            with pytest.raises(ValueError, match="max_number_of_rag_patterns must be a valid integer"):
                rag_templates_optimization.python_func(
                    extracted_text="/tmp/extracted",
                    test_data="/tmp/test_data.json",
                    search_space_prep_report="/tmp/report.yml",
                    rag_patterns=mock.MagicMock(path="/tmp/rag_patterns", metadata={}, uri=""),
                    embedded_artifact=mock.MagicMock(path="/tmp/embedded"),
                    test_data_key="small-dataset/benchmark.json",
                    chat_model_url="https://chat",
                    chat_model_token="token",
                    embedding_model_url="https://emb",
                    embedding_model_token="token",
                    optimization_settings={
                        "metric": "faithfulness",
                        "max_number_of_rag_patterns": "not-a-number",
                    },
                )

    @mock.patch.dict(
        os.environ,
        {
            "LLAMA_STACK_CLIENT_BASE_URL": "https://llama-stack.example.com",
            "LLAMA_STACK_CLIENT_API_KEY": "test-api-key",
        },
    )
    def test_max_number_of_rag_patterns_numeric_string_coerced_for_gam_opt(self, tmp_path):
        """Pipeline UI often sends numbers as strings; they must coerce to int for GAMOptSettings."""
        mocks = _make_all_mocks()
        llama_mod = _make_llama_stack_client_module()
        mock_ls = mock.MagicMock()
        mock_ls.models.list.return_value = []
        llama_mod.LlamaStackClient.return_value = mock_ls
        mocks["llama_stack_client"] = llama_mod
        mocks["openai"] = _make_openai_module()
        mocks["ai4rag.core.experiment.experiment"].AI4RAGExperiment.side_effect = _SentinelAbort

        search_space_report = tmp_path / "report.yml"
        search_space_report.write_text("{}")
        extracted_text = str(tmp_path / "extracted_text")
        test_data_path = tmp_path / "test_data.json"
        test_data_path.write_text("[]")
        test_data = str(test_data_path)
        rag_patterns = mock.MagicMock()
        embedded_artifact = mock.MagicMock()

        with mock.patch.dict(sys.modules, mocks):
            with pytest.raises(_SentinelAbort):
                rag_templates_optimization.python_func(
                    extracted_text=extracted_text,
                    test_data=test_data,
                    search_space_prep_report=str(search_space_report),
                    rag_patterns=rag_patterns,
                    embedded_artifact=embedded_artifact,
                    test_data_key="small-dataset/benchmark.json",
                    llama_stack_vector_io_provider_id="milvus",
                    optimization_settings={"metric": "faithfulness", "max_number_of_rag_patterns": "8"},
                )

        mocks["ai4rag.core.hpo.gam_opt"].GAMOptSettings.assert_called_once_with(max_evals=8)
