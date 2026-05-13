"""Tests for the search_space_preparation component."""

import ssl
import sys
import types
from unittest import mock

import pytest

from ..component import search_space_preparation


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
        "ai4rag.core.experiment.benchmark_data",
        "ai4rag.core.experiment.mps",
        "ai4rag.rag",
        "ai4rag.rag.embedding",
        "ai4rag.rag.embedding.base_model",
        "ai4rag.rag.embedding.openai_model",
        "ai4rag.rag.foundation_models",
        "ai4rag.rag.foundation_models.base_model",
        "ai4rag.rag.foundation_models.openai_model",
        "ai4rag.search_space",
        "ai4rag.search_space.prepare",
        "ai4rag.search_space.prepare.prepare_search_space",
        "ai4rag.search_space.src",
        "ai4rag.search_space.src.parameter",
        "ai4rag.search_space.src.search_space",
        "langchain_core",
        "langchain_core.documents",
        "pandas",
        "yaml",
    ]:
        mocks[name] = mock.MagicMock()

    httpx_mod = _make_httpx_module()
    mocks["httpx"] = httpx_mod
    return mocks


def _minimal_dependency_modules():
    """Mock imported heavy third-party modules for validation-path tests."""
    return {
        "pandas": mock.MagicMock(),
        "yaml": mock.MagicMock(),
        "ai4rag": mock.MagicMock(),
        "ai4rag.core": mock.MagicMock(),
        "ai4rag.core.experiment": mock.MagicMock(),
        "ai4rag.core.experiment.benchmark_data": mock.MagicMock(BenchmarkData=mock.MagicMock()),
        "ai4rag.core.experiment.mps": mock.MagicMock(ModelsPreSelector=mock.MagicMock()),
        "ai4rag.rag": mock.MagicMock(),
        "ai4rag.rag.embedding": mock.MagicMock(),
        "ai4rag.rag.embedding.base_model": mock.MagicMock(BaseEmbeddingModel=mock.MagicMock()),
        "ai4rag.rag.embedding.openai_model": mock.MagicMock(OpenAIEmbeddingModel=mock.MagicMock()),
        "ai4rag.rag.foundation_models": mock.MagicMock(),
        "ai4rag.rag.foundation_models.base_model": mock.MagicMock(BaseFoundationModel=mock.MagicMock()),
        "ai4rag.rag.foundation_models.openai_model": mock.MagicMock(OpenAIFoundationModel=mock.MagicMock()),
        "ai4rag.search_space": mock.MagicMock(),
        "ai4rag.search_space.prepare": mock.MagicMock(),
        "ai4rag.search_space.prepare.prepare_search_space": mock.MagicMock(
            prepare_search_space_with_llama_stack=mock.MagicMock()
        ),
        "ai4rag.search_space.src": mock.MagicMock(),
        "ai4rag.search_space.src.parameter": mock.MagicMock(Parameter=mock.MagicMock()),
        "ai4rag.search_space.src.search_space": mock.MagicMock(AI4RAGSearchSpace=mock.MagicMock()),
        "langchain_core": mock.MagicMock(),
        "langchain_core.documents": mock.MagicMock(Document=mock.MagicMock()),
        "llama_stack_client": mock.MagicMock(LlamaStackClient=mock.MagicMock()),
        "openai": mock.MagicMock(OpenAI=mock.MagicMock()),
        "httpx": _make_minimal_httpx_module(),
    }


class TestSearchSpacePreparationUnitTests:
    """Unit tests for component logic."""

    def test_component_function_exists(self):
        """Test that the component function is properly imported."""
        assert callable(search_space_preparation)
        assert hasattr(search_space_preparation, "python_func")

    def test_component_with_default_parameters(self):
        """Test component has expected interface (required args)."""
        import inspect

        sig = inspect.signature(search_space_preparation.python_func)
        params = list(sig.parameters)
        assert "test_data" in params
        assert "extracted_text" in params
        assert "search_space_prep_report" in params

    def test_non_list_embeddings_models_raises_type_error(self):
        """embeddings_models must be a list when provided."""
        with mock.patch.dict(sys.modules, _minimal_dependency_modules()):
            with pytest.raises(TypeError, match="embeddings_models must be a list"):
                search_space_preparation.python_func(
                    test_data=mock.MagicMock(path="/tmp/test_data.json"),
                    extracted_text=mock.MagicMock(path="/tmp/extracted"),
                    search_space_prep_report=mock.MagicMock(path="/tmp/report.yaml"),
                    embeddings_models="not-a-list",
                )

    def test_non_list_generation_models_raises_type_error(self):
        """generation_models must be a list."""
        with mock.patch.dict(sys.modules, _minimal_dependency_modules()):
            with pytest.raises(TypeError, match="generation_models must be a list"):
                search_space_preparation.python_func(
                    test_data=mock.MagicMock(path="/tmp/test_data.json"),
                    extracted_text=mock.MagicMock(path="/tmp/extracted"),
                    search_space_prep_report=mock.MagicMock(path="/tmp/report.yaml"),
                    generation_models="not-a-list",
                )

    def test_unsupported_metric_raises_value_error(self):
        """Unsupported metric value raises ValueError with supported list."""
        with mock.patch.dict(sys.modules, _minimal_dependency_modules()):
            with pytest.raises(ValueError, match="is not supported"):
                search_space_preparation.python_func(
                    test_data=mock.MagicMock(path="/tmp/test_data.json"),
                    extracted_text=mock.MagicMock(path="/tmp/extracted"),
                    search_space_prep_report=mock.MagicMock(path="/tmp/report.yaml"),
                    embeddings_models=["embed-a"],
                    generation_models=["gen-a"],
                    metric="unsupported_metric",
                )

    @mock.patch.dict(
        "os.environ",
        {
            "LLAMA_STACK_CLIENT_BASE_URL": "https://llama-stack.example.com",
            "LLAMA_STACK_CLIENT_API_KEY": "test-api-key",
        },
    )
    def test_no_models_with_llama_stack_does_not_raise(self, tmp_path):
        """When no model lists are provided, LlamaStack auto-discovers models — no early validation error."""
        mocks = _make_all_mocks()

        llama_mod = _make_llama_stack_client_module()
        mock_ls = mock.MagicMock()
        mock_ls.models.list.return_value = []
        llama_mod.LlamaStackClient.return_value = mock_ls
        mocks["llama_stack_client"] = llama_mod
        mocks["openai"] = _make_openai_module()

        # Abort after search space preparation to avoid full pipeline execution
        mocks[
            "ai4rag.search_space.prepare.prepare_search_space"
        ].prepare_search_space_with_llama_stack.side_effect = _SentinelAbort

        with mock.patch.dict(sys.modules, mocks):
            with pytest.raises(_SentinelAbort):
                search_space_preparation.python_func(
                    test_data=mock.MagicMock(path=str(tmp_path / "test_data.json")),
                    extracted_text=mock.MagicMock(path=str(tmp_path / "extracted")),
                    search_space_prep_report=mock.MagicMock(path=str(tmp_path / "report.yml")),
                )

    @mock.patch.dict(
        "os.environ",
        {
            "LLAMA_STACK_CLIENT_BASE_URL": "https://llama-stack.example.com",
            "LLAMA_STACK_CLIENT_API_KEY": "test-api-key",
        },
    )
    def test_partial_model_lists_with_llama_stack(self, tmp_path):
        """Only generation_models provided — LlamaStack discovers embedding models automatically."""
        mocks = _make_all_mocks()

        llama_mod = _make_llama_stack_client_module()
        mock_ls = mock.MagicMock()
        mock_ls.models.list.return_value = []
        llama_mod.LlamaStackClient.return_value = mock_ls
        mocks["llama_stack_client"] = llama_mod
        mocks["openai"] = _make_openai_module()

        mocks[
            "ai4rag.search_space.prepare.prepare_search_space"
        ].prepare_search_space_with_llama_stack.side_effect = _SentinelAbort

        with mock.patch.dict(sys.modules, mocks):
            with pytest.raises(_SentinelAbort):
                search_space_preparation.python_func(
                    test_data=mock.MagicMock(path=str(tmp_path / "test_data.json")),
                    extracted_text=mock.MagicMock(path=str(tmp_path / "extracted")),
                    search_space_prep_report=mock.MagicMock(path=str(tmp_path / "report.yml")),
                    generation_models=["gen-model-a"],
                )

    def test_no_models_no_llama_stack_no_endpoints_raises(self):
        """Without LlamaStack env vars or model endpoints, ValueError is raised."""
        with mock.patch.dict(sys.modules, _minimal_dependency_modules()):
            with pytest.raises(ValueError, match="have to be defined"):
                search_space_preparation.python_func(
                    test_data=mock.MagicMock(path="/tmp/test_data.json"),
                    extracted_text=mock.MagicMock(path="/tmp/extracted"),
                    search_space_prep_report=mock.MagicMock(path="/tmp/report.yaml"),
                )


class TestSSLFallbackSearchSpacePreparation:
    """Tests for SSL retry logic in _create_llama_stack_client and _create_openai_client."""

    def _make_artifacts(self, tmp_path):
        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test_data.json")
        extracted_text = mock.MagicMock()
        extracted_text.path = str(tmp_path / "extracted")
        search_space_prep_report = mock.MagicMock()
        search_space_prep_report.path = str(tmp_path / "report.yml")
        return test_data, extracted_text, search_space_prep_report

    @mock.patch.dict(
        "os.environ",
        {
            "LLAMA_STACK_CLIENT_BASE_URL": "https://llama-stack.example.com",
            "LLAMA_STACK_CLIENT_API_KEY": "test-api-key",
        },
    )
    def test_llama_stack_client_api_connection_error_wrapping_ssl_retries(self, tmp_path):
        """LSAPIConnectionError wrapping an SSL cause triggers the verify=False retry (production case)."""
        mocks = _make_all_mocks()

        llama_mod = _make_llama_stack_client_module()
        LSAPIConnectionError = llama_mod.APIConnectionError
        ssl_err = ssl.SSLCertVerificationError("CERTIFICATE_VERIFY_FAILED: self-signed certificate")
        api_err = LSAPIConnectionError("Connection error.")
        api_err.__cause__ = ssl_err

        mock_ls_client_fail = mock.MagicMock()
        mock_ls_client_fail.models.list.side_effect = api_err
        mock_ls_client_ok = mock.MagicMock()
        mock_ls_client_ok.models.list.return_value = []

        ls_call_count = 0
        ls_kwargs_history = []

        def fake_ls_client(**kwargs):
            nonlocal ls_call_count
            ls_call_count += 1
            ls_kwargs_history.append(kwargs)
            if ls_call_count == 1:
                return mock_ls_client_fail
            return mock_ls_client_ok

        llama_mod.LlamaStackClient.side_effect = fake_ls_client
        mocks["llama_stack_client"] = llama_mod
        mocks["openai"] = _make_openai_module()

        mocks[
            "ai4rag.search_space.prepare.prepare_search_space"
        ].prepare_search_space_with_llama_stack.side_effect = _SentinelAbort

        test_data, extracted_text, search_space_prep_report = self._make_artifacts(tmp_path)

        with mock.patch.dict(sys.modules, mocks):
            with pytest.raises(_SentinelAbort):
                search_space_preparation.python_func(
                    test_data=test_data,
                    extracted_text=extracted_text,
                    search_space_prep_report=search_space_prep_report,
                    chat_model_url="http://chat.example.com",
                    chat_model_token="chat-token",
                    embedding_model_url="http://embed.example.com",
                    embedding_model_token="embed-token",
                )

        assert ls_call_count == 2, "LlamaStackClient should be instantiated twice (initial + SSL retry)"
        assert ls_kwargs_history[0].get("http_client") is None
        assert isinstance(ls_kwargs_history[1].get("http_client"), mocks["httpx"].Client)
        assert ls_kwargs_history[1]["http_client"].kwargs.get("verify") is False

    @mock.patch.dict(
        "os.environ",
        {
            "LLAMA_STACK_CLIENT_BASE_URL": "https://llama-stack.example.com",
            "LLAMA_STACK_CLIENT_API_KEY": "test-api-key",
        },
    )
    def test_llama_stack_client_api_connection_error_non_ssl_cause_is_reraised(self, tmp_path):
        """LSAPIConnectionError whose cause is NOT SSL propagates without retry."""
        mocks = _make_all_mocks()

        llama_mod = _make_llama_stack_client_module()
        LSAPIConnectionError = llama_mod.APIConnectionError
        err = LSAPIConnectionError("Connection timeout")
        err.__cause__ = TimeoutError("timed out")

        mock_ls_client = mock.MagicMock()
        mock_ls_client.models.list.side_effect = err
        llama_mod.LlamaStackClient.return_value = mock_ls_client
        mocks["llama_stack_client"] = llama_mod
        mocks["openai"] = _make_openai_module()

        test_data, extracted_text, search_space_prep_report = self._make_artifacts(tmp_path)

        with mock.patch.dict(sys.modules, mocks):
            with pytest.raises(LSAPIConnectionError):
                search_space_preparation.python_func(
                    test_data=test_data,
                    extracted_text=extracted_text,
                    search_space_prep_report=search_space_prep_report,
                    chat_model_url="http://chat.example.com",
                    chat_model_token="chat-token",
                    embedding_model_url="http://embed.example.com",
                    embedding_model_token="embed-token",
                )

    def test_openai_client_api_connection_error_wrapping_ssl_retries(self, tmp_path):
        """OAIAPIConnectionError wrapping an SSL cause triggers the verify=False retry (production case)."""
        mocks = _make_all_mocks()

        openai_mod = _make_openai_module()
        OAIAPIConnectionError = openai_mod.APIConnectionError
        ssl_err = ssl.SSLCertVerificationError("CERTIFICATE_VERIFY_FAILED: self-signed certificate")
        api_err = OAIAPIConnectionError("Connection error.")
        api_err.__cause__ = ssl_err

        openai_call_count = 0
        openai_kwargs_history = []

        mock_openai_client_fail = mock.MagicMock()
        mock_openai_client_fail.models.list.side_effect = api_err
        mock_openai_client_ok = mock.MagicMock()
        # _get_model_metadata_from calls models.list() on the ok client and accesses .data
        _models_list_ok = mock.MagicMock()
        _models_list_ok.data = [mock.MagicMock(id="test-model", max_model_len=8192)]
        mock_openai_client_ok.models.list.return_value = _models_list_ok

        def fake_openai(**kwargs):
            nonlocal openai_call_count
            openai_call_count += 1
            openai_kwargs_history.append(kwargs)
            if openai_call_count % 2 == 1:
                return mock_openai_client_fail
            return mock_openai_client_ok

        openai_mod.OpenAI.side_effect = fake_openai
        mocks["openai"] = openai_mod
        mocks["llama_stack_client"] = _make_llama_stack_client_module()

        mocks["ai4rag.search_space.src.search_space"].AI4RAGSearchSpace.side_effect = _SentinelAbort

        test_data, extracted_text, search_space_prep_report = self._make_artifacts(tmp_path)

        with mock.patch.dict(sys.modules, mocks):
            with pytest.raises(_SentinelAbort):
                search_space_preparation.python_func(
                    test_data=test_data,
                    extracted_text=extracted_text,
                    search_space_prep_report=search_space_prep_report,
                    chat_model_url="http://chat.example.com",
                    chat_model_token="chat-token",
                    embedding_model_url="http://embed.example.com",
                    embedding_model_token="embed-token",
                )

        assert openai_call_count == 8
        for retry_idx in [1, 3, 5, 7]:
            assert isinstance(openai_kwargs_history[retry_idx].get("http_client"), mocks["httpx"].Client)
            assert openai_kwargs_history[retry_idx]["http_client"].kwargs.get("verify") is False

    def test_openai_client_api_connection_error_non_ssl_cause_is_reraised(self, tmp_path):
        """OAIAPIConnectionError whose cause is NOT SSL propagates without retry."""
        mocks = _make_all_mocks()

        openai_mod = _make_openai_module()
        OAIAPIConnectionError = openai_mod.APIConnectionError
        err = OAIAPIConnectionError("Connection timeout")
        err.__cause__ = TimeoutError("timed out")

        mock_openai_client = mock.MagicMock()
        mock_openai_client.models.list.side_effect = err
        openai_mod.OpenAI.return_value = mock_openai_client
        mocks["openai"] = openai_mod
        mocks["llama_stack_client"] = _make_llama_stack_client_module()

        test_data, extracted_text, search_space_prep_report = self._make_artifacts(tmp_path)

        with mock.patch.dict(sys.modules, mocks):
            with pytest.raises(OAIAPIConnectionError):
                search_space_preparation.python_func(
                    test_data=test_data,
                    extracted_text=extracted_text,
                    search_space_prep_report=search_space_prep_report,
                    chat_model_url="http://chat.example.com",
                    chat_model_token="chat-token",
                    embedding_model_url="http://embed.example.com",
                    embedding_model_token="embed-token",
                )
