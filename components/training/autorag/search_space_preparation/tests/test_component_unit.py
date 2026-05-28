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


def _make_ogx_client_module():
    """Stub ogx_client with a real APIConnectionError (MagicMock breaks except clauses)."""
    mod = types.ModuleType("ogx_client")

    class APIConnectionError(Exception):
        pass

    mod.APIConnectionError = APIConnectionError
    mod.OgxClient = mock.MagicMock()
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
        "ai4rag.rag.embedding.ogx",
        "ai4rag.rag.foundation_models",
        "ai4rag.rag.foundation_models.base_model",
        "ai4rag.rag.foundation_models.ogx",
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
        "ai4rag.rag.embedding.ogx": mock.MagicMock(OGXEmbeddingModel=mock.MagicMock()),
        "ai4rag.rag.foundation_models": mock.MagicMock(),
        "ai4rag.rag.foundation_models.base_model": mock.MagicMock(BaseFoundationModel=mock.MagicMock()),
        "ai4rag.rag.foundation_models.ogx": mock.MagicMock(OGXFoundationModel=mock.MagicMock()),
        "ai4rag.search_space": mock.MagicMock(),
        "ai4rag.search_space.prepare": mock.MagicMock(),
        "ai4rag.search_space.prepare.prepare_search_space": mock.MagicMock(
            prepare_search_space_with_ogx=mock.MagicMock()
        ),
        "ai4rag.search_space.src": mock.MagicMock(),
        "ai4rag.search_space.src.parameter": mock.MagicMock(Parameter=mock.MagicMock()),
        "ai4rag.search_space.src.search_space": mock.MagicMock(AI4RAGSearchSpace=mock.MagicMock()),
        "langchain_core": mock.MagicMock(),
        "langchain_core.documents": mock.MagicMock(Document=mock.MagicMock()),
        "ogx_client": mock.MagicMock(OgxClient=mock.MagicMock()),
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

    def test_non_list_embedding_models_raises_type_error(self):
        """embedding_models must be a list when provided."""
        with mock.patch.dict(sys.modules, _minimal_dependency_modules()):
            with pytest.raises(TypeError, match="embedding_models must be a list"):
                search_space_preparation.python_func(
                    test_data=mock.MagicMock(path="/tmp/test_data.json"),
                    extracted_text=mock.MagicMock(path="/tmp/extracted"),
                    search_space_prep_report=mock.MagicMock(path="/tmp/report.yaml"),
                    embedding_models="not-a-list",
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
                    embedding_models=["embed-a"],
                    generation_models=["gen-a"],
                    metric="unsupported_metric",
                )

    @mock.patch.dict(
        "os.environ",
        {
            "OGX_CLIENT_BASE_URL": "https://ogx.example.com",
            "OGX_CLIENT_API_KEY": "test-api-key",
        },
    )
    def test_no_models_with_ogx_does_not_raise(self, tmp_path):
        """When no model lists are provided, OGX auto-discovers models — no early validation error."""
        mocks = _make_all_mocks()

        ogx_mod = _make_ogx_client_module()
        mock_ogx = mock.MagicMock()
        mock_ogx.models.list.return_value = []
        ogx_mod.OgxClient.return_value = mock_ogx
        mocks["ogx_client"] = ogx_mod

        # Abort after search space preparation to avoid full pipeline execution
        mocks[
            "ai4rag.search_space.prepare.prepare_search_space"
        ].prepare_search_space_with_ogx.side_effect = _SentinelAbort

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
            "OGX_CLIENT_BASE_URL": "https://ogx.example.com",
            "OGX_CLIENT_API_KEY": "test-api-key",
        },
    )
    def test_partial_model_lists_with_ogx(self, tmp_path):
        """Only generation_models provided — OGX discovers embedding models automatically."""
        mocks = _make_all_mocks()

        ogx_mod = _make_ogx_client_module()
        mock_ogx = mock.MagicMock()
        mock_ogx.models.list.return_value = []
        ogx_mod.OgxClient.return_value = mock_ogx
        mocks["ogx_client"] = ogx_mod

        mocks[
            "ai4rag.search_space.prepare.prepare_search_space"
        ].prepare_search_space_with_ogx.side_effect = _SentinelAbort

        with mock.patch.dict(sys.modules, mocks):
            with pytest.raises(_SentinelAbort):
                search_space_preparation.python_func(
                    test_data=mock.MagicMock(path=str(tmp_path / "test_data.json")),
                    extracted_text=mock.MagicMock(path=str(tmp_path / "extracted")),
                    search_space_prep_report=mock.MagicMock(path=str(tmp_path / "report.yml")),
                    generation_models=["gen-model-a"],
                )

    def test_missing_ogx_env_vars_raises_value_error(self):
        """Missing OGX environment variables raises ValueError."""
        with mock.patch.dict(sys.modules, _minimal_dependency_modules()):
            with mock.patch.dict("os.environ", {}, clear=True):
                with pytest.raises(ValueError, match="OGX_CLIENT_BASE_URL and OGX_CLIENT_API_KEY"):
                    search_space_preparation.python_func(
                        test_data=mock.MagicMock(path="/tmp/test_data.json"),
                        extracted_text=mock.MagicMock(path="/tmp/extracted"),
                        search_space_prep_report=mock.MagicMock(path="/tmp/report.yaml"),
                    )


class TestSSLFallbackSearchSpacePreparation:
    """Tests for SSL retry logic in _create_ogx_client."""

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
            "OGX_CLIENT_BASE_URL": "https://ogx.example.com",
            "OGX_CLIENT_API_KEY": "test-api-key",
        },
    )
    def test_ogx_client_ssl_retry_with_verify_false(self, tmp_path):
        """SSL error on models.list() retries OgxClient with verify=False."""
        mocks = _make_all_mocks()

        mock_ogx_client_ok = mock.MagicMock()
        mock_ogx_client_fail = mock.MagicMock()
        mock_ogx_client_fail.models.list.side_effect = ssl.SSLCertVerificationError(
            "CERTIFICATE_VERIFY_FAILED: self-signed certificate"
        )
        mock_ogx_client_ok.models.list.return_value = []

        ogx_call_count = 0
        ogx_kwargs_history = []

        def fake_ogx_client(**kwargs):
            nonlocal ogx_call_count
            ogx_call_count += 1
            ogx_kwargs_history.append(kwargs)
            if ogx_call_count == 1:
                return mock_ogx_client_fail
            return mock_ogx_client_ok

        ogx_mod = _make_ogx_client_module()
        ogx_mod.OgxClient.side_effect = fake_ogx_client

        # Abort after client creation by making prepare_search_space_with_ogx raise
        mocks[
            "ai4rag.search_space.prepare.prepare_search_space"
        ].prepare_search_space_with_ogx.side_effect = _SentinelAbort
        mocks["ogx_client"] = ogx_mod

        test_data, extracted_text, search_space_prep_report = self._make_artifacts(tmp_path)

        with mock.patch.dict(sys.modules, mocks):
            with pytest.raises(_SentinelAbort):
                search_space_preparation.python_func(
                    test_data=test_data,
                    extracted_text=extracted_text,
                    search_space_prep_report=search_space_prep_report,
                )

        assert ogx_call_count == 2, "OgxClient should be instantiated twice (initial + SSL retry)"
        assert ogx_kwargs_history[0].get("http_client") is None, "First call should not disable SSL"
        assert isinstance(ogx_kwargs_history[1].get("http_client"), mocks["httpx"].Client), (
            "Second call should pass httpx.Client"
        )
        assert ogx_kwargs_history[1]["http_client"].kwargs.get("verify") is False

    @mock.patch.dict(
        "os.environ",
        {
            "OGX_CLIENT_BASE_URL": "https://ogx.example.com",
            "OGX_CLIENT_API_KEY": "test-api-key",
        },
    )
    def test_ogx_client_api_connection_error_wrapping_ssl_retries(self, tmp_path):
        """OGXAPIConnectionError wrapping an SSL cause triggers the verify=False retry (production case)."""
        mocks = _make_all_mocks()

        ogx_mod = _make_ogx_client_module()
        OGXAPIConnectionError = ogx_mod.APIConnectionError
        ssl_err = ssl.SSLCertVerificationError("CERTIFICATE_VERIFY_FAILED: self-signed certificate")
        api_err = OGXAPIConnectionError("Connection error.")
        api_err.__cause__ = ssl_err

        mock_ogx_client_fail = mock.MagicMock()
        mock_ogx_client_fail.models.list.side_effect = api_err
        mock_ogx_client_ok = mock.MagicMock()
        mock_ogx_client_ok.models.list.return_value = []

        ogx_call_count = 0
        ogx_kwargs_history = []

        def fake_ogx_client(**kwargs):
            nonlocal ogx_call_count
            ogx_call_count += 1
            ogx_kwargs_history.append(kwargs)
            if ogx_call_count == 1:
                return mock_ogx_client_fail
            return mock_ogx_client_ok

        ogx_mod.OgxClient.side_effect = fake_ogx_client
        mocks["ogx_client"] = ogx_mod

        mocks[
            "ai4rag.search_space.prepare.prepare_search_space"
        ].prepare_search_space_with_ogx.side_effect = _SentinelAbort

        test_data, extracted_text, search_space_prep_report = self._make_artifacts(tmp_path)

        with mock.patch.dict(sys.modules, mocks):
            with pytest.raises(_SentinelAbort):
                search_space_preparation.python_func(
                    test_data=test_data,
                    extracted_text=extracted_text,
                    search_space_prep_report=search_space_prep_report,
                )

        assert ogx_call_count == 2, "OgxClient should be instantiated twice (initial + SSL retry)"
        assert ogx_kwargs_history[0].get("http_client") is None
        assert isinstance(ogx_kwargs_history[1].get("http_client"), mocks["httpx"].Client)
        assert ogx_kwargs_history[1]["http_client"].kwargs.get("verify") is False

    @mock.patch.dict(
        "os.environ",
        {
            "OGX_CLIENT_BASE_URL": "https://ogx.example.com",
            "OGX_CLIENT_API_KEY": "test-api-key",
        },
    )
    def test_ogx_client_non_ssl_error_is_reraised(self, tmp_path):
        """Non-SSL error from models.list() is not swallowed — it propagates."""
        mocks = _make_all_mocks()

        mock_ogx_client = mock.MagicMock()
        mock_ogx_client.models.list.side_effect = ConnectionRefusedError("Connection refused")

        ogx_mod = _make_ogx_client_module()
        ogx_mod.OgxClient.return_value = mock_ogx_client
        mocks["ogx_client"] = ogx_mod

        test_data, extracted_text, search_space_prep_report = self._make_artifacts(tmp_path)

        with mock.patch.dict(sys.modules, mocks):
            with pytest.raises(ConnectionRefusedError):
                search_space_preparation.python_func(
                    test_data=test_data,
                    extracted_text=extracted_text,
                    search_space_prep_report=search_space_prep_report,
                )

    @mock.patch.dict(
        "os.environ",
        {
            "OGX_CLIENT_BASE_URL": "https://ogx.example.com",
            "OGX_CLIENT_API_KEY": "test-api-key",
        },
    )
    def test_ogx_client_api_connection_error_non_ssl_cause_is_reraised(self, tmp_path):
        """OGXAPIConnectionError whose cause is NOT SSL propagates without retry."""
        mocks = _make_all_mocks()

        ogx_mod = _make_ogx_client_module()
        OGXAPIConnectionError = ogx_mod.APIConnectionError
        err = OGXAPIConnectionError("Connection timeout")
        err.__cause__ = TimeoutError("timed out")

        mock_ogx_client = mock.MagicMock()
        mock_ogx_client.models.list.side_effect = err
        ogx_mod.OgxClient.return_value = mock_ogx_client
        mocks["ogx_client"] = ogx_mod

        test_data, extracted_text, search_space_prep_report = self._make_artifacts(tmp_path)

        with mock.patch.dict(sys.modules, mocks):
            with pytest.raises(OGXAPIConnectionError):
                search_space_preparation.python_func(
                    test_data=test_data,
                    extracted_text=extracted_text,
                    search_space_prep_report=search_space_prep_report,
                )
