"""Tests for the rag_templates_optimization component."""

import os
import ssl
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
        "ai4rag.core.experiment.experiment",
        "ai4rag.core.experiment.results",
        "ai4rag.core.hpo",
        "ai4rag.core.hpo.gam_opt",
        "ai4rag.rag",
        "ai4rag.rag.embedding",
        "ai4rag.rag.embedding.base_model",
        "ai4rag.rag.embedding.ogx",
        "ai4rag.rag.foundation_models",
        "ai4rag.rag.foundation_models.base_model",
        "ai4rag.rag.foundation_models.ogx",
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
        "ai4rag.rag.embedding.ogx": mock.MagicMock(OGXEmbeddingModel=mock.MagicMock()),
        "ai4rag.rag.foundation_models": mock.MagicMock(),
        "ai4rag.rag.foundation_models.base_model": mock.MagicMock(BaseFoundationModel=mock.MagicMock()),
        "ai4rag.rag.foundation_models.ogx": mock.MagicMock(OGXFoundationModel=mock.MagicMock()),
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
        "ogx_client": mock.MagicMock(OgxClient=mock.MagicMock()),
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

    def _setup_ogx_mocks(self, tmp_path, abort_at_experiment=True):
        """Set up mocks and temp files for OGX vector store tests.

        Returns (mocks, extracted_text, test_data_path, search_space_report).
        """
        mocks = _make_all_mocks()
        ogx_mod = _make_ogx_client_module()
        mock_ogx = mock.MagicMock()
        mock_ogx.models.list.return_value = []
        ogx_mod.OgxClient.return_value = mock_ogx
        mocks["ogx_client"] = ogx_mod
        if abort_at_experiment:
            mocks["ai4rag.core.experiment.experiment"].AI4RAGExperiment.side_effect = _SentinelAbort

        search_space_report = tmp_path / "report.yml"
        search_space_report.write_text("{}")
        test_data_path = tmp_path / "test_data.json"
        test_data_path.write_text("[]")
        extracted_text = str(tmp_path / "extracted_text")

        return mocks, extracted_text, str(test_data_path), str(search_space_report)

    def _run_with_ogx(self, mocks, extracted_text, test_data, search_space_report, **kwargs):
        """Run the component with OGX env vars and the given mocks."""
        with (
            mock.patch.dict(sys.modules, mocks),
            mock.patch.dict(
                os.environ,
                {
                    "OGX_CLIENT_BASE_URL": "https://ogx.example.com",
                    "OGX_CLIENT_API_KEY": "test-api-key",
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
        """Any non-empty vector_io_provider_id string is accepted (no allowlist)."""
        mocks, extracted_text, test_data, report = self._setup_ogx_mocks(tmp_path)
        with pytest.raises(_SentinelAbort):
            self._run_with_ogx(mocks, extracted_text, test_data, report, vector_io_provider_id="my_custom_milvus")

    def test_vector_store_type_set_to_ogx(self, tmp_path):
        """AI4RAGExperiment receives vector_store_type 'ogx'."""
        mocks, extracted_text, test_data, report = self._setup_ogx_mocks(tmp_path)
        with pytest.raises(_SentinelAbort):
            self._run_with_ogx(mocks, extracted_text, test_data, report, vector_io_provider_id="milvus")

        ai4rag_exp = mocks["ai4rag.core.experiment.experiment"].AI4RAGExperiment
        ai4rag_exp.assert_called_once()
        assert ai4rag_exp.call_args.kwargs["vector_store_type"] == "ogx"

    def test_missing_provider_id_raises_value_error(self, tmp_path):
        """None provider_id raises ValueError."""
        mocks, extracted_text, test_data, report = self._setup_ogx_mocks(tmp_path, abort_at_experiment=False)
        with pytest.raises(ValueError, match="vector_io_provider_id must be a non-empty string"):
            self._run_with_ogx(mocks, extracted_text, test_data, report, vector_io_provider_id=None)

    def test_whitespace_provider_id_raises_value_error(self, tmp_path):
        """Whitespace-only provider_id raises ValueError."""
        mocks, extracted_text, test_data, report = self._setup_ogx_mocks(tmp_path, abort_at_experiment=False)
        with pytest.raises(ValueError, match="vector_io_provider_id must be a non-empty string"):
            self._run_with_ogx(mocks, extracted_text, test_data, report, vector_io_provider_id="   ")

    def test_max_number_of_rag_patterns_non_numeric_string_raises_value_error(self):
        """UI may pass string parameters; non-numeric strings are rejected with a clear error."""
        with mock.patch.dict(sys.modules, _minimal_dependency_modules()):
            with mock.patch.dict(
                os.environ,
                {
                    "OGX_CLIENT_BASE_URL": "https://ogx.example.com",
                    "OGX_CLIENT_API_KEY": "test-api-key",
                },
            ):
                with pytest.raises(ValueError, match="max_number_of_rag_patterns must be a valid integer"):
                    rag_templates_optimization.python_func(
                        extracted_text="/tmp/extracted",
                        test_data="/tmp/test_data.json",
                        search_space_prep_report="/tmp/report.yml",
                        rag_patterns=mock.MagicMock(path="/tmp/rag_patterns", metadata={}, uri=""),
                        embedded_artifact=mock.MagicMock(path="/tmp/embedded"),
                        test_data_key="small-dataset/benchmark.json",
                        vector_io_provider_id="milvus",
                        optimization_settings={
                            "metric": "faithfulness",
                            "max_number_of_rag_patterns": "not-a-number",
                        },
                    )

    @mock.patch.dict(
        os.environ,
        {
            "OGX_CLIENT_BASE_URL": "https://ogx.example.com",
            "OGX_CLIENT_API_KEY": "test-api-key",
        },
    )
    def test_max_number_of_rag_patterns_numeric_string_coerced_for_gam_opt(self, tmp_path):
        """Pipeline UI often sends numbers as strings; they must coerce to int for GAMOptSettings."""
        mocks = _make_all_mocks()
        ogx_mod = _make_ogx_client_module()
        mock_ogx = mock.MagicMock()
        mock_ogx.models.list.return_value = []
        ogx_mod.OgxClient.return_value = mock_ogx
        mocks["ogx_client"] = ogx_mod
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
                    vector_io_provider_id="milvus",
                    optimization_settings={"metric": "faithfulness", "max_number_of_rag_patterns": "8"},
                )

        mocks["ai4rag.core.hpo.gam_opt"].GAMOptSettings.assert_called_once_with(max_evals=8)

    def test_missing_ogx_env_vars_raises_value_error(self):
        """Missing OGX environment variables raises ValueError."""
        with mock.patch.dict(sys.modules, _minimal_dependency_modules()):
            with mock.patch.dict(os.environ, {}, clear=True):
                with pytest.raises(ValueError, match="OGX_CLIENT_BASE_URL and OGX_CLIENT_API_KEY"):
                    rag_templates_optimization.python_func(
                        extracted_text="/tmp/extracted",
                        test_data="/tmp/test_data.json",
                        search_space_prep_report="/tmp/report.yml",
                        rag_patterns=mock.MagicMock(path="/tmp/rag_patterns", metadata={}, uri=""),
                        embedded_artifact=mock.MagicMock(path="/tmp/embedded"),
                        test_data_key="small-dataset/benchmark.json",
                        vector_io_provider_id="milvus",
                        optimization_settings={"metric": "faithfulness", "max_number_of_rag_patterns": 8},
                    )


class TestSSLFallbackRagTemplatesOptimization:
    """Tests for SSL retry logic in _create_ogx_client."""

    def _make_paths(self, tmp_path):
        """Create minimal real files needed by the component."""
        search_space_report = tmp_path / "report.yml"
        search_space_report.write_text("{}")
        return (
            str(tmp_path / "extracted_text"),  # non-existent dir → load_as_langchain_doc returns []
            str(tmp_path / "test_data.json"),
            str(search_space_report),
        )

    def _make_output_artifacts(self):
        rag_patterns = mock.MagicMock()
        embedded_artifact = mock.MagicMock()
        return rag_patterns, embedded_artifact

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

        mock_ogx_client_fail = mock.MagicMock()
        mock_ogx_client_fail.models.list.side_effect = ssl.SSLCertVerificationError(
            "CERTIFICATE_VERIFY_FAILED: self-signed certificate"
        )
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

        ogx_mod = _make_ogx_client_module()
        ogx_mod.OgxClient.side_effect = fake_ogx_client
        mocks["ogx_client"] = ogx_mod

        # Abort after client creation via AI4RAGSearchSpace
        mocks["ai4rag.search_space.src.search_space"].AI4RAGSearchSpace.side_effect = _SentinelAbort

        extracted_text, test_data, search_space_report = self._make_paths(tmp_path)
        rag_patterns, embedded_artifact = self._make_output_artifacts()

        with mock.patch.dict(sys.modules, mocks):
            with pytest.raises(_SentinelAbort):
                rag_templates_optimization.python_func(
                    extracted_text=extracted_text,
                    test_data=test_data,
                    search_space_prep_report=search_space_report,
                    rag_patterns=rag_patterns,
                    embedded_artifact=embedded_artifact,
                    test_data_key="small-dataset/benchmark.json",
                    vector_io_provider_id="milvus",
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

        mocks["ai4rag.search_space.src.search_space"].AI4RAGSearchSpace.side_effect = _SentinelAbort

        extracted_text, test_data, search_space_report = self._make_paths(tmp_path)
        rag_patterns, embedded_artifact = self._make_output_artifacts()

        with mock.patch.dict(sys.modules, mocks):
            with pytest.raises(_SentinelAbort):
                rag_templates_optimization.python_func(
                    extracted_text=extracted_text,
                    test_data=test_data,
                    search_space_prep_report=search_space_report,
                    rag_patterns=rag_patterns,
                    embedded_artifact=embedded_artifact,
                    test_data_key="small-dataset/benchmark.json",
                    vector_io_provider_id="milvus",
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
        """Non-SSL error from models.list() propagates without retry."""
        mocks = _make_all_mocks()

        mock_ogx_client = mock.MagicMock()
        mock_ogx_client.models.list.side_effect = ConnectionRefusedError("Connection refused")

        ogx_mod = _make_ogx_client_module()
        ogx_mod.OgxClient.return_value = mock_ogx_client
        mocks["ogx_client"] = ogx_mod

        extracted_text, test_data, search_space_report = self._make_paths(tmp_path)
        rag_patterns, embedded_artifact = self._make_output_artifacts()

        with mock.patch.dict(sys.modules, mocks):
            with pytest.raises(ConnectionRefusedError):
                rag_templates_optimization.python_func(
                    extracted_text=extracted_text,
                    test_data=test_data,
                    search_space_prep_report=search_space_report,
                    rag_patterns=rag_patterns,
                    embedded_artifact=embedded_artifact,
                    test_data_key="small-dataset/benchmark.json",
                    vector_io_provider_id="milvus",
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

        extracted_text, test_data, search_space_report = self._make_paths(tmp_path)
        rag_patterns, embedded_artifact = self._make_output_artifacts()

        with mock.patch.dict(sys.modules, mocks):
            with pytest.raises(OGXAPIConnectionError):
                rag_templates_optimization.python_func(
                    extracted_text=extracted_text,
                    test_data=test_data,
                    search_space_prep_report=search_space_report,
                    rag_patterns=rag_patterns,
                    embedded_artifact=embedded_artifact,
                    test_data_key="small-dataset/benchmark.json",
                    vector_io_provider_id="milvus",
                )
