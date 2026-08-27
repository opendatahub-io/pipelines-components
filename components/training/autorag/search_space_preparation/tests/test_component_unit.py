"""Tests for the search_space_preparation component.

The component inlines the search-space orchestration: it builds a MaaS payload
from the preset and model lists, calls
``ai4rag.search_space.prepare.prepare_search_space_with_maas`` (where model
discovery + responsiveness validation happen — the fail-fast checkpoint), then
serializes the result with ``build_search_space_report``. Model pre-selection is
no longer part of this component.
"""

import inspect
from types import SimpleNamespace
from unittest import mock

import pytest

from ..component import search_space_preparation

MOCKED_ENV_VARIABLES = {
    "MAAS_BASE_URL": "https://maas.example.com/v1",
    "MAAS_API_KEY": "test-api-key",
}


def _make_ai4rag_mocks() -> SimpleNamespace:
    """Build mock ai4rag modules wired into sys.modules for the component body."""
    create_maas_client = mock.MagicMock(name="create_maas_client")
    prepare_search_space_with_maas = mock.MagicMock(name="prepare_search_space_with_maas")
    build_search_space_report = mock.MagicMock(name="build_search_space_report")
    ensure_sqlite3 = mock.MagicMock(name="ensure_sqlite3")

    utils = mock.MagicMock()
    utils.create_maas_client = create_maas_client

    prepare_module = mock.MagicMock()
    prepare_module.prepare_search_space_with_maas = prepare_search_space_with_maas
    prepare_module.build_search_space_report = build_search_space_report

    compat = mock.MagicMock()
    compat.ensure_sqlite3 = ensure_sqlite3

    modules = {
        "ai4rag": mock.MagicMock(),
        "ai4rag.components": mock.MagicMock(),
        "ai4rag.components.utils": utils,
        "ai4rag.search_space": mock.MagicMock(),
        "ai4rag.search_space.prepare": prepare_module,
        "ai4rag.utils": mock.MagicMock(),
        "ai4rag.utils.compat": compat,
        "pandas": mock.MagicMock(name="pandas"),
    }
    return SimpleNamespace(
        modules=modules,
        create_maas_client=create_maas_client,
        prepare=prepare_search_space_with_maas,
        build=build_search_space_report,
        ensure_sqlite3=ensure_sqlite3,
    )


class TestSearchSpacePreparationUnitTests:
    """Unit tests for the search_space_preparation component."""

    def test_component_function_exists(self):
        """Component factory exists and exposes python_func."""
        assert callable(search_space_preparation)
        assert hasattr(search_space_preparation, "python_func")

    def test_component_has_expected_interface(self):
        """Component has expected parameters and no longer takes extracted_text."""
        sig = inspect.signature(search_space_preparation.python_func)
        params = list(sig.parameters)
        assert "test_data" in params
        assert "extracted_text" not in params
        assert "search_space_report" in params
        assert "embedding_models" in params
        assert "generation_models" in params
        assert "preset" in params
        assert sig.parameters["preset"].default == "speed"

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_delegates_to_prepare_search_space_with_maas(self, tmp_path):
        """Delegates to create_maas_client, prepare_search_space_with_maas, and build_search_space_report."""
        m = _make_ai4rag_mocks()
        client = mock.MagicMock(name="maas_client_instance")
        m.create_maas_client.return_value = client
        search_space = mock.MagicMock(name="search_space")
        m.prepare.return_value = search_space
        report = mock.MagicMock(name="report")
        m.build.return_value = report

        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test_data.json")
        report_artifact = mock.MagicMock()
        report_artifact.path = str(tmp_path / "report.json")

        with mock.patch.dict("sys.modules", m.modules):
            search_space_preparation.python_func(
                test_data=test_data,
                search_space_report=report_artifact,
                embedding_models=["embed-1", "embed-2"],
                generation_models=["gen-1"],
            )

        m.ensure_sqlite3.assert_called_once()
        m.create_maas_client.assert_called_once_with(
            base_url="https://maas.example.com/v1",
            api_key="test-api-key",
        )

        m.prepare.assert_called_once()
        payload = m.prepare.call_args.args[0]
        assert payload["foundation_models"] == [{"model_id": "gen-1"}]
        assert payload["embedding_models"] == [{"model_id": "embed-1"}, {"model_id": "embed-2"}]
        assert payload["chunking_methods"] == ["recursive"]
        assert m.prepare.call_args.kwargs["client"] is client
        assert "benchmark_data" in m.prepare.call_args.kwargs

        m.build.assert_called_once_with(search_space)
        report.save_json.assert_called_once_with(str(tmp_path / "report.json"))

    def test_embedding_and_generation_models_are_required(self):
        """embedding_models and generation_models have no default: KFP marks them required inputs."""
        sig = inspect.signature(search_space_preparation.python_func)
        assert sig.parameters["embedding_models"].default is inspect.Parameter.empty
        assert sig.parameters["generation_models"].default is inspect.Parameter.empty

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_empty_model_list_raises_value_error(self, tmp_path):
        """An empty model list is rejected before any MaaS work."""
        m = _make_ai4rag_mocks()

        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test.json")
        report = mock.MagicMock()
        report.path = str(tmp_path / "report.json")

        with mock.patch.dict("sys.modules", m.modules):
            with pytest.raises(ValueError, match="non-empty list"):
                search_space_preparation.python_func(
                    test_data=test_data,
                    search_space_report=report,
                    embedding_models=[],
                    generation_models=["gen-1"],
                )
        m.create_maas_client.assert_not_called()

    def test_missing_maas_env_raises_key_error(self, tmp_path):
        """Missing MaaS env vars raise KeyError."""
        m = _make_ai4rag_mocks()

        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test.json")
        report = mock.MagicMock()
        report.path = str(tmp_path / "report.json")

        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch.dict("sys.modules", m.modules):
                with pytest.raises(KeyError):
                    search_space_preparation.python_func(
                        test_data=test_data,
                        search_space_report=report,
                        embedding_models=["embed-1"],
                        generation_models=["gen-1"],
                    )

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_propagates_ai4rag_exception(self, tmp_path):
        """Exceptions from ai4rag are propagated to the caller."""
        m = _make_ai4rag_mocks()
        m.create_maas_client.return_value = mock.MagicMock()
        m.prepare.side_effect = ValueError("models do not respond")

        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test.json")
        report = mock.MagicMock()
        report.path = str(tmp_path / "report.json")

        with mock.patch.dict("sys.modules", m.modules):
            with pytest.raises(ValueError, match="models do not respond"):
                search_space_preparation.python_func(
                    test_data=test_data,
                    search_space_report=report,
                    embedding_models=["embed-1"],
                    generation_models=["gen-1"],
                )

    def test_component_status_defaults_to_none(self):
        """component_status defaults to None, enabling direct notebook usage."""
        sig = inspect.signature(search_space_preparation.python_func)
        assert sig.parameters["component_status"].default is None

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_preset_validation_rejects_invalid(self, tmp_path):
        """Invalid preset raises ValueError."""
        m = _make_ai4rag_mocks()

        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test.json")
        report = mock.MagicMock()
        report.path = str(tmp_path / "report.json")

        with mock.patch.dict("sys.modules", m.modules):
            with pytest.raises(ValueError, match="preset must be one of"):
                search_space_preparation.python_func(
                    test_data=test_data,
                    search_space_report=report,
                    embedding_models=["embed-1"],
                    generation_models=["gen-1"],
                    preset="invalid",
                )

    @pytest.mark.parametrize(
        ("preset_value", "expected_chunking", "expected_chunk_sizes", "expected_chunk_overlaps"),
        [
            ("speed", ["recursive"], [128, 256, 512], [32, 64]),
            ("balanced", ["recursive", "hybrid"], [512, 1024, 2048], [0, 128, 256]),
        ],
    )
    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_preset_sets_payload_chunking(
        self, tmp_path, preset_value, expected_chunking, expected_chunk_sizes, expected_chunk_overlaps
    ):
        """Preset controls the chunking dimensions of the MaaS payload."""
        m = _make_ai4rag_mocks()
        m.create_maas_client.return_value = mock.MagicMock()
        m.prepare.return_value = mock.MagicMock()
        m.build.return_value = mock.MagicMock()

        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test.json")
        report = mock.MagicMock()
        report.path = str(tmp_path / "report.json")

        with mock.patch.dict("sys.modules", m.modules):
            search_space_preparation.python_func(
                test_data=test_data,
                search_space_report=report,
                embedding_models=["embed-1"],
                generation_models=["gen-1"],
                preset=preset_value,
            )

        payload = m.prepare.call_args.args[0]
        assert payload["chunking_methods"] == expected_chunking
        assert payload["chunk_sizes"] == expected_chunk_sizes
        assert payload["chunk_overlaps"] == expected_chunk_overlaps
