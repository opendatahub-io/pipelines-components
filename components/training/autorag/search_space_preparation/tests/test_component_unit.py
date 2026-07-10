"""Tests for the search_space_preparation thin wrapper component."""

import inspect
from unittest import mock

import pytest

from ..component import search_space_preparation

MOCKED_ENV_VARIABLES = {
    "OGX_CLIENT_BASE_URL": "https://ogx.example.com",
    "OGX_CLIENT_API_KEY": "test-api-key",
}


def _make_ai4rag_mocks():
    """Build mock modules for ai4rag search_space_preparation, ogx_client, and compat."""
    mock_create_ogx_client = mock.MagicMock(name="create_ogx_client")
    mock_prepare_report = mock.MagicMock(name="prepare_search_space_report")
    mock_ensure_sqlite3 = mock.MagicMock(name="ensure_sqlite3")

    mock_ogx_module = mock.MagicMock()
    mock_ogx_module.create_ogx_client = mock_create_ogx_client

    mock_report_module = mock.MagicMock()
    mock_report_module.prepare_search_space_report = mock_prepare_report

    mock_compat = mock.MagicMock()
    mock_compat.ensure_sqlite3 = mock_ensure_sqlite3

    modules = {
        "ai4rag": mock.MagicMock(),
        "ai4rag.components": mock.MagicMock(),
        "ai4rag.components.utils": mock.MagicMock(),
        "ai4rag.components.utils.ogx_client": mock_ogx_module,
        "ai4rag.components.optimization": mock.MagicMock(),
        "ai4rag.components.optimization.search_space_preparation": mock_report_module,
        "ai4rag.utils": mock.MagicMock(),
        "ai4rag.utils.compat": mock_compat,
    }
    return modules, mock_create_ogx_client, mock_prepare_report, mock_ensure_sqlite3


class TestSearchSpacePreparationUnitTests:
    """Unit tests for the search_space_preparation thin wrapper."""

    def test_component_function_exists(self):
        """Component factory exists and exposes python_func."""
        assert callable(search_space_preparation)
        assert hasattr(search_space_preparation, "python_func")

    def test_component_has_expected_interface(self):
        """Component has expected parameters."""
        sig = inspect.signature(search_space_preparation.python_func)
        params = list(sig.parameters)
        assert "test_data" in params
        assert "extracted_text" in params
        assert "search_space_prep_report" in params
        assert "embedding_models" in params
        assert "generation_models" in params
        assert "metric" in params
        assert "evaluator" not in params
        assert "judge_model_id" not in params
        assert "preset" in params
        assert sig.parameters["preset"].default == "speed"

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_delegates_to_ai4rag_prepare_search_space_report(self, tmp_path):
        """Wrapper calls ensure_sqlite3, create_ogx_client, and prepare_search_space_report."""
        modules, mock_create_ogx, mock_prepare, mock_sqlite = _make_ai4rag_mocks()
        mock_ogx_client = mock.MagicMock(name="ogx_client_instance")
        mock_create_ogx.return_value = mock_ogx_client
        mock_report = mock.MagicMock()
        mock_prepare.return_value = mock_report

        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test_data.json")
        extracted_text = mock.MagicMock()
        extracted_text.path = str(tmp_path / "extracted")
        report_artifact = mock.MagicMock()
        report_artifact.path = str(tmp_path / "report.yml")

        with mock.patch.dict("sys.modules", modules):
            search_space_preparation.python_func(
                test_data=test_data,
                extracted_text=extracted_text,
                search_space_prep_report=report_artifact,
                embedding_models=["embed-1", "embed-2"],
                generation_models=["gen-1"],
                metric="answer_correctness",
            )

        mock_sqlite.assert_called_once()
        mock_create_ogx.assert_called_once_with(
            base_url="https://ogx.example.com",
            api_key="test-api-key",
        )
        mock_prepare.assert_called_once_with(
            test_data_path=str(tmp_path / "test_data.json"),
            extracted_text_path=str(tmp_path / "extracted"),
            ogx_client=mock_ogx_client,
            embedding_models=["embed-1", "embed-2"],
            generation_models=["gen-1"],
            metric="answer_correctness",
            chunking_methods=["recursive"],
            chunk_sizes=[128, 256, 512],
            chunk_overlaps=[32, 64],
            inference_max_threads=10,
        )
        mock_report.save_json.assert_called_once_with(str(tmp_path / "report.yml"))

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_default_metric_is_faithfulness(self, tmp_path):
        """When metric is None, 'faithfulness' is passed as default."""
        modules, mock_create_ogx, mock_prepare, _ = _make_ai4rag_mocks()
        mock_create_ogx.return_value = mock.MagicMock()
        mock_prepare.return_value = mock.MagicMock()

        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test.json")
        extracted_text = mock.MagicMock()
        extracted_text.path = str(tmp_path / "ext")
        report = mock.MagicMock()
        report.path = str(tmp_path / "report.yml")

        with mock.patch.dict("sys.modules", modules):
            search_space_preparation.python_func(
                test_data=test_data,
                extracted_text=extracted_text,
                search_space_prep_report=report,
            )

        assert mock_prepare.call_args.kwargs["metric"] == "faithfulness"
        assert "evaluator" not in mock_prepare.call_args.kwargs

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_none_models_passed_through(self, tmp_path):
        """None embedding_models and generation_models are forwarded as None."""
        modules, mock_create_ogx, mock_prepare, _ = _make_ai4rag_mocks()
        mock_create_ogx.return_value = mock.MagicMock()
        mock_prepare.return_value = mock.MagicMock()

        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test.json")
        extracted_text = mock.MagicMock()
        extracted_text.path = str(tmp_path / "ext")
        report = mock.MagicMock()
        report.path = str(tmp_path / "report.yml")

        with mock.patch.dict("sys.modules", modules):
            search_space_preparation.python_func(
                test_data=test_data,
                extracted_text=extracted_text,
                search_space_prep_report=report,
                embedding_models=None,
                generation_models=None,
            )

        call_kwargs = mock_prepare.call_args.kwargs
        assert call_kwargs["embedding_models"] is None
        assert call_kwargs["generation_models"] is None

    def test_missing_ogx_env_raises_key_error(self, tmp_path):
        """Missing OGX env vars raise KeyError."""
        modules, _, _, _ = _make_ai4rag_mocks()

        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test.json")
        extracted_text = mock.MagicMock()
        extracted_text.path = str(tmp_path / "ext")
        report = mock.MagicMock()
        report.path = str(tmp_path / "report.yml")

        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch.dict("sys.modules", modules):
                with pytest.raises(KeyError):
                    search_space_preparation.python_func(
                        test_data=test_data,
                        extracted_text=extracted_text,
                        search_space_prep_report=report,
                    )

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_propagates_ai4rag_exception(self, tmp_path):
        """Exceptions from ai4rag are propagated to the caller."""
        modules, mock_create_ogx, mock_prepare, _ = _make_ai4rag_mocks()
        mock_create_ogx.return_value = mock.MagicMock()
        mock_prepare.side_effect = ValueError("Metric not_real is not supported")

        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test.json")
        extracted_text = mock.MagicMock()
        extracted_text.path = str(tmp_path / "ext")
        report = mock.MagicMock()
        report.path = str(tmp_path / "report.yml")

        with mock.patch.dict("sys.modules", modules):
            with pytest.raises(ValueError, match="Metric not_real is not supported"):
                search_space_preparation.python_func(
                    test_data=test_data,
                    extracted_text=extracted_text,
                    search_space_prep_report=report,
                    metric="not_real",
                )

    def test_component_status_defaults_to_none(self):
        """component_status defaults to None, enabling direct notebook usage."""
        sig = inspect.signature(search_space_preparation.python_func)
        param = sig.parameters["component_status"]
        assert param.default is None, "component_status should default to None for notebook usage"

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_preset_validation_rejects_invalid(self, tmp_path):
        """Invalid preset raises ValueError."""
        modules, _, _, _ = _make_ai4rag_mocks()

        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test.json")
        extracted_text = mock.MagicMock()
        extracted_text.path = str(tmp_path / "ext")
        report = mock.MagicMock()
        report.path = str(tmp_path / "report.yml")

        with mock.patch.dict("sys.modules", modules):
            with pytest.raises(ValueError, match="preset must be one of"):
                search_space_preparation.python_func(
                    test_data=test_data,
                    extracted_text=extracted_text,
                    search_space_prep_report=report,
                    preset="invalid",
                )

    @pytest.mark.parametrize("preset_value", ["speed", "balanced"])
    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_valid_presets_accepted(self, tmp_path, preset_value):
        """Both 'speed' and 'balanced' presets are accepted without error."""
        modules, mock_create_ogx, mock_prepare, _ = _make_ai4rag_mocks()
        mock_create_ogx.return_value = mock.MagicMock()
        mock_prepare.return_value = mock.MagicMock()

        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test.json")
        extracted_text = mock.MagicMock()
        extracted_text.path = str(tmp_path / "ext")
        report = mock.MagicMock()
        report.path = str(tmp_path / "report.yml")

        with mock.patch.dict("sys.modules", modules):
            search_space_preparation.python_func(
                test_data=test_data,
                extracted_text=extracted_text,
                search_space_prep_report=report,
                preset=preset_value,
            )

        mock_prepare.assert_called_once()

    @pytest.mark.parametrize(
        ("preset_value", "expected_chunking", "expected_chunk_sizes", "expected_chunk_overlaps", "expected_threads"),
        [
            ("speed", ["recursive"], [128, 256, 512], [32, 64], 10),
            ("balanced", ["recursive", "hybrid"], [512, 1024, 2048], [0, 128, 256], 4),
        ],
    )
    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_preset_sets_search_space_params(
        self, tmp_path, preset_value, expected_chunking, expected_chunk_sizes, expected_chunk_overlaps, expected_threads
    ):
        """Preset controls chunking_methods, chunk_sizes, chunk_overlaps, and inference_max_threads."""
        modules, mock_create_ogx, mock_prepare, _ = _make_ai4rag_mocks()
        mock_create_ogx.return_value = mock.MagicMock()
        mock_prepare.return_value = mock.MagicMock()

        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test.json")
        extracted_text = mock.MagicMock()
        extracted_text.path = str(tmp_path / "ext")
        report = mock.MagicMock()
        report.path = str(tmp_path / "report.yml")

        with mock.patch.dict("sys.modules", modules):
            search_space_preparation.python_func(
                test_data=test_data,
                extracted_text=extracted_text,
                search_space_prep_report=report,
                preset=preset_value,
            )

        assert mock_prepare.call_args.kwargs["chunking_methods"] == expected_chunking
        assert mock_prepare.call_args.kwargs["chunk_sizes"] == expected_chunk_sizes
        assert mock_prepare.call_args.kwargs["chunk_overlaps"] == expected_chunk_overlaps
        assert mock_prepare.call_args.kwargs["inference_max_threads"] == expected_threads
