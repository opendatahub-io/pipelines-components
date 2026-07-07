"""Tests for the rag_templates_optimization thin wrapper component."""

import inspect
from types import SimpleNamespace
from unittest import mock

import pytest

from ..component import rag_templates_optimization

MOCKED_ENV_VARIABLES = {
    "OGX_CLIENT_BASE_URL": "https://ogx.example.com",
    "OGX_CLIENT_API_KEY": "test-api-key",
}


def _make_ai4rag_mocks():
    """Build mock modules for ai4rag optimization, utils, compat, and leaderboard."""
    mock_create_ogx_client = mock.MagicMock(name="create_ogx_client")
    mock_run_rag_optimization = mock.MagicMock(name="run_rag_optimization")
    mock_ensure_sqlite3 = mock.MagicMock(name="ensure_sqlite3")

    mock_utils = mock.MagicMock()
    mock_utils.create_ogx_client = mock_create_ogx_client

    mock_optimization_module = mock.MagicMock()
    mock_optimization_module.run_rag_optimization = mock_run_rag_optimization

    mock_compat = mock.MagicMock()
    mock_compat.ensure_sqlite3 = mock_ensure_sqlite3

    mock_leaderboard = mock.MagicMock()
    mock_leaderboard.build_leaderboard_html = mock.MagicMock(return_value="<html></html>")
    mock_leaderboard.DEFAULT_METRIC = "faithfulness"

    modules = {
        "ai4rag": mock.MagicMock(),
        "ai4rag.components": mock.MagicMock(),
        "ai4rag.components.utils": mock_utils,
        "ai4rag.components.utils.ogx_client": mock_utils,
        "ai4rag.components.optimization": mock.MagicMock(),
        "ai4rag.components.optimization.rag_templates_optimization": mock_optimization_module,
        "ai4rag.components.assets_generator": mock.MagicMock(),
        "ai4rag.components.assets_generator.leaderboard": mock_leaderboard,
        "ai4rag.utils": mock.MagicMock(),
        "ai4rag.utils.compat": mock_compat,
    }
    return modules, mock_create_ogx_client, mock_run_rag_optimization, mock_ensure_sqlite3


class TestRagTemplatesOptimizationUnitTests:
    """Unit tests for the rag_templates_optimization thin wrapper."""

    def test_component_function_exists(self):
        """Component factory exists and exposes python_func."""
        assert callable(rag_templates_optimization)
        assert hasattr(rag_templates_optimization, "python_func")

    def test_component_has_expected_interface(self):
        """Component has expected required parameters."""
        sig = inspect.signature(rag_templates_optimization.python_func)
        params = list(sig.parameters)
        assert "extracted_text" in params
        assert "test_data" in params
        assert "search_space_prep_report" in params
        assert "rag_patterns" in params
        assert "embedded_artifact" in params
        assert "test_data_key" in params
        assert "vector_io_provider_id" in params
        assert "optimization_settings" in params
        assert "input_data_key" in params
        assert "evaluator" in params
        assert "judge_model_id" in params

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_delegates_to_ai4rag_run_rag_optimization(self, tmp_path):
        """Wrapper calls ensure_sqlite3, create_ogx_client, and run_rag_optimization."""
        modules, mock_create_ogx, mock_run_opt, mock_sqlite = _make_ai4rag_mocks()
        mock_ogx_client = mock.MagicMock(name="ogx_client_instance")
        mock_create_ogx.return_value = mock_ogx_client

        patterns_list = [{"name": "pattern_a", "scores": {"faithfulness": {"mean": 0.9}}}]
        mock_run_opt.return_value = SimpleNamespace(patterns=patterns_list)

        output_dir = tmp_path / "rag_patterns"
        rag_patterns = mock.MagicMock()
        rag_patterns.path = str(output_dir)
        rag_patterns.uri = "gs://bucket/rag_patterns"
        rag_patterns.metadata = {}

        html_path = tmp_path / "leaderboard.html"
        html_artifact = mock.MagicMock()
        html_artifact.path = str(html_path)
        html_artifact.metadata = {}

        with mock.patch.dict("sys.modules", modules):
            rag_templates_optimization.python_func(
                extracted_text=str(tmp_path / "extracted"),
                test_data=str(tmp_path / "test_data.json"),
                search_space_prep_report=str(tmp_path / "report.yml"),
                rag_patterns=rag_patterns,
                test_data_key="data/test.json",
                vector_io_provider_id="milvus-provider",
                html_artifact=html_artifact,
                optimization_settings={"max_number_of_rag_patterns": 8},
                input_data_key="data/docs/",
            )

        mock_sqlite.assert_called_once()
        mock_create_ogx.assert_called_once_with(
            base_url="https://ogx.example.com",
            api_key="test-api-key",
        )
        mock_run_opt.assert_called_once()
        call_kwargs = mock_run_opt.call_args.kwargs
        assert call_kwargs["extracted_text_path"] == str(tmp_path / "extracted")
        assert call_kwargs["test_data_path"] == str(tmp_path / "test_data.json")
        assert call_kwargs["search_space_report_path"] == str(tmp_path / "report.yml")
        assert call_kwargs["ogx_client"] is mock_ogx_client
        assert call_kwargs["vector_io_provider_id"] == "milvus-provider"
        assert call_kwargs["test_data_key"] == "data/test.json"
        assert call_kwargs["input_data_key"] == "data/docs/"
        assert call_kwargs["optimization_settings"] == {"max_number_of_rag_patterns": 8}
        assert call_kwargs["evaluator"] == "judge"
        assert call_kwargs["judge_model_id"] is None
        assert call_kwargs["llm_judge_model"] is None

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_passes_judge_model_when_evaluator_is_judge(self, tmp_path):
        """evaluator=judge forwards judge_model_id as llm_judge_model to ai4rag."""
        modules, mock_create_ogx, mock_run_opt, _ = _make_ai4rag_mocks()
        mock_create_ogx.return_value = mock.MagicMock()
        mock_run_opt.return_value = SimpleNamespace(patterns=[])

        rag_patterns = mock.MagicMock()
        rag_patterns.path = str(tmp_path / "out")
        rag_patterns.uri = "uri"
        rag_patterns.metadata = {}

        html_artifact = mock.MagicMock()
        html_artifact.path = str(tmp_path / "leaderboard.html")
        html_artifact.metadata = {}

        with mock.patch.dict("sys.modules", modules):
            rag_templates_optimization.python_func(
                extracted_text=str(tmp_path / "ext"),
                test_data=str(tmp_path / "td.json"),
                search_space_prep_report=str(tmp_path / "r.yml"),
                rag_patterns=rag_patterns,
                test_data_key="key.json",
                vector_io_provider_id="provider",
                html_artifact=html_artifact,
                evaluator="judge",
                judge_model_id="granite-3.3-8b-instruct",
            )

        assert mock_run_opt.call_args.kwargs["evaluator"] == "judge"
        assert mock_run_opt.call_args.kwargs["judge_model_id"] == "granite-3.3-8b-instruct"
        assert mock_run_opt.call_args.kwargs["llm_judge_model"] == "granite-3.3-8b-instruct"

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_unitxt_evaluator_disables_llm_judge(self, tmp_path):
        """evaluator=unitxt does not pass a judge model to ai4rag."""
        modules, mock_create_ogx, mock_run_opt, _ = _make_ai4rag_mocks()
        mock_create_ogx.return_value = mock.MagicMock()
        mock_run_opt.return_value = SimpleNamespace(patterns=[])

        rag_patterns = mock.MagicMock()
        rag_patterns.path = str(tmp_path / "out")
        rag_patterns.uri = "uri"
        rag_patterns.metadata = {}

        html_artifact = mock.MagicMock()
        html_artifact.path = str(tmp_path / "leaderboard.html")
        html_artifact.metadata = {}

        with mock.patch.dict("sys.modules", modules):
            rag_templates_optimization.python_func(
                extracted_text=str(tmp_path / "ext"),
                test_data=str(tmp_path / "td.json"),
                search_space_prep_report=str(tmp_path / "r.yml"),
                rag_patterns=rag_patterns,
                test_data_key="key.json",
                vector_io_provider_id="provider",
                html_artifact=html_artifact,
                evaluator="unitxt",
                judge_model_id="granite-3.3-8b-instruct",
            )

        assert mock_run_opt.call_args.kwargs["evaluator"] == "unitxt"
        assert mock_run_opt.call_args.kwargs["llm_judge_model"] is None

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_rejects_invalid_evaluator(self, tmp_path):
        """Unknown evaluator values raise ValueError."""
        modules, mock_create_ogx, mock_run_opt, _ = _make_ai4rag_mocks()
        mock_create_ogx.return_value = mock.MagicMock()

        rag_patterns = mock.MagicMock()
        rag_patterns.path = str(tmp_path / "out")
        rag_patterns.metadata = {}

        html_artifact = mock.MagicMock()
        html_artifact.path = str(tmp_path / "leaderboard.html")
        html_artifact.metadata = {}

        with mock.patch.dict("sys.modules", modules):
            with pytest.raises(ValueError, match="evaluator must be"):
                rag_templates_optimization.python_func(
                    extracted_text=str(tmp_path / "ext"),
                    test_data=str(tmp_path / "td.json"),
                    search_space_prep_report=str(tmp_path / "r.yml"),
                    rag_patterns=rag_patterns,
                    test_data_key="key.json",
                    vector_io_provider_id="provider",
                    html_artifact=html_artifact,
                    evaluator="invalid",
                )

        mock_run_opt.assert_not_called()

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_sets_artifact_metadata(self, tmp_path):
        """Wrapper sets rag_patterns.metadata correctly from result.patterns."""
        modules, mock_create_ogx, mock_run_opt, _ = _make_ai4rag_mocks()
        mock_create_ogx.return_value = mock.MagicMock()

        patterns_list = [
            {"name": "pattern_a", "final_score": 0.95},
            {"name": "pattern_b", "final_score": 0.88},
        ]
        mock_run_opt.return_value = SimpleNamespace(patterns=patterns_list)

        rag_patterns = mock.MagicMock()
        rag_patterns.path = str(tmp_path / "rag_patterns")
        rag_patterns.uri = "gs://bucket/rag_patterns"
        rag_patterns.metadata = {}

        html_artifact = mock.MagicMock()
        html_artifact.path = str(tmp_path / "leaderboard.html")
        html_artifact.metadata = {}

        with mock.patch.dict("sys.modules", modules):
            rag_templates_optimization.python_func(
                extracted_text=str(tmp_path / "ext"),
                test_data=str(tmp_path / "td.json"),
                search_space_prep_report=str(tmp_path / "r.yml"),
                rag_patterns=rag_patterns,
                test_data_key="key.json",
                vector_io_provider_id="provider",
                html_artifact=html_artifact,
            )

        assert rag_patterns.metadata["name"] == "rag_patterns_artifact"
        assert rag_patterns.metadata["uri"] == "gs://bucket/rag_patterns"
        assert rag_patterns.metadata["metadata"]["patterns"] == patterns_list

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_creates_output_directory(self, tmp_path):
        """Output directory is created before calling run_rag_optimization."""
        modules, mock_create_ogx, mock_run_opt, _ = _make_ai4rag_mocks()
        mock_create_ogx.return_value = mock.MagicMock()
        mock_run_opt.return_value = SimpleNamespace(patterns=[])

        output_dir = tmp_path / "nested" / "rag_patterns"
        rag_patterns = mock.MagicMock()
        rag_patterns.path = str(output_dir)
        rag_patterns.uri = "uri"
        rag_patterns.metadata = {}

        html_artifact = mock.MagicMock()
        html_artifact.path = str(tmp_path / "leaderboard.html")
        html_artifact.metadata = {}

        with mock.patch.dict("sys.modules", modules):
            rag_templates_optimization.python_func(
                extracted_text=str(tmp_path / "ext"),
                test_data=str(tmp_path / "td.json"),
                search_space_prep_report=str(tmp_path / "r.yml"),
                rag_patterns=rag_patterns,
                test_data_key="key.json",
                vector_io_provider_id="provider",
                html_artifact=html_artifact,
            )

        assert output_dir.exists()

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_none_keys_default_to_empty_string(self, tmp_path):
        """None test_data_key and input_data_key default to empty strings."""
        modules, mock_create_ogx, mock_run_opt, _ = _make_ai4rag_mocks()
        mock_create_ogx.return_value = mock.MagicMock()
        mock_run_opt.return_value = SimpleNamespace(patterns=[])

        rag_patterns = mock.MagicMock()
        rag_patterns.path = str(tmp_path / "out")
        rag_patterns.uri = "uri"
        rag_patterns.metadata = {}

        html_artifact = mock.MagicMock()
        html_artifact.path = str(tmp_path / "leaderboard.html")
        html_artifact.metadata = {}

        with mock.patch.dict("sys.modules", modules):
            rag_templates_optimization.python_func(
                extracted_text=str(tmp_path / "ext"),
                test_data=str(tmp_path / "td.json"),
                search_space_prep_report=str(tmp_path / "r.yml"),
                rag_patterns=rag_patterns,
                test_data_key=None,
                vector_io_provider_id="provider",
                html_artifact=html_artifact,
                input_data_key=None,
            )

        call_kwargs = mock_run_opt.call_args.kwargs
        assert call_kwargs["test_data_key"] == ""
        assert call_kwargs["input_data_key"] == ""

    def test_missing_ogx_env_raises_key_error(self, tmp_path):
        """Missing OGX env vars raise KeyError."""
        modules, _, _, _ = _make_ai4rag_mocks()

        rag_patterns = mock.MagicMock()
        rag_patterns.path = str(tmp_path / "out")
        rag_patterns.metadata = {}

        html_artifact = mock.MagicMock()
        html_artifact.path = str(tmp_path / "leaderboard.html")
        html_artifact.metadata = {}

        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch.dict("sys.modules", modules):
                with pytest.raises(KeyError):
                    rag_templates_optimization.python_func(
                        extracted_text=str(tmp_path / "ext"),
                        test_data=str(tmp_path / "td.json"),
                        search_space_prep_report=str(tmp_path / "r.yml"),
                        rag_patterns=rag_patterns,
                        test_data_key="key.json",
                        vector_io_provider_id="provider",
                        html_artifact=html_artifact,
                    )

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_propagates_ai4rag_exception(self, tmp_path):
        """Exceptions from ai4rag are propagated to the caller."""
        modules, mock_create_ogx, mock_run_opt, _ = _make_ai4rag_mocks()
        mock_create_ogx.return_value = mock.MagicMock()
        mock_run_opt.side_effect = ValueError("test_data_path must point to a JSON file")

        rag_patterns = mock.MagicMock()
        rag_patterns.path = str(tmp_path / "out")
        rag_patterns.metadata = {}

        html_artifact = mock.MagicMock()
        html_artifact.path = str(tmp_path / "leaderboard.html")
        html_artifact.metadata = {}

        with mock.patch.dict("sys.modules", modules):
            with pytest.raises(ValueError, match="test_data_path must point to a JSON file"):
                rag_templates_optimization.python_func(
                    extracted_text=str(tmp_path / "ext"),
                    test_data=str(tmp_path / "td.json"),
                    search_space_prep_report=str(tmp_path / "r.yml"),
                    rag_patterns=rag_patterns,
                    test_data_key="key.json",
                    vector_io_provider_id="provider",
                    html_artifact=html_artifact,
                )

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_leaderboard_html_written_with_correct_args(self, tmp_path):
        """build_leaderboard_html receives output_dir (Path) and HTML is written to html_artifact."""
        from pathlib import Path

        modules, mock_create_ogx, mock_run_opt, _ = _make_ai4rag_mocks()
        mock_create_ogx.return_value = mock.MagicMock()
        mock_run_opt.return_value = SimpleNamespace(patterns=[{"name": "p1"}])

        mock_leaderboard = modules["ai4rag.components.assets_generator.leaderboard"]
        expected_html = "<html><body>leaderboard</body></html>"
        mock_leaderboard.build_leaderboard_html.return_value = expected_html

        output_dir = tmp_path / "rag_patterns"
        rag_patterns = mock.MagicMock()
        rag_patterns.path = str(output_dir)
        rag_patterns.uri = "uri"
        rag_patterns.metadata = {}

        html_path = tmp_path / "leaderboard.html"
        html_artifact = mock.MagicMock()
        html_artifact.path = str(html_path)
        html_artifact.metadata = {}

        with mock.patch.dict("sys.modules", modules):
            rag_templates_optimization.python_func(
                extracted_text=str(tmp_path / "ext"),
                test_data=str(tmp_path / "td.json"),
                search_space_prep_report=str(tmp_path / "r.yml"),
                rag_patterns=rag_patterns,
                test_data_key="key.json",
                vector_io_provider_id="provider",
                html_artifact=html_artifact,
            )

        mock_leaderboard.build_leaderboard_html.assert_called_once()
        call_kwargs = mock_leaderboard.build_leaderboard_html.call_args.kwargs
        assert isinstance(call_kwargs["patterns_dir"], Path)
        assert str(call_kwargs["patterns_dir"]) == str(output_dir)

        assert html_path.exists()
        assert html_path.read_text(encoding="utf-8") == expected_html
        assert html_artifact.metadata["display_name"] == "autorag_leaderboard"
