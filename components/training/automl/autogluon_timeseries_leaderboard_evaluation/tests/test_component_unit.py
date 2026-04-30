"""Tests for the timeseries_leaderboard_evaluation component."""

import json
import sys
from pathlib import Path
from unittest import mock

import pytest


@pytest.fixture(autouse=True, scope="module")
def isolated_sys_modules():
    """Patch pandas in sys.modules only for this test module; restored on module teardown."""
    with mock.patch.dict(sys.modules, clear=False) as mocked_modules:
        mocked_modules["pandas"] = mock.MagicMock()
        yield


from ..component import timeseries_leaderboard_evaluation  # noqa: E402


def _make_model_artifact(base_path: Path, model_name: str, metrics: dict, *, uri: str = "http://example.com/artifacts"):
    """Create the filesystem layout written by autogluon_timeseries_models_full_refit and return a mock artifact.

    Layout:
        base_path/{model_name_full}/metrics/metrics.json
    """
    model_name_full = f"{model_name}_FULL"
    metrics_dir = base_path / model_name_full / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics))

    m = mock.MagicMock()
    m.uri = f"{uri}/{model_name_full}"
    m.path = str(base_path)
    m.metadata = {}  # empty — KFP does not propagate metadata for collected artifact lists
    return m


def _make_mock_sorted_df(rows, columns):
    """Build a mock sorted DataFrame with the given rows and columns."""
    mock_df_sorted = mock.MagicMock()
    mock_df_sorted.__len__ = lambda self: len(rows)
    mock_df_sorted.index.name = "rank"
    mock_df_sorted.columns = columns
    mock_df_sorted.iterrows.return_value = list(rows)
    row_dicts = [r[1] for r in rows]
    mock_df_sorted.iloc.__getitem__ = lambda _, idx: row_dicts[idx]
    return mock_df_sorted


@pytest.fixture()
def html_output_path(tmp_path):
    """Provide a temporary HTML output path."""
    return str(tmp_path / "leaderboard.html")


@pytest.fixture()
def embedded_artifact():
    """Provide mock embedded artifact pointing to shared dir (for leaderboard_html_template.html)."""
    shared_dir = Path(__file__).resolve().parent.parent.parent / "shared"
    mock_artifact = mock.MagicMock()
    mock_artifact.path = str(shared_dir)
    return mock_artifact


class TestTimeseriesLeaderboardEvaluationUnitTests:
    """Unit tests for timeseries_leaderboard_evaluation component logic."""

    @mock.patch("pandas.DataFrame")
    def test_single_model(self, mock_dataframe_class, tmp_path, html_output_path, embedded_artifact):
        """Test leaderboard with a single model: return value, metadata, HTML output."""
        artifact = _make_model_artifact(
            tmp_path / "ets",
            "ETS",
            {"mean_absolute_scaled_error": -0.85, "weighted_absolute_percentage_error": -0.12},
        )

        columns = ["model", "mean_absolute_scaled_error", "weighted_absolute_percentage_error", "notebook", "predictor"]
        rows = [
            (
                1,
                {
                    "model": "ETS_FULL",
                    "mean_absolute_scaled_error": -0.85,
                    "weighted_absolute_percentage_error": -0.12,
                    "notebook": "http://example.com/artifacts/ETS_FULL/ETS_FULL/notebooks/automl_predictor_notebook.ipynb",
                    "predictor": "http://example.com/artifacts/ETS_FULL/ETS_FULL/predictor",
                },
            ),
        ]
        mock_df_sorted = _make_mock_sorted_df(rows, columns)
        mock_df = mock.MagicMock()
        mock_df.sort_values.return_value = mock_df_sorted
        mock_dataframe_class.return_value = mock_df

        mock_html = mock.MagicMock()
        mock_html.path = html_output_path
        mock_html.metadata = {}

        result = timeseries_leaderboard_evaluation.python_func(
            models=[artifact],
            eval_metric="mean_absolute_scaled_error",  # legacy acronym — normalized to snake_case internally
            html_artifact=mock_html,
            embedded_artifact=embedded_artifact,
        )

        # Verify DataFrame constructed with correct data read from metrics.json
        mock_dataframe_class.assert_called_once()
        call_args = mock_dataframe_class.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0]["model"] == "ETS_FULL"
        assert call_args[0]["mean_absolute_scaled_error"] == -0.85
        assert call_args[0]["weighted_absolute_percentage_error"] == -0.12

        # Verify URIs derived from artifact.uri and model directory name
        assert "ETS_FULL/notebooks/automl_predictor_notebook.ipynb" in call_args[0]["notebook"]
        assert "ETS_FULL/predictor" in call_args[0]["predictor"]

        # Verify sort by normalized snake_case key
        mock_df.sort_values.assert_called_once_with(by="mean_absolute_scaled_error", ascending=False)

        # Verify best_model return value
        assert result.best_model == "ETS_FULL"

        # Verify HTML artifact metadata
        assert mock_html.metadata["display_name"] == "automl_timeseries_leaderboard"
        assert "data" in mock_html.metadata

        # Verify HTML file content
        html = Path(html_output_path).read_text()
        assert "Notebook" in html
        assert "Predictor" in html
        assert "ETS_FULL" in html
        assert "uri-cell" in html

    @mock.patch("pandas.DataFrame")
    def test_multiple_models(self, mock_dataframe_class, tmp_path, html_output_path, embedded_artifact):
        """Test leaderboard with multiple models and best_model selection."""
        artifacts = [
            _make_model_artifact(
                tmp_path / "ets",
                "ETS",
                {"mean_absolute_scaled_error": -0.85, "weighted_absolute_percentage_error": -0.12},
            ),
            _make_model_artifact(
                tmp_path / "deepar",
                "DeepAR",
                {"mean_absolute_scaled_error": -0.72, "weighted_absolute_percentage_error": -0.09},
            ),
            _make_model_artifact(
                tmp_path / "theta",
                "Theta",
                {"mean_absolute_scaled_error": -0.95, "weighted_absolute_percentage_error": -0.15},
            ),
        ]

        columns = ["model", "mean_absolute_scaled_error", "weighted_absolute_percentage_error", "notebook", "predictor"]
        rows = [
            (
                1,
                {
                    "model": "DeepAR_FULL",
                    "mean_absolute_scaled_error": -0.72,
                    "weighted_absolute_percentage_error": -0.09,
                    "notebook": "nb2",
                    "predictor": "p2",
                },
            ),
            (
                2,
                {
                    "model": "ETS_FULL",
                    "mean_absolute_scaled_error": -0.85,
                    "weighted_absolute_percentage_error": -0.12,
                    "notebook": "nb1",
                    "predictor": "p1",
                },
            ),
            (
                3,
                {
                    "model": "Theta_FULL",
                    "mean_absolute_scaled_error": -0.95,
                    "weighted_absolute_percentage_error": -0.15,
                    "notebook": "nb3",
                    "predictor": "p3",
                },
            ),
        ]
        mock_df_sorted = _make_mock_sorted_df(rows, columns)
        mock_df = mock.MagicMock()
        mock_df.sort_values.return_value = mock_df_sorted
        mock_dataframe_class.return_value = mock_df

        mock_html = mock.MagicMock()
        mock_html.path = html_output_path
        mock_html.metadata = {}

        result = timeseries_leaderboard_evaluation.python_func(
            models=artifacts,
            eval_metric="mean_absolute_scaled_error",
            html_artifact=mock_html,
            embedded_artifact=embedded_artifact,
        )

        call_args = mock_dataframe_class.call_args[0][0]
        assert len(call_args) == 3
        model_names = [r["model"] for r in call_args]
        assert "ETS_FULL" in model_names
        assert "DeepAR_FULL" in model_names
        assert "Theta_FULL" in model_names

        # Best model is first after descending sort
        assert result.best_model == "DeepAR_FULL"

        mock_df.sort_values.assert_called_once_with(by="mean_absolute_scaled_error", ascending=False)

        html = Path(html_output_path).read_text()
        assert "ETS_FULL" in html
        assert "DeepAR_FULL" in html
        assert "Theta_FULL" in html

    @mock.patch("pandas.DataFrame")
    def test_partial_missing_metrics_file_skips_bad_artifact(
        self, mock_dataframe_class, tmp_path, html_output_path, embedded_artifact
    ):
        """Artifact with no metrics.json is skipped; valid artifacts still appear in the leaderboard."""
        # Simulate a failed refit task: artifact path exists but has no metrics.json
        bad_path = tmp_path / "tft"
        bad_path.mkdir()
        bad_artifact = mock.MagicMock()
        bad_artifact.uri = "http://example.com/artifacts/TFT_FULL"
        bad_artifact.path = str(bad_path)
        bad_artifact.metadata = {}

        good_artifact = _make_model_artifact(tmp_path / "ets", "ETS", {"mean_absolute_scaled_error": -0.85})

        columns = ["model", "mean_absolute_scaled_error", "notebook", "predictor"]
        rows = [(1, {"model": "ETS_FULL", "mean_absolute_scaled_error": -0.85, "notebook": "nb", "predictor": "p"})]
        mock_df_sorted = _make_mock_sorted_df(rows, columns)
        mock_df = mock.MagicMock()
        mock_df.sort_values.return_value = mock_df_sorted
        mock_dataframe_class.return_value = mock_df

        mock_html = mock.MagicMock()
        mock_html.path = html_output_path
        mock_html.metadata = {}

        result = timeseries_leaderboard_evaluation.python_func(
            models=[bad_artifact, good_artifact],
            eval_metric="mean_absolute_scaled_error",
            html_artifact=mock_html,
            embedded_artifact=embedded_artifact,
        )

        call_args = mock_dataframe_class.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0]["model"] == "ETS_FULL"
        assert result.best_model == "ETS_FULL"

    def test_empty_models_raises(self, html_output_path, embedded_artifact):
        """Test that ValueError is raised when models list is empty."""
        mock_html = mock.MagicMock()
        mock_html.path = html_output_path
        mock_html.metadata = {}

        with pytest.raises(ValueError, match="models list must not be empty"):
            timeseries_leaderboard_evaluation.python_func(
                models=[],
                eval_metric="mean_absolute_scaled_error",
                html_artifact=mock_html,
                embedded_artifact=embedded_artifact,
            )

    def test_empty_eval_metric_raises(self, tmp_path, embedded_artifact):
        """Test that TypeError is raised when eval_metric is empty or not a string."""
        artifact = _make_model_artifact(tmp_path / "ets", "ETS", {"mean_absolute_scaled_error": -0.85})
        mock_html = mock.MagicMock()
        mock_html.path = "/tmp/out.html"

        with pytest.raises(TypeError, match=r"eval_metric must be a non-empty string\."):
            timeseries_leaderboard_evaluation.python_func(
                models=[artifact],
                eval_metric="",
                html_artifact=mock_html,
                embedded_artifact=embedded_artifact,
            )

        with pytest.raises(TypeError, match=r"eval_metric must be a non-empty string\."):
            timeseries_leaderboard_evaluation.python_func(
                models=[artifact],
                eval_metric="   ",
                html_artifact=mock_html,
                embedded_artifact=embedded_artifact,
            )

    def test_all_missing_metrics_files_raises(self, tmp_path, html_output_path, embedded_artifact):
        """ValueError raised when every artifact is missing its metrics.json (all refit tasks failed)."""
        empty_path = tmp_path / "empty"
        empty_path.mkdir()
        artifact = mock.MagicMock()
        artifact.uri = "http://example.com/artifacts/Model_FULL"
        artifact.path = str(empty_path)
        artifact.metadata = {}

        mock_html = mock.MagicMock()
        mock_html.path = html_output_path
        mock_html.metadata = {}

        with pytest.raises(ValueError, match="No valid model artifacts found"):
            timeseries_leaderboard_evaluation.python_func(
                models=[artifact],
                eval_metric="mean_absolute_scaled_error",
                html_artifact=mock_html,
                embedded_artifact=embedded_artifact,
            )

    def test_uri_construction(self, tmp_path, embedded_artifact):
        """Predictor and notebook URIs are derived from artifact.uri and the model directory name."""
        artifact = _make_model_artifact(
            tmp_path / "ets",
            "ETS",
            {"mean_absolute_scaled_error": -0.85},
            uri="http://s3.example.com/bucket/run-id/artifacts",
        )

        with mock.patch("pandas.DataFrame") as mock_df_class:
            columns = ["model", "mean_absolute_scaled_error", "notebook", "predictor"]
            rows = [
                (
                    1,
                    {
                        "model": "ETS_FULL",
                        "mean_absolute_scaled_error": -0.85,
                        "notebook": "http://s3.example.com/bucket/run-id/artifacts/ETS_FULL/ETS_FULL/notebooks/automl_predictor_notebook.ipynb",
                        "predictor": "http://s3.example.com/bucket/run-id/artifacts/ETS_FULL/ETS_FULL/predictor",
                    },
                )
            ]
            mock_df_sorted = _make_mock_sorted_df(rows, columns)
            mock_df = mock.MagicMock()
            mock_df.sort_values.return_value = mock_df_sorted
            mock_df_class.return_value = mock_df

            mock_html = mock.MagicMock()
            mock_html.path = str(tmp_path / "out.html")
            mock_html.metadata = {}

            timeseries_leaderboard_evaluation.python_func(
                models=[artifact],
                eval_metric="mean_absolute_scaled_error",
                html_artifact=mock_html,
                embedded_artifact=embedded_artifact,
            )

            call_args = mock_df_class.call_args[0][0]
            expected_base = "http://s3.example.com/bucket/run-id/artifacts/ETS_FULL"
            assert call_args[0]["predictor"] == f"{expected_base}/ETS_FULL/predictor"
            assert call_args[0]["notebook"] == f"{expected_base}/ETS_FULL/notebooks/automl_predictor_notebook.ipynb"

    def test_artifact_missing_metrics_json_skipped_with_warning(self, tmp_path, html_output_path, embedded_artifact):
        """Artifact without metrics/metrics.json is skipped with a warning, not an error."""
        # One valid artifact, one with no metrics.json
        good_artifact = _make_model_artifact(
            tmp_path / "ets", "ETS", {"mean_absolute_scaled_error": -0.85, "weighted_absolute_percentage_error": -0.12}
        )
        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        bad_artifact = mock.MagicMock()
        bad_artifact.uri = "http://example.com/artifacts/Bad_FULL"
        bad_artifact.path = str(bad_dir)
        bad_artifact.metadata = {}

        columns = ["model", "mean_absolute_scaled_error", "weighted_absolute_percentage_error", "notebook", "predictor"]
        rows = [
            (
                1,
                {
                    "model": "ETS_FULL",
                    "mean_absolute_scaled_error": -0.85,
                    "weighted_absolute_percentage_error": -0.12,
                    "notebook": "nb",
                    "predictor": "pred",
                },
            ),
        ]
        mock_df_sorted = _make_mock_sorted_df(rows, columns)

        with mock.patch("pandas.DataFrame") as mock_df_class:
            mock_df = mock.MagicMock()
            mock_df.sort_values.return_value = mock_df_sorted
            mock_df_class.return_value = mock_df

            mock_html = mock.MagicMock()
            mock_html.path = html_output_path
            mock_html.metadata = {}

            result = timeseries_leaderboard_evaluation.python_func(
                models=[bad_artifact, good_artifact],
                eval_metric="mean_absolute_scaled_error",
                html_artifact=mock_html,
                embedded_artifact=embedded_artifact,
            )

            # Only the good artifact should be in results
            call_args = mock_df_class.call_args[0][0]
            assert len(call_args) == 1
            assert call_args[0]["model"] == "ETS_FULL"
            assert result.best_model == "ETS_FULL"

    def test_all_artifacts_missing_metrics_raises_value_error(self, tmp_path, html_output_path, embedded_artifact):
        """ValueError is raised when all artifacts are missing metrics.json."""
        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        bad_artifact = mock.MagicMock()
        bad_artifact.uri = "http://example.com/artifacts/Bad_FULL"
        bad_artifact.path = str(bad_dir)
        bad_artifact.metadata = {}

        mock_html = mock.MagicMock()
        mock_html.path = html_output_path
        mock_html.metadata = {}

        with pytest.raises(ValueError, match="No valid model artifacts found"):
            timeseries_leaderboard_evaluation.python_func(
                models=[bad_artifact],
                eval_metric="mean_absolute_scaled_error",
                html_artifact=mock_html,
                embedded_artifact=embedded_artifact,
            )

    def test_component_imports_correctly(self):
        """Test that the component can be imported and has required attributes."""
        assert callable(timeseries_leaderboard_evaluation)
        assert hasattr(timeseries_leaderboard_evaluation, "python_func")
        assert hasattr(timeseries_leaderboard_evaluation, "component_spec")
