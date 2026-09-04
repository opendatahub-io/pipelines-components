"""Tests for shared leaderboard utility functions in leaderboard_utils.py."""

from pathlib import Path
from unittest import mock

import pytest

# Importable via the sys.path insertion in components/training/automl/conftest.py,
# which replicates what KFP does at container runtime (adding the embedded
# artifact directory to sys.path).
from ..leaderboard_utils import _build_leaderboard_html, _build_leaderboard_table, _format_metric_value


@pytest.fixture()
def template_path():
    """Path to the shared leaderboard HTML template."""
    return Path(__file__).resolve().parent.parent / "leaderboard_html_template.html"


class TestBuildLeaderboardTable:
    """Tests for _build_leaderboard_table."""

    def _make_df(self, rows, columns, index_name="rank"):
        """Build a mock DataFrame usable by _build_leaderboard_table."""
        df = mock.MagicMock()
        df.columns = columns
        df.index.name = index_name
        df.iterrows.return_value = list(rows)
        return df

    def test_basic_html_structure(self):
        """Output is a valid table with thead and tbody."""
        columns = ["model", "rmse", "notebook", "predictor"]
        rows = [(1, {"model": "M1", "rmse": 0.5, "notebook": "http://nb", "predictor": "http://pred"})]
        html = _build_leaderboard_table(self._make_df(rows, columns))
        assert html.startswith("<table>")
        assert html.endswith("</table>")
        assert "<thead>" in html
        assert "<tbody>" in html

    def test_header_contains_rank_and_metric_columns(self):
        """Header row has rank, metric columns, Notebook, Predictor."""
        columns = ["model", "rmse", "notebook", "predictor"]
        rows = [(1, {"model": "M1", "rmse": 0.5, "notebook": "nb", "predictor": "pred"})]
        html = _build_leaderboard_table(self._make_df(rows, columns))
        assert "<th>rank</th>" in html
        assert "<th>model</th>" in html
        assert "<th>rmse</th>" in html
        assert "<th>Notebook</th>" in html
        assert "<th>Predictor</th>" in html

    def test_notebook_and_predictor_not_regular_columns(self):
        """'notebook' and 'predictor' do not appear as plain <th> column headers."""
        columns = ["model", "notebook", "predictor"]
        rows = [(1, {"model": "M1", "notebook": "nb", "predictor": "pred"})]
        html = _build_leaderboard_table(self._make_df(rows, columns))
        assert "<th>notebook</th>" not in html
        assert "<th>predictor</th>" not in html

    def test_uri_cells_have_popover_structure(self):
        """Notebook and predictor cells use uri-cell/uri-link/uri-popover markup."""
        columns = ["model", "notebook", "predictor"]
        rows = [(1, {"model": "M1", "notebook": "http://example.com/nb", "predictor": "http://example.com/pred"})]
        html = _build_leaderboard_table(self._make_df(rows, columns))
        assert 'class="uri-cell"' in html
        assert 'class="uri-link"' in html
        assert 'class="uri-popover"' in html
        assert 'data-uri="http://example.com/nb"' in html
        assert 'data-uri="http://example.com/pred"' in html

    def test_html_special_characters_escaped(self):
        """Values containing HTML special characters are escaped."""
        columns = ["model", "notebook", "predictor"]
        rows = [(1, {"model": 'Model<1>&"', "notebook": "http://nb", "predictor": "http://pred"})]
        html = _build_leaderboard_table(self._make_df(rows, columns))
        assert "Model&lt;1&gt;&amp;&quot;" in html
        assert "Model<1>&" not in html

    def test_multiple_rows_all_present(self):
        """All rows appear in the output and row count matches."""
        columns = ["model", "rmse", "notebook", "predictor"]
        rows = [
            (1, {"model": "ModelA", "rmse": 0.3, "notebook": "nb1", "predictor": "pred1"}),
            (2, {"model": "ModelB", "rmse": 0.5, "notebook": "nb2", "predictor": "pred2"}),
            (3, {"model": "ModelC", "rmse": 0.8, "notebook": "nb3", "predictor": "pred3"}),
        ]
        html = _build_leaderboard_table(self._make_df(rows, columns))
        assert "ModelA" in html
        assert "ModelB" in html
        assert "ModelC" in html
        # 3 data rows + 1 header row in <thead>
        assert html.count("<tr>") == 4

    def test_row_index_included_as_first_cell(self):
        """The row index (rank) is rendered as the first <td> in each row."""
        columns = ["model", "notebook", "predictor"]
        rows = [(7, {"model": "M1", "notebook": "nb", "predictor": "pred"})]
        html = _build_leaderboard_table(self._make_df(rows, columns))
        assert "<td>7</td>" in html

    def test_metric_values_truncated_in_output(self):
        """Metric cell values go through the same truncation as _format_metric_value."""
        columns = ["model", "rmse", "notebook", "predictor"]
        rows = [(1, {"model": "M1", "rmse": 0.1238, "notebook": "nb", "predictor": "pred"})]
        html = _build_leaderboard_table(self._make_df(rows, columns))
        assert "<td>0.123</td>" in html
        assert "0.1238" not in html
        assert "0.124" not in html

    def test_custom_index_name(self):
        """A custom index name is used instead of 'rank'."""
        columns = ["model", "notebook", "predictor"]
        rows = [(1, {"model": "M1", "notebook": "nb", "predictor": "pred"})]
        html = _build_leaderboard_table(self._make_df(rows, columns, index_name="position"))
        assert "<th>position</th>" in html
        assert "<th>rank</th>" not in html


class TestFormatMetricValue:
    """Tests for _format_metric_value."""

    def test_truncates_not_rounds(self):
        """0.1238 truncates to 0.123, it does not round up to 0.124."""
        assert _format_metric_value(0.1238) == "0.123"

    def test_truncates_not_rounds_up_edge(self):
        """0.1239999 truncates down to 0.123 instead of rounding to 0.124."""
        assert _format_metric_value(0.1239999) == "0.123"

    def test_strips_trailing_zeros(self):
        """0.3 stays 0.3 instead of being padded to 0.300."""
        assert _format_metric_value(0.3) == "0.3"

    def test_strips_trailing_zeros_with_float_imprecision(self):
        """A float that is nearly 0.3 due to binary imprecision still renders as 0.3."""
        assert _format_metric_value(0.30000000004) == "0.3"

    def test_exact_three_decimals_kept(self):
        """A value with exactly 3 significant decimals is kept as-is."""
        assert _format_metric_value(0.123) == "0.123"

    def test_more_than_three_decimals_truncated(self):
        """Values with more than 3 decimals are truncated to 3, no rounding."""
        assert _format_metric_value(0.123456789) == "0.123"

    def test_whole_number_float_has_no_decimal_point(self):
        """A float with no fractional part (e.g. 1.0) renders without a trailing '.0'."""
        assert _format_metric_value(1.0) == "1"

    def test_negative_value_truncated(self):
        """Negative floats are truncated toward zero, not rounded, and keep their sign."""
        assert _format_metric_value(-0.1239) == "-0.123"

    def test_negative_value_strips_trailing_zeros(self):
        """Negative floats also strip trailing zeros instead of padding."""
        assert _format_metric_value(-0.3) == "-0.3"

    def test_small_value_below_precision(self):
        """A value smaller than the truncation precision renders as 0."""
        assert _format_metric_value(0.0001) == "0"

    def test_value_at_precision_boundary(self):
        """A value exactly at the 3-decimal boundary is kept."""
        assert _format_metric_value(0.001) == "0.001"

    def test_nan_not_mangled(self):
        """NaN is passed through as a string rather than truncated/garbled."""
        assert _format_metric_value(float("nan")) == "nan"

    def test_infinity_not_mangled(self):
        """Infinity is passed through as a string rather than truncated/garbled."""
        assert _format_metric_value(float("inf")) == "inf"
        assert _format_metric_value(float("-inf")) == "-inf"

    def test_integer_passthrough(self):
        """Plain ints are not treated as floats and are stringified as-is."""
        assert _format_metric_value(5) == "5"

    def test_boolean_passthrough(self):
        """Booleans are not treated as floats (bool is a subclass of int)."""
        assert _format_metric_value(True) == "True"

    def test_string_passthrough(self):
        """Non-numeric values (e.g. model names) are stringified unchanged."""
        assert _format_metric_value("model_a") == "model_a"

    def test_large_float_value(self):
        """Large floats are truncated to 3 decimals like any other float."""
        assert _format_metric_value(12345.6789) == "12345.678"


class TestBuildLeaderboardHtml:
    """Tests for _build_leaderboard_html."""

    def test_all_placeholders_replaced(self, template_path):
        """No placeholder tokens remain after substitution."""
        html = _build_leaderboard_html(
            template_path=template_path,
            table_html="<table></table>",
            eval_metric="rmse",
            best_model_name="BestModel",
            num_models=3,
        )
        assert "__TABLE_HTML__" not in html
        assert "__NUM_MODELS__" not in html
        assert "__EVAL_METRIC__" not in html
        assert "__BEST_MODEL_NAME__" not in html

    def test_values_appear_in_output(self, template_path):
        """Substituted values are present in the rendered HTML."""
        html = _build_leaderboard_html(
            template_path=template_path,
            table_html="<table>MY_TABLE</table>",
            eval_metric="accuracy",
            best_model_name="TopModel",
            num_models=5,
        )
        assert "MY_TABLE" in html
        assert "accuracy" in html
        assert "TopModel" in html
        assert "5" in html

    def test_output_is_complete_html_document(self, template_path):
        """Output contains standard HTML document markers."""
        html = _build_leaderboard_html(
            template_path=template_path,
            table_html="<table></table>",
            eval_metric="r2",
            best_model_name="M1",
            num_models=1,
        )
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html

    def test_missing_template_raises(self, tmp_path):
        """FileNotFoundError is raised when the template path does not exist."""
        with pytest.raises(FileNotFoundError):
            _build_leaderboard_html(
                template_path=tmp_path / "nonexistent.html",
                table_html="<table></table>",
                eval_metric="rmse",
                best_model_name="M",
                num_models=1,
            )
