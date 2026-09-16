"""Unit tests for the GRPO evaluation component."""

import json
import math
import sys
import tempfile
from pathlib import Path

import pytest
from kfp import compiler

from ..component import grpo_eval


class MockMetrics:
    """Minimal KFP Metrics substitute recording logged scalar values."""

    def __init__(self):
        """Create an empty metric collection."""
        self.logged_metrics = {}

    def log_metric(self, name: str, value: float) -> None:
        """Record a scalar metric using the KFP Metrics interface."""
        self.logged_metrics[name] = value


class MockHtml:
    """Minimal KFP HTML substitute exposing an artifact output path."""

    def __init__(self, path: Path):
        """Create an HTML artifact at the supplied output path."""
        self.path = str(path)


def write_results(path: Path, payload: object) -> Path:
    """Write a test training-results payload and return its path."""
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def run_component(path: Path):
    """Run the component's Python function with a mock Metrics artifact."""
    metrics = MockMetrics()
    reward_chart = MockHtml(path.with_name("reward_chart.html"))
    result = grpo_eval.python_func(
        training_results_path=str(path),
        output_metrics=metrics,
        output_reward_chart=reward_chart,
    )
    return result, metrics, Path(reward_chart.path)


def test_component_compiles():
    """The component YAML compiles without a cluster or accelerator."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as component_file:
        compiler.Compiler().compile(grpo_eval, component_file.name)
        assert Path(component_file.name).stat().st_size > 0


def test_component_signature():
    """The public component contract has one path input and two outputs."""
    spec = grpo_eval.component_spec

    assert set(spec.inputs) == {"training_results_path"}
    assert set(spec.outputs) == {"output_metrics", "output_reward_chart", "promotion_passed"}


def test_component_uses_lightweight_ubi_python_image():
    """The component uses a lightweight CPU-only Python image."""
    assert (
        grpo_eval.component_spec.implementation.container.image == "registry.access.redhat.com/ubi9/python-311:latest"
    )


def test_component_installs_its_kfp_runtime_from_pypi():
    """The Training Hub image does not bundle the KFP runtime."""
    command = " ".join(grpo_eval.component_spec.implementation.container.command)

    assert "kfp==2.16.1" in command
    assert "https://pypi.org/simple" in command
    assert "--no-deps" not in command


def test_improving_results_log_metrics_and_pass_promotion(tmp_path: Path):
    """An improving reward history is promoted and produces expected KFP metrics."""
    fixture_path = Path(__file__).parent / "fixtures" / "training_results.json"
    results_path = tmp_path / "training_results.json"
    results_path.write_text(fixture_path.read_text(encoding="utf-8"), encoding="utf-8")

    result, metrics, reward_chart = run_component(results_path)

    assert result.promotion_passed is True
    assert metrics.logged_metrics == {
        "mean_reward": 0.67,
        "full_match_rate": 0.75,
        "initial_reward": 0.33,
        "final_reward": 0.67,
        "reward_improvement": pytest.approx(0.34),
        "training_iterations": 3.0,
        "initial_iteration_time_seconds": 12.4,
        "final_iteration_time_seconds": 11.5,
        "mean_iteration_time_seconds": pytest.approx(11.933333333333334),
        "promotion_passed": 1.0,
    }
    chart_html = reward_chart.read_text(encoding="utf-8")
    assert "<title>Reward curve</title>" in chart_html
    assert "Mean reward by training iteration" in chart_html
    assert "Iteration 1: 0.33" in chart_html
    assert "Iteration 3: 0.67" in chart_html
    assert 'text-anchor="middle">1</text>' in chart_html
    assert 'text-anchor="middle">2</text>' in chart_html
    assert 'text-anchor="middle">3</text>' in chart_html


@pytest.mark.parametrize(
    ("reward_history", "expected_promotion"),
    [
        ([0.8, 0.4], False),
        ([0.5, 0.5], False),
        ([0.5], False),
    ],
)
def test_non_improving_or_single_reward_does_not_pass_promotion(
    tmp_path: Path,
    reward_history: list[float],
    expected_promotion: bool,
):
    """Only strict, multi-iteration reward improvement passes promotion."""
    results_path = write_results(
        tmp_path / "training_results.json",
        {
            "final_mean_reward": 0.5,
            "reward_history": reward_history,
            "full_match_history": [0.5] * len(reward_history),
            "timing_history": [],
        },
    )

    result, _, _ = run_component(results_path)

    assert result.promotion_passed is expected_promotion


def test_empty_timing_history_omits_timing_metrics_and_keeps_rewards_distinct(tmp_path: Path):
    """Empty timing history produces no timing metrics and preserves ART's aggregate."""
    results_path = write_results(
        tmp_path / "training_results.json",
        {
            "final_mean_reward": 0.72,
            "reward_history": [0.33, 0.48, 0.67],
            "full_match_history": [0.40, 0.60, 0.75],
            "timing_history": [],
        },
    )

    result, metrics, _ = run_component(results_path)

    assert result.promotion_passed is True
    assert metrics.logged_metrics["mean_reward"] == 0.72
    assert metrics.logged_metrics["final_reward"] == 0.67
    assert (
        not {
            "initial_iteration_time_seconds",
            "final_iteration_time_seconds",
            "mean_iteration_time_seconds",
        }
        & metrics.logged_metrics.keys()
    )


def test_extreme_reward_range_raises_value_error(tmp_path: Path):
    """Unrenderable finite reward ranges fail before SVG coordinates are created."""
    results_path = write_results(
        tmp_path / "training_results.json",
        {
            "final_mean_reward": 0.0,
            "reward_history": [-sys.float_info.max, sys.float_info.max],
            "full_match_history": [0.0, 0.0],
            "timing_history": [],
        },
    )

    with pytest.raises(ValueError, match="range is too large to render"):
        run_component(results_path)


@pytest.mark.parametrize(
    ("payload", "error_message"),
    [
        ({}, "missing required field"),
        ([], "top-level object"),
        (
            {
                "final_mean_reward": True,
                "reward_history": [0.2, 0.3],
                "full_match_history": [0.3, 0.4],
                "timing_history": [],
            },
            "final_mean_reward.*finite number",
        ),
        (
            {
                "final_mean_reward": 0.5,
                "reward_history": [0.2, 0.3],
                "full_match_history": [],
                "timing_history": [],
            },
            "full_match_history.*non-empty list",
        ),
        (
            {
                "final_mean_reward": 0.5,
                "reward_history": [],
                "full_match_history": [],
                "timing_history": [],
            },
            "reward_history.*non-empty list",
        ),
        (
            {
                "final_mean_reward": 0.5,
                "reward_history": [0.2, math.inf],
                "full_match_history": [0.3, 0.4],
                "timing_history": [],
            },
            "reward_history\\[1\\].*finite number",
        ),
        (
            {
                "final_mean_reward": 0.5,
                "reward_history": [0.2, 0.3],
                "full_match_history": [0.3, 0.4],
                "timing_history": {},
            },
            "timing_history.*list",
        ),
        (
            {
                "final_mean_reward": 0.5,
                "reward_history": [0.2, 0.3],
                "full_match_history": [0.4],
                "timing_history": [],
            },
            "full_match_history.*same length",
        ),
        (
            {
                "final_mean_reward": 0.5,
                "reward_history": [0.2, 0.3],
                "full_match_history": [0.3, math.inf],
                "timing_history": [],
            },
            "full_match_history\\[1\\].*finite number",
        ),
        (
            {
                "final_mean_reward": 0.5,
                "reward_history": [0.2, 0.3],
                "full_match_history": [0.3, 0.4],
                "timing_history": [math.inf],
            },
            "timing_history\\[0\\].*finite number",
        ),
    ],
)
def test_invalid_results_contract_raises_value_error(
    tmp_path: Path,
    payload: object,
    error_message: str,
):
    """Invalid JSON contracts fail before metrics are logged."""
    results_path = write_results(tmp_path / "training_results.json", payload)
    metrics = MockMetrics()

    with pytest.raises(ValueError, match=error_message):
        grpo_eval.python_func(
            training_results_path=str(results_path),
            output_metrics=metrics,
            output_reward_chart=MockHtml(tmp_path / "reward_chart.html"),
        )

    assert metrics.logged_metrics == {}


def test_invalid_json_raises_value_error(tmp_path: Path):
    """Malformed training-results JSON is reported as a contract error."""
    results_path = tmp_path / "training_results.json"
    results_path.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        run_component(results_path)


def test_missing_results_file_raises_file_not_found_error(tmp_path: Path):
    """A missing mounted results file fails clearly."""
    with pytest.raises(FileNotFoundError):
        run_component(tmp_path / "missing.json")


def test_empty_results_path_raises_value_error():
    """A blank input path is rejected before file access."""
    with pytest.raises(ValueError, match="non-empty string"):
        grpo_eval.python_func(
            training_results_path="",
            output_metrics=MockMetrics(),
            output_reward_chart=MockHtml(Path("reward_chart.html")),
        )
