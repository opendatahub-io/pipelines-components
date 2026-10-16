"""GRPO evaluation component."""

from typing import NamedTuple

from kfp import dsl


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-311:latest",
    kfp_package_path="kfp==2.16.1",
    pip_index_urls=["https://pypi.org/simple"],
)
def grpo_eval(
    training_results_path: str,
    output_metrics: dsl.Output[dsl.Metrics],
    output_reward_chart: dsl.Output[dsl.HTML],
) -> NamedTuple("GrpoEvalOutputs", [("promotion_passed", bool)]):
    """Evaluate GRPO training results from the shared pipeline workspace.

    Reads ART's ``training_results.json`` from the caller-provided shared-PVC path. The
    GRPO pipeline supplies ``{workspace_path}/checkpoints/training_results.json``, while
    the complete path input keeps the component independent of the PVC mount location.

    Required JSON fields are ``final_mean_reward``, ``reward_history``,
    ``full_match_history``, and ``timing_history``. The final reward and every history
    entry must be a finite number. Reward and full-match histories must have the same
    number of iterations. The logged ``mean_reward`` is ART's reported final aggregate
    reward, while ``final_reward`` is the last entry in ``reward_history`` used for the
    promotion comparison.

    Args:
        training_results_path: Mounted path to ART's
            ``checkpoints/training_results.json`` file.
        output_metrics: KFP Metrics artifact receiving the GRPO scalar metrics.
        output_reward_chart: KFP HTML artifact containing mean reward by training
            iteration.

    Returns:
        Named output ``promotion_passed``. It is true only when the final reward is
        strictly greater than the initial reward. A single reward value does not
        demonstrate improvement and returns false.

    Raises:
        FileNotFoundError: If ``training_results_path`` does not exist.
        ValueError: If the path is empty, the file is invalid JSON, or required result
            fields do not satisfy the provisional contract.
    """
    import json
    import math

    def require_finite_number(value: object, field_name: str) -> float:
        """Return a JSON number as a float or raise a field-specific error."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"'{field_name}' must be a finite number")

        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"'{field_name}' must be a finite number")
        return number

    def write_reward_chart(rewards: list[float], path: str) -> None:
        """Write a self-contained SVG reward curve as a KFP HTML artifact."""
        chart_width, chart_height, margin = 800, 400, 70
        plot_width = chart_width - 2 * margin
        plot_height = chart_height - 2 * margin
        reward_min, reward_max = min(rewards), max(rewards)
        reward_range = reward_max - reward_min
        padding = reward_range * 0.05 if reward_range else max(abs(reward_min) * 0.05, 0.05)
        y_min, y_max = reward_min - padding, reward_max + padding
        if not all(math.isfinite(value) for value in (reward_range, y_min, y_max)):
            raise ValueError("'reward_history' range is too large to render")

        def x_coordinate(index: int) -> float:
            if len(rewards) == 1:
                return margin + plot_width / 2
            return margin + index * plot_width / (len(rewards) - 1)

        def y_coordinate(reward: float) -> float:
            return margin + (y_max - reward) * plot_height / (y_max - y_min)

        y_ticks = "".join(
            f'<line class="grid" x1="{margin}" y1="{margin + tick * plot_height / 4:.2f}" '
            f'x2="{chart_width - margin}" y2="{margin + tick * plot_height / 4:.2f}" />'
            f'<text x="{margin - 12}" y="{margin + tick * plot_height / 4 + 5:.2f}" '
            f'text-anchor="end">{y_max - tick * (y_max - y_min) / 4:.6g}</text>'
            for tick in range(5)
        )
        x_tick_count = min(len(rewards), 10)
        x_tick_indices = {
            round(tick * (len(rewards) - 1) / (x_tick_count - 1)) if x_tick_count > 1 else 0
            for tick in range(x_tick_count)
        }
        x_ticks = "".join(
            f'<line class="grid" x1="{x_coordinate(index):.2f}" y1="{margin}" '
            f'x2="{x_coordinate(index):.2f}" y2="{chart_height - margin}" />'
            f'<text x="{x_coordinate(index):.2f}" y="{chart_height - margin + 22}" '
            f'text-anchor="middle">{index + 1}</text>'
            for index in sorted(x_tick_indices)
        )
        points = " ".join(
            f"{x_coordinate(index):.2f},{y_coordinate(reward):.2f}" for index, reward in enumerate(rewards)
        )
        circles = "".join(
            f'<circle cx="{x_coordinate(index):.2f}" cy="{y_coordinate(reward):.2f}" r="4">'
            f"<title>Iteration {index + 1}: {reward:.6g}</title></circle>"
            for index, reward in enumerate(rewards)
        )
        html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Reward curve</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px}}svg{{max-width:100%;height:auto}}.axis{{stroke:#555}}.grid{{stroke:#ddd;stroke-dasharray:4}}.curve{{fill:none;stroke:#0066cc;stroke-width:3}}circle{{fill:#0066cc}}text{{fill:#333;font-size:14px}}</style>
</head><body><h1>Reward curve</h1><p>Mean reward by training iteration</p>
<svg viewBox="0 0 {chart_width} {chart_height}" role="img" aria-label="Mean reward by training iteration">
<line class="axis" x1="{margin}" y1="{chart_height - margin}"
      x2="{chart_width - margin}" y2="{chart_height - margin}" />
<line class="axis" x1="{margin}" y1="{margin}" x2="{margin}" y2="{chart_height - margin}" />
{y_ticks}{x_ticks}
<text x="{chart_width / 2}" y="{chart_height - 18}" text-anchor="middle">Iteration</text>
<text x="20" y="{chart_height / 2}" text-anchor="middle"
      transform="rotate(-90 20 {chart_height / 2})">Mean reward</text>
<polyline class="curve" points="{points}" />{circles}
</svg></body></html>"""
        with open(path, "w", encoding="utf-8") as chart_file:
            chart_file.write(html)

    if not isinstance(training_results_path, str) or not training_results_path.strip():
        raise ValueError("'training_results_path' must be a non-empty string")

    try:
        with open(training_results_path, encoding="utf-8") as results_file:
            results = json.load(results_file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in training results file: {training_results_path}") from error

    if not isinstance(results, dict):
        raise ValueError("Training results JSON must contain a top-level object")

    required_fields = (
        "final_mean_reward",
        "reward_history",
        "full_match_history",
        "timing_history",
    )
    for field_name in required_fields:
        if field_name not in results:
            raise ValueError(f"Training results JSON missing required field: '{field_name}'")

    mean_reward = require_finite_number(results["final_mean_reward"], "final_mean_reward")
    reward_history = results["reward_history"]
    full_match_history = results["full_match_history"]
    timing_history = results["timing_history"]

    if not isinstance(reward_history, list) or not reward_history:
        raise ValueError("'reward_history' must be a non-empty list")
    if not isinstance(full_match_history, list) or not full_match_history:
        raise ValueError("'full_match_history' must be a non-empty list")
    if len(reward_history) != len(full_match_history):
        raise ValueError("'reward_history' and 'full_match_history' must have the same length")
    if not isinstance(timing_history, list):
        raise ValueError("'timing_history' must be a list")

    normalized_rewards = [
        require_finite_number(reward, f"reward_history[{index}]") for index, reward in enumerate(reward_history)
    ]
    normalized_match_rates = [
        require_finite_number(match_rate, f"full_match_history[{index}]")
        for index, match_rate in enumerate(full_match_history)
    ]
    normalized_timings = [
        require_finite_number(timing, f"timing_history[{index}]") for index, timing in enumerate(timing_history)
    ]
    initial_reward = normalized_rewards[0]
    final_reward = normalized_rewards[-1]
    full_match_rate = normalized_match_rates[-1]
    reward_improvement = final_reward - initial_reward
    promotion_passed = len(normalized_rewards) > 1 and final_reward > initial_reward

    write_reward_chart(normalized_rewards, output_reward_chart.path)
    output_metrics.log_metric("mean_reward", mean_reward)
    output_metrics.log_metric("full_match_rate", full_match_rate)
    output_metrics.log_metric("initial_reward", initial_reward)
    output_metrics.log_metric("final_reward", final_reward)
    output_metrics.log_metric("reward_improvement", reward_improvement)
    output_metrics.log_metric("training_iterations", float(len(normalized_rewards)))
    if normalized_timings:
        output_metrics.log_metric("initial_iteration_time_seconds", normalized_timings[0])
        output_metrics.log_metric("final_iteration_time_seconds", normalized_timings[-1])
        output_metrics.log_metric("mean_iteration_time_seconds", sum(normalized_timings) / len(normalized_timings))
    output_metrics.log_metric("promotion_passed", float(promotion_passed))

    return NamedTuple("GrpoEvalOutputs", [("promotion_passed", bool)])(
        promotion_passed=promotion_passed,
    )


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        grpo_eval,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
