from typing import NamedTuple

from kfp import dsl
from kfp_components.utils.consts import AUTOML_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=AUTOML_IMAGE,  # noqa: E501
)
def leaderboard_evaluation(
    models_artifact: dsl.Input[dsl.Model],
    eval_metric: str,
    html_artifact: dsl.Output[dsl.HTML],
) -> NamedTuple("outputs", best_model=str):
    """Evaluate refitted AutoGluon models and generate a leaderboard.

    This component reads pre-computed metrics from a combined models artifact
    produced by ``autogluon_models_training`` and generates an HTML-formatted
    leaderboard ranking the models by their performance metric.

    The artifact layout expected under ``models_artifact.path``::

        models_artifact.path /
          <model_name>_FULL /
            metrics / metrics.json
            predictor / predictor.pkl
            notebooks / automl_predictor_notebook.ipynb

    ``models_artifact.metadata["model_names"]`` must contain the list of
    refitted model display names (e.g. ``["LightGBM_BAG_L1_FULL", ...]``).

    Args:
        models_artifact: Combined Model artifact from ``autogluon_models_training``
            with ``metadata["model_names"]`` and per-model subdirectories.
        eval_metric: Metric name for ranking (e.g. ``"accuracy"``, ``"root_mean_squared_error"``);
            leaderboard sorted descending (AutoGluon uses higher-is-better convention).
        html_artifact: Output artifact for the HTML-formatted leaderboard.

    Raises:
        FileNotFoundError: If any model metrics path cannot be found.
        KeyError: If ``metadata["model_names"]`` is missing or metrics JSON lacks
            the ``eval_metric`` key.

    Example:
        from kfp import dsl
        from components.training.automl.autogluon_leaderboard_evaluation import (
            leaderboard_evaluation
        )

        @dsl.pipeline(name="model-evaluation-pipeline")
        def evaluation_pipeline(models_artifact):
            leaderboard = leaderboard_evaluation(
                models_artifact=models_artifact,
                eval_metric="root_mean_squared_error",
            )
            return leaderboard
    """  # noqa: E501
    import importlib.resources
    import json
    import logging
    from pathlib import Path

    import pandas as pd
    from kfp_components.components.training.automl.shared.leaderboard_utils import (
        _build_leaderboard_html,
        _build_leaderboard_table,
        _round_metrics,
    )

    logger = logging.getLogger(__name__)

    # Input validation
    if not isinstance(eval_metric, str) or not eval_metric.strip():
        raise TypeError("eval_metric must be a non-empty string.")
    model_names_raw = models_artifact.metadata.get("model_names", "[]")
    # KFP/MLMD serializes lists as JSON strings to avoid the {"list": [...]} round-trip bug.
    model_names = json.loads(model_names_raw) if isinstance(model_names_raw, str) else list(model_names_raw)
    if not model_names:
        raise KeyError("models_artifact.metadata must contain a non-empty 'model_names' list.")

    results = []
    base_uri = models_artifact.uri.rstrip("/")
    for display_name in model_names:
        metrics_path = Path(models_artifact.path) / display_name / "metrics" / "metrics.json"
        if not metrics_path.exists():
            logger.warning(
                "Skipping model '%s': no metrics/metrics.json found. The refit task may have failed for this model.",
                display_name,
            )
            continue
        with metrics_path.open("r") as f:
            eval_results = json.load(f)
        model_uri = f"{base_uri}/{display_name}"
        predictor_uri = f"{model_uri}/predictor"
        notebook_uri = f"{model_uri}/notebooks/automl_predictor_notebook.ipynb"
        results.append(
            {
                "model": display_name,
                **_round_metrics(eval_results),
                "notebook": notebook_uri,
                "predictor": predictor_uri,
            }
        )

    if not results:
        raise ValueError(
            "No valid model artifacts found. All models may have failed or produced no metrics/metrics.json output."
        )

    # Notice: AutoGluon follows the "higher is better" strategy for all metrics.
    # This means that some metrics—like log_loss and root_mean_squared_error—will have their signs FLIPPED and are shown as negative. # noqa: E501
    # This is to ensure that a higher value always means a better model, so users do not need to know about the metric's normal directionality when interpreting the leaderboard. # noqa: E501
    leaderboard_df = pd.DataFrame(results).sort_values(by=eval_metric, ascending=False)
    n = len(leaderboard_df)
    leaderboard_df.index = pd.RangeIndex(start=1, stop=n + 1, name="rank")

    html_table = _build_leaderboard_table(leaderboard_df)

    best_model_name = leaderboard_df.iloc[0]["model"]
    _template_ref = (
        importlib.resources.files("kfp_components.components.training.automl.shared") / "leaderboard_html_template.html"
    )
    with importlib.resources.as_file(_template_ref) as template_path:
        html_content = _build_leaderboard_html(
            template_path=template_path,
            table_html=html_table,
            eval_metric=eval_metric,
            best_model_name=best_model_name,
            num_models=len(leaderboard_df),
        )
    with open(html_artifact.path, "w", encoding="utf-8") as f:
        f.write(html_content)

    html_artifact.metadata["data"] = leaderboard_df.to_dict()
    html_artifact.metadata["display_name"] = "automl_leaderboard"
    return NamedTuple("outputs", best_model=str)(best_model=best_model_name)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        leaderboard_evaluation,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
