from typing import List, NamedTuple, Optional

from kfp import dsl
from kfp_components.utils.consts import AUTOML_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=AUTOML_IMAGE,  # noqa: E501
    install_kfp_package=False,
)
def autogluon_timeseries_models_training(
    target: str,
    id_column: str,
    timestamp_column: str,
    train_data_path: str,
    test_data: dsl.Input[dsl.Dataset],
    top_n: int,
    workspace_path: str,
    pipeline_name: str,
    run_id: str,
    models_artifact: dsl.Output[dsl.Model],
    extra_train_data_path: str,
    html_artifact: dsl.Output[dsl.HTML],
    component_status: dsl.Output[dsl.Artifact],
    sample_rows: str = "[]",
    sampling_config: Optional[dict] = None,
    split_config: Optional[dict] = None,
    prediction_length: int = 1,
    known_covariates_names: Optional[List[str]] = None,
    preset: str = "speed",
    eval_metric: str = "mean_absolute_scaled_error",
    evaluation_mode: str = "auto-split",
) -> NamedTuple(
    "outputs",
    top_models=List[str],
    predictor_path=str,
    eval_metric=str,
    model_config=dict,
    best_model_name=str,
):
    """Train, select, and full-refit top N AutoGluon timeseries models.

    This component combines model generation and full-refit into one step to avoid
    pipeline-level ``ParallelFor``. It trains multiple AutoGluon TimeSeries models on
    the selection training data, ranks them on the test set, then sequentially refits
    the top N models on full train data (selection split + extra split).

    Refit outputs for all selected models are written under one ``models_artifact``,
    and a ranked HTML leaderboard is written to ``html_artifact``.

    Args:
        target: Name of the target column to forecast.
        id_column: Name of the column identifying each time series (item_id).
        timestamp_column: Name of the timestamp/datetime column.
        train_data_path: Path to the selection training CSV file.
        test_data: Test dataset artifact for evaluation.
        top_n: Number of top models to select for full refit.
        workspace_path: Workspace directory where predictor will be saved.
        pipeline_name: Pipeline name used in generated notebook placeholders.
        run_id: Pipeline run id used in generated notebook placeholders.
        models_artifact: Combined output artifact containing all refitted models.
        extra_train_data_path: Path to extra train split for full refit.
        html_artifact: Output HTML artifact containing the ranked leaderboard page.
        sample_rows: Sample rows JSON string used in generated notebook placeholders.
        sampling_config: Optional sampling config stored in artifact metadata.
        split_config: Optional split config stored in artifact metadata.
        prediction_length: Forecast horizon (number of timesteps).
        known_covariates_names: Optional list of known covariate column names.
        component_status: Output artifact containing stage-level progress tracking for this component.
        preset: Training quality tier. ``"speed"`` (default) or ``"balanced"``
            (may run more than 2x longer).
        eval_metric: Metric for model ranking (e.g. ``"mean_absolute_scaled_error"``,
            ``"weighted_quantile_loss"``). Defaults to ``"mean_absolute_scaled_error"``.
            Legacy uppercase acronyms (e.g. ``"MASE"``) are accepted and normalized to snake_case.
        evaluation_mode: Evaluation mode from data loader ("user-provided" or "auto-split").
            "user-provided" indicates test data was provided by the user, "auto-split" indicates
            test data was created by automatically splitting training data. Used by downstream
            pipeline components for decision making. Defaults to "auto-split".

    Returns:
        NamedTuple: top_models list, predictor_path, eval_metric, model_config.
    """
    import json
    import logging
    import math
    from pathlib import Path

    import pandas as pd
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    from autogluon.timeseries.metrics import AVAILABLE_METRICS, METRIC_ALIASES
    from kfp_components.components.training.automl.shared.component_status import ComponentStatusTracker
    from kfp_components.components.training.automl.shared.run_status import shared_automl_dir

    logger = logging.getLogger(__name__)

    from kfp_components.components.training.automl.shared.back_testing import build_back_testing_json
    from kfp_components.components.training.automl.shared.back_testing_charts import (
        notebook_backtest_charts_source,
    )
    from kfp_components.components.training.automl.shared.timeseries_notebook_utils import (
        build_predict_sample_artifact,
        notebook_timeseries_sample_helpers_source,
    )

    status = ComponentStatusTracker(component_status.path, "autogluon_timeseries_models_training")
    with status:
        status.set_metadata(display_name="Timeseries Models Training Status")
        component_status.metadata["display_name"] = "Timeseries Models Training Status"
        TOP_N_MAX = 7
        VALID_PRESETS = {"speed", "balanced"}
        PRESET_AG_NAMES = {"speed": "fast_training", "balanced": "medium_quality"}
        PRESET_TIME_LIMITS = {"speed": 10 * 60, "balanced": 60 * 60}
        VALID_EVALUATION_MODES = {"user-provided", "auto-split"}

        # Normalize eval_metric to snake_case; accept legacy uppercase acronyms (e.g. "MASE") for back-compat.
        _acronym_to_snake = {acronym: snake for snake, acronym in METRIC_ALIASES.items()}
        eval_metric = _acronym_to_snake.get(eval_metric, eval_metric)

        # Input validation
        if preset not in VALID_PRESETS:
            raise ValueError(f"preset must be one of {VALID_PRESETS}; got {preset!r}.")
        if evaluation_mode not in VALID_EVALUATION_MODES:
            raise ValueError(f"evaluation_mode must be one of {VALID_EVALUATION_MODES}; got {evaluation_mode!r}.")
        for param, value in (
            ("target", target),
            ("id_column", id_column),
            ("timestamp_column", timestamp_column),
            ("train_data_path", train_data_path),
            ("workspace_path", workspace_path),
            ("eval_metric", eval_metric),
        ):
            if not isinstance(value, str) or not value.strip():
                raise TypeError(f"{param} must be a non-empty string.")
        if eval_metric not in METRIC_ALIASES:
            raise ValueError(f"eval_metric must be one of {sorted(METRIC_ALIASES)}; got {eval_metric!r}.")
        if not isinstance(top_n, int):
            raise TypeError("top_n must be an integer.")
        if top_n <= 0 or top_n > TOP_N_MAX:
            raise ValueError(f"top_n must be an integer in the range (0, {TOP_N_MAX}]; got {top_n}.")
        if not isinstance(prediction_length, int):
            raise TypeError("prediction_length must be an integer.")
        if prediction_length <= 0:
            raise ValueError("prediction_length must be greater than 0.")
        if known_covariates_names is not None:
            if not isinstance(known_covariates_names, list) or any(
                (not isinstance(v, str) or not v.strip()) for v in known_covariates_names
            ):
                raise TypeError("known_covariates_names must be a list of non-empty strings or None.")
        for param, value in (
            ("pipeline_name", pipeline_name),
            ("run_id", run_id),
            ("extra_train_data_path", extra_train_data_path),
        ):
            if not isinstance(value, str) or not value.strip():
                raise TypeError(f"{param} must be a non-empty string.")
        if not isinstance(sample_rows, str):
            raise TypeError("sample_rows must be a string.")
        if models_artifact is not None and not hasattr(models_artifact, "path"):
            raise TypeError("models_artifact must be a KFP output artifact or None.")
        if not hasattr(component_status, "path"):
            raise TypeError("component_status must be a KFP output artifact.")
        if not hasattr(models_artifact, "path"):
            raise TypeError("models_artifact must be a KFP output artifact.")
        if sampling_config is not None and not isinstance(sampling_config, dict):
            raise TypeError("sampling_config must be a dictionary or None.")
        if split_config is not None and not isinstance(split_config, dict):
            raise TypeError("split_config must be a dictionary or None.")
        sampling_config = sampling_config or {}
        split_config = split_config or {}
        time_limit = PRESET_TIME_LIMITS[preset]

        status.record("load_data", "started")
        train_df = pd.read_csv(train_data_path)
        test_df = pd.read_csv(test_data.path)
        logger.info("Loaded train=%s test=%s rows", len(train_df), len(test_df))
        # pd.read_csv returns the timestamp column as object (string) dtype.
        # TimeSeriesDataFrame.from_data_frame requires datetime64; convert explicitly.
        # utc=True handles both tz-naive and tz-aware (e.g. ISO 8601 with Z suffix) strings;
        # tz_localize(None) then strips tz info to produce tz-naive datetime64[ns].
        train_df[timestamp_column] = pd.to_datetime(train_df[timestamp_column], utc=True).dt.tz_localize(None)
        test_df[timestamp_column] = pd.to_datetime(test_df[timestamp_column], utc=True).dt.tz_localize(None)

        train_ts = TimeSeriesDataFrame.from_data_frame(
            train_df,
            id_column=id_column,
            timestamp_column=timestamp_column,
        )
        test_ts = TimeSeriesDataFrame.from_data_frame(
            test_df,
            id_column=id_column,
            timestamp_column=timestamp_column,
        )
        logger.info(
            "Train TimeSeriesDataFrame: %s rows, %s items; test: %s rows, %s items",
            len(train_ts),
            train_ts.num_items,
            len(test_ts),
            test_ts.num_items,
        )
        status.record(
            "load_data",
            "completed",
            train_rows=len(train_ts),
            test_rows=len(test_ts),
        )

        # Create predictor path in workspace
        predictor_path = Path(workspace_path) / "timeseries_predictor"

        # Create TimeSeriesPredictor
        predictor = TimeSeriesPredictor(
            prediction_length=prediction_length,
            target=target,
            eval_metric=eval_metric,
            path=str(predictor_path),
            verbosity=2,
            known_covariates_names=known_covariates_names,
        )

        logger.info(
            "Timeseries selection: training (preset=%s, time_limit=%ss, prediction_length=%s)...",
            preset,
            time_limit,
            prediction_length,
        )
        status.record("model_selection", "started")
        try:
            predictor.fit(
                train_data=train_ts,
                presets=PRESET_AG_NAMES[preset],
                time_limit=time_limit,
                # exclude deep learning models pretrained on large time series datasets
                excluded_model_types=["Chronos", "Toto", "Chronos2"],
            )
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise ValueError(f"TimeSeriesPredictor training failed: {str(e)}") from e

        try:
            leaderboard = predictor.leaderboard(test_ts)
        except Exception as e:
            logger.error(f"Leaderboard generation failed: {str(e)}")
            raise ValueError(f"Failed to generate leaderboard: {str(e)}") from e

        if top_n > len(leaderboard):
            raise ValueError(
                f"top_n must be less than or equal to number_of_models_trained ({len(leaderboard)}); got {top_n}."
            )

        top_models = leaderboard.head(top_n)["model"].values.tolist()
        status.record(
            "model_selection",
            "completed",
            top_n=top_n,
            selected_models=top_models,
            steps=["feature_engineering", "model_training", "stacking", "evaluation"],
        )
        logger.info(
            "Timeseries selection done: top_%s=%s best_score_test=%s",
            top_n,
            top_models,
            leaderboard.iloc[0]["score_test"],
        )

        # Create model config
        model_config = {
            "prediction_length": prediction_length,
            "eval_metric": eval_metric,
            "target": target,
            "id_column": id_column,
            "timestamp_column": timestamp_column,
            "presets": preset,
            "time_limit": time_limit,
            "known_covariates_names": known_covariates_names or [],
            "num_models_trained": len(leaderboard),
        }

        # Stage 2: Full refit of selected models on full train data (selection + extra).
        from autogluon.timeseries.models.ensemble import AbstractTimeSeriesEnsembleModel

        selection_ts_df = train_ts
        extra_train_ts_df = TimeSeriesDataFrame.from_path(
            path=extra_train_data_path,
            id_column=id_column,
            timestamp_column=timestamp_column,
        )
        full_train_ts_df = TimeSeriesDataFrame(pd.concat([selection_ts_df, extra_train_ts_df], axis=0))
        model_hyperparams = predictor.fit_summary()["model_hyperparams"]

        sample_row_list = json.loads(sample_rows or "[]")
        if not isinstance(sample_row_list, list):
            raise ValueError("sample_rows must be a JSON list.")

        def retrieve_pipeline_name(name: str) -> str:
            if not name:
                return name
            trimmed_name = name.rstrip("-")
            if "-" not in trimmed_name:
                return trimmed_name
            tokens = trimmed_name.split("-")
            return "-".join(tokens[:-1]) if len(tokens) > 1 else tokens[0]

        pipeline_name_trimmed = retrieve_pipeline_name(pipeline_name)
        model_names_full = []
        models_metadata = []
        failed_models = []

        status.record("refit_and_evaluate", "started")

        def replace_placeholder_in_notebook(notebook, replacements):
            for cell in notebook.get("cells", []):
                if cell.get("cell_type") != "code":
                    continue
                new_source = []
                src = cell.get("source", [])
                if isinstance(src, str):
                    src = [src]
                for line in src:
                    for placeholder, value in replacements.items():
                        line = line.replace(placeholder, value)
                    new_source.append(line)
                cell["source"] = new_source
            return notebook

        for model_name in top_models:
            try:
                model_name_full = f"{model_name}_FULL"
                output_path = Path(models_artifact.path) / model_name_full
                output_path.mkdir(parents=True, exist_ok=True)
                predictor_output = output_path / "predictor"

                model_type = predictor._trainer.get_model_attribute(model_name, "type")
                is_ensemble = issubclass(model_type, AbstractTimeSeriesEnsembleModel)
                hyperparams_option = "ensemble_hyperparameters" if is_ensemble else "hyperparameters"
                additional_fit_params = {hyperparams_option: {model_name: model_hyperparams[model_name]}}

                predictor_refit = TimeSeriesPredictor(
                    prediction_length=prediction_length,
                    target=target,
                    eval_metric=eval_metric,
                    path=predictor_output,
                    verbosity=2,
                    known_covariates_names=known_covariates_names,
                )
                # fit() automatically saves the predictor to path specified in constructor
                predictor_refit.fit(
                    train_data=full_train_ts_df,
                    **additional_fit_params,
                    time_limit=time_limit,
                    excluded_model_types=["Chronos", "Chronos2", "Toto"],
                )
                metrics = predictor_refit.evaluate(test_ts, metrics=list(AVAILABLE_METRICS.keys()))
                # Keep raw AutoGluon evaluate() signs for metrics.json (higher-is-better / negated errors)
                # so Phase C leaderboard sorting (ascending=False) stays correct.
                # back_testing.json normalizes separately.
                # Remap uppercase acronym keys (e.g. "MASE") to snake_case (e.g. "mean_absolute_scaled_error").
                metrics_dict = {}
                for k, v in metrics.items():
                    if hasattr(v, "item"):
                        v = float(v)
                    if isinstance(v, (int, float)) and math.isfinite(v):
                        normalized_key = _acronym_to_snake.get(k)
                        if normalized_key is None:
                            if k not in AVAILABLE_METRICS:
                                logger.warning("Unrecognized metric key %r from evaluate(); storing as-is.", k)
                            normalized_key = k
                        metrics_dict[normalized_key] = v

                if eval_metric not in metrics_dict:
                    raise ValueError(
                        f"Eval metric '{eval_metric}' was NaN/Inf for model '{model_name_full}' "
                        "and was dropped from metrics. Cannot produce a valid leaderboard entry."
                    )

                # Save additional metadata about the selected model
                predictor_metadata = {
                    "model_name": model_name_full,
                    "base_model": model_name,
                    "prediction_length": prediction_length,
                    "eval_metric": eval_metric,
                    "target": target,
                    "id_column": id_column,
                    "timestamp_column": timestamp_column,
                    "known_covariates_names": known_covariates_names or [],
                }
                predictor_output.mkdir(parents=True, exist_ok=True)
                with (predictor_output / "predictor_metadata.json").open("w", encoding="utf-8") as f:
                    json.dump(predictor_metadata, f, indent=2)

                metrics_path = output_path / "metrics"
                metrics_path.mkdir(parents=True, exist_ok=True)
                with (metrics_path / "metrics.json").open("w", encoding="utf-8") as f:
                    json.dump(metrics_dict, f, indent=2)

                back_testing_available = False
                try:
                    back_testing_payload = build_back_testing_json(
                        predictor_refit,
                        model_name=model_name,
                        model_name_full=model_name_full,
                        train_data=full_train_ts_df,
                        eval_metric=eval_metric,
                        target=target,
                        id_column=id_column,
                        timestamp_column=timestamp_column,
                        prediction_length=prediction_length,
                        metrics=list(AVAILABLE_METRICS.keys()),
                    )
                    with (metrics_path / "back_testing.json").open("w", encoding="utf-8") as f:
                        json.dump(back_testing_payload, f, indent=2)
                    back_testing_available = True
                except Exception as backtest_exc:
                    logger.warning(
                        "Could not generate back_testing.json for model %r: %s. Skipping backtest artifact.",
                        model_name_full,
                        backtest_exc,
                    )

                notebook_file = "timeseries_notebook.ipynb"
                with (shared_automl_dir() / "notebook_templates" / notebook_file).open("r", encoding="utf-8") as f:
                    notebook = json.load(f)
                predict_sample = build_predict_sample_artifact(
                    predictor_refit,
                    sample_row_list,
                    id_column,
                    timestamp_column,
                    known_covariates_names,
                )
                replacements = {
                    "<REPLACE_RUN_ID>": run_id,
                    "<REPLACE_PIPELINE_NAME>": pipeline_name_trimmed,
                    "<REPLACE_MODEL_NAME>": model_name_full,
                    "<REPLACE_PREDICT_SAMPLE>": str(predict_sample),
                    "<REPLACE_BACKTEST_PLOT_HELPERS>": notebook_backtest_charts_source(),
                    "<REPLACE_TIMESERIES_SAMPLE_HELPERS>": notebook_timeseries_sample_helpers_source(),
                }
                notebook = replace_placeholder_in_notebook(notebook, replacements)

                notebook_path = output_path / "notebooks"
                notebook_path.mkdir(parents=True, exist_ok=True)
                with (notebook_path / "automl_predictor_notebook.ipynb").open("w", encoding="utf-8") as f:
                    json.dump(notebook, f)

                model_location = {
                    "model_directory": model_name_full,
                    "predictor": str(Path(model_name_full) / "predictor"),
                    "notebook": str(Path(model_name_full) / "notebooks" / "automl_predictor_notebook.ipynb"),
                    "metrics": str(Path(model_name_full) / "metrics"),
                }
                # Only include back_testing path if file was successfully written
                if back_testing_available:
                    model_location["back_testing"] = str(Path(model_name_full) / "metrics" / "back_testing.json")

                model_metadata = {
                    "name": model_name_full,
                    "location": model_location,
                    "metrics": {
                        "test_data": metrics_dict,
                    },
                }
                with (output_path / "model.json").open("w", encoding="utf-8") as f:
                    json.dump(model_metadata, f, indent=2)

                # Only append to successful models lists after all operations succeed
                model_names_full.append(model_name_full)
                models_metadata.append(model_metadata)

            except Exception as e:
                logger.error("Refit failed for model '%s': %s", model_name, e)
                failed_models.append(model_name)

        # Report partial failures
        if failed_models:
            logger.warning("The following models failed refit: %s", failed_models)

        # Ensure at least one model succeeded
        if not model_names_full:
            raise RuntimeError("All models failed refit. No artifacts written.")

        status.record(
            "refit_and_evaluate",
            "completed",
            model_count=len(model_names_full),
            eval_metric=eval_metric,
        )

        # Phase C: leaderboard generation - uses models_metadata already built in the refit loop
        status.record("build_leaderboard", "started")

        import importlib.resources

        from kfp_components.components.training.automl.shared.leaderboard_utils import (
            _build_leaderboard_html,
            _build_leaderboard_table,
        )

        eval_results_by_model = {m["name"]: m["metrics"]["test_data"] for m in models_metadata}
        base_uri = models_artifact.uri.rstrip("/")
        leaderboard_rows = []
        for model_name_full in model_names_full:
            model_uri = f"{base_uri}/{model_name_full}"
            leaderboard_rows.append(
                {
                    "model": model_name_full,
                    **eval_results_by_model[model_name_full],
                    "notebook": f"{model_uri}/notebooks/automl_predictor_notebook.ipynb",
                    "predictor": f"{model_uri}/predictor",
                }
            )

        # AutoGluon negates error metrics (e.g. mean_absolute_scaled_error -> negative)
        # so descending = best model first.
        # Sort on raw (unrounded) values so close scores cannot flip rank after rounding.
        if not leaderboard_rows:
            raise RuntimeError("Leaderboard rows are empty; no models available for ranking.")
        leaderboard_df = pd.DataFrame(leaderboard_rows)
        if eval_metric in leaderboard_rows[0]:
            leaderboard_df = leaderboard_df.sort_values(by=eval_metric, ascending=False, na_position="last")
        else:
            logger.warning(
                "eval_metric '%s' not found in row keys %s; preserving refit order.",
                eval_metric,
                list(leaderboard_rows[0].keys()),
            )
        n = len(leaderboard_df)
        best_model_name = str(leaderboard_df.iloc[0]["model"])
        leaderboard_df.index = pd.RangeIndex(start=1, stop=n + 1, name="rank")
        _metric_cols = [c for c in leaderboard_df.columns if c not in ("model", "notebook", "predictor")]
        leaderboard_df[_metric_cols] = leaderboard_df[_metric_cols].round(4)
        html_table = _build_leaderboard_table(leaderboard_df)

        _template_ref = (
            importlib.resources.files("kfp_components.components.training.automl.shared")
            / "leaderboard_html_template.html"
        )
        with importlib.resources.as_file(_template_ref) as template_path:
            html_content = _build_leaderboard_html(
                template_path=template_path,
                table_html=html_table,
                eval_metric=eval_metric,
                best_model_name=best_model_name,
                num_models=n,
            )
        with open(html_artifact.path, "w", encoding="utf-8") as f:
            f.write(html_content)

        html_artifact.metadata["data"] = leaderboard_df.to_json(orient="records")
        html_artifact.metadata["display_name"] = "automl_leaderboard"
        html_artifact.metadata["evaluation_mode"] = evaluation_mode
        status.record(
            "build_leaderboard",
            "completed",
            best_model=best_model_name,
            model_count=n,
        )

        models_artifact.metadata["model_names"] = json.dumps(model_names_full)
        models_artifact.metadata["context"] = {
            "data_config": {
                "sampling_config": sampling_config,
                "split_config": split_config,
            },
            "model_config": model_config,
            "best_model_name": best_model_name,
            "models": models_metadata,
            "evaluation_mode": evaluation_mode,
        }

        outputs = NamedTuple(
            "outputs",
            top_models=List[str],
            predictor_path=str,
            eval_metric=str,
            model_config=dict,
            best_model_name=str,
        )
        return outputs(
            top_models=top_models,
            predictor_path=str(predictor_path),
            eval_metric=eval_metric,
            model_config=model_config,
            best_model_name=best_model_name,
        )


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        autogluon_timeseries_models_training,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
