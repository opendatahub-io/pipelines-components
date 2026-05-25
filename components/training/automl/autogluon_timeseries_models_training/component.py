import pathlib
from typing import List, NamedTuple, Optional

from kfp import dsl
from kfp_components.utils.consts import AUTOML_IMAGE  # pyright: ignore[reportMissingImports]

_NOTEBOOKS_DIR = str(pathlib.Path(__file__).parent / "notebook_templates")


@dsl.component(
    base_image=AUTOML_IMAGE,  # noqa: E501
    embedded_artifact_path=_NOTEBOOKS_DIR,
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
    models_artifact: dsl.Output[dsl.Model] = None,
    notebooks: dsl.EmbeddedInput[dsl.Dataset] = None,
    sample_rows: str = "[]",
    sampling_config: Optional[dict] = None,
    split_config: Optional[dict] = None,
    extra_train_data_path: str = "",
    prediction_length: int = 1,
    known_covariates_names: Optional[List[str]] = None,
) -> NamedTuple(
    "outputs",
    top_models=List[str],
    predictor_path=str,
    eval_metric=str,
    model_config=dict,
):
    """Train, select, and full-refit top N AutoGluon timeseries models.

    This component combines model generation and full-refit into one step to avoid
    pipeline-level ``ParallelFor``. It trains multiple AutoGluon TimeSeries models on
    the selection training data, ranks them on the test set, then sequentially refits
    the top N models on full train data (selection split + extra split).

    Refit outputs for all selected models are written under one ``models_artifact`` in
    a layout compatible with ``autogluon_leaderboard_evaluation``.

    Args:
        target: Name of the target column to forecast.
        id_column: Name of the column identifying each time series (item_id).
        timestamp_column: Name of the timestamp/datetime column.
        train_data_path: Path to the selection training CSV file.
        test_data: Test dataset artifact for evaluation.
        top_n: Number of top models to select for full refit.
        workspace_path: Workspace directory where predictor will be saved.
        models_artifact: Combined output artifact containing all refitted models.
        notebooks: Embedded notebook templates.
        pipeline_name: Pipeline name used in generated notebook placeholders.
        run_id: Pipeline run id used in generated notebook placeholders.
        sample_rows: Optional sample rows JSON string used in generated notebook placeholders.
        sampling_config: Optional sampling config stored in artifact metadata.
        split_config: Optional split config stored in artifact metadata.
        extra_train_data_path: Path to extra train split for full refit.
        prediction_length: Forecast horizon (number of timesteps).
        known_covariates_names: Optional list of known covariate column names.

    Returns:
        NamedTuple: top_models list, predictor_path, eval_metric, model_config.
    """
    import json
    import logging
    import math
    import os
    from pathlib import Path

    import pandas as pd
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    logger = logging.getLogger(__name__)

    # Set constants
    DEFAULT_PRESETS = "fast_training"
    DEFAULT_EVAL_METRIC = "MASE"
    DEFAULT_TIME_LIMIT = 600  # 10 minutes

    TOP_N_MAX = 7

    # Input validation
    for param, value in (
        ("target", target),
        ("id_column", id_column),
        ("timestamp_column", timestamp_column),
        ("train_data_path", train_data_path),
        ("workspace_path", workspace_path),
    ):
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"{param} must be a non-empty string.")
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
    ):
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"{param} must be a non-empty string.")
    if not isinstance(sample_rows, str):
        raise TypeError("sample_rows must be a string.")
    if models_artifact is not None and not hasattr(models_artifact, "path"):
        raise TypeError("models_artifact must be a KFP output artifact or None.")
    if notebooks is not None and not hasattr(notebooks, "path"):
        raise TypeError("notebooks must be a KFP embedded artifact or None.")
    if sampling_config is not None and not isinstance(sampling_config, dict):
        raise TypeError("sampling_config must be a dictionary or None.")
    if split_config is not None and not isinstance(split_config, dict):
        raise TypeError("split_config must be a dictionary or None.")
    if not isinstance(extra_train_data_path, str):
        raise TypeError("extra_train_data_path must be a string.")
    sampling_config = sampling_config or {}
    split_config = split_config or {}

    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data.path)
    logger.info("Loaded train=%s test=%s rows", len(train_df), len(test_df))

    train_ts = TimeSeriesDataFrame.from_data_frame(
        train_df,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )
    logger.info("Train TimeSeriesDataFrame: %s rows, %s items", len(train_ts), train_ts.num_items)

    # Convert test data to TimeSeriesDataFrame
    test_ts = TimeSeriesDataFrame.from_data_frame(
        test_df,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )

    # Create predictor path in workspace
    predictor_path = Path(workspace_path) / "timeseries_predictor"

    eval_metric = DEFAULT_EVAL_METRIC
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
        DEFAULT_PRESETS,
        DEFAULT_TIME_LIMIT,
        prediction_length,
    )
    try:
        predictor.fit(
            train_data=train_ts,
            presets=DEFAULT_PRESETS,
            time_limit=DEFAULT_TIME_LIMIT,
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
        "presets": DEFAULT_PRESETS,
        "time_limit": DEFAULT_TIME_LIMIT,
        "known_covariates_names": known_covariates_names or [],
        "num_models_trained": len(leaderboard),
    }

    # Stage 2: Full refit of selected models on full train data (selection + extra).
    if models_artifact is None or notebooks is None or not extra_train_data_path.strip():
        logger.info(
            "Skipping combined full-refit stage; missing models_artifact/notebooks or extra_train_data_path. "
            "Returning selection-only outputs for backward compatibility."
        )
        outputs = NamedTuple(
            "outputs",
            top_models=List[str],
            predictor_path=str,
            eval_metric=str,
            model_config=dict,
        )
        return outputs(
            top_models=top_models,
            predictor_path=str(predictor_path),
            eval_metric=eval_metric,
            model_config=model_config,
        )
    from autogluon.timeseries.metrics import AVAILABLE_METRICS
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

    def replace_placeholder_in_notebook(notebook, replacements):
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            new_source = []
            for line in cell.get("source", []):
                for placeholder, value in replacements.items():
                    line = line.replace(placeholder, value)
                new_source.append(line)
            cell["source"] = new_source
        return notebook

    for model_name in top_models:
        model_name_full = f"{model_name}_FULL"
        model_names_full.append(model_name_full)
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
        predictor_refit.fit(
            train_data=full_train_ts_df,
            **additional_fit_params,
            time_limit=DEFAULT_TIME_LIMIT,
            excluded_model_types=["Chronos", "Chronos2", "Toto"],
        )
        predictor_refit.save()
        metrics = predictor_refit.evaluate(test_ts, metrics=list(AVAILABLE_METRICS.keys()))

        metrics_dict = {
            k: float(v) if hasattr(v, "item") else v
            for k, v in metrics.items()
            if not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
        }

        metrics_path = output_path / "metrics"
        metrics_path.mkdir(parents=True, exist_ok=True)
        with (metrics_path / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2)

        notebook_file = "timeseries_notebook.ipynb"
        with open(os.path.join(notebooks.path, notebook_file), "r", encoding="utf-8") as f:
            notebook = json.load(f)
        replacements = {
            "<REPLACE_RUN_ID>": run_id,
            "<REPLACE_PIPELINE_NAME>": pipeline_name_trimmed,
            "<REPLACE_MODEL_NAME>": model_name_full,
            "<REPLACE_SAMPLE_ROW>": str(sample_row_list),
            "<REPLACE_ID_COLUMN>": id_column,
            "<REPLACE_TIMESTAMP_COLUMN>": timestamp_column,
            "<REPLACE_KNOWN_COVARIATES_NAMES>": str(known_covariates_names or []),
        }
        notebook = replace_placeholder_in_notebook(notebook, replacements)

        notebook_path = output_path / "notebooks"
        notebook_path.mkdir(parents=True, exist_ok=True)
        with (notebook_path / "automl_predictor_notebook.ipynb").open("w", encoding="utf-8") as f:
            json.dump(notebook, f)

        model_metadata = {
            "name": model_name_full,
            "location": {
                "model_directory": model_name_full,
                "predictor": str(Path(model_name_full) / "predictor"),
                "notebook": str(Path(model_name_full) / "notebooks" / "automl_predictor_notebook.ipynb"),
                "metrics": str(Path(model_name_full) / "metrics"),
            },
            "metrics": {
                "test_data": metrics_dict,
            },
        }
        models_metadata.append(model_metadata)
        with (output_path / "model.json").open("w", encoding="utf-8") as f:
            json.dump(model_metadata, f, indent=2)

    models_artifact.metadata["model_names"] = json.dumps(model_names_full)
    models_artifact.metadata["context"] = {
        "data_config": {
            "sampling_config": sampling_config,
            "split_config": split_config,
        },
        "model_config": model_config,
        "models": models_metadata,
    }

    outputs = NamedTuple(
        "outputs",
        top_models=List[str],
        predictor_path=str,
        eval_metric=str,
        model_config=dict,
    )
    return outputs(
        top_models=top_models,
        predictor_path=str(predictor_path),
        eval_metric=eval_metric,
        model_config=model_config,
    )


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        autogluon_timeseries_models_training,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
