from typing import List, NamedTuple, Optional

from kfp import dsl
from kfp_components.utils.consts import AUTOML_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=AUTOML_IMAGE,  # noqa: E501
)
def autogluon_timeseries_models_selection(
    target: str,
    id_column: str,
    timestamp_column: str,
    train_data_path: str,
    test_data: dsl.Input[dsl.Dataset],
    top_n: int,
    workspace_path: str,
    prediction_length: int = 1,
    known_covariates_names: Optional[List[str]] = None,
) -> NamedTuple(
    "outputs",
    top_models=List[str],
    predictor_path=str,
    eval_metric_name=str,
    model_config=dict,
):
    """Train and select top N AutoGluon timeseries models based on leaderboard.

    This component trains multiple AutoGluon TimeSeries models using TimeSeriesPredictor
    on the selection training data, evaluates them on the test set, and selects the
    top N performers based on the leaderboard ranking. Training uses the ``fast_training``
    preset for shorter wall-clock time versus ``medium_quality`` (trade-off: accuracy).

    The TimeSeriesPredictor automatically trains various model types (DeepAR, TFT,
    ARIMA, ETS, Theta, etc.) and ranks them by the evaluation metric. This component
    selects the top N models from the leaderboard for refitting on the full dataset.

    Args:
        target: Name of the target column to forecast.
        id_column: Name of the column identifying each time series (item_id).
        timestamp_column: Name of the timestamp/datetime column.
        train_data_path: Path to the selection training CSV file.
        test_data: Test dataset artifact for evaluation.
        top_n: Number of top models to select for refitting.
        workspace_path: Workspace directory where predictor will be saved.
        prediction_length: Forecast horizon (number of timesteps).
        known_covariates_names: Optional list of known covariate column names.

    Returns:
        NamedTuple: top_models list, predictor_path, eval_metric_name, model_config.
    """
    import logging
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

    outputs = NamedTuple(
        "outputs",
        top_models=List[str],
        predictor_path=str,
        eval_metric_name=str,
        model_config=dict,
    )
    return outputs(
        top_models=top_models,
        predictor_path=str(predictor_path),
        eval_metric_name=eval_metric,
        model_config=model_config,
    )


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        autogluon_timeseries_models_selection,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
