import pathlib

from kfp import dsl
from kfp_components.utils.consts import AUTOML_IMAGE  # pyright: ignore[reportMissingImports]

_NOTEBOOKS_DIR = str(pathlib.Path(__file__).parent / "notebook_templates")


@dsl.component(
    base_image=AUTOML_IMAGE,  # noqa: E501
    embedded_artifact_path=_NOTEBOOKS_DIR,
)
def autogluon_timeseries_models_full_refit(
    model_name: str,
    test_dataset: dsl.Input[dsl.Dataset],
    predictor_path: str,
    sampling_config: dict,
    split_config: dict,
    model_config: dict,
    pipeline_name: str,
    run_id: str,
    models_selection_train_data_path: str,
    extra_train_data_path: str,
    sample_rows: str,
    notebooks: dsl.EmbeddedInput[dsl.Dataset],
    model_artifact: dsl.Output[dsl.Model],
):
    """Refit a single AutoGluon timeseries model on full training data.

    This component takes a model selected during the selection phase and
    refits it on the full training dataset (selection + extra train data)
    for improved performance. The refitted model is optimized and saved
    for deployment. Each model directory contains a ``model.json`` file
    with model metadata (name, location, metrics).

    Args:
        model_name: Name of the model to refit.
        test_dataset: Test dataset artifact for evaluation.
        predictor_path: Path to the predictor from selection phase.
        sampling_config: Configuration used for data sampling.
        split_config: Configuration used for data splitting.
        model_config: Model configuration from selection phase.
        pipeline_name: Pipeline name for metadata.
        run_id: Pipeline run ID for metadata.
        models_selection_train_data_path: Path to the model-selection train split CSV
            (earlier segment of the train portion).
        extra_train_data_path: Path to the extra train split CSV (later segment of the train portion).
        sample_rows: Sample rows from test dataset as JSON string.
        model_artifact: Output artifact for the refitted model.
        notebooks: Embedded notebook templates (injected by the runtime from the component's embedded_artifact_path).
    """
    import json
    import logging
    import math
    import os
    from pathlib import Path

    import pandas as pd
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    from autogluon.timeseries.metrics import AVAILABLE_METRICS
    from autogluon.timeseries.models.ensemble import AbstractTimeSeriesEnsembleModel

    logger = logging.getLogger(__name__)

    logger.info("Timeseries refit: model=%s", model_name)

    # Load the predictor from selection phase
    try:
        predictor = TimeSeriesPredictor.load(predictor_path)
    except Exception as e:
        logger.error("Failed to load predictor: %s", e)
        raise ValueError(f"Could not load predictor from {predictor_path}: {e}") from e
    logger.debug(
        "Loaded selection predictor from %s; selection_train=%s extra_train=%s",
        predictor_path,
        models_selection_train_data_path,
        extra_train_data_path,
    )

    id_column = model_config.get("id_column")
    timestamp_column = model_config.get("timestamp_column")

    selection_ts_df = TimeSeriesDataFrame.from_path(
        path=models_selection_train_data_path,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )
    extra_train_ts_df = TimeSeriesDataFrame.from_path(
        path=extra_train_data_path,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )
    # Full train portion (excluding held-out test) = selection split then extra split, in time order.
    full_train_ts_df = TimeSeriesDataFrame(
        pd.concat([selection_ts_df, extra_train_ts_df], axis=0),
    )
    test_ts_df = TimeSeriesDataFrame.from_path(
        path=test_dataset.path,
        id_column=id_column,
        timestamp_column=timestamp_column,
    )
    logger.debug(
        "Train rows=%s (selection=%s extra=%s) test rows=%s (id=%s ts=%s)",
        len(full_train_ts_df),
        len(selection_ts_df),
        len(extra_train_ts_df),
        len(test_ts_df),
        id_column,
        timestamp_column,
    )

    # Create model output directory
    model_name_full = f"{model_name}_FULL"
    output_path = Path(model_artifact.path) / model_name_full
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the predictor with the selected model
    predictor_output = output_path / "predictor"

    def is_ensemble_model(predictor, model_name: str) -> bool:
        """Return True if the named model is an ensemble."""
        model_type = predictor._trainer.get_model_attribute(model_name, "type")
        return issubclass(model_type, AbstractTimeSeriesEnsembleModel)

    is_ensemble = is_ensemble_model(predictor, model_name)

    predictor_refit = TimeSeriesPredictor(
        prediction_length=model_config.get("prediction_length"),  # 7 days
        path=predictor_output,
        target=model_config.get("target"),
        eval_metric=model_config.get("eval_metric", "MASE"),
    )

    hyperparams_option = "ensemble_hyperparameters" if is_ensemble else "hyperparameters"
    # TODO: Save model_hyperparams in the output of the previous step & remove the predictor usage here
    additional_fit_params = {hyperparams_option: {model_name: predictor.fit_summary()["model_hyperparams"][model_name]}}
    logger.debug("Refit hyperparameters: %s", additional_fit_params)

    predictor_refit.fit(
        train_data=full_train_ts_df,
        **additional_fit_params,
        # exclude deep learning models pretrained on large time series datasets
        excluded_model_types=[
            "Chronos",
            "Chronos2",
            "Toto",
        ],
    )

    try:
        predictor_refit.save()
    except Exception as e:
        logger.error("Failed to save predictor: %s", e)
        raise ValueError(f"Could not save predictor to {predictor_output}: {e}") from e

    try:
        metrics = predictor_refit.evaluate(test_ts_df, metrics=list(AVAILABLE_METRICS.keys()))
    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        raise ValueError(f"Failed to evaluate model: {e}") from e
    logger.debug("Evaluation metrics: %s", metrics)

    # Save additional metadata about the selected model
    predictor_metadata = {
        "model_name": model_name_full,
        "base_model": model_name,
        "selected_model": model_name,
        "prediction_length": model_config.get("prediction_length", 1),
        "eval_metric": model_config.get("eval_metric", "MASE"),
        "target": model_config.get("target"),
        "id_column": model_config.get("id_column"),
        "timestamp_column": model_config.get("timestamp_column"),
    }

    with open(predictor_output / "predictor_metadata.json", "w") as f:
        json.dump(predictor_metadata, f, indent=2)

    metrics_path = output_path / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)

    # Convert metrics to JSON-serializable format; drop NaN/Inf which break Protobuf Struct serialization
    metrics_dict = {
        k: float(v) if hasattr(v, "item") else v
        for k, v in metrics.items()
        if not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
    }

    with open(metrics_path / "metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # Notebook generation

    notebook_file = "timeseries_notebook.ipynb"

    with open(os.path.join(notebooks.path, notebook_file), "r", encoding="utf-8") as f:
        notebook = json.load(f)

        # Improved retrieve_pipeline_name: trims only the run id or suffix
        def retrieve_pipeline_name(pipeline_name: str) -> str:
            """Attempts to infer the original pipeline name from a name that may have a run id or suffix at the end.

            Removes only the last dash-separated element (the run id or variant),
            handling trailing dashes gracefully to avoid dropping real name segments.
            If only a single element exists, returns as is.
            """
            if not pipeline_name:
                return pipeline_name
            # Strip trailing dashes for robust splitting
            name = pipeline_name.rstrip("-")
            if "-" not in name:
                return name
            tokens = name.split("-")
            if len(tokens) <= 1:
                return tokens[0] if tokens else ""
            return "-".join(tokens[:-1])

        pipeline_name = retrieve_pipeline_name(pipeline_name)

        # Replace <REPLACE_* placeholders (run id, pipeline name, model, sample row, extra pip index, …) in code cells. # noqa: E501
        def replace_placeholder_in_notebook(notebook, replacements):
            for cell in notebook.get("cells", []):
                if cell.get("cell_type") != "code":
                    continue
                # Replace in every string of the source list
                new_source = []
                for line in cell.get("source", []):
                    for placeholder, value in replacements.items():
                        line = line.replace(placeholder, value)
                    new_source.append(line)
                cell["source"] = new_source
            return notebook

        sample_row_list = json.loads(sample_rows)

        replacements = {
            "<REPLACE_RUN_ID>": run_id,
            "<REPLACE_PIPELINE_NAME>": pipeline_name,
            "<REPLACE_MODEL_NAME>": model_name_full,
            "<REPLACE_SAMPLE_ROW>": str(sample_row_list),
            "<REPLACE_ID_COLUMN>": model_config.get("id_column"),
            "<REPLACE_TIMESTAMP_COLUMN>": model_config.get("timestamp_column"),
            "<REPLACE_KNOWN_COVARIATES_NAMES>": str(model_config.get("known_covariates_names") or []),
        }
        notebook = replace_placeholder_in_notebook(notebook, replacements)

    notebook_path = output_path / "notebooks"
    notebook_path.mkdir(parents=True, exist_ok=True)
    with (notebook_path / "automl_predictor_notebook.ipynb").open("w", encoding="utf-8") as f:
        json.dump(notebook, f)

    # Write model.json alongside predictor/, metrics/, notebooks/
    model_metadata = {
        "name": model_name_full,
        "location": {
            "model_directory": model_name_full,
            "predictor": f"{model_name_full}/predictor",
            "notebook": f"{model_name_full}/notebooks/automl_predictor_notebook.ipynb",
            "metrics": f"{model_name_full}/metrics",
        },
        "metrics": {
            "test_data": metrics_dict,
        },
    }
    with (output_path / "model.json").open("w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)

    # Set artifact metadata
    model_artifact.metadata["display_name"] = model_name_full
    model_artifact.metadata["context"] = {
        "model_config": model_config,
        "sampling_config": sampling_config,
        "split_config": split_config,
        "metrics": {"test_data": metrics_dict},
        "location": {
            "model_directory": model_name_full,
            "predictor": f"{model_name_full}/predictor",
            "notebook": f"{model_name_full}/notebooks/automl_predictor_notebook.ipynb",
            "metrics": f"{model_name_full}/metrics",
        },
        "pipeline_info": {
            "pipeline_name": pipeline_name,
            "run_id": run_id,
        },
    }

    logger.info("Timeseries refit done: %s (artifact under %s)", model_name_full, output_path)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        autogluon_timeseries_models_full_refit,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
