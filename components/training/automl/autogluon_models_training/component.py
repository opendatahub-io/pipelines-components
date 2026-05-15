import pathlib
from typing import NamedTuple, Optional

from kfp import dsl
from kfp_components.utils.consts import AUTOML_IMAGE  # pyright: ignore[reportMissingImports]

# Reuse notebook templates from autogluon_models_full_refit
_NOTEBOOKS_DIR = str(pathlib.Path(__file__).parent / "notebook_templates")


@dsl.component(
    base_image=AUTOML_IMAGE,  # noqa: E501
    embedded_artifact_path=_NOTEBOOKS_DIR,
)
def autogluon_models_training(
    label_column: str,
    task_type: str,
    top_n: int,
    train_data_path: str,
    test_data: dsl.Input[dsl.Dataset],
    workspace_path: str,
    pipeline_name: str,
    run_id: str,
    sample_row: str,
    models_artifact: dsl.Output[dsl.Model],
    notebooks: dsl.EmbeddedInput[dsl.Dataset],
    sampling_config: Optional[dict] = None,
    split_config: Optional[dict] = None,
    extra_train_data_path: str = "",
) -> NamedTuple("outputs", eval_metric=str):
    """Train AutoGluon models, select the top N, and refit each on the full dataset.

    Expects pre-cleaned CSV data from the tabular data loader (infinite values replaced,
    duplicates removed, missing labels dropped). Reads train/test/extra-train CSVs and
    validates that the label column exists in each dataset.

    This component combines the model selection and full-refit stages into a single
    step. It trains a TabularPredictor on sampled data, ranks all models on the test
    set, then refits each of the top N models on the full training data in a single
    ``refit_full`` call. Post-refit work (predict, evaluate, feature importance,
    confusion matrix, notebook generation) runs concurrently across all top-N models
    via ``ThreadPoolExecutor``. The deployment clone (``set_model_best`` +
    ``clone_for_deployment``) is serialized afterward because it mutates predictor
    state. All artifacts are written under a single output artifact so the pipeline
    does not require a ParallelFor loop. Each model directory contains a ``model.json``
    file with model metadata (name, location, metrics).

    Args:
        label_column: Target/label column name in train and test datasets.
        task_type: ML task type: ``"binary"``, ``"multiclass"``, or ``"regression"``.
        top_n: Number of top models to select and refit (1-10).
        train_data_path: Path to the selection-train CSV on the PVC workspace.
        test_data: Dataset artifact (CSV) used for leaderboard ranking and evaluation.
        workspace_path: PVC workspace directory; predictor saved at ``workspace_path/autogluon_predictor``.
        pipeline_name: Pipeline run name; last dash-segment stripped for the notebook.
        run_id: Pipeline run ID written into the generated notebook.
        sample_row: JSON array of row dicts for the notebook example input; label column is stripped.
        models_artifact: Output Model artifact containing all refitted model subdirectories.
        notebooks: Embedded notebook templates injected by the KFP runtime.
        sampling_config: Data sampling config stored in artifact metadata.
        split_config: Data split config stored in artifact metadata.
        extra_train_data_path: Optional path to extra training CSV passed to ``refit_full``.

    Returns:
        NamedTuple with ``eval_metric`` (the metric used for ranking, e.g. ``"r2"`` or ``"accuracy"``).

    Raises:
        TypeError: If any required string parameter is empty or configs have wrong types.
        ValueError: If ``task_type`` is invalid, ``top_n`` is out of range, ``sample_row``
            is not a JSON list, ``problem_type`` is unsupported for notebook generation,
            label column not found in CSV, or train/test data is empty.
        FileNotFoundError: If train/test data or predictor paths cannot be found.
    """  # noqa: E501
    import json
    import logging
    import math
    import shutil
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    import pandas as pd
    from autogluon.tabular import TabularPredictor

    VALID_TASK_TYPES = {"binary", "multiclass", "regression"}
    TOP_N_MAX = 10

    # Input parameters validation
    if not isinstance(label_column, str) or not label_column.strip():
        raise TypeError("label_column must be a non-empty string.")
    if task_type not in VALID_TASK_TYPES:
        raise ValueError(f"task_type must be one of {VALID_TASK_TYPES}; got {task_type!r}.")
    if not train_data_path or not isinstance(train_data_path, str) or not train_data_path.strip():
        raise TypeError("train_data_path must be a non-empty string.")
    if not workspace_path or not isinstance(workspace_path, str) or not workspace_path.strip():
        raise TypeError("workspace_path must be a non-empty string.")
    if top_n <= 0 or top_n > TOP_N_MAX:
        raise ValueError(f"top_n must be an integer in the range (0, {TOP_N_MAX}]; got {top_n}.")
    for param, value in (
        ("pipeline_name", pipeline_name),
        ("run_id", run_id),
        ("sample_row", sample_row),
    ):
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"{param} must be a non-empty string.")
    if sampling_config is not None and not isinstance(sampling_config, dict):
        raise TypeError("sampling_config must be a dictionary or None.")
    if split_config is not None and not isinstance(split_config, dict):
        raise TypeError("split_config must be a dictionary or None.")

    sampling_config = sampling_config or {}
    split_config = split_config or {}

    try:
        sample_row_list = json.loads(sample_row)
    except json.JSONDecodeError as e:
        raise TypeError(f"sample_row must be valid JSON array: {e}") from e
    if not isinstance(sample_row_list, list):
        raise ValueError("sample_row must be a JSON array list of row objects.")

    logger = logging.getLogger(__name__)

    DEFAULT_PRESET = "medium_quality"
    DEFAULT_TIME_LIMIT = 30 * 60  # 30 minutes

    # 1. models selection stage

    train_data_df = pd.read_csv(train_data_path)
    if label_column not in train_data_df.columns:
        raise ValueError(
            f"Label column {label_column!r} not found in train CSV. Available columns: {list(train_data_df.columns)}"
        )
    if train_data_df.empty:
        raise ValueError("Training CSV is empty. Ensure the data loader produced valid training data.")

    test_data_df = pd.read_csv(test_data.path)
    if label_column not in test_data_df.columns:
        raise ValueError(
            f"Label column {label_column!r} not found in test CSV. Available columns: {list(test_data_df.columns)}"
        )
    if test_data_df.empty:
        raise ValueError("Test CSV is empty. Ensure the data loader produced valid test data.")

    extra_train_df = None
    if extra_train_data_path.strip():
        extra_train_df = pd.read_csv(extra_train_data_path)
        if label_column not in extra_train_df.columns:
            raise ValueError(
                f"Label column {label_column!r} not found in extra-train CSV. "
                f"Available columns: {list(extra_train_df.columns)}"
            )
        if extra_train_df.empty:
            logger.warning("Extra train CSV is empty; passing train_data_extra=None to refit_full.")
            extra_train_df = None

    eval_metric = "r2" if task_type == "regression" else "accuracy"

    predictor_path = Path(workspace_path) / "autogluon_predictor"
    predictor = TabularPredictor(
        problem_type=task_type,
        label=label_column,
        eval_metric=eval_metric,
        path=predictor_path,
        verbosity=2,
    ).fit(
        train_data=train_data_df,
        num_stack_levels=1,
        num_bag_folds=4,
        use_bag_holdout=True,
        holdout_frac=0.2,
        time_limit=DEFAULT_TIME_LIMIT,
        presets=DEFAULT_PRESET,
    )

    # Select top N models
    leaderboard = predictor.leaderboard(test_data_df)
    logger.info("Leaderboard:\n\n %s", leaderboard.head(top_n).to_string())
    top_models = leaderboard.head(top_n)["model"].values.tolist()

    model_config = {
        "preset": DEFAULT_PRESET,
        "eval_metric": eval_metric,
        "time_limit": DEFAULT_TIME_LIMIT,
    }

    def retrieve_pipeline_name(name: str) -> str:
        """Strip the last dash-separated segment (run id / suffix) from a pipeline name."""
        if not name:
            return name
        name = name.rstrip("-")
        if "-" not in name:
            return name
        tokens = name.split("-")
        return "-".join(tokens[:-1]) if len(tokens) > 1 else tokens[0]

    pipeline_name_trimmed = retrieve_pipeline_name(pipeline_name)

    # Strip label column from sample row -- same for all models
    sample_row_formatted = [
        {col: value for col, value in row.items() if col != predictor.label} for row in sample_row_list
    ]

    problem_type = predictor.problem_type
    match problem_type:
        case "regression":
            notebook_file = "regression_notebook.ipynb"
        case "binary" | "multiclass":
            notebook_file = "classification_notebook.ipynb"
        case _:
            raise ValueError(f"Invalid problem type: {problem_type}")

    model_names_full = [m + "_FULL" for m in top_models]

    # 2. models refit stage

    # Clone once to PVC (same filesystem as predictor_path) to avoid S3 FUSE file-dropping
    # during shutil.copytree inside predictor.clone().
    work_path = predictor_path.parent / "refit_work"
    predictor_clone = predictor.clone(path=work_path, return_clone=True, dirs_exist_ok=True)

    # Refit all top models in a single call:  AutoGluon resolves stacking dependencies internally.
    predictor_clone.refit_full(model=top_models, train_data_extra=extra_train_df)

    if problem_type in {"binary", "multiclass"}:
        from autogluon.core.metrics import confusion_matrix as ag_confusion_matrix

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

    def _process_model(model_name_full: str) -> tuple[str, dict]:
        """Compute metrics and write metric files + notebook for one refitted model.

        Safe to run concurrently across models: only reads from the shared predictor
        and test data, and writes to isolated per-model directories under
        models_artifact.path. Does NOT call set_model_best / clone_for_deployment,
        which mutate predictor state and must stay sequential.

        Returns (model_name_full, eval_results).
        """
        output_path = Path(models_artifact.path) / model_name_full

        predictions = predictor_clone.predict(test_data_df, model=model_name_full)
        eval_results = predictor_clone.evaluate_predictions(
            y_true=test_data_df[predictor.label],
            y_pred=predictions,
        )
        feature_importance = predictor_clone.feature_importance(
            test_data_df, model=model_name_full, subsample_size=2000
        )

        # Filter NaN/Inf which break Protobuf Struct serialization
        eval_results = {
            k: float(v) if hasattr(v, "item") else v
            for k, v in eval_results.items()
            if not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
        }

        (output_path / "metrics").mkdir(parents=True, exist_ok=True)
        with (output_path / "metrics" / "metrics.json").open("w") as f:
            json.dump(eval_results, f)
        with (output_path / "metrics" / "feature_importance.json").open("w") as f:
            json.dump(feature_importance.to_dict(), f)

        if problem_type in {"binary", "multiclass"}:
            confusion_matrix_res = ag_confusion_matrix(
                solution=test_data_df[predictor.label],
                prediction=predictions,
                output_format="pandas_dataframe",
            )
            with (output_path / "metrics" / "confusion_matrix.json").open("w") as f:
                json.dump(confusion_matrix_res.to_dict(), f)

        with (Path(notebooks.path) / notebook_file).open("r", encoding="utf-8") as f:
            notebook = json.load(f)
        replacements = {
            "<REPLACE_RUN_ID>": run_id,
            "<REPLACE_PIPELINE_NAME>": pipeline_name_trimmed,
            "<REPLACE_MODEL_NAME>": model_name_full,
            "<REPLACE_SAMPLE_ROW>": str(sample_row_formatted),
        }
        notebook = replace_placeholder_in_notebook(notebook, replacements)
        notebook_path = output_path / "notebooks"
        notebook_path.mkdir(parents=True, exist_ok=True)
        with (notebook_path / "automl_predictor_notebook.ipynb").open("w", encoding="utf-8") as f:
            json.dump(notebook, f)

        return model_name_full, eval_results

    # Phase A: metrics + notebooks - all models run concurrently.
    with ThreadPoolExecutor(max_workers=len(model_names_full)) as executor:
        futures = [executor.submit(_process_model, name) for name in model_names_full]
        eval_results_by_model = dict(f.result() for f in futures)

    # Phase B: clone for deployment - sequential because set_model_best mutates predictor state.
    for model_name_full in model_names_full:
        output_path = Path(models_artifact.path) / model_name_full
        predictor_clone.set_model_best(model=model_name_full, save_trainer=True)
        predictor_clone.clone_for_deployment(path=output_path / "predictor", dirs_exist_ok=True)

    shutil.rmtree(work_path, ignore_errors=True)

    # Build ordered models_metadata (preserves top-N ranking order).
    models_metadata = []
    for model_name_full in model_names_full:
        eval_results = eval_results_by_model[model_name_full]
        model_metadata = {
            "name": model_name_full,
            "location": {
                "model_directory": model_name_full,
                "predictor": str(Path(model_name_full) / "predictor"),
                "notebook": str(Path(model_name_full) / "notebooks" / "automl_predictor_notebook.ipynb"),
                "metrics": str(Path(model_name_full) / "metrics"),
            },
            "metrics": {
                "test_data": eval_results,
            },
        }
        models_metadata.append(model_metadata)
        with (Path(models_artifact.path) / model_name_full / "model.json").open("w", encoding="utf-8") as f:
            json.dump(model_metadata, f, indent=2)

    # Serialize as a JSON string and parse back in downstream components.
    models_artifact.metadata["model_names"] = json.dumps(model_names_full)
    models_artifact.metadata["context"] = {
        "data_config": {
            "sampling_config": sampling_config,
            "split_config": split_config,
        },
        "task_type": problem_type,
        "label_column": predictor.label,
        "model_config": model_config,
        "models": models_metadata,
    }

    return NamedTuple("outputs", eval_metric=str)(eval_metric=str(predictor.eval_metric))


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        autogluon_models_training,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
