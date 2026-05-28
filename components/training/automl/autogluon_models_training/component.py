import pathlib
from typing import NamedTuple, Optional

from kfp import dsl
from kfp_components.utils.consts import AUTOML_IMAGE  # pyright: ignore[reportMissingImports]

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
    positive_class: Optional[str] = None,
) -> NamedTuple("outputs", eval_metric=str):
    """Train AutoGluon models, select the top N, and refit each on the full dataset.

    Expects pre-cleaned CSV data from the tabular data loader (infinite values replaced,
    duplicates removed, missing labels dropped). Reads train/test/extra-train CSVs and
    validates that the label column exists in each dataset.

    This component combines the model selection and full-refit stages into a single
    step. It trains a TabularPredictor on sampled data, ranks all models on the test
    set, then refits each of the top N models on the full training data in a single
    ``refit_full`` call. Post-refit work (predict, evaluate, feature importance,
    confusion matrix (via evaluate_predictions detailed_report), classification curves,
    notebook generation) runs concurrently across all top-N models
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
        positive_class: Optional label value for the positive class in **binary** classification
            (``int`` or ``str``, e.g. ``"1"`` or ``"yes"``). Passed to ``TabularPredictor`` when set.
            If ``None`` or empty, AutoGluon infers the positive class when ``fit`` runs (see note below).
            Ignored for ``multiclass`` and ``regression``.

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
    from typing import Any

    import numpy as np
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
    if positive_class is not None and not isinstance(positive_class, str):
        raise TypeError("positive_class must be a string or None.")

    sampling_config = sampling_config or {}
    split_config = split_config or {}

    def _coerce_positive_class(value: Optional[str]) -> str | int | None:
        """Normalize pipeline input; empty string means unset (AutoGluon will infer)."""
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.lstrip("-").isdigit():
            return int(stripped)
        return stripped

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

    coerced_positive_class = _coerce_positive_class(positive_class)
    if coerced_positive_class is not None and task_type != "binary":
        logger.warning(
            "positive_class=%r is ignored when task_type=%r (only used for binary classification).",
            coerced_positive_class,
            task_type,
        )

    predictor_path = Path(workspace_path) / "autogluon_predictor"
    predictor_init_kwargs: dict[str, Any] = {
        "problem_type": task_type,
        "label": label_column,
        "eval_metric": eval_metric,
        "path": predictor_path,
        "verbosity": 2,
    }
    if task_type == "binary" and coerced_positive_class is not None:
        predictor_init_kwargs["positive_class"] = coerced_positive_class
        logger.info("Using explicit positive_class=%r for TabularPredictor.", coerced_positive_class)
    elif task_type == "binary":
        logger.info(
            "positive_class not set; AutoGluon will infer it when fit runs as the second unique "
            "label value after sorting classes (e.g. classes [0, 1] -> positive_class=1; "
            "classes ['abc', 'def'] -> positive_class='def')."
        )

    predictor = TabularPredictor(**predictor_init_kwargs).fit(
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

    def _scalar_eval_metrics(eval_results: dict) -> dict:
        """Keep only finite scalar scores for metrics.json (KFP-safe)."""
        metrics = {}
        for key, value in eval_results.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                continue
            if hasattr(value, "item"):
                value = float(value)
                if math.isnan(value) or math.isinf(value):
                    continue
            if isinstance(value, (int, float)):
                metrics[key] = value
        return metrics

    def _confusion_matrix_to_dict(confusion_matrix_res) -> dict | None:
        if confusion_matrix_res is None:
            return None
        if hasattr(confusion_matrix_res, "to_dict"):
            return confusion_matrix_res.to_dict()
        if isinstance(confusion_matrix_res, dict):
            return confusion_matrix_res
        raise TypeError(f"Unexpected confusion_matrix type: {type(confusion_matrix_res)!r}")

    def _serialize_curve_array(values: Any) -> list[float]:
        return [float(v) for v in np.asarray(values).tolist()]

    def _serialize_curve_thresholds(values: Any) -> list[float | str]:
        serialized: list[float | str] = []
        for value in np.asarray(values).tolist():
            if isinstance(value, (float, np.floating)) and np.isposinf(value):
                serialized.append("inf")
            elif isinstance(value, (float, np.floating)) and np.isneginf(value):
                serialized.append("-inf")
            else:
                serialized.append(float(value))
        return serialized

    def _binarize_labels(y_true, positive_label: Any) -> np.ndarray:
        import pandas as pd

        labels = y_true if isinstance(y_true, pd.Series) else pd.Series(y_true)
        return (labels == positive_label).astype(int).to_numpy()

    def _resolve_positive_column(classes: list[Any], positive_class: Any | None) -> Any:
        if positive_class is not None and positive_class in classes:
            return positive_class
        if len(classes) == 2:
            logger.warning(
                "positive_class not specified for binary classification; defaulting to %r as positive class "
                "(available classes: %r). Set positive_class explicitly for clarity.",
                classes[1],
                classes,
            )
            return classes[1]
        raise ValueError(
            f"Cannot resolve positive class from proba columns {classes!r}; "
            f"pass positive_class explicitly (got {positive_class!r})."
        )

    def _roc_curve_block(y_true_binary: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
        import math

        from sklearn.metrics import roc_auc_score, roc_curve

        fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
        auc = float(roc_auc_score(y_true_binary, y_score))

        if math.isnan(auc):
            raise ValueError(
                "Cannot compute ROC AUC: only one class present in test data. "
                "ROC curves require both positive and negative examples."
            )

        return {
            "auc": auc,
            "fpr": _serialize_curve_array(fpr),
            "tpr": _serialize_curve_array(tpr),
            "thresholds": _serialize_curve_thresholds(thresholds),
        }

    def _pr_curve_block(y_true_binary: np.ndarray, y_score: np.ndarray, baseline_precision: float) -> dict[str, Any]:
        import math

        from sklearn.metrics import average_precision_score, precision_recall_curve

        precision, recall, thresholds = precision_recall_curve(y_true_binary, y_score)
        average_precision = float(average_precision_score(y_true_binary, y_score))

        if math.isnan(average_precision):
            raise ValueError(
                "Cannot compute precision-recall curve: only one class present in test data. "
                "Precision-recall curves require both positive and negative examples."
            )

        return {
            "average_precision": average_precision,
            "precision": _serialize_curve_array(precision),
            "recall": _serialize_curve_array(recall),
            "thresholds": _serialize_curve_array(thresholds),
            "baseline_precision": baseline_precision,
        }

    def _macro_weighted_curve_metric(values: list[float], weights: list[int]) -> float:
        if not values:
            return 0.0
        if sum(weights) == 0:
            return float(np.mean(values))
        return float(np.average(values, weights=weights))

    def _build_binary_curves_json(y_true, proba_df, positive_class: Any | None) -> dict[str, Any]:
        classes = list(proba_df.columns)
        positive_label = _resolve_positive_column(classes, positive_class)
        y_true_binary = _binarize_labels(y_true, positive_label)
        y_score = proba_df[positive_label].to_numpy()
        num_positive = int(y_true_binary.sum())
        num_samples = len(y_true_binary)
        roc_block = _roc_curve_block(y_true_binary, y_score)
        roc_block["support"] = num_positive
        pr_block = _pr_curve_block(
            y_true_binary,
            y_score,
            baseline_precision=num_positive / num_samples if num_samples else 0.0,
        )
        pr_block["support"] = num_positive
        return {
            "task_type": "binary",
            "positive_class": positive_label.item() if hasattr(positive_label, "item") else positive_label,
            "num_samples": num_samples,
            "num_positive": num_positive,
            "num_negative": num_samples - num_positive,
            "roc_curve": roc_block,
            "precision_recall_curve": pr_block,
        }

    def _build_multiclass_curves_json(y_true, proba_df) -> dict[str, Any]:
        classes = list(proba_df.columns)
        num_samples = len(y_true)
        per_class_roc: dict[str, dict[str, Any]] = {}
        per_class_pr: dict[str, dict[str, Any]] = {}
        roc_aucs: list[float] = []
        ap_scores: list[float] = []
        weights: list[int] = []
        skipped_classes: list[str] = []

        for label in classes:
            key = str(label.item() if hasattr(label, "item") else label)
            y_true_binary = _binarize_labels(y_true, label)
            support = int(y_true_binary.sum())
            y_score = proba_df[label].to_numpy()
            try:
                roc = _roc_curve_block(y_true_binary, y_score)
                roc["support"] = support
                per_class_roc[key] = roc
                roc_aucs.append(roc["auc"])
                pr = _pr_curve_block(
                    y_true_binary,
                    y_score,
                    baseline_precision=support / num_samples if num_samples else 0.0,
                )
                per_class_pr[key] = pr
                ap_scores.append(pr["average_precision"])
                weights.append(support)
            except ValueError as e:
                logger.warning(
                    "Skipping ROC/PR curves for class %r: %s",
                    key,
                    e,
                )
                skipped_classes.append(key)

        if not per_class_roc:
            raise ValueError(
                "Cannot compute multiclass curves: no class had both positive and negative "
                f"examples in test data (skipped: {skipped_classes})."
            )

        serializable_classes = [c.item() if hasattr(c, "item") else c for c in classes]

        payload: dict[str, Any] = {
            "task_type": "multiclass",
            "strategy": "ovr",
            "num_classes": len(classes),
            "classes": serializable_classes,
            "num_samples": num_samples,
            "roc_curve": {
                "auc_macro": float(np.mean(roc_aucs)),
                "auc_weighted": _macro_weighted_curve_metric(roc_aucs, weights),
                "per_class": per_class_roc,
            },
            "precision_recall_curve": {
                "average_precision_macro": float(np.mean(ap_scores)),
                "average_precision_weighted": _macro_weighted_curve_metric(ap_scores, weights),
                "per_class": per_class_pr,
            },
        }
        if skipped_classes:
            payload["skipped_classes"] = skipped_classes
        return payload

    def build_curves_json(
        y_true,
        y_proba,
        curves_task_type: str,
        positive_class: Any | None = None,
    ) -> dict[str, Any]:
        """Build curves.json for binary or multiclass classification."""
        import pandas as pd

        if curves_task_type not in {"binary", "multiclass"}:
            raise ValueError(f"task_type must be 'binary' or 'multiclass'; got {curves_task_type!r}.")

        y_true_series = pd.Series(y_true)
        proba_df = y_proba if hasattr(y_proba, "columns") else pd.DataFrame(y_proba)

        if curves_task_type == "binary":
            return _build_binary_curves_json(y_true_series, proba_df, positive_class)
        return _build_multiclass_curves_json(y_true_series, proba_df)

    def _process_model(model_name_full: str) -> tuple[str, dict]:
        """Compute metrics and write metric files + notebook for one refitted model.

        Safe to run concurrently across models: only reads from the shared predictor
        and test data, and writes to isolated per-model directories under
        models_artifact.path. Does NOT call set_model_best / clone_for_deployment,
        which mutate predictor state and must stay sequential.

        Returns (model_name_full, eval_results).
        """
        output_path = Path(models_artifact.path) / model_name_full
        y_true = test_data_df[predictor.label]

        if problem_type in {"binary", "multiclass"}:
            y_proba = predictor_clone.predict_proba(test_data_df, model=model_name_full)
            eval_results = predictor_clone.evaluate_predictions(
                y_true=y_true,
                y_pred=y_proba,
                detailed_report=True,
            )
            confusion_matrix_dict = _confusion_matrix_to_dict(eval_results.pop("confusion_matrix", None))
            eval_results.pop("classification_report", None)
        else:
            y_proba = None
            predictions = predictor_clone.predict(test_data_df, model=model_name_full)
            eval_results = predictor_clone.evaluate_predictions(y_true=y_true, y_pred=predictions)
            confusion_matrix_dict = None

        feature_importance = predictor_clone.feature_importance(
            test_data_df, model=model_name_full, subsample_size=2000
        )
        metrics_for_json = _scalar_eval_metrics(eval_results)

        (output_path / "metrics").mkdir(parents=True, exist_ok=True)
        with (output_path / "metrics" / "metrics.json").open("w") as f:
            json.dump(metrics_for_json, f)
        with (output_path / "metrics" / "feature_importance.json").open("w") as f:
            json.dump(feature_importance.to_dict(), f)

        if confusion_matrix_dict is not None:
            with (output_path / "metrics" / "confusion_matrix.json").open("w") as f:
                json.dump(confusion_matrix_dict, f)

        if y_proba is not None:
            try:
                curves_payload = build_curves_json(
                    y_true=y_true,
                    y_proba=y_proba,
                    curves_task_type=problem_type,
                    positive_class=getattr(predictor_clone, "positive_class", None),
                )
                with (output_path / "metrics" / "curves.json").open("w") as f:
                    json.dump(curves_payload, f)
            except ValueError as e:
                logger.warning(
                    "Could not generate curves.json for model %r: %s. Skipping curve generation.",
                    model_name_full,
                    str(e),
                )

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

        return model_name_full, metrics_for_json

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
