"""Headless rendering of AutoML visualization plots for MLflow artifact logging.

The training components persist plot *data* as JSON (``confusion_matrix.json``,
``curves.json``, ``back_testing.json``); the notebooks render those into figures when a
user runs them. For MLflow artifact logging we render the same data into PNG files here,
without any notebook/IPython dependency, using the non-interactive ``Agg`` backend.

Every function is best-effort: rendering failures are logged and skipped so MLflow logging
never fails the pipeline. Matplotlib ships on the AutoML runtime image (AutoGluon depends
on it); if it is somehow missing, these functions return an empty list.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to read plot data JSON at %s", path)
        return None


def _matplotlib():
    """Import matplotlib configured for headless PNG rendering."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def render_confusion_matrix(confusion_matrix: dict[str, Any], out_path: Path) -> Path | None:
    """Render a confusion-matrix heatmap from a ``DataFrame.to_dict()`` payload.

    The payload maps predicted-label -> {true-label -> count}. Rows are true labels,
    columns are predicted labels.
    """
    if not confusion_matrix:
        return None
    try:
        plt = _matplotlib()

        col_labels = list(confusion_matrix.keys())
        row_labels: list[str] = []
        for column in confusion_matrix.values():
            for row in column:
                if row not in row_labels:
                    row_labels.append(row)

        matrix = [[float(confusion_matrix[col].get(row, 0)) for col in col_labels] for row in row_labels]

        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(image, ax=ax)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Confusion Matrix")
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, f"{int(value)}", ha="center", va="center", color="black")
        fig.tight_layout()
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        return out_path
    except Exception:
        logger.exception("Failed to render confusion matrix plot")
        return None


def _plot_binary_roc(plt: Any, roc: dict[str, Any], out_path: Path) -> Path | None:
    fpr = roc.get("fpr")
    tpr = roc.get("tpr")
    if not fpr or not tpr:
        return None
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {roc.get('auc', float('nan')):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def _plot_multiclass_roc(plt: Any, roc: dict[str, Any], out_path: Path) -> Path | None:
    per_class = roc.get("per_class") or {}
    if not per_class:
        return None
    fig, ax = plt.subplots(figsize=(6, 5))
    plotted = False
    for label, block in per_class.items():
        fpr = block.get("fpr")
        tpr = block.get("tpr")
        if not fpr or not tpr:
            continue
        ax.plot(fpr, tpr, label=f"{label} (AUC = {block.get('auc', float('nan')):.3f})")
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves (one-vs-rest, macro AUC = {roc.get('auc_macro', float('nan')):.3f})")
    ax.legend(loc="lower right", fontsize="small")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def render_roc_curve(curves: dict[str, Any], out_path: Path) -> Path | None:
    """Render an ROC curve PNG from a ``curves.json`` payload (binary or multiclass)."""
    roc = curves.get("roc_curve")
    if not isinstance(roc, dict):
        return None
    try:
        plt = _matplotlib()
        if curves.get("task_type") == "multiclass":
            return _plot_multiclass_roc(plt, roc, out_path)
        return _plot_binary_roc(plt, roc, out_path)
    except Exception:
        logger.exception("Failed to render ROC curve plot")
        return None


def render_classification_plots(model_dir: Path, out_dir: Path) -> list[Path]:
    """Render confusion-matrix and ROC plots for one classification model directory.

    Reads ``metrics/confusion_matrix.json`` and ``metrics/curves.json`` under
    ``model_dir`` and writes PNGs into ``out_dir``. Returns the list of files written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    confusion_matrix = _load_json(model_dir / "metrics" / "confusion_matrix.json")
    if confusion_matrix:
        rendered = render_confusion_matrix(confusion_matrix, out_dir / "confusion_matrix.png")
        if rendered is not None:
            written.append(rendered)

    curves = _load_json(model_dir / "metrics" / "curves.json")
    if curves:
        rendered = render_roc_curve(curves, out_dir / "roc_curve.png")
        if rendered is not None:
            written.append(rendered)

    return written


def render_timeseries_plots(model_dir: Path, out_dir: Path) -> list[Path]:
    """Render a back-testing forecast-vs-actual plot for one timeseries model directory.

    Reads ``metrics/back_testing.json`` under ``model_dir`` and writes a PNG into
    ``out_dir``. Returns the list of files written.
    """
    back_testing = _load_json(model_dir / "metrics" / "back_testing.json")
    if not back_testing:
        return []
    windows = back_testing.get("windows") or back_testing.get("forecasts")
    if not windows:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        plt = _matplotlib()
        fig, ax = plt.subplots(figsize=(8, 4))
        # plotted: was any series drawn at all (keep the figure). forecast_labeled /
        # actual_labeled: track each legend entry independently so an actual-only window is
        # not discarded and the "actual" label is not suppressed when a forecast drew first.
        plotted = False
        forecast_labeled = False
        actual_labeled = False
        for index, window in enumerate(windows):
            forecast = window.get("forecast_data") or window.get("forecast") or []
            for series in forecast if isinstance(forecast, list) else []:
                timestamps = series.get("timestamps") or series.get("timestamp")
                mean = series.get("mean") or series.get("0.5")
                if timestamps and mean:
                    ax.plot(range(len(mean)), mean, label=f"window {index} forecast" if not forecast_labeled else None)
                    forecast_labeled = True
                    plotted = True
                actual = series.get("actual") or series.get("target")
                if actual:
                    ax.plot(range(len(actual)), actual, linestyle="--", label="actual" if not actual_labeled else None)
                    actual_labeled = True
                    plotted = True
        if not plotted:
            plt.close(fig)
            return []
        ax.set_title("Back-testing: forecast vs actual")
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.legend(loc="best", fontsize="small")
        out_path = out_dir / "back_testing.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        return [out_path]
    except Exception:
        logger.exception("Failed to render timeseries back-testing plot")
        return []


def render_model_plots(task_type: str, model_dir: Path, out_dir: Path) -> list[Path]:
    """Render task-appropriate plots for one model directory. Best-effort."""
    if task_type in {"binary", "multiclass"}:
        return render_classification_plots(model_dir, out_dir)
    if task_type == "time_series":
        return render_timeseries_plots(model_dir, out_dir)
    return []
