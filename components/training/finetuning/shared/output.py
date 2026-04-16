"""Output utilities: model persistence, metrics extraction, loss plotting."""

import json
import logging
import os
import shutil
from typing import Dict, List, Optional


def find_model_dir(root: str) -> Optional[str]:
    """Find the most recent model checkpoint directory.

    Args:
        root: Root directory to search.

    Returns:
        Path to model directory, or None if not found.
    """
    if not os.path.isdir(root):
        return None
    cands = []
    for r, _, fs in os.walk(root):
        if "config.json" in fs:
            try:
                cands.append((os.path.getmtime(os.path.join(r, "config.json")), r))
            except OSError:
                pass
    if not cands:
        latest = None
        for e in os.listdir(root):
            p = os.path.join(root, e)
            if os.path.isdir(p):
                try:
                    m = os.path.getmtime(p)
                    if latest is None or m > latest[0]:
                        latest = (m, p)
                except OSError:
                    pass
        return latest[1] if latest else None
    cands.sort(reverse=True)
    return cands[0][1]


def persist_model(
    ckpt_dir: str,
    pvc_path: str,
    training_base_model: str,
    output_model,
    log: logging.Logger,
) -> None:
    """Persist trained model to PVC and output artifact.

    Args:
        ckpt_dir: Checkpoint directory.
        pvc_path: PVC root path.
        training_base_model: Base model name.
        output_model: Output model artifact.
        log: Logger instance.
    """
    latest = find_model_dir(ckpt_dir)
    if not latest:
        raise RuntimeError(f"No model found in {ckpt_dir}")
    log.info(f"Model: {latest}")
    pvc_out = os.path.join(pvc_path, "final_model")
    if os.path.exists(pvc_out):
        shutil.rmtree(pvc_out, ignore_errors=True)
    shutil.copytree(latest, pvc_out, dirs_exist_ok=True)
    log.info(f"Copied to {pvc_out}")
    output_model.name = f"{training_base_model}-checkpoint"
    shutil.copytree(latest, output_model.path, dirs_exist_ok=True)
    log.info(f"Artifact: {output_model.path}")
    output_model.metadata["model_name"] = training_base_model
    output_model.metadata["artifact_path"] = output_model.path
    output_model.metadata["pvc_model_dir"] = pvc_out


def extract_metrics_from_jsonl(metrics_file: str) -> tuple[Dict, List[float]]:
    """Extract training metrics from JSONL file.

    Args:
        metrics_file: Path to metrics JSONL file.

    Returns:
        Tuple of (metrics dict, loss values list).
    """
    if not os.path.exists(metrics_file):
        return {}, []

    met, loss = {}, []
    with open(metrics_file) as f:
        for ln in f:
            if ln.strip():
                try:
                    e = json.loads(ln)
                    for s, d in [
                        ("loss", "loss"),
                        ("avg_loss", "loss"),
                        ("lr", "learning_rate"),
                        ("grad_norm", "grad_norm"),
                        ("gradnorm", "grad_norm"),
                        ("val_loss", "eval_loss"),
                        ("epoch", "epoch"),
                        ("step", "step"),
                    ]:
                        if s in e and d not in met:
                            try:
                                met[d] = float(e[s])
                            except (ValueError, TypeError):
                                pass
                    lv = e.get("loss") or e.get("avg_loss")
                    if lv:
                        try:
                            loss.append(float(lv))
                        except (ValueError, TypeError):
                            pass
                except Exception:
                    pass
    if loss:
        met["final_loss"], met["min_loss"] = loss[-1], min(loss)
        met["final_perplexity"] = __import__("math").exp(min(loss[-1], 10))
    return met, loss


def plot_training_loss(loss: list, path: str) -> None:
    """Plot training loss curve and save as HTML.

    Args:
        loss: List of loss values.
        path: Output HTML file path.
    """
    import base64
    from io import BytesIO

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not loss:
        with open(path, "w") as f:
            f.write("<html><body><p>No loss data.</p></body></html>")
        return
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(range(1, len(loss) + 1), loss, "b-", linewidth=2, label="Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ml, mi = min(loss), loss.index(min(loss))
    ax.annotate(
        f"Min:{ml:.4f}",
        xy=(mi + 1, ml),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="green"),
        color="green",
    )
    ax.annotate(
        f"Final:{loss[-1]:.4f}",
        xy=(len(loss), loss[-1]),
        xytext=(-60, 10),
        textcoords="offset points",
        fontsize=9,
        color="red",
    )
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    html = (
        f"<!DOCTYPE html><html><head><style>*{{margin:0;padding:0}}"
        f"body{{font-family:sans-serif;padding:5px}}.s{{font-size:10px;margin-bottom:5px}}"
        f".c{{width:100%}}</style></head><body>"
        f'<div class="s">Loss:{loss[0]:.4f}-{loss[-1]:.4f}(min {ml:.4f})|{len(loss)} steps</div>'
        f'<img class="c" src="data:image/png;base64,{img}"/></body></html>'
    )
    with open(path, "w") as f:
        f.write(html)
