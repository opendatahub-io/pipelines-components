"""Data utilities: dataset resolution and JSONL preparation."""

import json
import logging
import os
import shutil


def resolve_dataset(inp, out_dir: str, log: logging.Logger) -> None:
    """Resolve and prepare dataset from various sources.

    Args:
        inp: Input dataset artifact.
        out_dir: Output directory.
        log: Logger instance.
    """
    from datasets import load_dataset

    if os.path.isdir(out_dir) and any(os.scandir(out_dir)):
        log.info(f"Using existing ds: {out_dir}")
        return
    if inp and getattr(inp, "path", None) and os.path.exists(inp.path):
        src = inp.path
        if os.path.isdir(src):
            log.info(f"Copy ds dir: {src}")
            shutil.copytree(src, out_dir, dirs_exist_ok=True)
        else:
            log.info(f"Copy ds file: {src}")
            dst = os.path.join(out_dir, os.path.basename(src))
            if not os.path.splitext(dst)[1]:
                dst = os.path.join(out_dir, "train.jsonl")
            shutil.copy2(src, dst)
        return
    rp = ""
    try:
        if inp and hasattr(inp, "metadata") and isinstance(inp.metadata, dict):
            pvc_m = (inp.metadata.get("pvc_path") or inp.metadata.get("pvc_dir") or "").strip()
            if pvc_m and os.path.exists(pvc_m):
                if os.path.isdir(pvc_m) and any(os.scandir(pvc_m)):
                    log.info(f"PVC ds dir: {pvc_m}")
                    shutil.copytree(pvc_m, out_dir, dirs_exist_ok=True)
                    return
                elif os.path.isfile(pvc_m):
                    log.info(f"PVC ds file: {pvc_m}")
                    dst = os.path.join(out_dir, os.path.basename(pvc_m))
                    if not os.path.splitext(dst)[1]:
                        dst = os.path.join(out_dir, "train.jsonl")
                    shutil.copy2(pvc_m, dst)
                    return
            rp = (inp.metadata.get("artifact_path") or "").strip()
    except Exception:
        rp = ""
    if rp:
        if rp.startswith("s3://") or rp.startswith("http://") or rp.startswith("https://"):
            log.info(f"Remote ds: {rp}")
            ext = rp.lower()
            if ext.endswith(".json") or ext.endswith(".jsonl"):
                ds = load_dataset("json", data_files=rp, split="train")
            elif ext.endswith(".parquet"):
                ds = load_dataset("parquet", data_files=rp, split="train")
            else:
                raise ValueError("Unsupported remote format")
            ds.save_to_disk(out_dir)
            return
        else:
            log.info(f"HF ds: {rp}")
            load_dataset(rp, split="train").save_to_disk(out_dir)
            return
    raise ValueError(
        "No dataset provided or resolvable. Please supply an input artifact, a PVC path via metadata "
        "('pvc_path' or 'pvc_dir'), or a remote source via metadata['artifact_path'] (S3/HTTP/HF repo id)."
    )


def prepare_jsonl(ds_dir: str, jsonl_path: str, log: logging.Logger) -> None:
    """Prepare JSONL file from dataset.

    Args:
        ds_dir: Dataset directory.
        jsonl_path: Output JSONL path.
        log: Logger instance.
    """
    from datasets import load_from_disk

    try:
        dsk = load_from_disk(ds_dir)
        tr = dsk["train"] if isinstance(dsk, dict) else dsk
        try:
            tr.to_json(jsonl_path, lines=True)
            log.info(f"JSONL: {jsonl_path}")
        except AttributeError:
            with open(jsonl_path, "w") as f:
                for r in tr:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            log.info(f"JSONL manual: {jsonl_path}")
    except Exception as e:
        log.warning(f"JSONL export failed: {e}")
