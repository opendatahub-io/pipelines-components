"""Data utilities: dataset resolution, JSONL preparation, OCI model download."""

import json
import logging
import os
import shutil
import subprocess
from typing import List, Optional


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


def _skopeo_copy(ref: str, dest: str, auth: Optional[str], log: logging.Logger) -> None:
    """Copy OCI image using skopeo.

    Args:
        ref: Image reference.
        dest: Destination directory.
        auth: Authentication JSON.
        log: Logger instance.
    """
    os.makedirs(dest, exist_ok=True)
    af = None
    if auth:
        af = "/tmp/skopeo-auth.json"
        with open(af, "w") as f:
            f.write(auth)
    cmd = ["skopeo", "copy"]
    if af:
        cmd.extend(["--authfile", af])
    cmd.extend([f"docker://{ref}", f"dir:{dest}"])
    log.info(f"Run: {' '.join(cmd)}")
    r = subprocess.run(cmd, text=True, capture_output=True)
    if r.returncode != 0:
        log.error(f"skopeo fail: {r.stderr}")
        r.check_returncode()


def _extract_tar(img_dir: str, out: str, log: logging.Logger) -> List[str]:
    """Extract model files from OCI image layers.

    Args:
        img_dir: Image directory.
        out: Output directory.
        log: Logger instance.

    Returns:
        List of extracted file paths.
    """
    import tarfile

    os.makedirs(out, exist_ok=True)
    ext = []
    for fn in os.listdir(img_dir):
        fp = os.path.join(img_dir, fn)
        if not os.path.isfile(fp) or fn.endswith(".json") or fn in {"manifest", "index.json"}:
            continue
        try:
            with tarfile.open(fp, mode="r:*") as tf:
                members = [m for m in tf.getmembers() if m.isfile() and m.name.startswith("models/")]
                tf.extractall(path=out, members=members, filter="data")
                ext.extend(m.name for m in members)
        except tarfile.FilterError:
            raise
        except tarfile.TarError:
            pass
    log.info(f"Extracted: {len(ext)}")
    return ext


def _find_hf_model(root: str) -> Optional[str]:
    """Find HuggingFace model directory.

    Args:
        root: Root directory to search.

    Returns:
        Path to model directory, or None if not found.
    """
    wt = {"pytorch_model.bin", "pytorch_model.bin.index.json", "model.safetensors", "model.safetensors.index.json"}
    tk = {"tokenizer.json", "tokenizer.model"}
    for dp, _, fns in os.walk(root):
        fn = set(fns)
        if "config.json" in fn and (fn & wt) and (fn & tk):
            return dp
    return None


def _get_oci_auth(log: logging.Logger) -> Optional[str]:
    """Get OCI authentication from environment.

    Args:
        log: Logger instance.

    Returns:
        Authentication JSON string, or None.
    """
    raw = os.environ.get("OCI_PULL_SECRET_MODEL_DOWNLOAD", "").strip()
    if not raw:
        log.warning("OCI_PULL_SECRET_MODEL_DOWNLOAD not set")
        return None
    try:
        p = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "OCI_PULL_SECRET_MODEL_DOWNLOAD is set but is not valid JSON. "
            "It must contain Docker config.json content for skopeo authentication."
        ) from exc
    if not isinstance(p, dict) or not p.get("auths"):
        raise ValueError(
            "OCI_PULL_SECRET_MODEL_DOWNLOAD is set but does not look like a Docker config.json. "
            "Expected a JSON object with a non-empty 'auths' field."
        )
    log.info("OCI authentication configuration loaded successfully")
    return raw


def download_oci_model(model_ref: str, pvc_path: str, log: logging.Logger) -> str:
    """Download model from OCI registry.

    Args:
        model_ref: OCI model reference (oci://...).
        pvc_path: PVC root path.
        log: Logger instance.

    Returns:
        Path to extracted model.
    """
    ref = model_ref[6:]  # Remove "oci://" prefix
    img_dir = os.path.join(pvc_path, "model-dir")
    mod_out = os.path.join(pvc_path, "model")
    if os.path.isdir(mod_out):
        shutil.rmtree(mod_out, ignore_errors=True)
    _skopeo_copy(ref, img_dir, _get_oci_auth(log), log)
    _extract_tar(img_dir, mod_out, log)
    cand = os.path.join(mod_out, "models")
    hfd = _find_hf_model(cand if os.path.isdir(cand) else mod_out)
    return hfd if hfd else mod_out
