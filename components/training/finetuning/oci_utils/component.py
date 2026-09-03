"""Lightweight KFP components for OCI model handling via dsl.importer.

These components support the dsl.If/dsl.Else/dsl.OneOf pattern needed
to conditionally run dsl.importer for OCI URIs while passing HuggingFace
model IDs through unchanged.
"""

from kfp import dsl


@dsl.component(base_image="registry.access.redhat.com/ubi9/python-311:latest")
def is_oci_uri(uri: str) -> str:
    """Check if a URI is an OCI reference.

    Returns "true" or "false" as a string for use with dsl.If conditions,
    since dsl.If only supports comparing task outputs (not Python methods).
    """
    return "true" if uri.startswith("oci://") else "false"


@dsl.component(base_image="registry.access.redhat.com/ubi9/python-311:latest")
def copy_oci_model_to_pvc(
    model: dsl.Input[dsl.Model],
    pvc_mount_path: str,
) -> str:
    """Copy model files from dsl.importer sidecar to workspace PVC.

    The dsl.importer modelcar sidecar makes model files available at
    model.path (/models). This component copies them to the workspace PVC
    and searches for a valid HuggingFace model directory structure.

    Returns the resolved model directory path on the PVC.
    """
    import os
    import shutil

    src = model.path
    dest = os.path.join(pvc_mount_path, "model")
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest, exist_ok=True)

    if not os.path.exists(src):
        raise RuntimeError(f"Model path not found: {src}")

    for dirpath, dirnames, filenames in os.walk(src):
        for name in [*dirnames, *filenames]:
            candidate = os.path.join(dirpath, name)
            if os.path.islink(candidate):
                raise RuntimeError(f"Refusing symlink in OCI model contents: {candidate}")

    count = 0
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dest, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
        count += 1
    print(f"Copied {count} items from {src} to {dest}")

    weight_files = {
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
        "model.safetensors",
        "model.safetensors.index.json",
    }
    tokenizer_files = {"tokenizer.json", "tokenizer.model"}
    for dirpath, _, filenames in os.walk(dest):
        fset = set(filenames)
        if "config.json" in fset and (fset & weight_files) and (fset & tokenizer_files):
            print(f"Found HuggingFace model directory: {dirpath}")
            return dirpath

    print(f"No HuggingFace model directory found, using: {dest}")
    return dest


@dsl.component(base_image="registry.access.redhat.com/ubi9/python-311:latest")
def passthrough_uri(value: str) -> str:
    """Return the input value unchanged.

    Exists to satisfy dsl.OneOf's requirement that both dsl.If and dsl.Else
    branches produce a task output. For HuggingFace model IDs, no download
    is needed — this component simply passes the ID through.
    """
    return value
