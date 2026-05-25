"""KFP component: Download a HuggingFace model to a PVC for caching.

Downloads the model once to a PVC path. On subsequent runs, detects the
model is already cached and skips the download. This avoids re-downloading
14GB+ model weights every time the InferenceService pod restarts.
"""

from kfp import dsl
from kfp_components.utils.consts import RAY_RAG_BASE_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=RAY_RAG_BASE_IMAGE,
    packages_to_install=["huggingface_hub>=0.20.0"],
)
def download_model(
    model_name: str,
    model_cache_pvc: str,
    model_cache_mount: str = "/mnt/models",
) -> str:
    """Download a HuggingFace model to a PVC for caching.

    If the model is already present on the PVC, skips the download.

    Args:
        model_name: HuggingFace model ID (e.g. 'mistralai/Mistral-7B-Instruct-v0.3').
        model_cache_pvc: Name of the PVC to store models (unused here, mounted via pipeline).
        model_cache_mount: Mount path for the model cache PVC.

    Returns:
        The PVC sub-path where the model is stored.
    """
    import os

    from huggingface_hub import snapshot_download

    model_dir_name = model_name.replace("/", "--")
    model_path = os.path.join(model_cache_mount, model_dir_name)
    sentinel = os.path.join(model_path, ".download_complete")

    if os.path.exists(sentinel):
        file_count = sum(1 for _ in os.scandir(model_path) if _.is_file())
        print(f"Model '{model_name}' already cached at {model_path} ({file_count} files). Skipping download.")
        return model_dir_name

    print(f"Downloading model '{model_name}' to {model_path}...")
    snapshot_download(
        repo_id=model_name,
        local_dir=model_path,
        local_dir_use_symlinks=False,
    )

    with open(sentinel, "w") as f:
        f.write(model_name)

    file_count = sum(1 for _ in os.scandir(model_path) if _.is_file())
    print(f"Model '{model_name}' downloaded to {model_path} ({file_count} files).")
    return model_dir_name


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        download_model,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
