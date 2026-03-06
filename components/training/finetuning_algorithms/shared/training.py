"""Training utilities: runtime selection, nproc computation, job waiting."""

import logging


def safe_int(v, default: int) -> int:
    """Safely convert a value to int with a default fallback.

    Args:
        v: Value to convert.
        default: Default value if conversion fails.

    Returns:
        Integer value.
    """
    if v is None:
        return default
    if isinstance(v, int):
        return v
    s = str(v).strip()
    return int(s) if s else default


def select_runtime(client, log: logging.Logger, runtime_name: str = "training-hub"):
    """Find and return the named runtime from a TrainerClient.

    Args:
        client: TrainerClient instance.
        log: Logger instance.
        runtime_name: Name of the ClusterTrainingRuntime to find.

    Returns:
        The matching runtime object.

    Raises:
        RuntimeError: If the named runtime is not found.
    """
    for r in client.list_runtimes():
        if getattr(r, "name", "") == runtime_name:
            log.info(f"Runtime: {r}")
            return r
    raise RuntimeError(f"Runtime '{runtime_name}' not found")


def compute_nproc(
    gpu_per_worker: int,
    num_procs_per_worker: str,
    num_workers: int = 1,
    single_node: bool = False,
) -> tuple:
    """Compute nproc_per_node and nnodes for training jobs.

    Args:
        gpu_per_worker: GPUs per worker.
        num_procs_per_worker: Processes per worker ('auto' or int).
        num_workers: Number of training workers.
        single_node: Force single node (e.g. for LoRA/unsloth).

    Returns:
        Tuple of (nproc_per_node, nnodes).
    """
    auto = str(num_procs_per_worker).strip().lower() == "auto"
    np = gpu_per_worker if auto else safe_int(num_procs_per_worker, 1)
    nn = 1 if single_node else safe_int(num_workers, 1)
    return max(np, 1), max(nn, 1)


def wait_for_training_job(client, job, log: logging.Logger) -> None:
    """Wait for a training job to complete and raise on failure.

    Args:
        client: TrainerClient instance.
        job: Job name/identifier.
        log: Logger instance.

    Raises:
        RuntimeError: If job fails or ends in unexpected status.
    """
    client.wait_for_job_status(name=job, status={"Running"}, timeout=900)
    client.wait_for_job_status(name=job, status={"Complete", "Failed"}, timeout=1800)
    j = client.get_job(name=job)
    if getattr(j, "status", None) == "Failed":
        log.error(f"Job failed: {j.status}")
        raise RuntimeError(f"Job failed: {j.status}")
    elif getattr(j, "status", None) != "Complete":
        log.error(f"Unexpected status: {j.status}")
        raise RuntimeError(f"Unexpected status: {j.status}")
    log.info("Training completed successfully")
