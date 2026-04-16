"""Training utilities: runtime selection, nproc computation, job waiting."""

import logging
import time


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


_SEPARATOR = "─" * 50


def _log_job_details(client, train_job, log: logging.Logger, has_node_0: bool) -> None:
    """Log TrainJob details including pod names and kubectl commands.

    Args:
        client: TrainerClient instance.
        train_job: TrainJob object.
        log: Logger instance.
        has_node_0: Whether the job has a node-0 step for log streaming.
    """
    namespace = getattr(getattr(client, "backend", None), "namespace", None) or "unknown"

    log.info(_SEPARATOR)
    log.info(f"TrainJob '{train_job.name}' is {train_job.status} in namespace '{namespace}'")
    created = getattr(train_job, "creation_timestamp", None)
    if created:
        log.info(f"Created at: {created}")
    log.info("")

    steps = getattr(train_job, "steps", None) or []
    if steps:
        log.info("Training pod(s):")
        for step in steps:
            log.info(f"  - {step.name}: {step.pod_name} ({step.status})")
        log.info("")
        log.info("To follow logs from a specific pod:")
        for step in steps:
            log.info(f"  kubectl -n {namespace} logs {step.pod_name} -f")
    else:
        log.info("No training pods found yet.")

    if has_node_0:
        log.info("")
        log.info("Streaming logs for node-0 below. For other nodes, use the kubectl commands above.")
    log.info(_SEPARATOR)


def wait_for_training_job(client, job: str, log: logging.Logger) -> None:
    """Wait for a training job to complete and raise on failure.

    After the job reaches Running status, logs TrainJob details (name,
    namespace, pod names) and streams node-0 logs into the pipeline step
    output. Falls back to status polling if log streaming fails.

    Args:
        client: TrainerClient instance.
        job: Job name/identifier.
        log: Logger instance.

    Raises:
        RuntimeError: If job fails or ends in unexpected status.
    """
    client.wait_for_job_status(name=job, status={"Running"}, timeout=900)

    # Fetch the TrainJob once and reuse for both logging and node-0 check.
    try:
        train_job = client.get_job(name=job)
    except Exception as e:
        log.warning(f"Could not retrieve TrainJob details: {e}")
        train_job = None

    # Check for node-0 before logging so the info block message is accurate.
    steps = (getattr(train_job, "steps", None) or []) if train_job else []
    node_0_exists = any(step.name == "node-0" for step in steps)

    if train_job:
        _log_job_details(client, train_job, log, has_node_0=node_0_exists)

    if train_job and not node_0_exists:
        log.warning("node-0 step not found in TrainJob, skipping log streaming.")

    if node_0_exists:
        # Retry log streaming to handle the brief window where the pod
        # container may still be in ContainerCreating after the job
        # reaches Running status.
        max_retries = 3
        retry_delay = 5
        for attempt in range(1, max_retries + 1):
            try:
                for line in client.get_job_logs(name=job, step="node-0", follow=True):
                    print(line, flush=True)
                break
            except Exception as e:
                if attempt < max_retries:
                    log.warning(
                        f"Log streaming attempt {attempt}/{max_retries} failed: {e}. Retrying in {retry_delay}s..."
                    )
                    time.sleep(retry_delay)
                else:
                    log.warning(f"Log streaming failed after {max_retries} attempts: {e}")

    client.wait_for_job_status(name=job, status={"Complete", "Failed"}, timeout=1800)

    j = client.get_job(name=job)
    if train_job:
        log.info(_SEPARATOR)
    if getattr(j, "status", None) == "Failed":
        log.error(f"Job failed: {j.status}")
        raise RuntimeError(f"Job failed: {j.status}")
    elif getattr(j, "status", None) != "Complete":
        log.error(f"Unexpected status: {j.status}")
        raise RuntimeError(f"Unexpected status: {j.status}")
    log.info("Training completed successfully")
