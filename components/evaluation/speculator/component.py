"""Benchmark speculative decoding with the upstream Speculators evaluator."""

from kfp import dsl


def _resolve_evaluation_dataset_path(uri: str, mount_path: str, persistent_pvc: str = "") -> str:
    """Resolve a mounted-PVC path and reject traversal or mismatched PVC URIs."""
    import os

    path = uri
    if path.startswith("pvc://"):
        pvc_uri = path.removeprefix("pvc://")
        pvc_name, separator, path = pvc_uri.partition("/")
        if not persistent_pvc:
            raise ValueError("A mounted PVC name is required to resolve pvc:// URIs")
        if pvc_name != persistent_pvc:
            raise ValueError(f"PVC URI references claim {pvc_name!r}, but the mounted PVC is {persistent_pvc!r}")
        if not separator:
            path = ""

    mount_root = os.path.realpath(mount_path)
    candidate = path if os.path.isabs(path) else os.path.join(mount_root, path)
    resolved_path = os.path.realpath(candidate)
    try:
        common_path = os.path.commonpath((mount_root, resolved_path))
    except ValueError as exc:
        raise ValueError(f"Path {uri!r} resolves outside mounted PVC {mount_root!r}") from exc
    if common_path != mount_root:
        raise ValueError(f"Path {uri!r} resolves outside mounted PVC {mount_root!r}")
    return resolved_path


def _extract_evaluation_prompt(record: object) -> str:
    """Extract a prompt string from common JSONL chat record shapes."""
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        raise ValueError(f"Unsupported evaluation record type: {type(record).__name__}")

    prompt = record.get("prompt") or record.get("text")
    if prompt:
        return str(prompt)

    messages = record.get("messages") or record.get("conversations") or []
    user_messages = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role", message.get("from")) in ("user", "human")
    ]
    if user_messages:
        prompt = user_messages[-1].get("content", user_messages[-1].get("value"))
    if not prompt:
        raise ValueError("Each evaluation record must contain a prompt or user message")
    return str(prompt)


def _prepare_local_evaluation_dataset(
    uri: str,
    mount_path: str,
    persistent_pvc: str,
    max_requests: int,
    sample_seed: int,
    output_path: str,
) -> tuple[str, bool, int]:
    """Normalize a local JSONL file to prompt rows for GuideLLM.

    Returns the original URI unchanged when it is an upstream dataset ID.
    """
    import json
    import os
    import random
    from pathlib import Path

    if not uri:
        raise ValueError("evaluation_dataset_uri must identify a dataset")
    if max_requests <= 0:
        raise ValueError("evaluation_max_samples must be greater than zero")

    if uri.startswith("pvc://") or os.path.isabs(uri):
        dataset_path = Path(_resolve_evaluation_dataset_path(uri, mount_path, persistent_pvc))
        if not dataset_path.exists():
            raise FileNotFoundError(f"Evaluation dataset was not found on the PVC: {dataset_path}")
    else:
        mounted_path = Path(_resolve_evaluation_dataset_path(uri, mount_path, persistent_pvc))
        if not mounted_path.exists():
            return uri, False, 0
        dataset_path = mounted_path

    if dataset_path.is_dir():
        candidates = sorted(dataset_path.glob("*.jsonl"))
        if not candidates:
            raise ValueError(f"No JSONL dataset found in {dataset_path}")
        dataset_path = candidates[0]
    if not dataset_path.is_file():
        raise ValueError(f"Evaluation dataset is not a JSONL file: {dataset_path}")

    prompts = []
    with dataset_path.open() as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            try:
                prompts.append(_extract_evaluation_prompt(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid evaluation record on line {line_number}: {exc}") from exc
    if not prompts:
        raise ValueError(f"The evaluation dataset contains no prompts: {dataset_path}")
    if len(prompts) > max_requests:
        prompts = random.Random(sample_seed).sample(prompts, max_requests)

    normalized_path = Path(output_path)
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    with normalized_path.open("w") as normalized_file:
        for prompt in prompts:
            normalized_file.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
    return str(normalized_path), True, len(prompts)


def _download_upstream_file(url: str, filename: str, attempts: int = 3) -> bytes:
    """Download one upstream file with short retries for transient failures."""
    import http.client
    import time
    import urllib.error
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": "kfp-speculator-evaluator"})
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                contents = response.read()
            if contents:
                return contents
            last_error = RuntimeError(f"Downloaded an empty upstream evaluation file: {filename}")
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and not 500 <= exc.code < 600:
                raise RuntimeError(f"Could not download upstream evaluation file {filename}: HTTP {exc.code}") from exc
            last_error = exc
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            last_error = exc

        if attempt < attempts:
            delay_seconds = attempt
            print(
                f"Download attempt {attempt}/{attempts} for {filename} failed: {last_error}; "
                f"retrying in {delay_seconds}s",
                flush=True,
            )
            time.sleep(delay_seconds)

    raise RuntimeError(
        f"Failed to download upstream evaluation file {filename} after {attempts} attempts"
    ) from last_error


def _download_upstream_evaluator(destination: str) -> str:
    """Download the pinned upstream evaluator and its adjacent Python modules.

    The cluster package index does not publish ``speculators>0.7.1``. The
    evaluator only imports ``speculators.provenance`` for its best-effort
    command metadata, so use that exact pinned upstream module as a local
    compatibility namespace rather than installing an unavailable package.
    """
    from pathlib import Path

    revision = "7883cd44264e6958cfc003963e5cdb895eb53bf1"
    source_dir = Path(destination)
    source_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("evaluate.py", "perf_utils.py", "mrcr.py"):
        url = f"https://raw.githubusercontent.com/vllm-project/speculators/{revision}/scripts/evaluate/{filename}"
        contents = _download_upstream_file(url, filename)
        (source_dir / filename).write_bytes(contents)
    provenance_package = source_dir / "compat" / "speculators"
    provenance_package.mkdir(parents=True, exist_ok=True)
    (provenance_package / "__init__.py").write_text('"""Pinned upstream provenance helper namespace."""\n')
    provenance_url = (
        f"https://raw.githubusercontent.com/vllm-project/speculators/{revision}/src/speculators/provenance.py"
    )
    provenance_contents = _download_upstream_file(provenance_url, "provenance.py")
    (provenance_package / "provenance.py").write_bytes(provenance_contents)
    return str(source_dir / "evaluate.py")


def _build_upstream_evaluate_command(
    python_executable: str,
    script_path: str,
    target: str,
    dataset: str,
    subsets: str,
    output_dir: str,
    max_requests: int,
    max_tokens: int,
    temperature: float,
) -> list[str]:
    """Build Fynn's documented ``evaluate.py sweep`` command."""
    import json

    return [
        python_executable,
        script_path,
        "--target",
        target,
        "--dataset",
        dataset,
        "--subsets",
        subsets,
        "--output-dir",
        output_dir,
        "--max-requests",
        str(max_requests),
        "--gen-kwargs",
        json.dumps({"temperature": temperature, "max_tokens": max_tokens}),
        "sweep",
    ]


def _run_upstream_evaluate(command: list[str], environment: dict[str, str], timeout_seconds: int) -> None:
    """Run the pinned upstream sweep with a bounded timeout."""
    import subprocess

    try:
        subprocess.run(command, check=True, env=environment, timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"Upstream Speculators evaluation timed out after {timeout_seconds} seconds") from exc


@dsl.component(
    base_image=(
        "quay.io/opendatahub/odh-th-torch-cuda-py312:odh-3.6-ea.1@"
        "sha256:9ad2d72ebe892dffd3554eeb81e2a47c9fcf0d97f89ed261d48f4e8bbc367b4b"
    ),
    packages_to_install=["kfp==2.17.0", "requests>=2.32.0", "guidellm~=0.7.1"],
    install_kfp_package=False,
    additional_funcs=[
        _resolve_evaluation_dataset_path,
        _extract_evaluation_prompt,
        _prepare_local_evaluation_dataset,
        _download_upstream_file,
        _download_upstream_evaluator,
        _build_upstream_evaluate_command,
        _run_upstream_evaluate,
    ],
)
def evaluate_speculator(
    output_metrics: dsl.Output[dsl.Metrics],
    output_results: dsl.Output[dsl.Artifact],
    verifier_model: str,
    draft_model_path: str,
    evaluation_dataset_uri: str,
    evaluation_speculator_type: str,
    evaluation_dataset_pvc: str = "",
    evaluation_max_samples: int = 80,
    evaluation_max_tokens: int = 256,
    evaluation_temperature: float = 0.0,
    evaluation_num_speculative_tokens: int = 5,
    evaluation_gpu_memory_utilization: float = 0.9,
    evaluation_port: int = 8000,
    evaluation_startup_timeout_seconds: int = 900,
    evaluation_timeout_seconds: int = 7200,
    evaluation_sample_seed: int = 42,
    evaluation_model_mount_path: str = "/mnt/persistent",
    evaluation_dataset_mount_path: str = "/mnt/persistent",
) -> None:
    """Run the upstream Speculators performance sweep against a speculative vLLM server.

    The component starts vLLM with the verifier and draft model, then runs the
    pinned upstream ``scripts/evaluate/evaluate.py sweep`` against its OpenAI
    endpoint. Upstream results (acceptance.csv, perf_results.csv, raw GuideLLM
    JSON, and provenance) are returned in ``output_results``.

    A local JSONL/PVC dataset may contain prompts, chat messages, or conversation
    records. Local records are normalized to prompt JSONL and deterministically
    capped at ``evaluation_max_samples``. A Hugging Face dataset ID is passed
    directly to the upstream script, using its HumanEval example subset.

    Use paths relative to the configured mount without a PVC prefix. With
    ``pvc://`` syntax, prefix the path with the exact mounted claim name; for
    example, use ``pvc://shared/models/ultrachat.jsonl`` when the claim is
    ``shared``.

    Args:
        output_metrics: KFP metrics for acceptance and sweep performance results.
        output_results: Directory artifact containing upstream CSVs and raw GuideLLM output.
        verifier_model: Verifier directory under the model mount, optionally using pvc://<claim>/<path> syntax.
        draft_model_path: Draft directory under the model mount, optionally using pvc://<claim>/<path> syntax.
        evaluation_dataset_uri: Hugging Face dataset ID, relative path under the dataset mount, or
            ``pvc://<claim>/<path>``.
        evaluation_speculator_type: Speculator method configured on the vLLM server.
        evaluation_dataset_pvc: Mounted PVC claim name. Every ``pvc://`` URI must use this exact
            claim name; relative paths need no PVC prefix.
        evaluation_max_samples: Maximum GuideLLM requests at each sweep point.
        evaluation_max_tokens: Maximum generation tokens for each request.
        evaluation_temperature: Sampling temperature passed through to GuideLLM.
        evaluation_num_speculative_tokens: Number of draft tokens proposed by vLLM.
        evaluation_gpu_memory_utilization: vLLM GPU memory utilization fraction.
        evaluation_port: Local port for the temporary vLLM endpoint.
        evaluation_startup_timeout_seconds: Maximum wait for the vLLM health endpoint.
        evaluation_timeout_seconds: Maximum time allowed for the upstream evaluation sweep.
        evaluation_sample_seed: Seed used when sampling a local JSONL dataset.
        evaluation_model_mount_path: Root directory for verifier and draft models; resolved paths must stay inside it.
        evaluation_dataset_mount_path: Root directory for local datasets; resolved paths must stay inside it.
    """
    import csv
    import json
    import math
    import os
    import shutil
    import signal
    import subprocess
    import sys
    import tempfile
    import time
    from pathlib import Path

    import requests

    if shutil.which("vllm") is None:
        raise RuntimeError("vllm binary not found - ensure vLLM is installed in the base image")
    if evaluation_max_samples <= 0:
        raise ValueError("evaluation_max_samples must be greater than zero")
    if evaluation_max_tokens <= 0:
        raise ValueError("evaluation_max_tokens must be greater than zero")
    if evaluation_timeout_seconds <= 0:
        raise ValueError("evaluation_timeout_seconds must be greater than zero")

    verifier_path = _resolve_evaluation_dataset_path(
        verifier_model,
        evaluation_model_mount_path,
        evaluation_dataset_pvc,
    )
    draft_path = _resolve_evaluation_dataset_path(
        draft_model_path,
        evaluation_model_mount_path,
        evaluation_dataset_pvc,
    )
    if not os.path.isdir(verifier_path):
        raise FileNotFoundError(f"Verifier model directory does not exist: {verifier_path}")
    if not os.path.isdir(draft_path):
        raise FileNotFoundError(f"Draft model directory does not exist: {draft_path}")

    methods = {"eagle3": "eagle3", "dflash": "dflash", "mtp": "mtp", "peagle": "eagle3"}
    method = methods.get(evaluation_speculator_type.lower())
    if method is None:
        raise ValueError(f"Unsupported evaluation_speculator_type: {evaluation_speculator_type}")

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    endpoint = f"http://127.0.0.1:{evaluation_port}"
    server_process = None

    def stop_server() -> None:
        nonlocal server_process
        if server_process is None:
            return
        if server_process.poll() is None:
            try:
                os.killpg(server_process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(server_process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                server_process.wait()
        server_process = None

    with tempfile.TemporaryDirectory(prefix="speculator-evaluate-") as temp_dir:
        work_dir = Path(temp_dir)
        upstream_dir = work_dir / "scripts" / "evaluate"
        upstream_script = _download_upstream_evaluator(str(upstream_dir))
        local_dataset_path, is_local_dataset, local_request_count = _prepare_local_evaluation_dataset(
            evaluation_dataset_uri,
            evaluation_dataset_mount_path,
            evaluation_dataset_pvc,
            evaluation_max_samples,
            evaluation_sample_seed,
            str(work_dir / "evaluation_prompts.jsonl"),
        )
        subsets = "local" if is_local_dataset else "HumanEval"
        results_dir = work_dir / "upstream-results"
        command = [
            "vllm",
            "serve",
            verifier_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(evaluation_port),
            "--gpu-memory-utilization",
            str(evaluation_gpu_memory_utilization),
            "--speculative-config",
            json.dumps(
                {
                    "method": method,
                    "model": draft_path,
                    "draft_tensor_parallel_size": 1,
                    "num_speculative_tokens": evaluation_num_speculative_tokens,
                }
            ),
        ]
        evaluate_command = _build_upstream_evaluate_command(
            sys.executable,
            upstream_script,
            f"{endpoint}/v1",
            local_dataset_path,
            subsets,
            str(results_dir),
            local_request_count if is_local_dataset else evaluation_max_samples,
            evaluation_max_tokens,
            evaluation_temperature,
        )

        try:
            server_process = subprocess.Popen(command, start_new_session=True)
            deadline = time.monotonic() + evaluation_startup_timeout_seconds
            while time.monotonic() < deadline:
                if server_process.poll() is not None:
                    raise RuntimeError(f"Speculative vLLM exited during startup with code {server_process.returncode}")
                try:
                    response = requests.get(f"{endpoint}/health", timeout=5)
                    if response.ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError(
                    f"Speculative vLLM did not become ready within {evaluation_startup_timeout_seconds} seconds"
                )

            dataset_description = (
                f"local dataset ({local_request_count} prompts)" if is_local_dataset else evaluation_dataset_uri
            )
            print(
                f"Running upstream Speculators evaluation at commit 7883cd44264e6958cfc003963e5cdb895eb53bf1: "
                f"{dataset_description}",
                flush=True,
            )
            evaluator_environment = os.environ.copy()
            compat_path = str(upstream_dir / "compat")
            existing_pythonpath = evaluator_environment.get("PYTHONPATH")
            evaluator_environment["PYTHONPATH"] = (
                compat_path if not existing_pythonpath else f"{compat_path}{os.pathsep}{existing_pythonpath}"
            )
            _run_upstream_evaluate(evaluate_command, evaluator_environment, evaluation_timeout_seconds)
        finally:
            stop_server()

        acceptance_path = results_dir / "acceptance.csv"
        if not acceptance_path.is_file():
            raise RuntimeError(f"Upstream evaluator did not produce {acceptance_path}")
        with acceptance_path.open(newline="") as acceptance_file:
            acceptance_rows = list(csv.DictReader(acceptance_file))
        if not acceptance_rows:
            raise RuntimeError("Upstream evaluator produced an empty acceptance.csv")

        total_drafts = sum(float(row.get("num_drafts") or 0) for row in acceptance_rows)
        total_draft_tokens = sum(float(row.get("num_draft_tokens") or 0) for row in acceptance_rows)
        total_accepted_tokens = sum(float(row.get("num_accepted_tokens") or 0) for row in acceptance_rows)
        if total_drafts > 0:
            output_metrics.log_metric("acceptance_length", 1.0 + total_accepted_tokens / total_drafts)
            output_metrics.log_metric("num_drafts", total_drafts)
        if total_draft_tokens > 0:
            output_metrics.log_metric("acceptance_rate", total_accepted_tokens / total_draft_tokens)
            output_metrics.log_metric("num_draft_tokens", total_draft_tokens)
            output_metrics.log_metric("num_accepted_tokens", total_accepted_tokens)
        for position in range(100):
            column = f"acceptance_at_pos_{position}"
            if not any(column in row for row in acceptance_rows):
                break
            weighted = sum(float(row.get(column) or 0) * float(row.get("num_drafts") or 0) for row in acceptance_rows)
            if total_drafts > 0:
                output_metrics.log_metric(column, weighted / total_drafts)

        performance_path = results_dir / "perf_results.csv"
        if performance_path.is_file():
            with performance_path.open(newline="") as performance_file:
                performance_rows = list(csv.DictReader(performance_file))
            metric_columns = {
                "rps_median": ("max_requests_per_second", max),
                "output_tps_median": ("max_output_tokens_per_second", max),
                "latency_median_s": ("min_latency_median_seconds", min),
                "itl_median_ms": ("min_inter_token_latency_milliseconds", min),
                "ttft_median_ms": ("min_time_to_first_token_milliseconds", min),
            }
            for column, (metric_name, selector) in metric_columns.items():
                values = []
                for row in performance_rows:
                    try:
                        value = float(row.get(column, ""))
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(value):
                        values.append(value)
                if values:
                    output_metrics.log_metric(metric_name, selector(values))

        artifact_path = Path(output_results.path)
        if artifact_path.exists() and artifact_path.is_file():
            artifact_path.unlink()
        shutil.copytree(results_dir, artifact_path, dirs_exist_ok=True)
        output_results.name = "speculator-performance-evaluation"
        output_results.metadata["evaluation_tool"] = "vllm-project/speculators/scripts/evaluate/evaluate.py"
        output_results.metadata["evaluation_tool_revision"] = "7883cd44264e6958cfc003963e5cdb895eb53bf1"
        output_results.metadata["evaluation_mode"] = "sweep"
        output_results.metadata["evaluation_dataset"] = evaluation_dataset_uri
        output_results.metadata["evaluation_subsets"] = subsets
        output_results.metadata["evaluation_max_samples"] = evaluation_max_samples
