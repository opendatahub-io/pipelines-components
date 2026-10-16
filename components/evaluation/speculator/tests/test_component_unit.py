"""Unit tests for the upstream Speculators evaluation component."""

import inspect
import json

from ..component import (
    _build_upstream_evaluate_command,
    _download_upstream_file,
    _extract_evaluation_prompt,
    _prepare_local_evaluation_dataset,
    _resolve_evaluation_dataset_path,
    _run_upstream_evaluate,
    evaluate_speculator,
)


def test_component_function_exists():
    """The component is exposed as a KFP Python function."""
    assert callable(evaluate_speculator)
    assert hasattr(evaluate_speculator, "python_func")


def test_component_has_upstream_evaluation_parameters():
    """The component exposes endpoint, dataset, sweep, and model controls."""
    params = inspect.signature(evaluate_speculator.python_func).parameters
    for name in (
        "output_metrics",
        "output_results",
        "verifier_model",
        "draft_model_path",
        "evaluation_dataset_uri",
        "evaluation_dataset_pvc",
        "evaluation_max_samples",
        "evaluation_max_tokens",
        "evaluation_temperature",
        "evaluation_num_speculative_tokens",
        "evaluation_speculator_type",
        "evaluation_startup_timeout_seconds",
        "evaluation_timeout_seconds",
    ):
        assert name in params


def test_component_defaults_match_upstream_example():
    """The request cap matches the upstream HumanEval example."""
    params = inspect.signature(evaluate_speculator.python_func).parameters
    assert params["evaluation_max_samples"].default == 80
    assert params["evaluation_max_tokens"].default == 256
    assert params["evaluation_temperature"].default == 0.0
    assert params["evaluation_num_speculative_tokens"].default == 5
    assert params["evaluation_timeout_seconds"].default == 7200


def test_local_chat_records_are_converted_to_user_prompts():
    """Local chat JSONL is normalized to the prompt column GuideLLM expects."""
    record = {
        "conversations": [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "latest question"},
        ]
    }

    assert _extract_evaluation_prompt(record) == "latest question"
    assert _extract_evaluation_prompt({"prompt": "plain prompt"}) == "plain prompt"


def test_local_dataset_is_sampled_and_normalized(tmp_path):
    """PVC JSONL is prepared as a deterministic prompt-only input file."""
    source = tmp_path / "ultrachat.jsonl"
    records = [{"conversations": [{"role": "user", "content": f"question {i}"}]} for i in range(10)]
    source.write_text("".join(json.dumps(record) + "\n" for record in records))
    output = tmp_path / "sampled.jsonl"

    dataset, is_local, count = _prepare_local_evaluation_dataset(
        str(source), str(tmp_path), "shared", 3, 42, str(output)
    )

    prompts = [json.loads(line)["prompt"] for line in output.read_text().splitlines()]
    assert dataset == str(output)
    assert is_local
    assert count == 3
    assert prompts == ["question 1", "question 0", "question 4"]


def test_huggingface_dataset_identifier_is_not_resolved_as_a_local_path(tmp_path):
    """Remote dataset IDs pass through to the upstream evaluator unchanged."""
    dataset, is_local, count = _prepare_local_evaluation_dataset(
        "RedHatAI/speculator_benchmarks", str(tmp_path), "shared", 80, 42, str(tmp_path / "local.jsonl")
    )

    assert dataset == "RedHatAI/speculator_benchmarks"
    assert not is_local
    assert count == 0


def test_upstream_file_download_retries_transient_failures(monkeypatch):
    """A transient CDN error is retried before the downloaded file is accepted."""
    import io
    import time
    import urllib.error
    import urllib.request

    attempts = []

    def flaky_urlopen(request, timeout):
        attempts.append(timeout)
        if len(attempts) < 3:
            raise urllib.error.URLError("temporary CDN failure")
        return io.BytesIO(b"upstream evaluator")

    monkeypatch.setattr(urllib.request, "urlopen", flaky_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    contents = _download_upstream_file("https://example.com/evaluate.py", "evaluate.py")

    assert contents == b"upstream evaluator"
    assert attempts == [60, 60, 60]


def test_upstream_file_download_fails_after_retry_limit(monkeypatch):
    """Repeated transient download failures produce a clear component error."""
    import time
    import urllib.error
    import urllib.request

    attempts = []

    def failing_urlopen(request, timeout):
        attempts.append(timeout)
        raise urllib.error.URLError("temporary CDN failure")

    monkeypatch.setattr(urllib.request, "urlopen", failing_urlopen)
    monkeypatch.setattr(time, "sleep", lambda _: None)

    import pytest

    with pytest.raises(RuntimeError, match="Failed to download upstream evaluation file evaluate.py after 3 attempts"):
        _download_upstream_file("https://example.com/evaluate.py", "evaluate.py")

    assert attempts == [60, 60, 60]


def test_upstream_command_runs_sweep_against_vllm_endpoint(tmp_path):
    """The command uses Fynn's endpoint-based full-performance workflow."""
    command = _build_upstream_evaluate_command(
        "python",
        "evaluate.py",
        "http://127.0.0.1:8000/v1",
        "RedHatAI/speculator_benchmarks",
        "HumanEval",
        str(tmp_path),
        80,
        256,
        0.0,
    )

    assert command[:4] == ["python", "evaluate.py", "--target", "http://127.0.0.1:8000/v1"]
    assert command[command.index("--dataset") + 1] == "RedHatAI/speculator_benchmarks"
    assert command[command.index("--subsets") + 1] == "HumanEval"
    assert command[-1] == "sweep"


def test_upstream_evaluation_timeout_raises_clear_error(monkeypatch):
    """A stalled upstream sweep fails with a useful timeout error."""
    import subprocess

    def raise_timeout(command, check, env, timeout):
        assert timeout == 30
        raise subprocess.TimeoutExpired(command, timeout=timeout)

    monkeypatch.setattr(subprocess, "run", raise_timeout)

    import pytest

    with pytest.raises(TimeoutError, match="timed out after 30 seconds"):
        _run_upstream_evaluate(["python", "evaluate.py", "sweep"], {}, 30)


def test_component_uses_upstream_script_and_no_custom_baseline_loop():
    """The component launches one speculative endpoint and delegates evaluation upstream."""
    source = inspect.getsource(evaluate_speculator.python_func)
    assert "_download_upstream_evaluator" in source
    assert "_build_upstream_evaluate_command" in source
    assert "_run_upstream_evaluate" in source
    assert "os.killpg(server_process.pid, signal.SIGKILL)" in source
    assert "server_process.kill()" not in source
    assert "baseline_seconds" not in source
    assert "run_requests" not in source


def test_component_checks_vllm_binary():
    """The component reports a clear error when vLLM is unavailable."""
    source = inspect.getsource(evaluate_speculator.python_func)
    assert 'shutil.which("vllm")' in source
    assert "vllm binary not found" in source


def test_component_resolves_dataset_uri_against_mounted_pvc():
    """Mounted PVC URIs and relative paths resolve under the configured mount."""
    mount_path = "/mnt/persistent"

    assert (
        _resolve_evaluation_dataset_path(
            "models/ultrachat.jsonl",
            mount_path,
            "shared",
        )
        == "/mnt/persistent/models/ultrachat.jsonl"
    )
    assert (
        _resolve_evaluation_dataset_path(
            "pvc://shared/models/ultrachat.jsonl",
            mount_path,
            "shared",
        )
        == "/mnt/persistent/models/ultrachat.jsonl"
    )


def test_dataset_resolver_rejects_mismatched_pvc_claim():
    """PVC URIs must identify the mounted claim rather than another PVC."""
    import pytest

    with pytest.raises(ValueError, match="mounted PVC is 'shared'"):
        _resolve_evaluation_dataset_path("pvc://other/models/data.jsonl", "/mnt/persistent", "shared")


def test_dataset_resolver_rejects_path_traversal(tmp_path):
    """Relative and absolute paths cannot escape the configured mount root."""
    import pytest

    with pytest.raises(ValueError, match="outside mounted PVC"):
        _resolve_evaluation_dataset_path("../outside.jsonl", str(tmp_path), "shared")
    with pytest.raises(ValueError, match="outside mounted PVC"):
        _resolve_evaluation_dataset_path(str(tmp_path.parent / "outside.jsonl"), str(tmp_path), "shared")


def test_dataset_resolver_rejects_symlink_escape(tmp_path):
    """Symlinks are canonicalized before enforcing the mount boundary."""
    import pytest

    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="outside mounted PVC"):
        _resolve_evaluation_dataset_path("escape/data.jsonl", str(tmp_path), "shared")
