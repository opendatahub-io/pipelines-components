"""Unit tests for the parse_and_chunk component."""

import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from kfp import compiler
from kfp_components.components.data_processing.parse_and_chunk import parse_and_chunk

# ---------------------------------------------------------------------------
# KFP compile / signature tests
# ---------------------------------------------------------------------------


def test_component_compiles():
    """Compiler().compile() succeeds and produces a non-empty YAML."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        compiler.Compiler().compile(parse_and_chunk, f.name)
        assert Path(f.name).stat().st_size > 0


def test_component_signature():
    """All 25 parameters present with expected names."""
    spec = parse_and_chunk.component_spec
    input_names = set(spec.inputs.keys())
    expected = {
        "pvc_name",
        "pvc_mount_path",
        "input_path",
        "ray_image",
        "namespace",
        "s3_endpoint",
        "s3_bucket",
        "s3_prefix",
        "s3_secret_name",
        "tokenizer",
        "chunk_max_tokens",
        "num_workers",
        "worker_cpus",
        "worker_memory_gb",
        "head_cpus",
        "head_memory_gb",
        "cpus_per_actor",
        "min_actors",
        "max_actors",
        "batch_size",
        "num_files",
        "timeout_seconds",
        "enable_profiling",
        "verbose",
        "bypass_kueue",
    }
    assert expected == input_names


# ---------------------------------------------------------------------------
# Helpers for logic tests
# ---------------------------------------------------------------------------

# Default kwargs for parse_and_chunk.python_func with only required args.
_DEFAULT_KWARGS = {
    "pvc_name": "test-pvc",
    "pvc_mount_path": "/mnt/data",
    "input_path": "input/pdfs",
    "ray_image": "test-image:latest",
    "namespace": "test-ns",
    "s3_endpoint": "http://minio:9000",
    "s3_bucket": "test-bucket",
}


def _make_rayjob_obj(*, kueue_label=False, job_status="SUCCEEDED", num_workers=2):
    """Build a fake RayJob custom resource dict.

    Args:
        kueue_label: If True, include the kueue.x-k8s.io/queue-name label.
        job_status: Value of status.jobStatus (SUCCEEDED, FAILED, etc.).
        num_workers: Number of ready workers reported in status.
    """
    labels = {}
    if kueue_label:
        labels["kueue.x-k8s.io/queue-name"] = "default"

    return {
        "metadata": {"labels": labels},
        "spec": {
            "rayClusterSpec": {
                "headGroupSpec": {
                    "template": {
                        "spec": {
                            "containers": [{"name": "ray-head"}],
                        },
                    },
                },
                "workerGroupSpecs": [
                    {
                        "template": {
                            "spec": {
                                "containers": [{"name": "ray-worker"}],
                            },
                        },
                    },
                ],
            },
        },
        "status": {
            "rayClusterStatus": {
                "readyWorkerReplicas": str(num_workers),
                "state": "ready",
            },
            "jobStatus": job_status,
        },
    }


def _build_mock_modules():
    """Build mock module hierarchy for codeflare_sdk, kubernetes, and ray.

    Returns a dict suitable for ``mock.patch.dict("sys.modules", ...)``.
    """
    # --- codeflare_sdk ---
    mock_codeflare = mock.MagicMock()

    # --- kubernetes ---
    mock_kubernetes = mock.MagicMock()
    # ``kubernetes.client`` and ``kubernetes.config`` need to be
    # importable as sub-modules *and* as attributes.
    mock_k8s_client = mock.MagicMock()
    mock_k8s_config = mock.MagicMock()
    # ConfigException must be a real exception class so the try/except works.
    mock_k8s_config.ConfigException = type("ConfigException", (Exception,), {})
    mock_kubernetes.client = mock_k8s_client
    mock_kubernetes.config = mock_k8s_config

    # --- ray ---
    mock_ray = mock.MagicMock()
    mock_ray_runtime_env = mock.MagicMock()
    mock_ray.runtime_env = mock_ray_runtime_env

    return {
        "codeflare_sdk": mock_codeflare,
        "kubernetes": mock_kubernetes,
        "kubernetes.client": mock_k8s_client,
        "kubernetes.config": mock_k8s_config,
        "ray": mock_ray,
        "ray.runtime_env": mock_ray_runtime_env,
    }


def _run_component(*, bypass_kueue=False, kueue_label=False, job_status="SUCCEEDED", extra_kwargs=None):
    """Invoke parse_and_chunk.python_func with all external deps mocked.

    Returns:
        (result, mocks_dict) -- result is the return value; mocks_dict has
        the mock objects keyed by name for assertion.
    """
    modules = _build_mock_modules()

    mock_codeflare = modules["codeflare_sdk"]
    mock_k8s_client = modules["kubernetes.client"]
    mock_k8s_config = modules["kubernetes.config"]

    # Configure the mock K8s CustomObjectsApi.
    mock_k8s_api = mock_k8s_client.CustomObjectsApi.return_value
    rayjob_obj = _make_rayjob_obj(kueue_label=kueue_label, job_status=job_status)
    mock_k8s_api.get_namespaced_custom_object.return_value = rayjob_obj

    # RayJob mock.
    mock_job = mock_codeflare.RayJob.return_value

    kwargs = {**_DEFAULT_KWARGS, "bypass_kueue": bypass_kueue}
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    with mock.patch("time.sleep"), mock.patch.dict(sys.modules, modules):
        result = parse_and_chunk.python_func(**kwargs)

    return result, {
        "rayjob_cls": mock_codeflare.RayJob,
        "job": mock_job,
        "k8s_api": mock_k8s_api,
        "cluster_config": mock_codeflare.ManagedClusterConfig,
        "runtime_env": modules["ray.runtime_env"].RuntimeEnv,
        "k8s_config": mock_k8s_config,
    }


# ---------------------------------------------------------------------------
# Logic tests
# ---------------------------------------------------------------------------


def test_rayjob_submitted():
    """RayJob.submit() is called exactly once."""
    _, mocks = _run_component()
    mocks["job"].submit.assert_called_once()


def test_s3_credentials_injected():
    """After submit, K8s patch injects S3 credentials via secretKeyRef."""
    _, mocks = _run_component()
    k8s_api = mocks["k8s_api"]

    # Find the patch call that injects the rayClusterSpec (S3 credentials).
    patch_calls = k8s_api.patch_namespaced_custom_object.call_args_list
    cluster_spec_patch = None
    for call in patch_calls:
        body = call.kwargs.get("body", {})
        if "spec" in body and "rayClusterSpec" in body.get("spec", {}):
            cluster_spec_patch = body
            break

    assert cluster_spec_patch is not None, "No patch call found with rayClusterSpec"

    # Verify the head container has the S3 env vars injected.
    head_containers = cluster_spec_patch["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"]["containers"]
    head_env = head_containers[0].get("env", [])
    env_names = {e["name"] for e in head_env}
    assert "S3_ACCESS_KEY" in env_names
    assert "S3_SECRET_KEY" in env_names

    # Verify secretKeyRef structure.
    for env_var in head_env:
        if env_var["name"] == "S3_ACCESS_KEY":
            assert env_var["valueFrom"]["secretKeyRef"]["key"] == "access_key"
        if env_var["name"] == "S3_SECRET_KEY":
            assert env_var["valueFrom"]["secretKeyRef"]["key"] == "secret_key"


def test_bypass_kueue_removes_label():
    """bypass_kueue=True with kueue label present triggers label removal patch."""
    _, mocks = _run_component(bypass_kueue=True, kueue_label=True)
    k8s_api = mocks["k8s_api"]

    patch_calls = k8s_api.patch_namespaced_custom_object.call_args_list
    label_removal_found = False
    for call in patch_calls:
        body = call.kwargs.get("body", {})
        labels = body.get("metadata", {}).get("labels", {})
        if "kueue.x-k8s.io/queue-name" in labels and labels["kueue.x-k8s.io/queue-name"] is None:
            label_removal_found = True
            break

    assert label_removal_found, "Expected patch to remove kueue label (set to None)"


def test_bypass_kueue_unsuspends_job():
    """bypass_kueue=True triggers a patch with spec.suspend=False."""
    _, mocks = _run_component(bypass_kueue=True, kueue_label=True)
    k8s_api = mocks["k8s_api"]

    patch_calls = k8s_api.patch_namespaced_custom_object.call_args_list
    unsuspend_found = False
    for call in patch_calls:
        body = call.kwargs.get("body", {})
        if body.get("spec", {}).get("suspend") is False:
            unsuspend_found = True
            break

    assert unsuspend_found, "Expected patch with spec.suspend=False"


def test_kueue_not_bypassed_by_default():
    """bypass_kueue=False does not remove kueue label or unsuspend."""
    _, mocks = _run_component(bypass_kueue=False)
    k8s_api = mocks["k8s_api"]

    patch_calls = k8s_api.patch_namespaced_custom_object.call_args_list
    for call in patch_calls:
        body = call.kwargs.get("body", {})

        # No label-removal patch.
        labels = body.get("metadata", {}).get("labels", {})
        assert "kueue.x-k8s.io/queue-name" not in labels, "Unexpected kueue label removal"

        # No unsuspend patch.
        assert body.get("spec", {}).get("suspend") is not False, "Unexpected spec.suspend=False"


def test_returns_s3_uri():
    """Component returns s3://{bucket}/{prefix}."""
    result, _ = _run_component()
    assert result == "s3://test-bucket/chunks"


def test_returns_s3_uri_custom_prefix():
    """Component returns s3://{bucket}/{custom_prefix} when prefix is overridden."""
    result, _ = _run_component(extra_kwargs={"s3_prefix": "my-chunks"})
    assert result == "s3://test-bucket/my-chunks"


def test_job_failure_raises():
    """When jobStatus is FAILED, RuntimeError is raised."""
    with pytest.raises(RuntimeError, match="failed"):
        _run_component(job_status="FAILED")
