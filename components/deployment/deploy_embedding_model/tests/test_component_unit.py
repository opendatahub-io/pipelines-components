"""Unit tests for the deploy_embedding_model component."""

import tempfile
from pathlib import Path
from unittest import mock

import pytest
from kfp import compiler
from kfp_components.components.deployment.deploy_embedding_model import deploy_embedding_model

# ---------------------------------------------------------------------------
# KFP compile / signature tests
# ---------------------------------------------------------------------------


def test_component_compiles():
    """Compiler().compile() succeeds and produces a non-empty YAML."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        compiler.Compiler().compile(deploy_embedding_model, f.name)
        assert Path(f.name).stat().st_size > 0


def test_component_signature():
    """All 12 parameters present with expected names."""
    spec = deploy_embedding_model.component_spec
    input_names = set(spec.inputs)
    expected = {
        "model_name",
        "namespace",
        "serving_runtime_name",
        "runtime_image",
        "min_replicas",
        "max_replicas",
        "cpu_requests",
        "cpu_limits",
        "memory_requests",
        "memory_limits",
        "gpu_count",
        "max_model_len",
    }
    assert expected == input_names


# ---------------------------------------------------------------------------
# Helpers for logic tests
# ---------------------------------------------------------------------------


def _api_404():
    """Create a kubernetes ApiException with status 404 (lazy import)."""
    from kubernetes.client.rest import ApiException

    return ApiException(status=404)


DEFAULT_KWARGS = {
    "model_name": "org/test-model",
    "namespace": "test-ns",
    "serving_runtime_name": "embedding-runtime",
    "runtime_image": "registry.example.com/vllm:latest",
    "min_replicas": 1,
    "max_replicas": 1,
    "cpu_requests": "2",
    "cpu_limits": "4",
    "memory_requests": "4Gi",
    "memory_limits": "8Gi",
    "gpu_count": 1,
    "max_model_len": 512,
}


def _ready_isvc(*args, **kwargs):
    """Return an InferenceService object with Ready status."""
    return {
        "status": {
            "conditions": [{"type": "Ready", "status": "True"}],
            "url": "http://embedding.test.svc",
        }
    }


def _not_ready_isvc(*args, **kwargs):
    """Return an InferenceService object that is not ready."""
    return {
        "status": {
            "conditions": [{"type": "Ready", "status": "False"}],
        }
    }


def _run_component(custom_api_mock, **overrides):
    """Invoke deploy_embedding_model.python_func with Kubernetes mocked.

    Args:
        custom_api_mock: A MagicMock for CustomObjectsApi().
        **overrides: Override any default kwargs.

    Returns:
        The result of the component function.
    """
    kwargs = {**DEFAULT_KWARGS, **overrides}

    with (
        mock.patch("kubernetes.config.load_incluster_config"),
        mock.patch("kubernetes.client.CustomObjectsApi", return_value=custom_api_mock),
        mock.patch("time.sleep"),
    ):
        return deploy_embedding_model.python_func(**kwargs)


# ---------------------------------------------------------------------------
# Logic tests
# ---------------------------------------------------------------------------


def test_creates_serving_runtime_when_not_found():
    """When ServingRuntime does not exist (404), create is called."""
    api = mock.MagicMock()

    # First get (ServingRuntime) raises 404; second get (InferenceService) also 404
    api.get_namespaced_custom_object.side_effect = [
        _api_404(),  # ServingRuntime lookup
        _api_404(),  # InferenceService lookup
        _ready_isvc(),  # readiness poll
    ]

    result = _run_component(api)

    # ServingRuntime created (first create call)
    create_calls = api.create_namespaced_custom_object.call_args_list
    sr_call = create_calls[0]
    assert sr_call[1]["group"] == "serving.kserve.io"
    assert sr_call[1]["plural"] == "servingruntimes"

    assert result == "http://embedding.test.svc"


def test_patches_serving_runtime_when_exists():
    """When ServingRuntime already exists, patch is called."""
    api = mock.MagicMock()

    api.get_namespaced_custom_object.side_effect = [
        {"metadata": {"name": "embedding-runtime"}},  # ServingRuntime exists
        _api_404(),  # InferenceService not found
        _ready_isvc(),  # readiness poll
    ]

    _run_component(api)

    patch_calls = api.patch_namespaced_custom_object.call_args_list
    sr_patch = patch_calls[0]
    assert sr_patch[1]["group"] == "serving.kserve.io"
    assert sr_patch[1]["plural"] == "servingruntimes"


def test_creates_isvc_when_not_found():
    """When InferenceService does not exist (404), create is called."""
    api = mock.MagicMock()

    api.get_namespaced_custom_object.side_effect = [
        _api_404(),  # ServingRuntime 404
        _api_404(),  # InferenceService 404
        _ready_isvc(),  # readiness poll
    ]

    _run_component(api)

    create_calls = api.create_namespaced_custom_object.call_args_list
    isvc_call = create_calls[1]
    assert isvc_call[1]["group"] == "serving.kserve.io"
    assert isvc_call[1]["plural"] == "inferenceservices"


def test_patches_isvc_when_exists():
    """When InferenceService already exists, patch is called."""
    api = mock.MagicMock()

    api.get_namespaced_custom_object.side_effect = [
        _api_404(),  # ServingRuntime 404
        {"metadata": {"name": "test-model"}},  # InferenceService exists
        _ready_isvc(),  # readiness poll
    ]

    _run_component(api)

    patch_calls = api.patch_namespaced_custom_object.call_args_list
    isvc_patch = patch_calls[0]
    assert isvc_patch[1]["group"] == "serving.kserve.io"
    assert isvc_patch[1]["plural"] == "inferenceservices"


def test_waits_for_ready_and_returns_url():
    """After CRs created, polls until Ready and returns the URL."""
    api = mock.MagicMock()

    api.get_namespaced_custom_object.side_effect = [
        _api_404(),  # ServingRuntime
        _api_404(),  # InferenceService
        _not_ready_isvc(),  # poll 1: not ready
        _not_ready_isvc(),  # poll 2: not ready
        _ready_isvc(),  # poll 3: ready
    ]

    result = _run_component(api)
    assert result == "http://embedding.test.svc"


def test_timeout_raises():
    """When InferenceService never becomes Ready, TimeoutError is raised."""
    api = mock.MagicMock()

    api.get_namespaced_custom_object.side_effect = [
        _api_404(),  # ServingRuntime
        _api_404(),  # InferenceService
    ] + [_not_ready_isvc()] * 60  # 60 poll iterations, all not ready

    with pytest.raises(TimeoutError, match="did not become ready"):
        _run_component(api)


def test_vllm_embedding_args():
    """The created ServingRuntime body contains vLLM embedding args."""
    api = mock.MagicMock()

    api.get_namespaced_custom_object.side_effect = [
        _api_404(),  # ServingRuntime 404 -> create
        _api_404(),  # InferenceService 404 -> create
        _ready_isvc(),
    ]

    _run_component(api)

    # Capture the body from the ServingRuntime create call
    create_calls = api.create_namespaced_custom_object.call_args_list
    sr_body = create_calls[0][1]["body"]
    container_args = sr_body["spec"]["containers"][0]["args"]

    assert "--task" in container_args
    assert "embedding" in container_args
    assert "--port" in container_args
    assert "8080" in container_args
    assert "--dtype" in container_args
    assert "float16" in container_args
    assert "--max-model-len" in container_args
    assert "512" in container_args


def test_isvc_name_derived_correctly():
    """model_name='org/Model_Name.v1' -> isvc_name='model-name-v1'."""
    api = mock.MagicMock()

    api.get_namespaced_custom_object.side_effect = [
        _api_404(),  # ServingRuntime
        _api_404(),  # InferenceService
        _ready_isvc(),
    ]

    _run_component(api, model_name="org/Model_Name.v1")

    # The InferenceService create call should use the derived name
    create_calls = api.create_namespaced_custom_object.call_args_list
    isvc_body = create_calls[1][1]["body"]
    assert isvc_body["metadata"]["name"] == "model-name-v1"
