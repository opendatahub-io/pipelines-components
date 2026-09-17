"""Unit tests for the speculative decoding serving component."""

import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from kfp import compiler

from ..component import speculative_decoding_serving


def _api_404():
    """Create a Kubernetes 404 exception."""
    from kubernetes.client.rest import ApiException

    return ApiException(status=404)


DEFAULT_KWARGS = {
    "namespace": "test-ns",
    "endpoint_name": "qwen3-speculative",
    "model_cache_pvc": "model-cache",
    "verifier_model_dir": "models/qwen3-0.6b",
    "draft_model_dir": "speculator/run-01/checkpoint_best",
    "runtime_image": "registry.example.com/vllm:speculative",
    "hardware_profile_name": "gpu-profile",
    "min_replicas": 1,
    "max_replicas": 1,
    "gpu_count": 1,
    "draft_tensor_parallel_size": 1,
    "max_model_len": 4096,
    "num_speculative_tokens": 3,
    "cpu_requests": "2",
    "memory_requests": "8Gi",
    "cpu_limits": "2",
    "memory_limits": "8Gi",
    "gpu_memory_utilization": 0.9,
    "max_num_seqs": 16,
    "trust_remote_code": False,
    "enable_auth": False,
    "enable_external_route": False,
}


def test_route_403_raises_informative_runtime_error():
    """A 403 on Route creation raises a RuntimeError pointing to rbac.yaml."""
    from kubernetes.client.rest import ApiException

    api = mock.MagicMock()
    api.get_namespaced_custom_object.side_effect = [
        {"metadata": {"resourceVersion": "1"}},  # HardwareProfile
        _api_404(),  # ServingRuntime upsert check → create
        _api_404(),  # InferenceService upsert check → create
        _ready_isvc(),  # readiness poll
        ApiException(status=403),  # Route upsert GET → 403
    ]
    api.create_namespaced_custom_object.return_value = {}

    with pytest.raises(RuntimeError, match="rbac.yaml"):
        _run_component(api, enable_external_route=True)


def test_route_non_403_api_exception_propagates():
    """A non-403 error during Route creation is not swallowed."""
    from kubernetes.client.rest import ApiException

    api = mock.MagicMock()
    api.get_namespaced_custom_object.side_effect = [
        {"metadata": {"resourceVersion": "1"}},
        _api_404(),
        _api_404(),
        _ready_isvc(),
        ApiException(status=500),  # Route upsert GET → 500
    ]
    api.create_namespaced_custom_object.return_value = {}

    with pytest.raises(ApiException):
        _run_component(api, enable_external_route=True)


def _ready_isvc(url="http://qwen3-speculative.test.svc"):
    """Return a ready InferenceService status."""
    return {"status": {"conditions": [{"type": "Ready", "status": "True"}], "url": url}}


def _not_ready_isvc():
    """Return a non-ready InferenceService status."""
    return {"status": {"conditions": [{"type": "Ready", "status": "False"}]}}


def _run_component(api, **overrides):
    """Run the component with Kubernetes and polling mocked."""
    kwargs = {**DEFAULT_KWARGS, **overrides}
    with (
        mock.patch("kubernetes.config.load_incluster_config"),
        mock.patch("kubernetes.client.CustomObjectsApi", return_value=api),
        mock.patch("time.sleep"),
    ):
        return speculative_decoding_serving.python_func(**kwargs)


def test_component_compiles():
    """The component compiles to a non-empty KFP specification."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as output:
        compiler.Compiler().compile(speculative_decoding_serving, output.name)
        assert Path(output.name).stat().st_size > 0


def test_component_signature():
    """The public component signature exposes the serving contract."""
    expected = set(DEFAULT_KWARGS)
    assert set(speculative_decoding_serving.component_spec.inputs) == expected


def test_external_route_creates_route_and_returns_https_url():
    """When enable_external_route=True the component creates an OpenShift Route and returns its HTTPS URL."""
    api = mock.MagicMock()
    api.get_namespaced_custom_object.side_effect = [
        {"metadata": {"resourceVersion": "1"}},  # HardwareProfile
        _api_404(),  # ServingRuntime upsert check → create
        _api_404(),  # InferenceService upsert check → create
        _ready_isvc(),  # readiness poll
        _api_404(),  # Route upsert check → create
        {"spec": {"host": "my-model.apps.example.com"}},  # Route host fetch
    ]
    api.create_namespaced_custom_object.return_value = {}

    result = _run_component(api, enable_external_route=True)

    assert result == "https://my-model.apps.example.com/v1"
    route_create = [
        call
        for call in api.create_namespaced_custom_object.call_args_list
        if call.kwargs.get("plural") == "routes" or call.kwargs.get("group") == "route.openshift.io"
    ]
    assert len(route_create) == 1, "Expected exactly one Route to be created"
    route_body = route_create[0].kwargs["body"]
    assert route_body["spec"]["to"]["name"] == "qwen3-speculative-predictor"
    assert route_body["spec"]["tls"]["termination"] == "edge"
    assert route_body["spec"]["port"]["targetPort"] == "http"


def test_creates_resources_with_speculative_config_and_shared_pvc():
    """Created resources configure vLLM and mount both models from the PVC."""
    api = mock.MagicMock()
    api_404 = _api_404()
    readiness_calls = {"count": 0}

    def get_object(*, plural, **_):
        if plural == "hardwareprofiles":
            return {"metadata": {"resourceVersion": "42"}}
        if plural == "servingruntimes":
            raise api_404
        if plural == "inferenceservices":
            readiness_calls["count"] += 1
            if readiness_calls["count"] == 1:
                raise api_404
            return _ready_isvc()
        raise AssertionError(f"Unexpected resource lookup: {plural}")

    api.get_namespaced_custom_object.side_effect = get_object
    api.create_namespaced_custom_object.return_value = {}

    result = _run_component(api)

    assert result == "http://qwen3-speculative.test.svc/v1"
    creates = api.create_namespaced_custom_object.call_args_list
    runtime_body = creates[0].kwargs["body"]
    assert runtime_body["spec"]["containers"][0]["command"] == [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
    ]
    runtime_env = runtime_body["spec"]["containers"][0]["env"]
    assert {entry["name"]: entry["value"] for entry in runtime_env}["USER"] == "default"
    assert {entry["name"]: entry["value"] for entry in runtime_env}["TORCHINDUCTOR_CACHE_DIR"] == "/tmp/torchinductor"
    runtime_args = runtime_body["spec"]["containers"][0]["args"]
    config_index = runtime_args.index("--speculative-config")
    assert json.loads(runtime_args[config_index + 1]) == {
        "method": "eagle3",
        "model": "/mnt/models/speculator/run-01/checkpoint_best",
        "draft_tensor_parallel_size": 1,
        "num_speculative_tokens": 3,
    }
    assert "--model=/mnt/models/models/qwen3-0.6b" in runtime_args
    assert "--trust-remote-code" not in runtime_args

    isvc_body = creates[1].kwargs["body"]
    predictor = isvc_body["spec"]["predictor"]
    assert predictor["model"]["storageUri"] == "pvc://model-cache"
    assert isvc_body["metadata"]["annotations"]["security.opendatahub.io/enable-auth"] == "false"
    assert isvc_body["metadata"]["annotations"]["opendatahub.io/hardware-profile-resource-version"] == "42"


def test_patches_existing_resources_and_can_enable_auth():
    """Existing resources are patched in place and auth is configurable."""
    api = mock.MagicMock()
    api.get_namespaced_custom_object.side_effect = [
        {"metadata": {"resourceVersion": "7"}},
        {"metadata": {"name": "qwen3-speculative-runtime"}},
        {"metadata": {"name": "qwen3-speculative"}},
        _ready_isvc("http://qwen3.test.svc/v1"),
    ]
    api.patch_namespaced_custom_object.return_value = {}

    result = _run_component(api, enable_auth=True)

    assert result == "http://qwen3.test.svc/v1"
    patches = api.patch_namespaced_custom_object.call_args_list
    assert [call.kwargs["plural"] for call in patches] == ["servingruntimes", "inferenceservices"]
    isvc_body = patches[1].kwargs["body"]
    assert isvc_body["metadata"]["annotations"]["security.opendatahub.io/enable-auth"] == "true"
    api.create_namespaced_custom_object.assert_not_called()


def test_hardware_profile_permission_error_raises():
    """A 403 on the HardwareProfile lookup propagates instead of being silently ignored."""
    from kubernetes.client.rest import ApiException

    api = mock.MagicMock()
    api.get_namespaced_custom_object.side_effect = ApiException(status=403)

    with pytest.raises(ApiException):
        _run_component(api)


def test_trust_remote_code_opt_in_adds_flag():
    """--trust-remote-code is only passed when explicitly requested."""
    api = mock.MagicMock()
    api.get_namespaced_custom_object.side_effect = [
        {"metadata": {"resourceVersion": "1"}},  # HardwareProfile
        _api_404(),  # ServingRuntime upsert check → create
        _api_404(),  # InferenceService upsert check → create
        _ready_isvc(),  # readiness poll
    ]
    api.create_namespaced_custom_object.return_value = {}

    _run_component(api, trust_remote_code=True)

    runtime_body = api.create_namespaced_custom_object.call_args_list[0].kwargs["body"]
    assert "--trust-remote-code" in runtime_body["spec"]["containers"][0]["args"]


def test_invalid_model_directory_is_rejected():
    """Model paths cannot escape the PVC directory."""
    api = mock.MagicMock()

    for draft_model_dir in ("../checkpoint_best", "/absolute/checkpoint_best"):
        with pytest.raises(ValueError, match="draft_model_dir"):
            _run_component(api, draft_model_dir=draft_model_dir)


def test_invalid_draft_tensor_parallel_size_is_rejected():
    """Draft tensor parallelism must be one GPU or match the target tensor parallelism."""
    api = mock.MagicMock()

    with pytest.raises(ValueError, match="draft_tensor_parallel_size"):
        _run_component(api, gpu_count=2, draft_tensor_parallel_size=3)


def test_timeout_raises():
    """A service that never becomes ready raises a timeout."""
    api = mock.MagicMock()
    api_404 = _api_404()
    api.get_namespaced_custom_object.side_effect = [
        {"metadata": {"resourceVersion": "7"}},
        api_404,
        api_404,
    ] + [_not_ready_isvc()] * 60
    api.create_namespaced_custom_object.return_value = {}

    with pytest.raises(TimeoutError, match="did not become ready"):
        _run_component(api)
