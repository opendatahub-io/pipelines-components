"""Unit tests for the speculative decoding serving component."""

import copy
import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest
from kfp import compiler

from ..component import speculative_decoding_serving

CONFIG_HASH_ANNOTATION = "pipelines.kubeflow.org/serving-config-hash"
HP_ANNOTATIONS = (
    "opendatahub.io/hardware-profile-name",
    "opendatahub.io/hardware-profile-namespace",
    "opendatahub.io/hardware-profile-resource-version",
)
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


def _api_error(status):
    """Create a Kubernetes API exception."""
    from kubernetes.client.rest import ApiException

    return ApiException(status=status)


def _ready_isvc(url="http://qwen3-speculative.test.svc"):
    """Return a ready InferenceService status."""
    return {"status": {"conditions": [{"type": "Ready", "status": "True"}], "url": url, "address": {"url": url}}}


def _create_api(*poll_responses, hardware_profile=True):
    """Mock profile lookup, initial resource creation, and subsequent polling."""
    api = mock.MagicMock()
    responses = [{"metadata": {"resourceVersion": "7"}}] if hardware_profile else []
    api.get_namespaced_custom_object.side_effect = [
        *responses,
        _api_error(404),
        _api_error(404),
        *(poll_responses or (_ready_isvc(),)),
    ]
    api.create_namespaced_custom_object.return_value = {}
    return api


def _resource_body(api, plural):
    """Return the last submitted resource body without reconstructing production logic."""
    for call in reversed(api.mock_calls):
        if call[0] in ("create_namespaced_custom_object", "patch_namespaced_custom_object"):
            if call.kwargs.get("plural") == plural:
                return call.kwargs["body"]
    raise AssertionError(f"No submitted {plural} body")


def _deployment(
    api,
    config_hash=None,
    generation=2,
    observed_generation=2,
    desired_replicas=1,
    replicas=1,
    updated_replicas=1,
    available_replicas=1,
    conditions=None,
):
    """Return an SDK Deployment model with configurable rollout status."""
    from kubernetes import client

    if config_hash is None:
        predictor = _resource_body(api, "inferenceservices")["spec"]["predictor"]
        config_hash = predictor.get("annotations", {}).get(CONFIG_HASH_ANNOTATION, "missing")
    runtime_container = _resource_body(api, "servingruntimes")["spec"]["containers"][0]
    endpoint_name = _resource_body(api, "inferenceservices")["metadata"]["name"]
    container = client.V1Container(
        name="kserve-container",
        image=runtime_container["image"],
        command=runtime_container["command"],
        args=[arg.replace("{{.Name}}", endpoint_name) for arg in runtime_container["args"]],
    )
    return client.V1Deployment(
        metadata=client.V1ObjectMeta(generation=generation),
        spec=client.V1DeploymentSpec(
            replicas=desired_replicas,
            selector=client.V1LabelSelector(match_labels={"app": "test"}),
            template=client.V1PodTemplateSpec(
                metadata=client.V1ObjectMeta(annotations={CONFIG_HASH_ANNOTATION: config_hash}),
                spec=client.V1PodSpec(containers=[container]),
            ),
        ),
        status=client.V1DeploymentStatus(
            observed_generation=observed_generation,
            replicas=replicas,
            updated_replicas=updated_replicas,
            available_replicas=available_replicas,
            conditions=conditions,
        ),
    )


def _route(auth=False, admitted=True, host="my-model.apps.example.com"):
    """Return the platform Route for the requested authentication setting."""
    return {
        "spec": {
            "host": host,
            "to": {"kind": "Service", "name": "qwen3-speculative-predictor"},
            "port": {"targetPort": "https" if auth else "http"},
            "tls": {"termination": "reencrypt" if auth else "edge"},
        },
        "status": {
            "ingress": [{"host": host, "conditions": [{"type": "Admitted", "status": "True" if admitted else "False"}]}]
        },
    }


def _run_component(api, apps_api=None, **overrides):
    """Run the component with an advancing virtual clock and Kubernetes mocked."""
    if apps_api is None:
        apps_api = mock.MagicMock()
        apps_api.read_namespaced_deployment.side_effect = lambda **_: _deployment(api)
    clock = {"now": 0.0}

    def advance(seconds):
        clock["now"] += seconds

    with (
        mock.patch("kubernetes.config.load_incluster_config"),
        mock.patch("kubernetes.client.CustomObjectsApi", return_value=api),
        mock.patch("kubernetes.client.AppsV1Api", return_value=apps_api),
        mock.patch("time.monotonic", side_effect=lambda: clock["now"]),
        mock.patch("time.sleep", side_effect=advance),
    ):
        return speculative_decoding_serving.python_func(**{**DEFAULT_KWARGS, **overrides})


def test_component_compiles():
    """The component compiles to a non-empty KFP specification."""
    with tempfile.TemporaryDirectory() as output_dir:
        output = Path(output_dir) / "component.yaml"
        compiler.Compiler().compile(speculative_decoding_serving, str(output))
        assert output.stat().st_size > 0


def test_component_signature():
    """The public component signature remains unchanged."""
    assert set(speculative_decoding_serving.component_spec.inputs) == set(DEFAULT_KWARGS)


def test_creates_resources_with_speculative_config_and_shared_pvc():
    """Created resources preserve vLLM arguments and the shared PVC."""
    api = _create_api()

    assert _run_component(api) == "http://qwen3-speculative.test.svc/v1"

    runtime = _resource_body(api, "servingruntimes")
    container = runtime["spec"]["containers"][0]
    assert container["command"] == ["python3", "-m", "vllm.entrypoints.openai.api_server"]
    env = {entry["name"]: entry["value"] for entry in container["env"]}
    assert env["USER"] == "default"
    assert env["TORCHINDUCTOR_CACHE_DIR"] == "/tmp/torchinductor"
    args = container["args"]
    assert json.loads(args[args.index("--speculative-config") + 1]) == {
        "method": "eagle3",
        "model": "/mnt/models/speculator/run-01/checkpoint_best",
        "draft_tensor_parallel_size": 1,
        "num_speculative_tokens": 3,
    }
    assert "--model=/mnt/models/models/qwen3-0.6b" in args
    assert "--trust-remote-code" not in args
    isvc = _resource_body(api, "inferenceservices")
    assert isvc["spec"]["predictor"]["model"]["storageUri"] == "pvc://model-cache"
    assert isvc["metadata"]["annotations"]["security.opendatahub.io/enable-auth"] == "false"
    assert isvc["metadata"]["annotations"][HP_ANNOTATIONS[2]] == "7"
    assert isvc["metadata"]["labels"]["networking.kserve.io/visibility"] == "cluster-local"
    assert [call.kwargs["plural"] for call in api.create_namespaced_custom_object.call_args_list] == [
        "servingruntimes",
        "inferenceservices",
    ]


def test_updates_allow_gpu_reuse():
    """Updates release one old replica before allocating replacement GPU resources."""
    api = _create_api()
    _run_component(api)
    assert _resource_body(api, "inferenceservices")["spec"]["predictor"]["deploymentStrategy"] == {
        "type": "RollingUpdate",
        "rollingUpdate": {"maxSurge": 0, "maxUnavailable": 1},
    }


def test_patches_existing_resources_and_can_enable_auth():
    """Existing serving resources are patched, with authentication still configurable."""
    api = mock.MagicMock()
    api.get_namespaced_custom_object.side_effect = [
        {"metadata": {"resourceVersion": "7"}},
        {"metadata": {"name": "qwen3-speculative-runtime"}},
        {"metadata": {"name": "qwen3-speculative"}},
        _ready_isvc("http://qwen3.test.svc/v1"),
    ]
    assert _run_component(api, enable_auth=True) == "http://qwen3.test.svc/v1"
    assert [call.kwargs["plural"] for call in api.patch_namespaced_custom_object.call_args_list] == [
        "servingruntimes",
        "inferenceservices",
    ]
    assert _resource_body(api, "inferenceservices")["metadata"]["annotations"][
        "security.opendatahub.io/enable-auth"
    ] == ("true")
    api.create_namespaced_custom_object.assert_not_called()


@pytest.mark.parametrize("enable_auth", [False, True])
def test_external_route_is_read_without_route_writes(enable_auth):
    """External serving uses the admitted platform Route for either authentication mode."""
    api = _create_api(_ready_isvc(), _route(auth=enable_auth))
    assert _run_component(api, enable_external_route=True, enable_auth=enable_auth) == (
        "https://my-model.apps.example.com/v1"
    )
    assert (
        _resource_body(api, "inferenceservices")["metadata"]["labels"]["networking.kserve.io/visibility"] == "exposed"
    )
    for method in (api.create_namespaced_custom_object, api.patch_namespaced_custom_object):
        assert all(call.kwargs["group"] != "route.openshift.io" for call in method.call_args_list)


@pytest.mark.parametrize("pending_route", [_api_error(404), _route(admitted=False), _route(host="")])
def test_waits_for_route_creation_host_and_admission(pending_route):
    """A missing or unadmitted Route is retried within the readiness deadline."""
    api = _create_api(_ready_isvc(), pending_route, _ready_isvc(), _route())
    assert _run_component(api, enable_external_route=True) == "https://my-model.apps.example.com/v1"
    assert sum(call.kwargs["plural"] == "routes" for call in api.get_namespaced_custom_object.call_args_list) == 2


@pytest.mark.parametrize("field", ["port", "tls", "to"])
def test_waits_for_authenticated_route_configuration(field):
    """An admitted Route with stale authentication or backend settings is not returned."""
    stale_route = _route(auth=True)
    stale_route["spec"][field] = _route(auth=False)["spec"][field]
    if field == "to":
        stale_route["spec"]["to"]["name"] = "another-predictor"
    api = _create_api(_ready_isvc(), stale_route, _ready_isvc(), _route(auth=True))
    assert _run_component(api, enable_external_route=True, enable_auth=True) == "https://my-model.apps.example.com/v1"
    assert sum(call.kwargs["plural"] == "routes" for call in api.get_namespaced_custom_object.call_args_list) == 2


def test_route_permission_error_is_actionable():
    """Route read failures ask for get permission rather than obsolete Route write RBAC."""
    api = _create_api(_ready_isvc(), _api_error(403))
    with pytest.raises(RuntimeError, match=r"get.*routes\.route\.openshift\.io"):
        _run_component(api, enable_external_route=True)


def test_route_other_api_errors_propagate():
    """Unexpected Route API failures remain visible."""
    from kubernetes.client.rest import ApiException

    api = _create_api(_ready_isvc(), _api_error(500))
    with pytest.raises(ApiException) as exc:
        _run_component(api, enable_external_route=True)
    assert exc.value.status == 500


def test_internal_endpoint_prefers_cluster_address():
    """Disabling external access returns the internal address despite an old external URL."""
    isvc = _ready_isvc("https://old-route.apps.example.com")
    isvc["status"]["address"] = {"url": "http://qwen3-speculative.test.svc"}
    api = _create_api(isvc)
    assert _run_component(api) == "http://qwen3-speculative.test.svc/v1"
    assert all(call.kwargs["plural"] != "routes" for call in api.get_namespaced_custom_object.call_args_list)


def test_waits_for_internal_url_publication():
    """A ready service without a published URL is retried."""
    api = _create_api(_ready_isvc(""), _ready_isvc())
    assert _run_component(api) == "http://qwen3-speculative.test.svc/v1"


@pytest.mark.parametrize("address", [None, {}, {"url": ""}, {"url": "/not-an-address"}])
def test_internal_endpoint_waits_instead_of_returning_stale_external_url(address):
    """Internal visibility must wait for its address rather than return an old external URL."""
    isvc = _ready_isvc("https://old-route.apps.example.com")
    if address is None:
        isvc["status"].pop("address")
    else:
        isvc["status"]["address"] = address
    api = _create_api(isvc, _ready_isvc())
    assert _run_component(api) == "http://qwen3-speculative.test.svc/v1"
    assert (
        sum(call.kwargs["plural"] == "inferenceservices" for call in api.get_namespaced_custom_object.call_args_list)
        == 3
    )


def test_missing_internal_address_times_out_despite_external_url():
    """An external URL cannot satisfy internal publication even when the predictor is ready."""
    isvc = _ready_isvc("https://old-route.apps.example.com")
    isvc["status"].pop("address")
    api = _create_api(*[isvc] * 60)
    with pytest.raises(TimeoutError, match="waiting for InferenceService internal URL"):
        _run_component(api)


@pytest.mark.parametrize(
    ("field", "stale_value"),
    [
        ("image", "registry.example.com/vllm:old"),
        ("command", ["python3", "-m", "old.server"]),
        ("args", ["--model=/mnt/models/old"]),
        ("args", None),
        ("containers", []),
    ],
)
def test_matching_hash_waits_for_requested_runtime_container(field, stale_value):
    """A new ISVC hash can reach the Deployment before the cached runtime configuration."""
    api = _create_api(_ready_isvc(), _ready_isvc())
    apps_api = mock.MagicMock()
    attempts = {"count": 0}

    def read_deployment(**_):
        attempts["count"] += 1
        deployment = _deployment(api)
        if attempts["count"] == 1:
            if field == "containers":
                deployment.spec.template.spec.containers = stale_value
            else:
                setattr(deployment.spec.template.spec.containers[0], field, stale_value)
        return deployment

    apps_api.read_namespaced_deployment.side_effect = read_deployment
    assert _run_component(api, apps_api=apps_api) == "http://qwen3-speculative.test.svc/v1"
    assert apps_api.read_namespaced_deployment.call_count == 2


def test_matching_hash_with_stale_runtime_times_out():
    """A matching annotation cannot hide a runtime that never reaches the requested configuration."""
    api = _create_api(*[_ready_isvc()] * 60)
    apps_api = mock.MagicMock()

    def read_deployment(**_):
        deployment = _deployment(api)
        deployment.spec.template.spec.containers[0].image = "registry.example.com/vllm:old"
        return deployment

    apps_api.read_namespaced_deployment.side_effect = read_deployment
    with pytest.raises(TimeoutError, match="runtime matches=False"):
        _run_component(api, apps_api=apps_api)


@pytest.mark.parametrize(
    "pending_status",
    [
        {"config_hash": "old-config"},
        {"observed_generation": 1},
        {"updated_replicas": 0},
        {"available_replicas": 0},
        {"replicas": 2},
        {"desired_replicas": 0, "replicas": 0, "updated_replicas": 0, "available_replicas": 0},
    ],
)
def test_stale_ready_does_not_complete_an_unfinished_rollout(pending_status):
    """Old readiness cannot hide stale configuration, status, replicas, or unavailable pods."""
    api = _create_api(_ready_isvc(), _ready_isvc())
    apps_api = mock.MagicMock()
    attempts = {"count": 0}

    def read_deployment(**_):
        attempts["count"] += 1
        return _deployment(api, **pending_status) if attempts["count"] == 1 else _deployment(api)

    apps_api.read_namespaced_deployment.side_effect = read_deployment
    assert _run_component(api, apps_api=apps_api) == "http://qwen3-speculative.test.svc/v1"
    assert apps_api.read_namespaced_deployment.call_count == 2


def test_deployment_not_found_is_retried():
    """KServe can create the predictor Deployment after the first poll."""
    api = _create_api(_ready_isvc(), _ready_isvc())
    apps_api = mock.MagicMock()
    responses = iter([_api_error(404), None])

    def read_deployment(**_):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return _deployment(api)

    apps_api.read_namespaced_deployment.side_effect = read_deployment
    assert _run_component(api, apps_api=apps_api) == "http://qwen3-speculative.test.svc/v1"
    assert apps_api.read_namespaced_deployment.call_count == 2


def test_inferenceservice_not_found_during_poll_is_retried():
    """A transient missing service during polling is not mistaken for a terminal failure."""
    api = _create_api(_api_error(404), _ready_isvc())
    assert _run_component(api) == "http://qwen3-speculative.test.svc/v1"


def test_deployment_read_permission_error_is_actionable():
    """The runner needs read permission for the predictor Deployment."""
    api = _create_api()
    apps_api = mock.MagicMock()
    apps_api.read_namespaced_deployment.side_effect = _api_error(403)
    with pytest.raises(RuntimeError, match=r"get.*deployments\.apps"):
        _run_component(api, apps_api=apps_api)


@pytest.mark.parametrize(
    "overrides",
    [
        {"runtime_image": "registry.example.com/vllm:new"},
        {"num_speculative_tokens": 5},
        {"max_model_len": 8192},
        {"draft_model_dir": "speculator/new-checkpoint"},
        {"enable_auth": True},
    ],
)
def test_configuration_changes_wait_for_the_new_predictor(overrides):
    """Changes including runtime-only inputs must complete their corresponding rollout."""
    first_api = _create_api()
    _run_component(first_api)
    previous_isvc = copy.deepcopy(_resource_body(first_api, "inferenceservices"))
    previous_runtime = copy.deepcopy(_resource_body(first_api, "servingruntimes"))
    previous_hash = previous_isvc["spec"]["predictor"]["annotations"][CONFIG_HASH_ANNOTATION]
    api = mock.MagicMock()
    api.get_namespaced_custom_object.side_effect = [
        {"metadata": {"resourceVersion": "7"}},
        previous_runtime,
        previous_isvc,
        _ready_isvc(),
        _ready_isvc(),
    ]
    apps_api = mock.MagicMock()
    attempts = {"count": 0}

    def read_deployment(**_):
        attempts["count"] += 1
        return _deployment(api, config_hash=previous_hash) if attempts["count"] == 1 else _deployment(api)

    apps_api.read_namespaced_deployment.side_effect = read_deployment
    _run_component(api, apps_api=apps_api, **overrides)
    assert _resource_body(api, "inferenceservices")["spec"]["predictor"]["annotations"][CONFIG_HASH_ANNOTATION] != (
        previous_hash
    )
    assert apps_api.read_namespaced_deployment.call_count == 2


def test_identical_rerun_keeps_the_completed_configuration():
    """Identical reruns retain the hash and can reuse the existing completed rollout."""
    first_api = _create_api()
    _run_component(first_api)
    previous_isvc = copy.deepcopy(_resource_body(first_api, "inferenceservices"))
    previous_runtime = copy.deepcopy(_resource_body(first_api, "servingruntimes"))
    previous_hash = previous_isvc["spec"]["predictor"]["annotations"][CONFIG_HASH_ANNOTATION]
    api = mock.MagicMock()
    api.get_namespaced_custom_object.side_effect = [
        {"metadata": {"resourceVersion": "7"}},
        previous_runtime,
        previous_isvc,
        _ready_isvc(),
    ]
    apps_api = mock.MagicMock()
    apps_api.read_namespaced_deployment.return_value = _deployment(first_api, config_hash=previous_hash)
    _run_component(api, apps_api=apps_api)
    assert _resource_body(api, "inferenceservices")["spec"]["predictor"]["annotations"][CONFIG_HASH_ANNOTATION] == (
        previous_hash
    )
    assert apps_api.read_namespaced_deployment.call_count == 1


def test_terminal_rollout_failure_is_actionable():
    """A controller-observed failed rollout raises with a diagnostic command."""
    from kubernetes import client

    api = _create_api()
    apps_api = mock.MagicMock()
    apps_api.read_namespaced_deployment.side_effect = lambda **_: _deployment(
        api,
        conditions=[
            client.V1DeploymentCondition(
                type="Progressing",
                status="False",
                reason="ProgressDeadlineExceeded",
                message="Replacement predictor could not start",
            )
        ],
    )
    with pytest.raises(RuntimeError, match="Replacement predictor could not start.*oc describe deployment"):
        _run_component(api, apps_api=apps_api)


def test_missing_hardware_profile_fails_before_resource_writes():
    """A missing requested profile fails immediately with an explicit opt-out."""
    api = mock.MagicMock()
    api.get_namespaced_custom_object.side_effect = _api_error(404)
    with pytest.raises(ValueError, match="gpu-profile.*redhat-ods-applications.*hardware_profile_name"):
        _run_component(api)
    api.create_namespaced_custom_object.assert_not_called()
    api.patch_namespaced_custom_object.assert_not_called()


def test_hardware_profile_permission_error_propagates():
    """A non-404 profile error remains visible."""
    from kubernetes.client.rest import ApiException

    api = mock.MagicMock()
    api.get_namespaced_custom_object.side_effect = _api_error(403)
    with pytest.raises(ApiException) as exc:
        _run_component(api)
    assert exc.value.status == 403


def test_hardware_profile_opt_out_omits_annotations_on_create():
    """Creating without a HardwareProfile skips lookup and omits its annotations."""
    api = _create_api(hardware_profile=False)
    _run_component(api, hardware_profile_name="")
    assert not any(
        call.kwargs["plural"] == "hardwareprofiles" for call in api.get_namespaced_custom_object.call_args_list
    )
    annotations = _resource_body(api, "inferenceservices")["metadata"]["annotations"]
    assert all(key not in annotations for key in HP_ANNOTATIONS)


def test_hardware_profile_opt_out_clears_old_annotations_on_patch():
    """JSON merge patches explicitly remove the previous profile reference."""
    api = mock.MagicMock()
    api.get_namespaced_custom_object.side_effect = [
        {},
        {"metadata": {"annotations": {**dict.fromkeys(HP_ANNOTATIONS, "old"), "example.org/unrelated": "keep"}}},
        _ready_isvc(),
    ]
    _run_component(api, hardware_profile_name="")
    annotations = _resource_body(api, "inferenceservices")["metadata"]["annotations"]
    assert all(key in annotations and annotations[key] is None for key in HP_ANNOTATIONS)
    assert "example.org/unrelated" not in annotations


def test_trust_remote_code_opt_in_adds_flag():
    """Remote-code execution remains an explicit opt-in."""
    api = _create_api()
    _run_component(api, trust_remote_code=True)
    assert "--trust-remote-code" in _resource_body(api, "servingruntimes")["spec"]["containers"][0]["args"]


@pytest.mark.parametrize("draft_model_dir", ["../checkpoint_best", "/absolute/checkpoint_best"])
def test_invalid_model_directory_is_rejected(draft_model_dir):
    """Model paths cannot escape the PVC directory."""
    with pytest.raises(ValueError, match="draft_model_dir"):
        _run_component(mock.MagicMock(), draft_model_dir=draft_model_dir)


def test_invalid_draft_tensor_parallel_size_is_rejected():
    """Draft parallelism must be one GPU or match verifier parallelism."""
    with pytest.raises(ValueError, match="draft_tensor_parallel_size"):
        _run_component(mock.MagicMock(), gpu_count=2, draft_tensor_parallel_size=3)


def test_readiness_timeout_raises():
    """A service that never becomes ready fails within the existing deadline."""
    not_ready = {"status": {"conditions": [{"type": "Ready", "status": "False"}]}}
    api = _create_api(*[not_ready] * 60)
    with pytest.raises(TimeoutError, match="did not become ready.*30 minutes"):
        _run_component(api)


def test_route_timeout_shares_the_readiness_deadline():
    """Route creation cannot restart the timeout after a slow predictor rollout."""
    api = _create_api(*[_ready_isvc()] * 60, _api_error(404))
    apps_api = mock.MagicMock()
    attempts = {"count": 0}

    def read_deployment(**_):
        attempts["count"] += 1
        return _deployment(api, config_hash="old") if attempts["count"] < 60 else _deployment(api)

    apps_api.read_namespaced_deployment.side_effect = read_deployment
    with pytest.raises(TimeoutError, match="Route"):
        _run_component(api, apps_api=apps_api, enable_external_route=True)
    assert apps_api.read_namespaced_deployment.call_count == 60
