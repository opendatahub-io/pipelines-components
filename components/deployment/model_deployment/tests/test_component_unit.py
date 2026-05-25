"""Unit tests for the model_deployment KFP component."""

import tempfile
from pathlib import Path
from unittest import mock

import pytest
from kfp import compiler
from kfp_components.components.deployment.model_deployment import model_deployment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _api_404():
    """Create a kubernetes ApiException with status 404 (lazy import)."""
    from kubernetes.client.rest import ApiException

    return ApiException(status=404)


# Default kwargs passed to model_deployment.python_func in logic tests.
_DEFAULT_KWARGS = {
    "model_name": "org/Model-Name",
    "namespace": "test-ns",
    "model_dir": "org--Model-Name",
    "model_cache_pvc": "cache-pvc",
    "hardware_profile_name": "gpu-profile",
    "hardware_profile_namespace": "redhat-ods-applications",
    "min_replicas": 1,
    "max_replicas": 1,
    "gpu_count": 1,
    "max_model_len": 4096,
    "cpu_requests": "2",
    "memory_requests": "8Gi",
    "cpu_limits": "2",
    "memory_limits": "8Gi",
    "force_recreate": False,
}

# The expected isvc_name derived from "org/Model-Name"
_ISVC_NAME = "model-name"


def _ready_isvc(url="http://test.svc"):
    """Return a mock ISVC object that has a Ready condition."""
    return {
        "status": {
            "conditions": [
                {"type": "Ready", "status": "True"},
            ],
            "url": url,
        },
    }


def _not_ready_isvc():
    """Return a mock ISVC object that is NOT ready."""
    return {
        "status": {
            "conditions": [
                {"type": "Ready", "status": "False"},
            ],
        },
    }


def _hp_object(rv="42"):
    """Return a mock HardwareProfile custom object."""
    return {"metadata": {"resourceVersion": rv}}


# ---------------------------------------------------------------------------
# Compilation / signature tests
# ---------------------------------------------------------------------------


def test_component_compiles():
    """The component YAML compiles without errors."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        compiler.Compiler().compile(model_deployment, f.name)
        assert Path(f.name).stat().st_size > 0


def test_component_signature():
    """All expected parameters are present on the component spec."""
    spec = model_deployment.component_spec
    # spec.inputs is a dict keyed by parameter name
    input_names = set(spec.inputs)
    expected = {
        "model_name",
        "namespace",
        "model_dir",
        "model_cache_pvc",
        "hardware_profile_name",
        "hardware_profile_namespace",
        "min_replicas",
        "max_replicas",
        "gpu_count",
        "max_model_len",
        "cpu_requests",
        "memory_requests",
        "cpu_limits",
        "memory_limits",
        "force_recreate",
    }
    assert expected == input_names


# ---------------------------------------------------------------------------
# Logic tests -- mock kubernetes client
# ---------------------------------------------------------------------------


@mock.patch("time.sleep")
@mock.patch("kubernetes.config.load_incluster_config")
@mock.patch("kubernetes.client.CustomObjectsApi")
def test_creates_serving_runtime_when_not_found(mock_api_cls, mock_config, mock_sleep):
    """When the ServingRuntime does not exist, create_namespaced_custom_object is called."""
    mock_api = mock_api_cls.return_value

    api_404 = _api_404()

    def _get_side_effect(group, version, namespace, plural, name):
        # HardwareProfile lookup -- return a valid HP
        if plural == "hardwareprofiles":
            return _hp_object()
        # ServingRuntime lookup -- not found
        if plural == "servingruntimes":
            raise api_404
        # InferenceService lookup -- not found either (must create)
        if plural == "inferenceservices":
            raise api_404
        raise AssertionError(f"Unexpected get call: {plural}")

    mock_api.get_namespaced_custom_object.side_effect = _get_side_effect
    mock_api.create_namespaced_custom_object.return_value = {}

    # After ISVC creation, the ready-check loop returns ready immediately
    ready_obj = _ready_isvc()
    original_side_effect = mock_api.get_namespaced_custom_object.side_effect

    call_count = {"n": 0}

    def _get_with_ready(group, version, namespace, plural, name):
        call_count["n"] += 1
        # After create calls have been made, the ready-check poll calls
        # get for inferenceservices again. Return ready object for those.
        if plural == "inferenceservices" and call_count["n"] > 2:
            return ready_obj
        return original_side_effect(group, version, namespace, plural, name)

    mock_api.get_namespaced_custom_object.side_effect = _get_with_ready

    result = model_deployment.python_func(**_DEFAULT_KWARGS)

    # ServingRuntime was created (not patched)
    create_calls = mock_api.create_namespaced_custom_object.call_args_list
    sr_creates = [c for c in create_calls if c.kwargs.get("plural") == "servingruntimes"]
    assert len(sr_creates) == 1

    # Verify patch was NOT called for serving runtimes
    patch_calls = mock_api.patch_namespaced_custom_object.call_args_list
    sr_patches = [c for c in patch_calls if c.kwargs.get("plural") == "servingruntimes"]
    assert len(sr_patches) == 0

    assert "/v1" in result


@mock.patch("time.sleep")
@mock.patch("kubernetes.config.load_incluster_config")
@mock.patch("kubernetes.client.CustomObjectsApi")
def test_patches_serving_runtime_when_exists(mock_api_cls, mock_config, mock_sleep):
    """When the ServingRuntime already exists, it is patched (not created)."""
    mock_api = mock_api_cls.return_value
    api_404 = _api_404()

    def _get_side_effect(group, version, namespace, plural, name):
        if plural == "hardwareprofiles":
            return _hp_object()
        if plural == "servingruntimes":
            return {"metadata": {"name": _ISVC_NAME}}  # exists
        if plural == "inferenceservices":
            raise api_404
        raise AssertionError(f"Unexpected get call: {plural}")

    mock_api.get_namespaced_custom_object.side_effect = _get_side_effect
    mock_api.patch_namespaced_custom_object.return_value = {}
    mock_api.create_namespaced_custom_object.return_value = {}

    # Ready on first poll
    ready_obj = _ready_isvc()
    original_side_effect = mock_api.get_namespaced_custom_object.side_effect
    call_count = {"n": 0}

    def _get_with_ready(group, version, namespace, plural, name):
        call_count["n"] += 1
        if plural == "inferenceservices" and call_count["n"] > 3:
            return ready_obj
        return original_side_effect(group, version, namespace, plural, name)

    mock_api.get_namespaced_custom_object.side_effect = _get_with_ready

    result = model_deployment.python_func(**_DEFAULT_KWARGS)

    # ServingRuntime was patched
    patch_calls = mock_api.patch_namespaced_custom_object.call_args_list
    sr_patches = [c for c in patch_calls if c.kwargs.get("plural") == "servingruntimes"]
    assert len(sr_patches) == 1

    # ServingRuntime was NOT created
    create_calls = mock_api.create_namespaced_custom_object.call_args_list
    sr_creates = [c for c in create_calls if c.kwargs.get("plural") == "servingruntimes"]
    assert len(sr_creates) == 0

    assert "/v1" in result


@mock.patch("time.sleep")
@mock.patch("kubernetes.config.load_incluster_config")
@mock.patch("kubernetes.client.CustomObjectsApi")
def test_patches_isvc_in_place_by_default(mock_api_cls, mock_config, mock_sleep):
    """With force_recreate=False (default), an existing ISVC is patched, not deleted."""
    mock_api = mock_api_cls.return_value

    def _get_side_effect(group, version, namespace, plural, name):
        if plural == "hardwareprofiles":
            return _hp_object()
        if plural == "servingruntimes":
            return {"metadata": {"name": _ISVC_NAME}}
        if plural == "inferenceservices":
            return _not_ready_isvc()  # exists but not ready yet
        raise AssertionError(f"Unexpected get call: {plural}")

    mock_api.get_namespaced_custom_object.side_effect = _get_side_effect
    mock_api.patch_namespaced_custom_object.return_value = {}

    # After patching, the ready-check returns ready
    ready_obj = _ready_isvc()
    original_side_effect = mock_api.get_namespaced_custom_object.side_effect
    call_count = {"n": 0}

    def _get_with_ready(group, version, namespace, plural, name):
        call_count["n"] += 1
        # The first ISVC get is the existence check (call 3, after HP + SR gets).
        # The second ISVC get onwards are ready-check polls.
        if plural == "inferenceservices" and call_count["n"] > 3:
            return ready_obj
        return original_side_effect(group, version, namespace, plural, name)

    mock_api.get_namespaced_custom_object.side_effect = _get_with_ready

    kwargs = {**_DEFAULT_KWARGS, "force_recreate": False}
    result = model_deployment.python_func(**kwargs)

    # ISVC was patched (not created or deleted)
    patch_calls = mock_api.patch_namespaced_custom_object.call_args_list
    isvc_patches = [c for c in patch_calls if c.kwargs.get("plural") == "inferenceservices"]
    assert len(isvc_patches) == 1

    # No delete was issued
    mock_api.delete_namespaced_custom_object.assert_not_called()

    assert "/v1" in result


@mock.patch("time.sleep")
@mock.patch("kubernetes.config.load_incluster_config")
@mock.patch("kubernetes.client.CustomObjectsApi")
def test_force_recreate_deletes_then_creates(mock_api_cls, mock_config, mock_sleep):
    """With force_recreate=True, existing ISVC is deleted, waited on, then re-created."""
    mock_api = mock_api_cls.return_value
    api_404 = _api_404()

    # Track phase: existence-check -> delete-wait -> ready-check
    isvc_get_count = {"n": 0}

    def _get_side_effect(group, version, namespace, plural, name):
        if plural == "hardwareprofiles":
            return _hp_object()
        if plural == "servingruntimes":
            return {"metadata": {"name": _ISVC_NAME}}
        if plural == "inferenceservices":
            isvc_get_count["n"] += 1
            if isvc_get_count["n"] == 1:
                # First call: existence check -- ISVC exists
                return _not_ready_isvc()
            if isvc_get_count["n"] == 2:
                # Second call: delete-wait poll -- 404 (deleted)
                raise api_404
            # Remaining calls: ready-check polls -- return ready
            return _ready_isvc()
        raise AssertionError(f"Unexpected get call: {plural}")

    mock_api.get_namespaced_custom_object.side_effect = _get_side_effect
    mock_api.delete_namespaced_custom_object.return_value = {}
    mock_api.patch_namespaced_custom_object.return_value = {}
    mock_api.create_namespaced_custom_object.return_value = {}

    kwargs = {**_DEFAULT_KWARGS, "force_recreate": True}
    result = model_deployment.python_func(**kwargs)

    # Delete was called for ISVC
    delete_calls = mock_api.delete_namespaced_custom_object.call_args_list
    isvc_deletes = [c for c in delete_calls if c.kwargs.get("plural") == "inferenceservices"]
    assert len(isvc_deletes) == 1

    # Create was called for ISVC (after delete)
    create_calls = mock_api.create_namespaced_custom_object.call_args_list
    isvc_creates = [c for c in create_calls if c.kwargs.get("plural") == "inferenceservices"]
    assert len(isvc_creates) == 1

    # Patch was NOT called for ISVC (force_recreate skips patch)
    patch_calls = mock_api.patch_namespaced_custom_object.call_args_list
    isvc_patches = [c for c in patch_calls if c.kwargs.get("plural") == "inferenceservices"]
    assert len(isvc_patches) == 0

    assert "/v1" in result


@mock.patch("time.sleep")
@mock.patch("kubernetes.config.load_incluster_config")
@mock.patch("kubernetes.client.CustomObjectsApi")
def test_creates_isvc_when_not_found(mock_api_cls, mock_config, mock_sleep):
    """When the ISVC does not exist, it is created regardless of force_recreate."""
    mock_api = mock_api_cls.return_value
    api_404 = _api_404()

    isvc_get_count = {"n": 0}

    def _get_side_effect(group, version, namespace, plural, name):
        if plural == "hardwareprofiles":
            return _hp_object()
        if plural == "servingruntimes":
            return {"metadata": {"name": _ISVC_NAME}}
        if plural == "inferenceservices":
            isvc_get_count["n"] += 1
            if isvc_get_count["n"] == 1:
                # Existence check -- not found
                raise api_404
            # Ready-check polls -- return ready
            return _ready_isvc()
        raise AssertionError(f"Unexpected get call: {plural}")

    mock_api.get_namespaced_custom_object.side_effect = _get_side_effect
    mock_api.patch_namespaced_custom_object.return_value = {}
    mock_api.create_namespaced_custom_object.return_value = {}

    result = model_deployment.python_func(**_DEFAULT_KWARGS)

    # ISVC was created
    create_calls = mock_api.create_namespaced_custom_object.call_args_list
    isvc_creates = [c for c in create_calls if c.kwargs.get("plural") == "inferenceservices"]
    assert len(isvc_creates) == 1

    # No delete
    mock_api.delete_namespaced_custom_object.assert_not_called()

    assert "/v1" in result


@mock.patch("time.sleep")
@mock.patch("kubernetes.config.load_incluster_config")
@mock.patch("kubernetes.client.CustomObjectsApi")
def test_hardware_profile_not_found_continues(mock_api_cls, mock_config, mock_sleep):
    """A 404 on the HardwareProfile lookup does not crash the component."""
    mock_api = mock_api_cls.return_value
    api_404 = _api_404()

    isvc_get_count = {"n": 0}

    def _get_side_effect(group, version, namespace, plural, name):
        if plural == "hardwareprofiles":
            raise api_404  # HP not found
        if plural == "servingruntimes":
            raise api_404  # also not found, will create
        if plural == "inferenceservices":
            isvc_get_count["n"] += 1
            if isvc_get_count["n"] == 1:
                raise api_404
            return _ready_isvc()
        raise AssertionError(f"Unexpected get call: {plural}")

    mock_api.get_namespaced_custom_object.side_effect = _get_side_effect
    mock_api.create_namespaced_custom_object.return_value = {}

    # Should not raise
    result = model_deployment.python_func(**_DEFAULT_KWARGS)
    assert "/v1" in result


@mock.patch("time.sleep")
@mock.patch("kubernetes.config.load_incluster_config")
@mock.patch("kubernetes.client.CustomObjectsApi")
def test_waits_for_ready_and_returns_url(mock_api_cls, mock_config, mock_sleep):
    """The component polls until Ready and returns '<url>/v1'."""
    mock_api = mock_api_cls.return_value
    api_404 = _api_404()

    isvc_get_count = {"n": 0}

    def _get_side_effect(group, version, namespace, plural, name):
        if plural == "hardwareprofiles":
            return _hp_object()
        if plural == "servingruntimes":
            raise api_404
        if plural == "inferenceservices":
            isvc_get_count["n"] += 1
            if isvc_get_count["n"] == 1:
                raise api_404  # existence check
            if isvc_get_count["n"] <= 3:
                return _not_ready_isvc()  # first few polls not ready
            return _ready_isvc(url="http://my-model.test.svc")
        raise AssertionError(f"Unexpected get call: {plural}")

    mock_api.get_namespaced_custom_object.side_effect = _get_side_effect
    mock_api.create_namespaced_custom_object.return_value = {}

    result = model_deployment.python_func(**_DEFAULT_KWARGS)

    assert result == "http://my-model.test.svc/v1"


@mock.patch("time.sleep")
@mock.patch("kubernetes.config.load_incluster_config")
@mock.patch("kubernetes.client.CustomObjectsApi")
def test_timeout_raises(mock_api_cls, mock_config, mock_sleep):
    """If the ISVC never becomes Ready, TimeoutError is raised."""
    mock_api = mock_api_cls.return_value
    api_404 = _api_404()

    isvc_get_count = {"n": 0}

    def _get_side_effect(group, version, namespace, plural, name):
        if plural == "hardwareprofiles":
            return _hp_object()
        if plural == "servingruntimes":
            raise api_404
        if plural == "inferenceservices":
            isvc_get_count["n"] += 1
            if isvc_get_count["n"] == 1:
                raise api_404  # existence check
            return _not_ready_isvc()  # never ready
        raise AssertionError(f"Unexpected get call: {plural}")

    mock_api.get_namespaced_custom_object.side_effect = _get_side_effect
    mock_api.create_namespaced_custom_object.return_value = {}

    with pytest.raises(TimeoutError, match="did not become ready"):
        model_deployment.python_func(**_DEFAULT_KWARGS)

    # time.sleep was called 60 times (the ready-check loop)
    assert mock_sleep.call_count >= 60
