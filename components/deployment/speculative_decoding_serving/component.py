"""KFP component for serving a verifier and Eagle3 draft model with vLLM.

The component creates a RHOAI/KServe ServingRuntime and InferenceService in
RawDeployment mode. The PVC is mounted once at the model root, so the verifier
and draft directories remain available to the predictor under that root.
"""

from kfp import dsl

_COMPONENT_BASE_IMAGE = "registry.access.redhat.com/ubi9/python-311:latest"
_DEFAULT_RUNTIME_IMAGE = (
    "registry.redhat.io/rhaii-fast/vllm-cuda-rhel9"
    "@sha256:e310f71b5f9424783982e81d8dc4cba657c3e98f17f8512b827d9fa499ac82a3"
)


@dsl.component(
    base_image=_COMPONENT_BASE_IMAGE,
    packages_to_install=["kubernetes>=28.1.0"],
)
def speculative_decoding_serving(
    namespace: str,
    endpoint_name: str,
    model_cache_pvc: str,
    verifier_model_dir: str,
    draft_model_dir: str,
    runtime_image: str = (
        "registry.redhat.io/rhaii-fast/vllm-cuda-rhel9"
        "@sha256:e310f71b5f9424783982e81d8dc4cba657c3e98f17f8512b827d9fa499ac82a3"
    ),
    hardware_profile_name: str = "gpu-profile",
    min_replicas: int = 1,
    max_replicas: int = 1,
    gpu_count: int = 1,
    draft_tensor_parallel_size: int = 1,
    max_model_len: int = 4096,
    num_speculative_tokens: int = 3,
    cpu_requests: str = "2",
    memory_requests: str = "8Gi",
    cpu_limits: str = "2",
    memory_limits: str = "8Gi",
    gpu_memory_utilization: float = 0.9,
    max_num_seqs: int = 16,
    trust_remote_code: bool = False,
    enable_auth: bool = False,
    enable_external_route: bool = False,
) -> str:
    """Serve a verifier and Eagle3 draft model through one vLLM endpoint.

    The verifier and draft directories must already exist on ``model_cache_pvc``.
    KServe mounts the PVC root at ``/mnt/models``. The draft directory should
    contain the speculator checkpoint's ``config.json`` and weights. vLLM receives
    an Eagle3 ``--speculative-config`` that points to the draft path under that
    root; the verifier is the primary ``--model``.

    The component waits for the requested predictor configuration to finish
    rolling out and for KServe readiness. Updates release an old replica before
    allocating replacement GPU resources, so single-replica updates have downtime.
    Rollout and endpoint publication share a 30-minute timeout.

    Args:
        namespace: Namespace in which to create the KServe resources.
        endpoint_name: DNS-compatible InferenceService name.
        model_cache_pvc: PVC containing both model directories.
        verifier_model_dir: Relative PVC directory containing the verifier model.
        draft_model_dir: Relative PVC directory containing the draft checkpoint.
        runtime_image: vLLM CUDA image supporting Eagle3 speculative decoding.
        hardware_profile_name: Existing RHOAI HardwareProfile name in
            ``redhat-ods-applications``, or empty to skip lookup and remove an old
            profile reference. A missing requested profile is an error.
        min_replicas: Minimum number of predictor replicas.
        max_replicas: Maximum number of predictor replicas.
        gpu_count: GPUs per predictor; also used as verifier vLLM tensor parallel size.
        draft_tensor_parallel_size: vLLM tensor parallel size for the draft model.
        max_model_len: Maximum verifier context length.
        num_speculative_tokens: Number of draft tokens proposed per step.
        cpu_requests: Predictor CPU request.
        memory_requests: Predictor memory request.
        cpu_limits: Predictor CPU limit.
        memory_limits: Predictor memory limit.
        gpu_memory_utilization: Fraction of GPU memory available to vLLM.
        max_num_seqs: Maximum concurrent sequences in the vLLM scheduler.
        trust_remote_code: Pass ``--trust-remote-code`` for the draft checkpoint.
        enable_auth: Enable RHOAI authentication for the exposed endpoint.
        enable_external_route: Request external visibility and wait for the
            platform-managed OpenShift Route to be admitted with the requested
            authentication settings. No Route write permissions are needed. When
            ``False`` (the default), request cluster-local visibility and return
            the internal cluster URL.

    Returns:
        The OpenAI-compatible ``/v1`` endpoint URL (external HTTPS Route URL when
        ``enable_external_route=True``, internal cluster URL otherwise).
    """
    import copy
    import hashlib
    import json
    import re
    import time
    from urllib.parse import urlparse

    from kubernetes import client as kclient
    from kubernetes import config

    dns_label = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")

    def validate_dns_label(value: str, field_name: str) -> str:
        value = value.strip()
        if not value or len(value) > 63 or not dns_label.fullmatch(value):
            raise ValueError(f"{field_name} must be a DNS-compatible name of at most 63 characters: {value!r}")
        return value

    def normalize_model_dir(value: str, field_name: str) -> str:
        value = value.strip()
        if value.startswith("/"):
            raise ValueError(f"{field_name} must be a non-empty relative PVC directory: {value!r}")
        value = value.strip("/")
        parts = value.split("/") if value else []
        if not value or any(part in {"", ".", ".."} for part in parts):
            raise ValueError(f"{field_name} must be a non-empty relative PVC directory: {value!r}")
        return value

    namespace = validate_dns_label(namespace, "namespace")
    endpoint_name = validate_dns_label(endpoint_name, "endpoint_name")
    model_cache_pvc = validate_dns_label(model_cache_pvc, "model_cache_pvc")
    verifier_model_dir = normalize_model_dir(verifier_model_dir, "verifier_model_dir")
    draft_model_dir = normalize_model_dir(draft_model_dir, "draft_model_dir")
    runtime_image = runtime_image.strip()
    if not runtime_image:
        raise ValueError("runtime_image must not be empty")
    if min_replicas < 1 or max_replicas < min_replicas:
        raise ValueError("replica settings must satisfy 1 <= min_replicas <= max_replicas")
    if gpu_count < 1:
        raise ValueError("gpu_count must be at least 1 for the CUDA serving runtime")
    if draft_tensor_parallel_size < 1 or draft_tensor_parallel_size not in (1, gpu_count):
        raise ValueError("draft_tensor_parallel_size must be 1 or equal to gpu_count")
    if max_model_len < 1 or num_speculative_tokens < 1 or max_num_seqs < 1:
        raise ValueError("max_model_len, num_speculative_tokens, and max_num_seqs must be positive")
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be greater than 0 and at most 1")

    serving_runtime_name = f"{endpoint_name}-runtime"

    config.load_incluster_config()
    custom_api = kclient.CustomObjectsApi()
    apps_api = kclient.AppsV1Api()

    _hp_namespace = "redhat-ods-applications"
    hp_annotation_keys = (
        "opendatahub.io/hardware-profile-name",
        "opendatahub.io/hardware-profile-namespace",
        "opendatahub.io/hardware-profile-resource-version",
    )
    hp_resource_version = ""
    if hardware_profile_name:
        try:
            hp = custom_api.get_namespaced_custom_object(
                group="infrastructure.opendatahub.io",
                version="v1",
                namespace=_hp_namespace,
                plural="hardwareprofiles",
                name=hardware_profile_name,
            )
            hp_resource_version = hp.get("metadata", {}).get("resourceVersion", "")
            print(f"Found HardwareProfile '{hardware_profile_name}' (rv={hp_resource_version}).")
        except kclient.rest.ApiException as error:
            if error.status != 404:
                raise
            raise ValueError(
                f"HardwareProfile '{hardware_profile_name}' was not found in namespace '{_hp_namespace}'. "
                'Use an existing profile or set hardware_profile_name="" to skip the HardwareProfile.'
            ) from error

    model_root = "/mnt/models"
    verifier_model_path = f"{model_root}/{verifier_model_dir}"
    draft_model_path = f"{model_root}/{draft_model_dir}"
    speculative_config = {
        "method": "eagle3",
        "model": draft_model_path,
        "draft_tensor_parallel_size": draft_tensor_parallel_size,
        "num_speculative_tokens": num_speculative_tokens,
    }
    runtime_args = [
        "--port=8080",
        f"--model={verifier_model_path}",
        "--speculative-config",
        json.dumps(speculative_config, separators=(",", ":")),
        "--served-model-name={{.Name}}",
        f"--max-model-len={max_model_len}",
        f"--gpu-memory-utilization={gpu_memory_utilization}",
        f"--max-num-seqs={max_num_seqs}",
        f"--tensor-parallel-size={gpu_count}",
    ]
    if trust_remote_code:
        runtime_args.append("--trust-remote-code")

    serving_runtime = {
        "apiVersion": "serving.kserve.io/v1alpha1",
        "kind": "ServingRuntime",
        "metadata": {
            "name": serving_runtime_name,
            "namespace": namespace,
            "annotations": {
                "opendatahub.io/apiProtocol": "REST",
                "opendatahub.io/recommended-accelerators": '["nvidia.com/gpu"]',
                "opendatahub.io/runtime-version": "v0.24.0",
                "opendatahub.io/serving-runtime-scope": "global",
                "opendatahub.io/template-display-name": "vLLM NVIDIA GPU ServingRuntime for KServe",
                "opendatahub.io/template-name": "vllm-cuda-runtime-template",
                "openshift.io/display-name": "vLLM NVIDIA GPU ServingRuntime for KServe",
            },
            "labels": {"opendatahub.io/dashboard": "true"},
        },
        "spec": {
            "annotations": {
                "opendatahub.io/kserve-runtime": "vllm",
                "prometheus.io/path": "/metrics",
                "prometheus.io/port": "8080",
            },
            "multiModel": False,
            "supportedModelFormats": [{"name": "vLLM", "autoSelect": True}],
            "containers": [
                {
                    "name": "kserve-container",
                    "image": runtime_image,
                    "command": ["python3", "-m", "vllm.entrypoints.openai.api_server"],
                    "args": runtime_args,
                    "env": [
                        {"name": "HOME", "value": "/tmp"},
                        {"name": "USER", "value": "default"},
                        {"name": "LOGNAME", "value": "default"},
                        {"name": "USERNAME", "value": "default"},
                        {"name": "HF_HOME", "value": "/tmp/hf_home"},
                        {"name": "HUGGINGFACE_HUB_CACHE", "value": "/tmp/hf_home"},
                        {"name": "XDG_CACHE_HOME", "value": "/tmp/.cache"},
                        {"name": "TORCHINDUCTOR_CACHE_DIR", "value": "/tmp/torchinductor"},
                        {"name": "HF_HUB_OFFLINE", "value": "1"},
                        {"name": "TRANSFORMERS_OFFLINE", "value": "1"},
                        {"name": "VLLM_ATTENTION_BACKEND", "value": "TRITON_ATTN"},
                        {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"},
                    ],
                    "ports": [{"containerPort": 8080, "protocol": "TCP"}],
                }
            ],
        },
    }

    def upsert_custom_object(group: str, version: str, plural: str, name: str, body: dict) -> None:
        try:
            existing = custom_api.get_namespaced_custom_object(
                group=group,
                version=version,
                namespace=namespace,
                plural=plural,
                name=name,
            )
        except kclient.rest.ApiException as error:
            if error.status != 404:
                raise
            custom_api.create_namespaced_custom_object(
                group=group,
                version=version,
                namespace=namespace,
                plural=plural,
                body=body,
            )
            print(f"Created {plural} '{name}'.")
        else:
            patch_body = copy.deepcopy(body)
            if plural == "inferenceservices":
                existing_annotations = existing.get("metadata", {}).get("annotations", {}) or {}
                desired_annotations = patch_body["metadata"]["annotations"]
                for key in hp_annotation_keys:
                    if key in existing_annotations and key not in desired_annotations:
                        desired_annotations[key] = None
            custom_api.patch_namespaced_custom_object(
                group=group,
                version=version,
                namespace=namespace,
                plural=plural,
                name=name,
                body=patch_body,
            )
            print(f"Updated {plural} '{name}'.")

    upsert_custom_object(
        group="serving.kserve.io",
        version="v1alpha1",
        plural="servingruntimes",
        name=serving_runtime_name,
        body=serving_runtime,
    )

    gpu_resources = {"nvidia.com/gpu": str(gpu_count)}
    resources = {
        "requests": {"cpu": cpu_requests, "memory": memory_requests, **gpu_resources},
        "limits": {"cpu": cpu_limits, "memory": memory_limits, **gpu_resources},
    }
    isvc_annotations = {
        "serving.kserve.io/deploymentMode": "RawDeployment",
        "security.opendatahub.io/enable-auth": str(enable_auth).lower(),
        "opendatahub.io/model-type": "generative",
        "opendatahub.io/genai-use-case": "speculative-decoding",
        "openshift.io/display-name": endpoint_name,
    }
    if hardware_profile_name:
        isvc_annotations.update(
            {
                "opendatahub.io/hardware-profile-name": hardware_profile_name,
                "opendatahub.io/hardware-profile-namespace": _hp_namespace,
            }
        )
        if hp_resource_version:
            isvc_annotations["opendatahub.io/hardware-profile-resource-version"] = hp_resource_version

    isvc = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": endpoint_name,
            "namespace": namespace,
            "annotations": isvc_annotations,
            "labels": {
                "opendatahub.io/dashboard": "true",
                "opendatahub.io/genai-asset": "true",
                "networking.kserve.io/visibility": "exposed" if enable_external_route else "cluster-local",
            },
        },
        "spec": {
            "predictor": {
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas,
                "automountServiceAccountToken": False,
                "deploymentStrategy": {
                    "type": "RollingUpdate",
                    "rollingUpdate": {"maxSurge": 0, "maxUnavailable": 1},
                },
                "model": {
                    "modelFormat": {"name": "vLLM"},
                    "name": "",
                    "runtime": serving_runtime_name,
                    "storageUri": f"pvc://{model_cache_pvc}",
                    "resources": resources,
                },
            }
        },
    }

    # Runtime-only changes must also change the predictor pod template. KServe
    # propagates predictor annotations to the Deployment template; a stable hash
    # links its completed rollout to this configuration without restarting reruns.
    config_hash_annotation = "pipelines.kubeflow.org/serving-config-hash"
    desired_config = {
        "runtime": serving_runtime["spec"],
        "predictor": isvc["spec"]["predictor"],
        "annotations": isvc_annotations,
        "visibility": isvc["metadata"]["labels"]["networking.kserve.io/visibility"],
    }
    config_hash = hashlib.sha256(json.dumps(desired_config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    isvc["spec"]["predictor"]["annotations"] = {config_hash_annotation: config_hash}

    upsert_custom_object(
        group="serving.kserve.io",
        version="v1beta1",
        plural="inferenceservices",
        name=endpoint_name,
        body=isvc,
    )

    def read_custom_object(group: str, version: str, plural: str) -> dict:
        try:
            return custom_api.get_namespaced_custom_object(
                group=group,
                version=version,
                namespace=namespace,
                plural=plural,
                name=endpoint_name,
            )
        except kclient.rest.ApiException as error:
            if error.status == 404:
                return {}
            if error.status == 403:
                raise RuntimeError(
                    f"Permission denied reading '{endpoint_name}' in namespace '{namespace}'. "
                    f"The pipeline ServiceAccount needs get permission for {plural}.{group}."
                ) from error
            raise

    predictor_name = f"{endpoint_name}-predictor"
    runtime_container = serving_runtime["spec"]["containers"][0]
    expected_args = [arg.replace("{{.Name}}", endpoint_name) for arg in runtime_args]
    deadline = time.monotonic() + 30 * 60
    last_status = "waiting for predictor Deployment"
    print("Waiting for the requested predictor rollout and InferenceService readiness...")
    while time.monotonic() < deadline:
        obj = read_custom_object("serving.kserve.io", "v1beta1", "inferenceservices")
        isvc_status = obj.get("status", {}) or {}
        conditions = isvc_status.get("conditions", []) or []
        isvc_ready = any(
            condition.get("type") == "Ready" and condition.get("status") == "True" for condition in conditions
        )
        deployment = None
        try:
            deployment = apps_api.read_namespaced_deployment(name=predictor_name, namespace=namespace)
        except kclient.rest.ApiException as error:
            if error.status == 403:
                raise RuntimeError(
                    f"Permission denied reading Deployment '{predictor_name}' in namespace '{namespace}'. "
                    "The pipeline ServiceAccount needs get permission for deployments.apps."
                ) from error
            if error.status != 404:
                raise

        rollout_ready = False
        last_status = f"predictor Deployment not yet published; InferenceService Ready={isvc_ready}"
        if deployment is not None and deployment.spec is not None and deployment.status is not None:
            template_metadata = deployment.spec.template.metadata
            template_annotations = (template_metadata.annotations or {}) if template_metadata else {}
            matches_config = template_annotations.get(config_hash_annotation) == config_hash
            template_spec = deployment.spec.template.spec
            container = (
                next((item for item in (template_spec.containers or []) if item.name == "kserve-container"), None)
                if template_spec
                else None
            )
            # KServe may observe the new ISVC hash before its separate runtime cache
            # catches up. Require the rendered container configuration as well.
            matches_runtime = (
                container is not None
                and container.image == runtime_image
                and container.command == runtime_container["command"]
                and container.args == expected_args
            )
            generation = deployment.metadata.generation or 0
            status = deployment.status
            observed = generation > 0 and (status.observed_generation or 0) >= generation
            desired_replicas = deployment.spec.replicas or 0
            last_status = (
                f"predictor config matches={matches_config}, runtime matches={matches_runtime}, "
                f"generation={generation}, observedGeneration={status.observed_generation}, "
                f"replicas={status.replicas}, updated={status.updated_replicas}, "
                f"available={status.available_replicas}, desired={desired_replicas}; "
                f"InferenceService Ready={isvc_ready}"
            )
            if matches_config and matches_runtime and observed:
                for condition in status.conditions or []:
                    if (
                        condition.type == "Progressing"
                        and condition.status == "False"
                        and condition.reason == "ProgressDeadlineExceeded"
                    ):
                        raise RuntimeError(
                            f"Predictor Deployment '{predictor_name}' failed its requested rollout: "
                            f"{condition.message or condition.reason}. "
                            f"Inspect with: oc describe deployment {predictor_name} -n {namespace}"
                        )
                rollout_ready = (
                    desired_replicas > 0
                    and (status.updated_replicas or 0) == desired_replicas
                    and (status.available_replicas or 0) >= desired_replicas
                    and (status.replicas or 0) == desired_replicas
                )

        if rollout_ready and isvc_ready:
            if enable_external_route:
                route = read_custom_object("route.openshift.io", "v1", "routes")
                route_spec = route.get("spec", {}) or {}
                host = route_spec.get("host", "")
                admitted = any(
                    ingress.get("host") == host
                    and any(
                        condition.get("type") == "Admitted" and condition.get("status") == "True"
                        for condition in ingress.get("conditions", []) or []
                    )
                    for ingress in (route.get("status", {}) or {}).get("ingress", []) or []
                )
                expected_port = "https" if enable_auth else "http"
                expected_tls = "reencrypt" if enable_auth else "edge"
                route_matches = (
                    route_spec.get("to", {}).get("name") == predictor_name
                    and route_spec.get("port", {}).get("targetPort") == expected_port
                    and route_spec.get("tls", {}).get("termination") == expected_tls
                )
                if host and admitted and route_matches:
                    endpoint = f"https://{host}/v1"
                    print(f"InferenceService ready (external route): {endpoint}")
                    return endpoint
                last_status = (
                    f"waiting for platform Route '{endpoint_name}': host={host!r}, admitted={admitted}, "
                    f"backend/auth configuration matches={route_matches} "
                    f"(expected {predictor_name}, {expected_port}, {expected_tls})"
                )
            else:
                url = (isvc_status.get("address", {}) or {}).get("url", "")
                parsed = urlparse(url.strip()) if url else None
                if parsed and parsed.scheme and parsed.netloc:
                    endpoint = f"{parsed.scheme}://{parsed.netloc}/v1"
                    print(f"InferenceService ready: {endpoint}")
                    return endpoint
                last_status = "predictor rollout complete; waiting for InferenceService internal URL publication"

        time.sleep(min(30, max(0, deadline - time.monotonic())))

    route_diagnostic = f"; oc describe route {endpoint_name} -n {namespace}" if enable_external_route else ""
    raise TimeoutError(
        f"InferenceService '{endpoint_name}' did not become ready within 30 minutes. Last status: {last_status}. "
        f"Inspect with: oc describe inferenceservice {endpoint_name} -n {namespace}; "
        f"oc describe deployment {predictor_name} -n {namespace}{route_diagnostic}"
    )


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        speculative_decoding_serving,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
