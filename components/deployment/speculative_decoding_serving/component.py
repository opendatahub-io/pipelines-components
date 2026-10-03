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

    Args:
        namespace: Namespace in which to create the KServe resources.
        endpoint_name: DNS-compatible InferenceService name.
        model_cache_pvc: PVC containing both model directories.
        verifier_model_dir: Relative PVC directory containing the verifier model.
        draft_model_dir: Relative PVC directory containing the draft checkpoint.
        runtime_image: vLLM CUDA image supporting Eagle3 speculative decoding.
        hardware_profile_name: RHOAI HardwareProfile name, or empty to skip lookup.
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
        enable_external_route: Create an OpenShift Route so the endpoint is reachable
            outside the cluster. Requires a one-time RBAC setup — see ``rbac.yaml`` in
            this component's directory. When ``False`` (the default) the internal
            cluster-local URL is returned, which is usable from any pod in the cluster
            or via ``oc port-forward``.

    Returns:
        The OpenAI-compatible ``/v1`` endpoint URL (external HTTPS Route URL when
        ``enable_external_route=True``, internal cluster URL otherwise).
    """
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

    _hp_namespace = "redhat-ods-applications"
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
            print(f"HardwareProfile '{hardware_profile_name}' not found, proceeding without it.")

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
            custom_api.get_namespaced_custom_object(
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
            custom_api.patch_namespaced_custom_object(
                group=group,
                version=version,
                namespace=namespace,
                plural=plural,
                name=name,
                body=body,
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
            },
        },
        "spec": {
            "predictor": {
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas,
                "automountServiceAccountToken": False,
                "deploymentStrategy": {"type": "RollingUpdate"},
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

    upsert_custom_object(
        group="serving.kserve.io",
        version="v1beta1",
        plural="inferenceservices",
        name=endpoint_name,
        body=isvc,
    )

    print("Waiting for InferenceService to become ready...")
    for _ in range(60):
        obj = custom_api.get_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=endpoint_name,
        )
        conditions = obj.get("status", {}).get("conditions", [])
        if any(condition.get("type") == "Ready" and condition.get("status") == "True" for condition in conditions):
            if enable_external_route:
                predictor_svc = f"{endpoint_name}-predictor"
                route_body = {
                    "apiVersion": "route.openshift.io/v1",
                    "kind": "Route",
                    "metadata": {
                        "name": endpoint_name,
                        "namespace": namespace,
                        "labels": {
                            "opendatahub.io/dashboard": "true",
                            "serving.kserve.io/inferenceservice": endpoint_name,
                        },
                    },
                    "spec": {
                        "to": {"kind": "Service", "name": predictor_svc, "weight": 100},
                        "port": {"targetPort": "http"},
                        "tls": {
                            "termination": "edge",
                            "insecureEdgeTerminationPolicy": "Redirect",
                        },
                        "wildcardPolicy": "None",
                    },
                }
                try:
                    upsert_custom_object(
                        group="route.openshift.io",
                        version="v1",
                        plural="routes",
                        name=endpoint_name,
                        body=route_body,
                    )
                    route = custom_api.get_namespaced_custom_object(
                        group="route.openshift.io",
                        version="v1",
                        namespace=namespace,
                        plural="routes",
                        name=endpoint_name,
                    )
                except kclient.rest.ApiException as exc:
                    if exc.status == 403:
                        raise RuntimeError(
                            f"Permission denied creating Route '{endpoint_name}' in "
                            f"namespace '{namespace}'. The pipeline-runner-dspa service "
                            "account lacks permission to manage routes.route.openshift.io "
                            "by default. Apply the one-time RBAC setup before retrying:\n\n"
                            f"  NAMESPACE={namespace}\n"
                            '  sed "s/<NAMESPACE>/$NAMESPACE/g" \\\n'
                            "    components/deployment/speculative_decoding_serving/"
                            "rbac.yaml | oc apply -f -\n\n"
                            "See the Prerequisites section of the component README for details."
                        ) from exc
                    raise
                host = route.get("spec", {}).get("host", "")
                if not host:
                    raise RuntimeError(f"Route '{endpoint_name}' was created but has no host assigned")
                endpoint = f"https://{host}/v1"
                print(f"InferenceService ready (external route): {endpoint}")
                return endpoint

            status = obj.get("status", {})
            url = status.get("url") or status.get("address", {}).get("url", "")
            if not url:
                raise RuntimeError(f"InferenceService '{endpoint_name}' is Ready but did not publish a URL")
            parsed = urlparse(url.strip())
            endpoint = f"{parsed.scheme}://{parsed.netloc}/v1"
            print(f"InferenceService ready: {endpoint}")
            return endpoint

        time.sleep(30)

    raise TimeoutError(f"InferenceService '{endpoint_name}' did not become ready within 30 minutes.")


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        speculative_decoding_serving,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
