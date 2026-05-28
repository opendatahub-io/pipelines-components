"""KFP component: Deploy a model on OpenShift AI using vLLM.

Creates a ServingRuntime (vLLM NVIDIA GPU) and InferenceService CR
matching the RHOAI dashboard deployment pattern with hardware profiles.

Uses in-cluster Kubernetes config (service account token) — no
explicit API URL or token required.
"""

from kfp import dsl
from kfp_components.utils.consts import RAY_RAG_BASE_IMAGE  # pyright: ignore[reportMissingImports]

_VLLM_IMAGE = (
    "registry.redhat.io/rhaiis/vllm-cuda-rhel9@sha256:094db84a1da5e8a575d0c9eade114fa30f4a2061064a338e3e032f3578f8082a"
)


@dsl.component(
    base_image=RAY_RAG_BASE_IMAGE,
    packages_to_install=["kubernetes>=28.1.0"],
)
def model_deployment(
    model_name: str,
    namespace: str,
    model_dir: str,
    model_cache_pvc: str,
    hardware_profile_name: str = "gpu-profile",
    hardware_profile_namespace: str = "redhat-ods-applications",
    min_replicas: int = 1,
    max_replicas: int = 1,
    gpu_count: int = 1,
    max_model_len: int = 4096,
    cpu_requests: str = "2",
    memory_requests: str = "8Gi",
    cpu_limits: str = "2",
    memory_limits: str = "8Gi",
    force_recreate: bool = False,
) -> str:
    """Deploy a model on OpenShift AI using vLLM InferenceService.

    Creates a vLLM NVIDIA GPU ServingRuntime and InferenceService
    matching the RHOAI dashboard deployment pattern. Uses a PVC-cached
    model to avoid re-downloading on every pod restart.

    Args:
        model_name: HuggingFace model name (e.g. 'mistralai/Mistral-7B-Instruct-v0.3').
        namespace: Namespace to deploy into.
        model_dir: Sub-path on the PVC where model files are stored (output from download_model).
        model_cache_pvc: Name of the PVC containing cached model weights.
        hardware_profile_name: HardwareProfile CR name for GPU resources.
        hardware_profile_namespace: Namespace of the HardwareProfile CR.
        min_replicas: Minimum number of replicas.
        max_replicas: Maximum number of replicas.
        gpu_count: Number of GPUs per replica.
        max_model_len: Maximum context length (limits KV cache memory usage).
        cpu_requests: CPU requests for the predictor pod.
        memory_requests: Memory requests for the predictor pod.
        cpu_limits: CPU limits for the predictor pod.
        memory_limits: Memory limits for the predictor pod.
        force_recreate: If True, delete and recreate the InferenceService
            (causes downtime). If False (default), patch in place.

    Returns:
        The inference endpoint URL.
    """
    import time

    from kubernetes import client as kclient
    from kubernetes import config

    config.load_incluster_config()
    custom_api = kclient.CustomObjectsApi()

    isvc_name = model_name.split("/")[-1].lower().replace(".", "-")

    # Look up the HardwareProfile resource version for the annotation
    try:
        hp = custom_api.get_namespaced_custom_object(
            group="infrastructure.opendatahub.io",
            version="v1",
            namespace=hardware_profile_namespace,
            plural="hardwareprofiles",
            name=hardware_profile_name,
        )
        hp_resource_version = hp["metadata"]["resourceVersion"]
        print(f"Found HardwareProfile '{hardware_profile_name}' (rv={hp_resource_version}).")
    except kclient.rest.ApiException:
        hp_resource_version = ""
        print(f"HardwareProfile '{hardware_profile_name}' not found, proceeding without it.")

    # --- Create ServingRuntime (vLLM NVIDIA GPU, matching RHOAI template) ---
    serving_runtime_name = isvc_name

    serving_runtime = {
        "apiVersion": "serving.kserve.io/v1alpha1",
        "kind": "ServingRuntime",
        "metadata": {
            "name": serving_runtime_name,
            "namespace": namespace,
            "annotations": {
                "opendatahub.io/apiProtocol": "REST",
                "opendatahub.io/recommended-accelerators": '["nvidia.com/gpu"]',
                "opendatahub.io/runtime-version": "v0.11.2",
                "opendatahub.io/serving-runtime-scope": "global",
                "opendatahub.io/template-display-name": "vLLM NVIDIA GPU ServingRuntime for KServe",
                "opendatahub.io/template-name": "vllm-cuda-runtime-template",
                "openshift.io/display-name": "vLLM NVIDIA GPU ServingRuntime for KServe",
            },
            "labels": {
                "opendatahub.io/dashboard": "true",
            },
        },
        "spec": {
            "annotations": {
                "opendatahub.io/kserve-runtime": "vllm",
                "prometheus.io/path": "/metrics",
                "prometheus.io/port": "8080",
            },
            "multiModel": False,
            "supportedModelFormats": [
                {
                    "name": "vLLM",
                    "autoSelect": True,
                },
            ],
            "containers": [
                {
                    "name": "kserve-container",
                    "image": _VLLM_IMAGE,
                    "command": ["python", "-m", "vllm.entrypoints.openai.api_server"],
                    "args": [
                        "--port=8080",
                        "--model=/mnt/models",
                        "--served-model-name={{.Name}}",
                        f"--max-model-len={max_model_len}",
                        "--gpu-memory-utilization=0.90",
                        "--max-num-seqs=16",
                    ],
                    "env": [
                        {"name": "HF_HOME", "value": "/tmp/hf_home"},
                        {"name": "VLLM_ATTENTION_BACKEND", "value": "TRITON_ATTN"},
                        {"name": "PYTORCH_CUDA_ALLOC_CONF", "value": "expandable_segments:True"},
                    ],
                    "ports": [
                        {"containerPort": 8080, "protocol": "TCP"},
                    ],
                },
            ],
        },
    }

    try:
        custom_api.get_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1alpha1",
            namespace=namespace,
            plural="servingruntimes",
            name=serving_runtime_name,
        )
        # Replace to ensure args (like --max-model-len) are up to date
        custom_api.patch_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1alpha1",
            namespace=namespace,
            plural="servingruntimes",
            name=serving_runtime_name,
            body=serving_runtime,
        )
        print(f"ServingRuntime '{serving_runtime_name}' updated.")
    except kclient.rest.ApiException as e:
        if e.status == 404:
            custom_api.create_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1alpha1",
                namespace=namespace,
                plural="servingruntimes",
                body=serving_runtime,
            )
            print(f"Created ServingRuntime '{serving_runtime_name}'.")
        else:
            raise

    # --- Create or update InferenceService ---
    resources = {
        "requests": {"cpu": cpu_requests, "memory": memory_requests},
        "limits": {"cpu": cpu_limits, "memory": memory_limits},
    }
    if gpu_count > 0:
        resources["requests"]["nvidia.com/gpu"] = str(gpu_count)
        resources["limits"]["nvidia.com/gpu"] = str(gpu_count)

    isvc_annotations = {
        "serving.kserve.io/deploymentMode": "RawDeployment",
        "security.opendatahub.io/enable-auth": "false",
        "opendatahub.io/hardware-profile-name": hardware_profile_name,
        "opendatahub.io/hardware-profile-namespace": hardware_profile_namespace,
        "opendatahub.io/model-type": "generative",
        "opendatahub.io/genai-use-case": "ray-rag",
        "openshift.io/display-name": isvc_name,
    }
    if hp_resource_version:
        isvc_annotations["opendatahub.io/hardware-profile-resource-version"] = hp_resource_version

    isvc = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": isvc_name,
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
                    "storageUri": f"pvc://{model_cache_pvc}/{model_dir}",
                    "resources": resources,
                },
            },
        },
    }

    # --- Create or update InferenceService CR ---
    try:
        custom_api.get_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=isvc_name,
        )
        existing = True
    except kclient.rest.ApiException as e:
        if e.status == 404:
            existing = False
        else:
            raise

    if existing and force_recreate:
        custom_api.delete_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=isvc_name,
        )
        print(f"Deleted existing InferenceService '{isvc_name}'. Waiting for cleanup...")
        for _ in range(24):
            time.sleep(5)
            try:
                custom_api.get_namespaced_custom_object(
                    group="serving.kserve.io",
                    version="v1beta1",
                    namespace=namespace,
                    plural="inferenceservices",
                    name=isvc_name,
                )
            except kclient.rest.ApiException as del_e:
                if del_e.status == 404:
                    print("Old InferenceService fully deleted.")
                    break
        else:
            print("Warning: old InferenceService still deleting, proceeding anyway.")
        existing = False

    if existing:
        custom_api.patch_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=isvc_name,
            body=isvc,
        )
        print(f"InferenceService '{isvc_name}' patched in place.")
    else:
        custom_api.create_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            body=isvc,
        )
        print(f"InferenceService '{isvc_name}' created.")

    # Wait for the InferenceService to become ready
    print("Waiting for InferenceService to become ready...")
    for attempt in range(60):
        time.sleep(30)
        obj = custom_api.get_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=isvc_name,
        )
        conditions = obj.get("status", {}).get("conditions", [])
        ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in conditions)
        if ready:
            url = obj.get("status", {}).get("url", "")
            print(f"InferenceService ready: {url}")
            return f"{url}/v1"

    raise TimeoutError(f"InferenceService '{isvc_name}' did not become ready within 30 minutes.")


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        model_deployment,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
