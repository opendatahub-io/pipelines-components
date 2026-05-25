"""KFP component: Deploy a text embedding model on OpenShift AI.

Creates a KServe InferenceService with a vLLM-based ServingRuntime
(``--task embedding``) for serving embedding requests via an
OpenAI-compatible ``/v1/embeddings`` API.

Uses in-cluster Kubernetes config (service account token) — no
explicit API URL or token required.
"""

from kfp import dsl
from kfp_components.utils.consts import RAY_RAG_BASE_IMAGE  # pyright: ignore[reportMissingImports]

_DEFAULT_RUNTIME_IMAGE = (
    "registry.redhat.io/rhaiis/vllm-cuda-rhel9@sha256:094db84a1da5e8a575d0c9eade114fa30f4a2061064a338e3e032f3578f8082a"
)


@dsl.component(
    base_image=RAY_RAG_BASE_IMAGE,
    packages_to_install=["kubernetes>=28.1.0"],
)
def deploy_embedding_model(
    model_name: str,
    namespace: str,
    serving_runtime_name: str = "embedding-runtime",
    runtime_image: str = _DEFAULT_RUNTIME_IMAGE,
    min_replicas: int = 1,
    max_replicas: int = 1,
    cpu_requests: str = "2",
    cpu_limits: str = "4",
    memory_requests: str = "4Gi",
    memory_limits: str = "8Gi",
    gpu_count: int = 1,
    max_model_len: int = 512,
) -> str:
    """Deploy a text embedding model using KServe InferenceService.

    Uses vLLM with ``--task embedding`` on GPU to serve an
    OpenAI-compatible ``/v1/embeddings`` endpoint.

    Args:
        model_name: HuggingFace model ID (e.g. 'ibm-granite/granite-embedding-125m-english').
        namespace: Namespace to deploy the InferenceService.
        serving_runtime_name: Name of the embedding ServingRuntime CR.
        runtime_image: Container image for the vLLM embedding server.
        min_replicas: Minimum number of replicas.
        max_replicas: Maximum number of replicas.
        cpu_requests: CPU requests per replica.
        cpu_limits: CPU limits per replica.
        memory_requests: Memory requests per replica.
        memory_limits: Memory limits per replica.
        gpu_count: Number of GPUs per replica.
        max_model_len: Maximum sequence length for the embedding model.

    Returns:
        The embedding service endpoint URL.
    """
    import time

    from kubernetes import client as kclient
    from kubernetes import config

    config.load_incluster_config()
    custom_api = kclient.CustomObjectsApi()

    # Derive InferenceService name from model name
    isvc_name = model_name.split("/")[-1].lower().replace("_", "-").replace(".", "-")

    # --- Create or update ServingRuntime ---
    gpu_resources = {}
    if gpu_count > 0:
        gpu_resources = {"nvidia.com/gpu": str(gpu_count)}

    serving_runtime = {
        "apiVersion": "serving.kserve.io/v1alpha1",
        "kind": "ServingRuntime",
        "metadata": {
            "name": serving_runtime_name,
            "namespace": namespace,
        },
        "spec": {
            "containers": [
                {
                    "name": "kserve-container",
                    "image": runtime_image,
                    "args": [
                        "--model",
                        model_name,
                        "--task",
                        "embedding",
                        "--port",
                        "8080",
                        "--dtype",
                        "float16",
                        "--max-model-len",
                        str(max_model_len),
                    ],
                    "env": [
                        {"name": "HF_HOME", "value": "/tmp/hf_home"},
                        {"name": "HUGGINGFACE_HUB_CACHE", "value": "/tmp/hf_home"},
                        {"name": "HF_HUB_OFFLINE", "value": "0"},
                        {"name": "TRANSFORMERS_OFFLINE", "value": "0"},
                    ],
                    "ports": [{"containerPort": 8080, "protocol": "TCP"}],
                    "resources": {
                        "requests": {
                            "cpu": cpu_requests,
                            "memory": memory_requests,
                            **gpu_resources,
                        },
                        "limits": {
                            "cpu": cpu_limits,
                            "memory": memory_limits,
                            **gpu_resources,
                        },
                    },
                }
            ],
            "supportedModelFormats": [{"name": "vLLM", "version": "1"}],
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
            print(f"ServingRuntime '{serving_runtime_name}' created.")
        else:
            raise

    # --- Create or update InferenceService ---
    isvc = {
        "apiVersion": "serving.kserve.io/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": isvc_name,
            "namespace": namespace,
            "annotations": {
                "serving.kserve.io/deploymentMode": "RawDeployment",
            },
        },
        "spec": {
            "predictor": {
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas,
                "model": {
                    "modelFormat": {"name": "vLLM"},
                    "runtime": serving_runtime_name,
                    "resources": {
                        "requests": {
                            "cpu": cpu_requests,
                            "memory": memory_requests,
                            **gpu_resources,
                        },
                        "limits": {
                            "cpu": cpu_limits,
                            "memory": memory_limits,
                            **gpu_resources,
                        },
                    },
                },
            },
        },
    }

    try:
        custom_api.get_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=isvc_name,
        )
        custom_api.patch_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=isvc_name,
            body=isvc,
        )
        print(f"InferenceService '{isvc_name}' updated.")
    except kclient.rest.ApiException as e:
        if e.status == 404:
            custom_api.create_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                body=isvc,
            )
            print(f"InferenceService '{isvc_name}' created.")
        else:
            raise

    # Wait for Ready
    print("Waiting for embedding InferenceService to become ready...")
    for _ in range(60):
        time.sleep(15)
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
            print(f"Embedding InferenceService ready: {url}")
            return url

    raise TimeoutError(f"Embedding InferenceService '{isvc_name}' did not become ready within 15 minutes.")


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        deploy_embedding_model,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
