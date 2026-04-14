"""EvalHub Evaluation Component.

Deploys a KServe InferenceService, evaluates the model via EvalHub, then
removes resources this run created (InferenceService and optionally
ServingRuntime).
"""

import kfp.compiler
from kfp import dsl


@dsl.component(
    packages_to_install=[
        "kubernetes>=28.1.0",
        "eval-hub-sdk[client] @ git+https://github.com/eval-hub/eval-hub-sdk.git@v0.1.4",
    ],
)
def evalhub_evaluate(
    model_id: str,
    model_name: str,
    namespace: str,
    evalhub_url: str,
    output_metrics: dsl.Output[dsl.Metrics],
    output_results: dsl.Output[dsl.Artifact],
    gpu_count: int = 1,
    max_model_len: int = 4096,
    wait_timeout: int = 900,
    serving_image: str = "vllm/vllm-openai:latest",
    memory_limit: str = "24Gi",
    memory_request: str = "16Gi",
    cpu_limit: str = "4",
    cpu_request: str = "2",
    accelerator_type: str = "nvidia.com/gpu",
    gpu_memory_utilization: float = 0.95,
    min_replicas: int = 1,
    max_replicas: int = 1,
    port: int = 8080,
    url_path: str = "/v1",
    extra_args: str = "",
    existing_runtime: str = "",
    model_format: str = "vLLM",
    tenant: str = "",
    benchmarks_json: str = "",
    tokenizer: str = "",
    num_examples: int = 0,
    timeout_seconds: float = 3600.0,
    poll_interval: float = 15.0,
    insecure: bool = True,
    replace_existing_serving: bool = True,
    delete_serving_resources_after_eval: bool = True,
) -> str:
    """Serve a model on KServe, run an EvalHub job, then tear down serving resources.

    **Evaluation duration:** EvalHub runs benchmarks (e.g. lm_evaluation_harness) in a
    separate job; time depends on dataset size, model speed, and GPU. The component
    also polls job status every ``poll_interval`` seconds (default 15) until completion.
    Use ``num_examples`` to cap work for quicker smoke tests.

    **Serving cleanup:** By default, an existing InferenceService with the same derived
    name is removed and recreated so you do not evaluate a stale deployment, and KServe
    objects are deleted after evaluation so the next run does not leave ``lora-model``
    behind. Set ``delete_serving_resources_after_eval=False`` only if you intentionally
    share a long-lived InferenceService.

    HuggingFace auth for serving uses
    ``HF_TOKEN`` from the environment (mount via ``kfp.kubernetes.use_secret_as_env``).
    EvalHub auth uses ``EVALHUB_TOKEN``.

    Args:
        model_id: Model path for serving (HuggingFace ID, S3 URI, or PVC path).
        model_name: Short name for the served model and EvalHub job naming.
        namespace: Kubernetes namespace for KServe resources.
        evalhub_url: EvalHub server base URL.
        output_metrics: KFP Metrics artifact with per-benchmark scores.
        output_results: KFP Artifact with the full JSON results payload.
        gpu_count: GPUs for the predictor (0 for CPU-only).
        max_model_len: Maximum context length (vLLM).
        wait_timeout: Seconds to wait for the InferenceService to become ready.
        serving_image: Container image when creating a ServingRuntime.
        memory_limit: Memory limit for the predictor pod.
        memory_request: Memory request for the predictor pod.
        cpu_limit: CPU limit for the predictor pod.
        cpu_request: CPU request for the predictor pod.
        accelerator_type: GPU resource name (empty for CPU-only).
        gpu_memory_utilization: GPU memory fraction (vLLM).
        min_replicas: Minimum predictor replicas.
        max_replicas: Maximum predictor replicas.
        port: Container port for the serving endpoint.
        url_path: Path suffix for the cluster-internal URL (e.g. ``/v1``).
        extra_args: Extra vLLM args as a comma-separated list.
        existing_runtime: Use a pre-registered ServingRuntime name; no runtime CR
            is created.
        model_format: Model format label for KServe (e.g. ``vLLM``).
        tenant: EvalHub ``X-Tenant`` header value (tenant namespace). If empty, ``namespace``
            is used so EvalHub receives a valid tenant when the pipeline omits
            ``phase_03_eval_opt_evalhub_tenant``.
        benchmarks_json: Optional JSON array of benchmark configs.
        tokenizer: HuggingFace tokenizer path for default benchmarks.
        num_examples: Examples per benchmark (0 = provider default).
        timeout_seconds: Max seconds to wait for the EvalHub job.
        poll_interval: Seconds between EvalHub status polls.
        insecure: Skip TLS verification for EvalHub.
        replace_existing_serving: If an InferenceService with the same name already exists,
            delete it (and the managed ServingRuntime when not using ``existing_runtime``)
            before creating a new one.
        delete_serving_resources_after_eval: If True, always delete the InferenceService
            (and managed ServingRuntime when applicable) in ``finally`` after eval. If False,
            only delete resources created in this run (legacy behavior).

    Returns:
        The EvalHub job ID.

    Raises:
        TimeoutError: If serving or evaluation does not complete in time.
        RuntimeError: If the evaluation job fails or is cancelled.
    """
    import json
    import os
    import re
    import time

    from evalhub import SyncEvalHubClient
    from evalhub.models.api import (
        BenchmarkConfig,
        JobStatus,
        JobSubmissionRequest,
        ModelConfig,
    )
    from kubernetes import client, config

    def log(msg: str) -> None:
        print(msg, flush=True)

    def teardown_served(
        api_client: client.CustomObjectsApi,
        ns: str,
        dep_name: str,
        del_isvc: bool,
        del_runtime: bool,
    ) -> None:
        if del_isvc:
            try:
                api_client.delete_namespaced_custom_object(
                    group="serving.kserve.io",
                    version="v1beta1",
                    namespace=ns,
                    plural="inferenceservices",
                    name=dep_name,
                )
                log(f"Removed InferenceService '{dep_name}'")
            except client.ApiException as e:
                if e.status != 404:
                    log(f"Warning: could not delete InferenceService: {e.status} {e.reason}")
        if del_runtime:
            try:
                api_client.delete_namespaced_custom_object(
                    group="serving.kserve.io",
                    version="v1alpha1",
                    namespace=ns,
                    plural="servingruntimes",
                    name=dep_name,
                )
                log(f"Removed ServingRuntime '{dep_name}'")
            except client.ApiException as e:
                if e.status != 404:
                    log(f"Warning: could not delete ServingRuntime: {e.status} {e.reason}")

    try:
        config.load_incluster_config()
        log("Loaded in-cluster Kubernetes config")
    except Exception:
        config.load_kube_config()
        log("Loaded local kubeconfig")

    api = client.CustomObjectsApi()
    core_api = client.CoreV1Api()

    deployment_name = re.sub(r"[^a-z0-9-]", "-", model_name.lower()).strip("-")
    runtime_name = existing_runtime if existing_runtime else deployment_name
    log(
        f"evalhub_evaluate: model_id={model_id}, model_name={model_name}, "
        f"namespace={namespace}, deployment_name={deployment_name}"
    )

    if not namespace:
        raise ValueError(
            "namespace is required but was empty. Set 'phase_03_eval_man_namespace' "
            "when creating the pipeline run."
        )

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        log(f"Ensuring hf-token secret in namespace {namespace}...")
        try:
            core_api.create_namespaced_secret(
                namespace=namespace,
                body=client.V1Secret(
                    metadata=client.V1ObjectMeta(name="hf-token"),
                    string_data={"token": hf_token},
                ),
            )
            log(f"Created hf-token secret in {namespace}")
        except client.ApiException as e:
            if e.status == 409:
                log("hf-token secret already exists")
            else:
                raise
    else:
        log("HF_TOKEN not set, skipping hf-token secret creation")

    created_isvc = False
    created_runtime = False
    model_url = ""

    isvc_exists = False
    log(f"Checking if InferenceService '{deployment_name}' exists in {namespace}...")
    try:
        api.get_namespaced_custom_object(
            group="serving.kserve.io",
            version="v1beta1",
            namespace=namespace,
            plural="inferenceservices",
            name=deployment_name,
        )
        isvc_exists = True
        log(f"InferenceService '{deployment_name}' already exists")
    except client.ApiException as e:
        if e.status != 404:
            raise
        log(f"InferenceService '{deployment_name}' not found; will create if needed")

    if isvc_exists and replace_existing_serving:
        log(
            f"replace_existing_serving=True: removing InferenceService '{deployment_name}' "
            f"(and managed ServingRuntime if applicable) before creating a new deployment"
        )
        teardown_served(
            api,
            namespace,
            deployment_name,
            del_isvc=True,
            del_runtime=not bool(str(existing_runtime).strip()),
        )
        isvc_exists = False

    try:
        if not isvc_exists:
            if not existing_runtime:
                vllm_args = [
                    f"--port={port}",
                    f"--model={model_id}",
                    f"--served-model-name={model_name}",
                    f"--gpu-memory-utilization={gpu_memory_utilization}",
                    f"--max-model-len={max_model_len}",
                    "--download-dir=/tmp/hf_home",
                ]
                if extra_args.strip():
                    vllm_args.extend(a.strip() for a in extra_args.split(",") if a.strip())

                serving_runtime = {
                    "apiVersion": "serving.kserve.io/v1alpha1",
                    "kind": "ServingRuntime",
                    "metadata": {
                        "name": deployment_name,
                        "annotations": {
                            "opendatahub.io/apiProtocol": "REST",
                            "openshift.io/display-name": deployment_name,
                        },
                        "labels": {"opendatahub.io/dashboard": "true"},
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": "kserve-container",
                                "image": serving_image,
                                "command": ["python3", "-m", "vllm.entrypoints.openai.api_server"],
                                "args": vllm_args,
                                "env": [
                                    {"name": "HF_HOME", "value": "/tmp/hf_home"},
                                    {"name": "HOME", "value": "/tmp"},
                                    {"name": "HF_HUB_OFFLINE", "value": "0"},
                                    {"name": "TRANSFORMERS_OFFLINE", "value": "0"},
                                    {
                                        "name": "HUGGING_FACE_HUB_TOKEN",
                                        "valueFrom": {
                                            "secretKeyRef": {
                                                "name": "hf-token",
                                                "key": "token",
                                                "optional": True,
                                            }
                                        },
                                    },
                                ],
                                "ports": [{"containerPort": port, "protocol": "TCP"}],
                                "volumeMounts": [{"name": "hf-cache", "mountPath": "/tmp/hf_home"}],
                            }
                        ],
                        "multiModel": False,
                        "supportedModelFormats": [{"autoSelect": True, "name": model_format}],
                        "volumes": [{"name": "hf-cache", "emptyDir": {}}],
                    },
                }

                if accelerator_type:
                    serving_runtime["metadata"]["annotations"][
                        "opendatahub.io/recommended-accelerators"
                    ] = f'["{accelerator_type}"]'

                log(f"Creating ServingRuntime '{deployment_name}' in {namespace}...")
                try:
                    api.create_namespaced_custom_object(
                        group="serving.kserve.io",
                        version="v1alpha1",
                        namespace=namespace,
                        plural="servingruntimes",
                        body=serving_runtime,
                    )
                    log(f"Created ServingRuntime '{deployment_name}'")
                    created_runtime = True
                except client.ApiException as e:
                    if e.status == 409:
                        log(f"ServingRuntime '{deployment_name}' already exists")
                    else:
                        raise
            else:
                log(f"Using existing ServingRuntime '{existing_runtime}'")

            resources = {
                "limits": {"cpu": cpu_limit, "memory": memory_limit},
                "requests": {"cpu": cpu_request, "memory": memory_request},
            }
            if gpu_count > 0 and accelerator_type:
                resources["limits"][accelerator_type] = str(gpu_count)
                resources["requests"][accelerator_type] = str(gpu_count)

            inference_service = {
                "apiVersion": "serving.kserve.io/v1beta1",
                "kind": "InferenceService",
                "metadata": {
                    "name": deployment_name,
                    "annotations": {
                        "openshift.io/display-name": model_name,
                        "serving.kserve.io/deploymentMode": "Standard",
                    },
                    "labels": {"opendatahub.io/dashboard": "true"},
                },
                "spec": {
                    "predictor": {
                        "maxReplicas": max_replicas,
                        "minReplicas": min_replicas,
                        "model": {
                            "modelFormat": {"name": model_format},
                            "resources": resources,
                            "runtime": runtime_name,
                        },
                    },
                },
            }

            log(f"Creating InferenceService '{deployment_name}' in {namespace}...")
            api.create_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                body=inference_service,
            )
            log(f"Created InferenceService '{deployment_name}'")
            created_isvc = True

        model_url = (
            f"http://{deployment_name}-predictor.{namespace}"
            f".svc.cluster.local:{port}{url_path}"
        )

        log(f"Waiting for InferenceService '{deployment_name}' to become ready...")
        start = time.time()
        while time.time() - start < wait_timeout:
            try:
                isvc = api.get_namespaced_custom_object(
                    group="serving.kserve.io",
                    version="v1beta1",
                    namespace=namespace,
                    plural="inferenceservices",
                    name=deployment_name,
                )
                conditions = isvc.get("status", {}).get("conditions", [])
                ready = any(
                    c.get("type") == "Ready" and c.get("status") == "True" for c in conditions
                )
                if ready:
                    log(f"Model ready at {model_url}")
                    break

                elapsed = int(time.time() - start)
                latest = next(
                    (c.get("message", "") for c in conditions if c.get("type") == "Ready"),
                    "",
                )
                log(f"  [{elapsed}s] waiting — {latest}")
            except client.ApiException as poll_err:
                log(f"  [{int(time.time() - start)}s] poll error: {poll_err.status} {poll_err.reason}")

            time.sleep(15)
        else:
            raise TimeoutError(
                f"InferenceService '{deployment_name}' not ready within {wait_timeout}s"
            )

        auth_token = os.environ.get("EVALHUB_TOKEN", "").strip()
        if not auth_token:
            raise RuntimeError(
                "EVALHUB_TOKEN is empty: the EvalHub API requires authentication. "
                "Create a secret in your pipeline namespace, e.g. "
                "oc create secret generic evalhub-auth --from-literal=EVALHUB_TOKEN='<your-token>' -n <namespace>, "
                "using a token your EvalHub server accepts (see EvalHub / cluster docs). "
                "The lora_minimal pipeline mounts this secret as env EVALHUB_TOKEN on the eval step."
            )

        if benchmarks_json.strip():
            raw = json.loads(benchmarks_json)
            benchmarks = [BenchmarkConfig(**b) for b in raw]
        else:
            params: dict = {}
            if tokenizer:
                params["tokenizer"] = tokenizer
            if num_examples > 0:
                params["num_examples"] = num_examples

            benchmarks = [
                BenchmarkConfig(
                    id="arc_easy",
                    provider_id="lm_evaluation_harness",
                    parameters={**params},
                ),
            ]

        tenant_effective = (tenant or "").strip() or (namespace or "").strip()
        if not tenant_effective:
            raise RuntimeError(
                "EvalHub requires a tenant (X-Tenant): set `tenant` / "
                "`phase_03_eval_opt_evalhub_tenant`, or a non-empty `namespace` / "
                "`phase_03_eval_man_namespace`."
            )

        log(f"EvalHub URL : {evalhub_url}")
        log(f"Model URL   : {model_url}")
        if (tenant or "").strip():
            log(f"Tenant      : {tenant_effective}")
        else:
            log(f"Tenant      : {tenant_effective} (from namespace; set tenant to override)")
        log(f"Benchmarks  : {[b.id for b in benchmarks]}")

        eh_client = SyncEvalHubClient(
            base_url=evalhub_url,
            auth_token=auth_token,
            tenant=tenant_effective,
            insecure=insecure,
        )

        providers = eh_client.providers.list()
        log(f"Connected — {len(providers)} providers available")

        request = JobSubmissionRequest(
            name=f"{model_name}-pipeline-eval",
            description=f"Pipeline evaluation for {model_name}",
            model=ModelConfig(url=model_url, name=model_name),
            benchmarks=benchmarks,
        )

        log(f"Submitting job: model_url={model_url}, benchmarks={[b.id for b in benchmarks]}")
        try:
            job = eh_client.jobs.submit(request)
        except Exception as submit_err:
            if hasattr(submit_err, "response"):
                log(f"Job submission failed — status: {submit_err.response.status_code}")
                log(f"Response body: {submit_err.response.text}")
            raise
        job_id = job.resource.id
        log(f"Submitted job {job_id} (state={job.status.state})")

        completed_job = eh_client.jobs.wait_for_completion(
            job_id,
            timeout=timeout_seconds,
            poll_interval=poll_interval,
        )

        state = completed_job.state
        log(f"Job {job_id} finished — state={state}")

        if state == JobStatus.FAILED:
            msg = ""
            if completed_job.status and completed_job.status.message:
                msg = completed_job.status.message.message
            raise RuntimeError(f"Evaluation job {job_id} failed: {msg}")

        if state == JobStatus.CANCELLED:
            raise RuntimeError(f"Evaluation job {job_id} was cancelled")

        result_payload = completed_job.model_dump(mode="json")
        with open(output_results.path, "w") as f:
            json.dump(result_payload, f, indent=2, default=str)
        log(f"Full results written to {output_results.path}")

        if completed_job.results and completed_job.results.benchmarks:
            for br in completed_job.results.benchmarks:
                log(f"  Benchmark: {br.id} ({br.provider_id})")
                for metric_name, metric_value in br.metrics.items():
                    safe_key = f"{br.id}_{metric_name}"
                    if isinstance(metric_value, (int, float)):
                        output_metrics.log_metric(safe_key, metric_value)
                        log(f"    {metric_name}: {metric_value}")
                    else:
                        log(f"    {metric_name}: {metric_value} (non-numeric, skipped)")
        else:
            log("  No benchmark results returned")

        return job_id

    finally:
        if delete_serving_resources_after_eval:
            teardown_served(
                api,
                namespace,
                deployment_name,
                del_isvc=True,
                del_runtime=not bool(str(existing_runtime).strip()),
            )
        else:
            teardown_served(
                api,
                namespace,
                deployment_name,
                del_isvc=created_isvc,
                del_runtime=created_runtime,
            )


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        evalhub_evaluate,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
