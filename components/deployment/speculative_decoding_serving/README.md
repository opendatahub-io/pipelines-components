# Speculative Decoding Serving ✨

> ⚠️ **Stability: experimental** — This asset is not yet stable and may change.

## Overview 🧾

Serve a verifier and Eagle3 draft model through one vLLM endpoint.

The verifier and draft directories must already exist on ``model_cache_pvc``. KServe mounts the PVC root at ``/mnt/models``. The draft directory should contain the speculator checkpoint's ``config.json`` and weights. vLLM receives an Eagle3 ``--speculative-config`` that points to the draft path
under that root; the verifier is the primary ``--model``.

The component waits for the requested predictor configuration to finish rolling out and for KServe readiness. Updates release an old replica before allocating replacement GPU resources, so single-replica updates have downtime. Rollout and endpoint publication share a 30-minute timeout. Reusing a
stopped InferenceService clears its stop annotation to resume serving.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `namespace` | `str` | `None` | Namespace in which to create the KServe resources. |
| `endpoint_name` | `str` | `None` | DNS-compatible InferenceService name. |
| `model_cache_pvc` | `str` | `None` | PVC containing both model directories. |
| `verifier_model_dir` | `str` | `None` | Relative PVC directory containing the verifier model. |
| `draft_model_dir` | `str` | `None` | Relative PVC directory containing the draft checkpoint. |
| `runtime_image` | `str` | `registry.redhat.io/rhaii-fast/vllm-cuda-rhel9@sha256:e310f71b5f9424783982e81d8dc4cba657c3e98f17f8512b827d9fa499ac82a3` | vLLM CUDA image supporting Eagle3 speculative decoding. |
| `hardware_profile_name` | `str` | `gpu-profile` | Existing RHOAI HardwareProfile name in ``redhat-ods-applications``, or empty to skip lookup and remove an old profile reference. A missing requested profile is an error. |
| `min_replicas` | `int` | `1` | Minimum number of predictor replicas. |
| `max_replicas` | `int` | `1` | Maximum number of predictor replicas. |
| `gpu_count` | `int` | `1` | GPUs per predictor; also used as verifier vLLM tensor parallel size. |
| `draft_tensor_parallel_size` | `int` | `1` | vLLM tensor parallel size for the draft model. |
| `max_model_len` | `int` | `4096` | Maximum verifier context length. |
| `num_speculative_tokens` | `int` | `3` | Number of draft tokens proposed per step. |
| `cpu_requests` | `str` | `2` | Predictor CPU request. |
| `memory_requests` | `str` | `8Gi` | Predictor memory request. |
| `cpu_limits` | `str` | `2` | Predictor CPU limit. |
| `memory_limits` | `str` | `8Gi` | Predictor memory limit. |
| `gpu_memory_utilization` | `float` | `0.9` | Fraction of GPU memory available to vLLM. |
| `max_num_seqs` | `int` | `16` | Maximum concurrent sequences in the vLLM scheduler. |
| `trust_remote_code` | `bool` | `False` | Pass ``--trust-remote-code`` for the draft checkpoint. |
| `enable_auth` | `bool` | `False` | Enable RHOAI authentication for the exposed endpoint. |
| `enable_external_route` | `bool` | `False` | Request external visibility and wait for the platform-managed OpenShift Route to be admitted with the requested authentication settings. No Route write permissions are needed. When ``False`` (the default), request cluster-local visibility and return the internal cluster URL. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `str` | The OpenAI-compatible ``/v1`` endpoint URL (external HTTPS Route URL when ``enable_external_route=True``, internal cluster URL otherwise). |

## Metadata 🗂️

- **Name**: speculative_decoding_serving
- **Description**: Serve a verifier model and an Eagle3 draft model together with vLLM speculative decoding through a RHOAI KServe InferenceService.

- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: OpenShift AI (KServe), Version: >=2.10.0
    - Name: vLLM ServingRuntime, Version: >=0.24.0
- **Tags**:
  - deployment
  - model_serving
  - speculative_decoding
  - vllm
  - kserve
- **Last Verified**: 2026-10-16 12:32:07+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - ChughShilpa
    - efazal
    - hrathina
    - JaZeeGH
    - Sridhar1030
  - Reviewers:
    - ChughShilpa
    - hrathina

## Additional Resources 📚

- **Documentation**: [https://github.com/kubeflow/pipelines-components](https://github.com/kubeflow/pipelines-components)

<!-- custom-content -->

## Serving Configuration

The draft directory must contain an inference-ready Eagle3 checkpoint, not an
unmodified training checkpoint. For Qwen3 Eagle3 checkpoints, this includes
the expected three auxiliary hidden-state layers and a `target_hidden_size` in
`config.json` matching the verifier hidden size. Export or prepare the
checkpoint before invoking this component; the component does not transform
training checkpoints.

The component mounts the PVC root at `/mnt/models`, then starts vLLM with paths
derived from the two directory inputs. For example, with
`verifier_model_dir=models/qwen3-0.6b` and
`draft_model_dir=speculator/run-01/checkpoint_best`, the configuration is
equivalent to:

```json
{
  "method": "eagle3",
  "model": "/mnt/models/speculator/run-01/checkpoint_best",
  "draft_tensor_parallel_size": 1,
  "num_speculative_tokens": 3
}
```

The draft checkpoint's `config.json` contains the Eagle3 algorithm, layer IDs, proposal defaults, and verifier
metadata such as `speculators_config.verifier.name_or_path`. That metadata identifies the verifier used during
training; it is not a filesystem mount path. The serving component supplies the actual verifier PVC directory as
vLLM's primary model. The runtime image must contain vLLM `>=0.24.0` (or a compatible vendor build) with
Qwen3 Eagle3 support; the `runtime_image` input is the place to provide that RHOAI runtime image.

When `enable_auth` is `True`, RHOAI protects endpoint requests with its platform authentication and authorization
layer. Clients then need a valid bearer token and permission to access the namespace. When it is `False`, callers
that can reach the endpoint can invoke the OpenAI-compatible `/v1` API without an RHOAI user token.

This endpoint authentication is separate from Kubernetes authentication used by the component: the pipeline task
uses its in-cluster ServiceAccount to create and watch the `ServingRuntime` and `InferenceService` resources. The
predictor itself has `automountServiceAccountToken` disabled and does not use that ServiceAccount to authenticate
model requests.

## Routing and Permissions

The component sets `networking.kserve.io/visibility` to `exposed` when
`enable_external_route=True`, or `cluster-local` otherwise. RHOAI's model
controller owns the external Route; the component never creates or patches it.
For external access, the component waits for a host and matching router admission,
the predictor backend, and the requested authentication configuration: `http` with
edge TLS when authentication is disabled, or `https` with reencrypt TLS when
authentication is enabled. It returns `https://<route-host>/v1`.

For internal access, the component waits for the InferenceService's cluster address
and appends `/v1`. Missing or invalid addresses are retried; an old external URL
is never used as a fallback.

The pipeline ServiceAccount needs read/create/patch access to `ServingRuntime` and
`InferenceService` resources, `get` access to `deployments.apps`, and, for external
access, `get` access to `routes.route.openshift.io`. Reading a requested
HardwareProfile also requires `get` access in `redhat-ods-applications`. The standard
RHOAI pipeline-runner role already includes Deployment and Route reads; custom
ServiceAccounts must provide the equivalent permissions. No additional Route write
Role or RoleBinding is required.

## HardwareProfile Selection

`hardware_profile_name` must identify an existing HardwareProfile in
`redhat-ods-applications`. A missing requested profile fails before serving resource
writes. Set `hardware_profile_name=""` to explicitly opt out; this skips the lookup
and removes any previous HardwareProfile annotations when updating an endpoint.
Explicit CPU, memory, and GPU resource inputs still apply when opting out.

## Updates and Readiness

The component uses a stable configuration hash propagated to the predictor pod
template. This includes runtime-only changes, such as the image, draft checkpoint,
and speculative token count. It also checks the deployed container's image,
command, and rendered arguments, since the controller can observe the service's
new hash before its separate runtime cache catches up. It waits until the
Deployment has observed its latest generation, all desired replicas are updated
and available, no old replicas remain, and the InferenceService reports
`Ready=True`. Identical reruns retain the hash and do not force a new rollout.

Updates use `maxSurge=0` and `maxUnavailable=1` so a replacement can reuse GPU
capacity released by an old replica. **Single-replica updates have downtime** while
the replacement starts and loads the models. Rollout and endpoint publication share
one 30-minute deadline. A controller-reported rollout failure or timeout includes
diagnostic `oc describe` commands.

The vLLM reference is [EAGLE Draft Models](https://docs.vllm.ai/en/stable/features/speculative_decoding/eagle/).
