# Speculator Online Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Train a draft online, with extraction and training in one Trainer job.

This online pipeline downloads the verifier from HuggingFace or reuses an existing PVC-backed verifier, then runs a single Trainer job that extracts hidden states and trains the draft model together. The managed vLLM sidecar generates hidden states on the fly, eliminating the need for pre-existing
extraction artifacts or separate data-only runs. The output draft is persisted to ``final_model`` for evaluation, optional Model Registry registration, and KServe serving.

### Required inputs and shared storage

``p01_persistent_pvc`` is an existing ReadWriteMany PVC in the pipeline namespace. It stores the verifier, Trainer outputs, and final draft, and must also be available in ``p08_serving_namespace``. The PVC is mounted at ``/mnt/persistent``. With ``p01_persistent_pvc="shared"``,
``p12_output_dir="speculator-output"`` resolves to ``pvc://shared/speculator-output``.

The parameter prefixes follow execution order: storage, verifier, dataset, extraction, output, architecture, training, vLLM, job config, evaluation, registry, and serving. The Registry address ``p43_model_registry_address`` is optional; leave it empty to skip registration. If provided, enter only
the hostname and omit the URL scheme and API path. The pipeline ServiceAccount needs a RoleBinding to ``registry-user-<registry-name>`` in the Registry namespace.

### Verifier model and online training

``p02_verifier_model`` is the optional HuggingFace model ID used for downloading or resolving the default cached directory. ``p04_download_model`` defaults to false. When downloads are disabled, provide an existing model directory with ``p03_verifier_model_pvc`` or use the default cache directory
derived from the model ID. Configure the optional ``hf-token`` secret if downloading requires authentication.

``p03_verifier_model_pvc`` accepts a path relative to the shared PVC or a ``pvc://<p01_persistent_pvc>/<path>`` URI. It is used when ``p04_download_model`` is false; when a path is not supplied, the model ID's default cache directory is used.

``p05_target_layer_ids`` is required: supply four comma-separated layer IDs matching the verifier. The online Trainer receives the cached model as a PVC URI and cannot auto-detect layer IDs from the PVC-backed model config, so this input must be populated.

``p06_dataset_name`` accepts ``magpie`` (Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered), ``ultrachat`` (HuggingFaceH4/ultrachat_200k), or ``gsm8k`` (openai/gsm8k), and can also identify a JSON/JSONL dataset via ``pvc://<p01_persistent_pvc>/<path>`` URI. Response regeneration
(``p10_data_extraction_regenerate_responses``) applies to built-in datasets. ``p09_data_extraction_max_samples=0`` processes all examples. ``p11_data_extraction_concurrency`` controls vLLM request concurrency and maps to Speculator's ``datagen_concurrency`` during training, affecting data-loading
throughput. Use ``1`` on shared/NFS PVCs to limit concurrent I/O.

The online Trainer uses ``p34_job_runtime`` (default ``vllm-extract-cuda``). Environment overrides and packages are shared through ``p36_job_envs`` and ``p35_packages_to_install``. Set ``PIP_INDEX_URL`` or ``PIP_EXTRA_INDEX_URL`` in ``p36_job_envs`` to use a custom package index. Package
installation runs in Trainer worker pods; restrict run access for these inputs to trusted users and approved package sources. The ``kubernetes-credentials`` secret with ``KUBERNETES_SERVER_URL`` and ``KUBERNETES_AUTH_TOKEN`` is mounted only in the KFP launcher task to submit and monitor the Trainer
job, not forwarded to the Trainer worker. Scope it to the minimum Trainer API permissions required for runtime selection, job submission, and monitoring. The optional ``hf-token`` secret is attached to both the model download task and evaluation.

### Training and evaluation

Training runs with a managed vLLM sidecar for extraction. A draft vocabulary size of ``0`` lets Trainer derive it; scheduler warmup and total steps of ``0`` let Trainer calculate them. Training uses ``0.5`` cosine cycles when the cosine scheduler is selected (hardcoded via
``training_scheduler_num_cosine_cycles``), checkpoints every epoch (hardcoded via ``training_checkpoint_freq=1.0``), and logs every step (hardcoded via ``training_log_freq=1``). The managed vLLM sidecar always requests one GPU (hardcoded via ``vllm_resource_gpu=1``). Evaluation uses a fixed sampling
seed of ``42`` (hardcoded via ``evaluation_sample_seed``). The completed draft is written to ``pvc://<p01_persistent_pvc>/final_model``. Each run replaces this shared path; use a separate PVC per revision if earlier Registry versions must continue referencing their original weights.

``p07_evaluation_dataset_uri`` must identify a HuggingFace dataset ID or a JSON/JSONL file on the PVC. Local paths are relative to the PVC mount, and ``pvc://`` URIs must reference ``p01_persistent_pvc``. Avoid leading or trailing whitespace. Evaluation receives only the optional ``hf-token``
secret, not Kubernetes Trainer credentials. ``p37``–``p42`` control the evaluation request limit, generation, sampling, and startup timeout.

### Model Registry and serving

In the RHOAI dashboard, open **AI Hub > Models > Model registries** and create or select a registry. Copy its HTTP REST service URL. Put only the hostname in ``p43_model_registry_address`` and the service port in ``p44_model_registry_port``; omit the scheme and API path. For example, for
``http://registry-service.registry-namespace.svc:8080/api/model_registry/v1alpha3``, use ``registry-service.registry-namespace.svc`` and ``8080``. Set a unique model name in ``p45_model_registry_name`` and use a new ``p46_model_registry_version`` for each revision.

Serving creates a cluster-local KServe InferenceService using the verifier available on the PVC and the draft at ``final_model``. ``p08_serving_namespace`` must access the PVC; the pipeline ServiceAccount needs permission to manage the ServingRuntime and InferenceService and read the predictor
Deployment. ``p47_serving_endpoint_name`` must be DNS-compatible. The pinned compatible vLLM CUDA image is the default for ``p48_serving_runtime_image``. ``p13_total_seq_len`` applies to training and serving. ``p40_evaluation_num_speculative_tokens`` and ``p41_evaluation_gpu_memory_utilization``
apply to evaluation and serving.

Zero-padded ``pNN_`` prefixes preserve parameter order in the KFP UI.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `p01_persistent_pvc` | `str` | `None` | Required name of the existing ReadWriteMany PVC in the pipeline namespace. It stores the downloaded verifier, Trainer outputs, and final draft; KServe must also access it in ``p08_serving_namespace``. The PVC is mounted at ``/mnt/persistent``. |
| `p05_target_layer_ids` | `str` | `None` | Required in online mode. Supply four comma-separated layer IDs matching the verifier. The online Trainer receives the cached model as a PVC URI and cannot auto-detect layer IDs from the PVC-backed model config; this input must be populated. |
| `p06_dataset_name` | `str` | `None` | Required built-in dataset name or PVC URI to a JSON/JSONL file. Supported built-ins are ``magpie`` (Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered), ``ultrachat`` (HuggingFaceH4/ultrachat_200k), and ``gsm8k`` (openai/gsm8k). Local JSON/JSONL paths are converted to Trainer PVC URIs; ``pvc://`` URIs must reference ``p01_persistent_pvc``. |
| `p07_evaluation_dataset_uri` | `str` | `None` | Required Hugging Face dataset ID or local JSON/JSONL file. A ``pvc://`` URI must reference ``p01_persistent_pvc``. Avoid leading or trailing whitespace. Evaluation receives the optional ``hf-token`` secret but not Kubernetes Trainer credentials. |
| `p08_serving_namespace` | `str` | `None` | Required KServe namespace; must access the PVC. The pipeline ServiceAccount needs permission to manage the ServingRuntime and InferenceService and read the predictor Deployment. |
| `p02_verifier_model` | `str` | `""` | Optional HuggingFace model ID used for downloading or resolving the default cached directory. Required when downloading or when no existing PVC path is supplied. |
| `p03_verifier_model_pvc` | `str` | `""` | Optional existing verifier directory on the shared PVC, as a relative path or matching ``pvc://`` URI. Used when ``p04_download_model`` is false. When empty and downloads are disabled, the default cached directory derived from ``p02_verifier_model`` is used. |
| `p04_download_model` | `bool` | `False` | Download the model specified by ``p02_verifier_model`` to the shared PVC. Defaults to false; when true, the model ID is downloaded before training. |
| `p09_data_extraction_max_samples` | `int` | `0` | Maximum training examples extracted by the managed vLLM sidecar. Set to ``0`` to process the full dataset. |
| `p10_data_extraction_regenerate_responses` | `bool` | `False` | Regenerate assistant responses with the verifier instead of using original responses. Supported only for the built-in datasets. |
| `p11_data_extraction_concurrency` | `int` | `1` | Managed vLLM request concurrency and training ``datagen_concurrency``; affects training data-loading throughput. Use ``1`` on shared or NFS PVCs. |
| `p12_output_dir` | `str` | `speculator-output` | PVC-relative directory for Trainer outputs and checkpoints. The final draft is copied to ``final_model`` at the PVC root. |
| `p13_total_seq_len` | `int` | `2048` | Positive maximum sequence length used for extraction, training, and serving. |
| `p14_training_epochs` | `int` | `3` | Number of training epochs. |
| `p15_training_lr` | `float` | `0.0001` | Learning rate for draft-model training. |
| `p16_training_draft_vocab_size` | `int` | `0` | Draft vocabulary size. ``0`` lets the Trainer SDK derive it. |
| `p17_training_num_layers` | `int` | `1` | Number of layers in the draft model. |
| `p18_training_test_time_steps` | `int` | `1` | Number of test-time training steps configured for Speculator. |
| `p19_training_norm_before_residual` | `bool` | `True` | Apply normalization before the residual connection. |
| `p20_training_norm_before_fc` | `bool` | `False` | Apply normalization before the fully connected layer. |
| `p21_training_embed_requires_grad` | `bool` | `False` | Whether draft embedding weights are trainable. |
| `p22_training_hidden_states_dtype` | `str` | `bfloat16` | Hidden-state dtype; it must match the verifier model. |
| `p23_training_scheduler_type` | `str` | `linear` | Learning-rate scheduler: ``linear``, ``cosine``, or ``none``. |
| `p24_training_scheduler_warmup_steps` | `int` | `0` | Warmup steps. ``0`` lets the SDK calculate them. |
| `p25_training_scheduler_total_steps` | `int` | `0` | Total scheduler steps. ``0`` lets the SDK calculate them. |
| `p26_training_save_best` | `bool` | `False` | Save the best checkpoint by validation loss. |
| `p27_training_resume_from_checkpoint` | `bool` | `False` | Resume from a saved checkpoint when supported by the runtime. |
| `p28_training_resource_cpu` | `str` | `4` | CPU request for the main Trainer worker. |
| `p29_training_resource_gpu` | `int` | `1` | Number of GPUs requested for the main Trainer worker. The managed vLLM sidecar always requests one GPU, so the total is ``p29_training_resource_gpu + 1``. |
| `p30_training_resource_memory` | `str` | `64Gi` | Memory request for the main Trainer worker. |
| `p31_vllm_resource_memory` | `str` | `96Gi` | Memory request for the managed vLLM sidecar. The sidecar always requests one GPU. |
| `p32_vllm_gpu_memory_utilization` | `float` | `0.9` | GPU memory fraction for the managed vLLM sidecar, greater than ``0`` and at most ``1.0``. |
| `p33_vllm_readiness_timeout_minutes` | `int` | `60` | Maximum wait in minutes for the managed vLLM sidecar to become ready. |
| `p34_job_runtime` | `str` | `vllm-extract-cuda` | ClusterTrainingRuntime for the online Trainer with vLLM extraction. |
| `p35_packages_to_install` | `str` | `""` | Comma-separated Python packages installed in the online Trainer job. |
| `p36_job_envs` | `str` | `NCCL_DEBUG=INFO,PYTHONUNBUFFERED=1` | Comma-separated ``KEY=VALUE`` environment overrides for the Trainer job. Set ``PIP_INDEX_URL`` or ``PIP_EXTRA_INDEX_URL`` here to select a custom index for ``p35_packages_to_install``. |
| `p37_evaluation_max_samples` | `int` | `80` | Maximum GuideLLM requests per evaluation sweep point; must be positive. |
| `p38_evaluation_max_tokens` | `int` | `256` | Maximum tokens generated per evaluation request; must be positive. |
| `p39_evaluation_temperature` | `float` | `0.0` | Sampling temperature passed to GuideLLM. ``0`` is deterministic. |
| `p40_evaluation_num_speculative_tokens` | `int` | `5` | Draft tokens proposed per step in evaluation and serving. |
| `p41_evaluation_gpu_memory_utilization` | `float` | `0.9` | GPU memory fraction for evaluation and serving, greater than ``0`` and at most ``1.0``. |
| `p42_evaluation_startup_timeout_seconds` | `int` | `900` | Maximum wait for the evaluation vLLM server to become ready. |
| `p43_model_registry_address` | `str` | `""` | Model Registry hostname. Enter only the host, not the scheme or API path. Its default is empty, which skips registration. |
| `p44_model_registry_port` | `int` | `8080` | HTTP REST service port copied from the Model Registry URL; ``8080`` is common. |
| `p45_model_registry_name` | `str` | `speculator-online-draft` | Name assigned to the registered draft model; use a unique name in the registry. |
| `p46_model_registry_version` | `str` | `1.0.0` | Version label for this registration. Use a new value for each revision. |
| `p47_serving_endpoint_name` | `str` | `speculator-online` | DNS-compatible name for the KServe InferenceService. |
| `p48_serving_runtime_image` | `str` | `registry.redhat.io/rhaii-fast/vllm-cuda-rhel9@sha256:97c7d6f66f3f0897b3ff10ddd135de930d45bdd116cf41a43e3ad7270029116a` | Pinned compatible vLLM CUDA image pull spec used by KServe; override only with a compatible image. |

## Metadata 🗂️

- **Name**: speculator_online_pipeline
- **Description**: Cache a verifier model on a shared PVC, train an Eagle3 draft in one online Speculator Trainer job with a managed vLLM sidecar, evaluate it, optionally register it, and serve it.
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
    - Name: Trainer, Version: >=0.1.0
  - External Services:
    - Name: Kubernetes, Version: >=1.28.0
    - Name: HuggingFace Hub, Version: >=0.20.0
    - Name: Model Registry, Version: >=0.3.4
    - Name: OpenShift AI (KServe), Version: >=2.10.0
    - Name: vLLM ServingRuntime, Version: >=0.24.0
    - Name: vLLM, Version: >=0.9.0
    - Name: GuideLLM, Version: ~=0.7.1
- **Tags**:
  - training
  - speculator
  - speculative_decoding
  - pipeline
- **Last Verified**: 2026-10-21 00:00:00+00:00
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
    - efazal
    - hrathina

## Additional Resources 📚

- **Documentation**: [https://github.com/kubeflow/trainer](https://github.com/kubeflow/trainer)
