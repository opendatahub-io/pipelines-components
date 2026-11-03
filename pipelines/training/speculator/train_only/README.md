# Speculator Training Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Train, evaluate, register, and serve a draft from existing artifacts.

This pipeline trains a draft from existing artifacts and expects the verifier model, hidden states, and the dataset from the data extraction pipeline to already exist on the shared PVC. It does not download the verifier. Relative storage paths are resolved against ``p01_persistent_pvc``. Explicit
``pvc://<claim>/<path>`` paths must reference that same PVC.

For example, with ``p01_persistent_pvc="shared"``, ``p06_verifier_model_pvc="models/Qwen--Qwen3-0.6B"`` is equivalent to ``pvc://shared/models/Qwen--Qwen3-0.6B``. Likewise, ``p10_output_dir="speculator-output"`` and ``p09_hidden_states_path="speculator-output/hidden_states"`` can be written as
``pvc://shared/speculator-output`` and ``pvc://shared/speculator-output/hidden_states``. The default ``p08_training_data_path`` and ``p10_output_dir`` are both ``speculator-output``, matching the data-only pipeline defaults. If you customize the data-only output directory, update
``p08_training_data_path`` and ``p09_hidden_states_path`` too. The hidden-state default matches the data-only pipeline's default output directory; it is not discovered automatically from a previous pipeline run.

The trained draft is copied to ``pvc://<p01_persistent_pvc>/final_model`` for evaluation, Model Registry, and serving. The resulting KServe endpoint uses cluster-local visibility.

Model Registry setup: In the RHOAI dashboard, open **AI Hub > Models > Model registries** and create or select a registry. Copy its HTTP REST service URL. This pipeline takes the host and port separately: put only the hostname in ``p02_model_registry_address`` and the service port in
``p45_model_registry_port``. For a URL such as ``http://registry-service.registry-namespace.svc:8080/api/model_registry/v1alpha3``, use ``registry-service.registry-namespace.svc`` and ``8080``; omit the scheme and API path. The pipeline ServiceAccount must have a RoleBinding to
``registry-user-<registry-name>`` in the registry namespace. ``p02_model_registry_address`` is required because this pipeline registers the evaluated model before serving it. Use a new ``p46_model_registry_version`` for each trained revision.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `p01_persistent_pvc` | `str` | `None` | Name of an existing ReadWriteMany PVC mounted at ``/mnt/persistent``; it stores the verifier, hidden states, training data, and model output. |
| `p02_model_registry_address` | `str` | `None` | Required Model Registry REST host; enter only the hostname. |
| `p03_model_registry_name` | `str` | `None` | Required model name, unique in the selected registry. |
| `p04_serving_namespace` | `str` | `None` | Namespace where KServe creates the endpoint; it must match the PVC namespace and have required permissions. |
| `p05_serving_endpoint_name` | `str` | `None` | Required DNS-compatible KServe InferenceService name. |
| `p06_verifier_model_pvc` | `str` | `None` | Required existing verifier directory on ``p01_persistent_pvc``. Use a relative path or matching ``pvc://`` URI. |
| `p07_evaluation_dataset_uri` | `str` | `None` | Required Hugging Face dataset ID or local JSONL file on ``p01_persistent_pvc``. Use a relative path or matching ``pvc://`` URI. |
| `p08_training_data_path` | `str` | `speculator-output` | Directory containing the preprocessed dataset from the data extraction pipeline. Default: ``speculator-output``. Use a relative path or matching URI. |
| `p09_hidden_states_path` | `str` | `speculator-output/hidden_states` | Existing hidden-state directory from data-only extraction. Default: ``speculator-output/hidden_states``. Use a relative path or matching URI. |
| `p10_output_dir` | `str` | `speculator-output` | Checkpoint and output directory on ``p01_persistent_pvc``. Default: ``speculator-output``. Use a relative path or matching URI. |
| `p11_total_seq_len` | `int` | `2048` | Positive sequence length for preprocessing and training; keep consistent with the data-only extraction run. |
| `p12_speculator_type` | `str` | `eagle3` | Draft architecture. ``eagle3`` is currently supported. |
| `p13_target_layer_ids` | `str` | `2,14,25,28` | Four comma-separated verifier layer IDs. Use the same IDs as data-only extraction; set matching IDs for another verifier. |
| `p14_training_epochs` | `int` | `3` | Positive number of training epochs. |
| `p15_training_lr` | `float` | `0.0001` | Positive learning rate for draft-model training. |
| `p16_training_draft_vocab_size` | `int` | `0` | Draft vocabulary size. ``0`` lets the SDK derive it. |
| `p17_training_num_layers` | `int` | `1` | Number of layers in the draft model. |
| `p18_training_test_time_steps` | `int` | `1` | Number of test-time training steps configured for Speculator. |
| `p19_training_norm_before_residual` | `bool` | `True` | Apply normalization before the residual connection. |
| `p20_training_norm_before_fc` | `bool` | `False` | Apply normalization before the fully connected layer. |
| `p21_training_embed_requires_grad` | `bool` | `False` | Whether draft embedding weights are trainable. |
| `p22_training_hidden_states_dtype` | `str` | `bfloat16` | Hidden-state dtype; it must match the verifier. |
| `p23_training_scheduler_type` | `str` | `linear` | Learning-rate scheduler: ``linear``, ``cosine``, or ``none``. |
| `p24_training_scheduler_warmup_steps` | `int` | `0` | Warmup steps; ``0`` lets the SDK calculate 1% of total steps. |
| `p25_training_scheduler_total_steps` | `int` | `0` | Total scheduler steps; ``0`` lets the SDK calculate them. |
| `p26_training_scheduler_num_cosine_cycles` | `float` | `0.5` | Cosine cycles, used only by the cosine scheduler. |
| `p27_training_checkpoint_freq` | `float` | `1.0` | Checkpoint interval in epochs. |
| `p28_training_save_best` | `bool` | `False` | Save only the best checkpoint by validation loss. |
| `p29_training_log_freq` | `int` | `1` | Training steps between log updates. |
| `p30_training_resume_from_checkpoint` | `bool` | `False` | Resume when the runtime supports it. |
| `p31_training_resource_cpu` | `str` | `4` | CPU request for the training worker. |
| `p32_training_resource_gpu` | `int` | `1` | Number of GPUs requested for the training worker. |
| `p33_training_resource_memory` | `str` | `64Gi` | Memory request for the training worker. |
| `p34_job_envs` | `str` | `NCCL_DEBUG=INFO,PYTHONUNBUFFERED=1` | Comma-separated ``KEY=VALUE`` environment overrides for the job. |
| `p35_job_runtime` | `str` | `speculator-model-opt-cuda` | Name of the runtime used by the job. |
| `p36_packages_to_install` | `str` | `""` | Comma-separated Python packages to install for the training job. |
| `p37_pip_index_urls` | `str` | `""` | Comma-separated package index URLs to use for installs in the training job. |
| `p38_evaluation_max_samples` | `int` | `80` | Maximum requests per evaluation sweep point; must be positive. |
| `p39_evaluation_max_tokens` | `int` | `256` | Maximum generated tokens per evaluation request; must be positive. |
| `p40_evaluation_temperature` | `float` | `0.0` | GuideLLM sampling temperature; ``0`` is deterministic. |
| `p41_num_speculative_tokens` | `int` | `5` | Draft tokens proposed per step for evaluation and serving. |
| `p42_gpu_memory_utilization` | `float` | `0.9` | GPU memory fraction for evaluation and serving. It must be greater than 0 and at most 1.0 (for example, 0.1, 0.2, 0.3, up to 1.0). |
| `p43_evaluation_startup_timeout_seconds` | `int` | `900` | Maximum wait for the evaluation vLLM server. |
| `p44_evaluation_sample_seed` | `int` | `42` | Seed for sampling from a local evaluation dataset. |
| `p45_model_registry_port` | `int` | `8080` | HTTP REST service port copied from the registry endpoint; ``8080`` is the default. |
| `p46_model_registry_version` | `str` | `1.0.0` | Version label for this registration; use a new version per revision. |
| `p47_serving_runtime_image` | `str` | `registry.redhat.io/rhaii-fast/vllm-cuda-rhel9@sha256:97c7d6f66f3f0897b3ff10ddd135de930d45bdd116cf41a43e3ad7270029116a` | Red Hat CUDA runtime image used by KServe for Eagle3 draft serving. The pinned default contains vLLM 0.28.0+rhaiv.7. When upgrading, select a compatible vLLM image and replace the digest. |

## Metadata 🗂️

- **Name**: speculator_training_pipeline
- **Description**: Train a Speculator Eagle3 draft from pre-extracted hidden states, benchmark it, register it, and serve it with vLLM.
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
    - Name: Trainer, Version: >=0.1.0
  - External Services:
    - Name: Kubernetes, Version: >=1.28.0
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
- **Last Verified**: 2026-10-20 00:00:00+00:00
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

- **Documentation**: [https://github.com/kubeflow/trainer](https://github.com/kubeflow/trainer)
