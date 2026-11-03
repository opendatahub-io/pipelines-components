# Speculator Hidden State Extraction Pipeline ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Extract verifier hidden states for speculative decoding training using managed or remote vLLM.

Managed mode runs vLLM in the Trainer job. When ``p04_download_model`` is true, ``p02_verifier_model`` is downloaded to ``p01_persistent_pvc``; otherwise, managed mode reuses ``p03_verifier_model_pvc``. Remote mode uses ``p11_vllm_endpoint`` and requires the verifier and hidden-state PVC paths to be
accessible to the external service. In remote mode, configure ``p07_hidden_states_path`` to match the hidden-state location used by the remote vLLM service on the shared PVC. Managed mode writes hidden states under ``p06_output_dir/hidden_states``.

Relative storage paths are resolved against ``p01_persistent_pvc``. For example, with ``p01_persistent_pvc="shared"``, ``p06_output_dir="speculator-output"`` resolves to ``pvc://shared/speculator-output``. Numeric prefixes preserve a logical parameter order in the KFP UI, which sorts inputs
alphabetically.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `p01_persistent_pvc` | `str` | `None` | Name of the existing PVC in the pipeline namespace, mounted at ``/mnt/persistent``, that stores models, datasets, and extraction outputs. It is required and has no default. |
| `p02_verifier_model` | `str` | `""` | Hugging Face Hub model ID. Set this when ``p04_download_model`` is true to download it to the PVC. In managed mode, when ``p04_download_model`` is false and ``p03_verifier_model_pvc`` is empty, this ID is passed directly to vLLM. |
| `p03_verifier_model_pvc` | `str` | `""` | Optional verifier directory on ``p01_persistent_pvc``. In managed mode with ``p04_download_model`` false, set this to reuse a cached model; if empty, ``p02_verifier_model`` is used directly. Remote mode always requires this PVC-backed path. It is ignored in managed mode when ``p04_download_model`` is true. Use a relative path or a ``pvc://`` URI. |
| `p04_download_model` | `bool` | `False` | Download ``p02_verifier_model`` to the PVC. When false, managed mode uses ``p03_verifier_model_pvc`` if set, or passes ``p02_verifier_model`` directly otherwise. Remote mode uses ``p03_verifier_model_pvc`` for the verifier visible to the external endpoint. |
| `p05_dataset_name` | `str` | `ultrachat` | Use ``magpie`` (Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered), ``ultrachat`` (HuggingFaceH4/ultrachat_200k; default), or ``gsm8k`` (openai/gsm8k). For custom data, use an existing JSON/JSONL file on the PVC (relative path or ``pvc://<claim>/<path>``); arbitrary Hugging Face dataset IDs are not supported. |
| `p06_output_dir` | `str` | `speculator-output` | Directory on ``p01_persistent_pvc`` where extraction outputs are written. The default is ``speculator-output``; use another relative path or URI to choose a different directory. |
| `p07_hidden_states_path` | `str` | `speculator-output/hidden_states` | Remote-mode output location for extracted hidden states. It must match the location on the shared PVC used by the remote vLLM service. Managed mode ignores this input and writes to ``<p06_output_dir>/hidden_states`` automatically. |
| `p08_max_samples` | `int` | `16` | Maximum samples to process; ``0`` processes the full dataset. |
| `p09_regenerate_responses` | `bool` | `False` | Re-generate the assistant responses in the dataset with the verifier model instead of using the original responses. Only supported for the built-in datasets ``magpie``, ``ultrachat``, and ``gsm8k``. |
| `p10_vllm_mode` | `str` | `managed` | Use ``managed`` for a vLLM sidecar in the job or ``remote`` for an external vLLM service. |
| `p11_vllm_endpoint` | `str` | `""` | URL of the external vLLM service; required for ``remote`` and blank for ``managed``. |
| `p12_vllm_resource_gpu` | `int` | `1` | GPUs requested by managed vLLM; the SDK currently supports exactly ``1``. |
| `p13_vllm_resource_memory` | `str` | `96Gi` | Memory requested by the managed vLLM sidecar. |
| `p14_vllm_gpu_memory_utilization` | `float` | `0.9` | Fraction of GPU memory available to vLLM, in the range ``(0, 1]``. |
| `p15_vllm_readiness_timeout_minutes` | `int` | `60` | Maximum wait for the managed or remote vLLM endpoint to become ready, in minutes; minimum ``1``. |
| `p16_total_seq_len` | `int` | `2048` | Maximum sequence length for dataset preprocessing and vLLM extraction; must be positive. |
| `p17_speculator_type` | `str` | `eagle3` | Draft architecture; ``eagle3`` is currently the only supported type. |
| `p18_target_layer_ids` | `str` | `""` | Exactly four comma-separated verifier layer indices. Required when using ``p03_verifier_model_pvc``, when ``p04_download_model`` is true, or in remote mode. It may be omitted only in managed mode when ``p03_verifier_model_pvc`` is empty, ``p02_verifier_model`` is a Hugging Face model ID, and ``p04_download_model`` is false; the SDK can then infer indices from the model config. |
| `p19_data_extraction_concurrency` | `int` | `1` | Number of concurrent vLLM requests for hidden-state extraction; use ``1`` on shared or NFS storage. |
| `p20_packages_to_install` | `str` | `""` | Comma-separated Python packages to install in the job. |
| `p21_pip_index_urls` | `str` | `""` | Comma-separated package-index URLs used for job installs. |
| `p22_job_envs` | `str` | `NCCL_DEBUG=INFO,PYTHONUNBUFFERED=1` | Comma-separated ``KEY=VALUE`` environment overrides for the job. |
| `p23_job_runtime` | `str` | `vllm-extract-cuda` | Name of the runtime used by the extraction job. |

## Metadata 🗂️

- **Name**: speculator_hidden_state_extraction_pipeline
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
    - Name: Trainer, Version: >=0.1.0
  - External Services:
    - Name: Kubernetes, Version: >=1.28.0
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
    - efazal
    - hrathina

## Additional Resources 📚

- **Documentation**: [https://github.com/kubeflow/trainer](https://github.com/kubeflow/trainer)
