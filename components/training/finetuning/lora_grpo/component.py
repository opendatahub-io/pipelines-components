"""LoRA GRPO Training Component.

Reusable inline LoRA GRPO (Group Relative Policy Optimization) training component.
- Wraps Training Hub's lora_grpo() via TrainerClient.train()
- Uses ART backend with co-located vLLM for rollout inference on a single GPU
- Mounts /dev/shm emptyDir for vLLM IPC
- Merges LoRA adapters into base model via PEFT merge_and_unload()
- Outputs merged model for downstream deployment/evaluation
- Preserves raw adapter checkpoint path for reproducibility
"""

import os
from typing import Optional

from kfp import dsl

_SHARED_DIR = os.path.join(os.path.dirname(__file__), "..", "shared")


@dsl.component(
    base_image="quay.io/opendatahub/odh-th-torch-cpu-py312:odh-3.6-ea.2",
    install_kfp_package=False,
    packages_to_install=[
        "kfp==2.17.0",
        "kubernetes",
        "olot",
    ],
    embedded_artifact_path=_SHARED_DIR,
    task_config_passthroughs=[
        dsl.TaskConfigField.RESOURCES,
        dsl.TaskConfigField.KUBERNETES_TOLERATIONS,
        dsl.TaskConfigField.KUBERNETES_NODE_SELECTOR,
        dsl.TaskConfigField.KUBERNETES_AFFINITY,
        dsl.TaskConfigPassthrough(field=dsl.TaskConfigField.ENV, apply_to_task=True),
        dsl.TaskConfigPassthrough(field=dsl.TaskConfigField.KUBERNETES_VOLUMES, apply_to_task=True),
    ],
)
def train_model(
    pvc_path: str,
    output_model: dsl.Output[dsl.Model],
    output_metrics: dsl.Output[dsl.Metrics],
    dataset: dsl.Input[dsl.Dataset] = None,
    training_base_model: str = "Qwen/Qwen3-4B",
    training_data_path: str = "",
    training_num_iterations: int = 5,
    training_group_size: int = 4,
    training_prompt_batch_size: int = 50,
    training_n_train: int = 200,
    training_learning_rate: Optional[float] = None,
    training_gpu_memory_utilization: float = 0.45,
    training_enforce_eager: bool = True,
    training_data_config: str = "Qwen3",
    training_lora_r: int = 16,
    training_lora_alpha: int = 8,
    training_envs: str = "",
    training_resource_cpu_per_worker: str = "4",
    training_resource_gpu_per_worker: int = 1,
    training_resource_memory_per_worker: str = "64Gi",
    training_metadata_labels: str = "",
    training_metadata_annotations: str = "",
    training_runtime: str = "training-hub",
    training_pvc_name: str = "",
    kubernetes_config: dsl.TaskConfig = None,
) -> str:
    """Train using LoRA GRPO (Group Relative Policy Optimization).

    Uses ART backend with co-located vLLM for single-GPU RLVR training.
    Trains LoRA adapters on tool-calling agents, then merges adapters into the
    base model via PEFT merge_and_unload() for downstream deployment.

    Training is delegated entirely to Training Hub's built-in lora_grpo()
    algorithm (no custom func is passed to TrainingHubTrainer). This ensures
    the SDK wraps the training call with a `__main__` guard, which is required
    because ART uses `multiprocessing.spawn` internally.

    After the TrainJob completes, this component reads the adapter checkpoints
    from the shared PVC and merges them into the base model on CPU.

    Args:
        pvc_path: Workspace PVC root path (use dsl.WORKSPACE_PATH_PLACEHOLDER).
        output_model: Output model artifact (merged model).
        output_metrics: Output training metrics artifact.
        dataset: Input training dataset artifact (tool-call traces).
        training_base_model: Base model (HuggingFace ID or local path).
        training_data_path: Dataset path or HuggingFace ID (e.g., Agent-Ark/Toucan-1.5M).
            When empty, uses dataset artifact. When set, overrides the artifact.
        training_num_iterations: Number of GRPO training iterations.
        training_group_size: Number of rollouts per prompt for GRPO advantage estimation.
        training_prompt_batch_size: Number of prompts per training batch.
        training_n_train: Number of training samples to use from the dataset.
        training_learning_rate: Learning rate (default: 1e-5).
        training_gpu_memory_utilization: Fraction of GPU memory for vLLM (rest for training).
        training_enforce_eager: Disable torch.compile/CUDAGraphs in vLLM (Qwen3 workaround).
        training_data_config: Dataset config name for HuggingFace datasets (e.g., Qwen3).
        training_lora_r: LoRA rank (controls adapter capacity).
        training_lora_alpha: LoRA scaling factor.
        training_envs: Environment overrides as KEY=VAL,KEY=VAL.
        training_resource_cpu_per_worker: CPU cores per worker.
        training_resource_gpu_per_worker: GPUs per worker (should be 1 for ART).
        training_resource_memory_per_worker: Memory per worker (e.g., 64Gi).
        training_metadata_labels: Pod labels as key=value,key=value.
        training_metadata_annotations: Pod annotations as key=value,key=value.
        training_runtime: Name of the ClusterTrainingRuntime to use.
        training_pvc_name: PVC name to mount on the TrainJob pod for shared storage.
        kubernetes_config: KFP TaskConfig for volumes/env/resources passthrough.

    Environment:
        HF_TOKEN: HuggingFace token for gated models (read from environment).
    """
    import glob
    import os
    from typing import Dict

    from data import download_oci_model, prepare_jsonl, resolve_dataset
    from output import persist_model
    from setup import configure_env, create_logger, init_k8s, parse_kv, setup_hf_token
    from training import select_runtime, wait_for_training_job

    log = create_logger("train_model")
    log.info(f"Initializing LoRA GRPO training component with: pvc={pvc_path}, model={training_base_model}")

    _api = init_k8s(log)

    cache = os.path.join(pvc_path, ".cache", "huggingface")
    default_env: Dict[str, str] = {
        "XDG_CACHE_HOME": "/tmp",
        "TRITON_CACHE_DIR": "/tmp/.triton",
        "HF_HOME": "/tmp/.cache/huggingface",
        "HF_DATASETS_CACHE": os.path.join(cache, "datasets"),
        "TRANSFORMERS_CACHE": os.path.join(cache, "transformers"),
        "TRANSFORMERS_ATTN_BACKEND": "sdpa",
        "PYTHONUNBUFFERED": "1",
        # Workaround: The training image (odh-th-torch-cuda-py312-rhel9) ships
        # the CUDA runtime but not the development headers (curand.h).
        # FlashInfer 0.6.8 tries to JIT-compile sampling kernels via ninja,
        # which fails with "fatal error: curand.h: No such file or directory".
        # Disabling the FlashInfer sampler forces vLLM to use PyTorch-based
        # sampling instead. Remove once the base image includes cuda-devel
        # headers or pre-compiled FlashInfer kernels.
        # Tracking: RHOAIENG-95424
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }

    merged_env = configure_env(training_envs, default_env, log)
    setup_hf_token(merged_env, training_base_model, log)

    ds_dir = os.path.join(pvc_path, "dataset", "train")
    os.makedirs(ds_dir, exist_ok=True)

    # Determine data_path for Training Hub:
    # 1. Explicit training_data_path (HF id or local path) takes priority
    # 2. Dataset artifact resolved to local JSONL
    # 3. Fallback -- Training Hub uses data_config default
    data_path = ""
    if training_data_path:
        data_path = training_data_path
        log.info(f"Using explicit data_path: {data_path}")
    elif dataset is not None:
        resolve_dataset(dataset, ds_dir, log)
        jsonl = os.path.join(ds_dir, "train.jsonl")
        prepare_jsonl(ds_dir, jsonl, log)
        data_path = jsonl if os.path.exists(jsonl) else ds_dir
        log.info(f"Using dataset artifact, resolved to: {data_path}")
    else:
        log.info("No dataset provided; Training Hub will use data_config default")

    resolved = training_base_model

    if isinstance(training_base_model, str) and training_base_model.startswith("oci://"):
        resolved = download_oci_model(training_base_model, pvc_path, log)

    ckpt_dir = os.path.join(pvc_path, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # -- Submit TrainJob ------------------------------------------------
    # We do NOT pass func= to TrainingHubTrainer. This lets the SDK use
    # its built-in lora_grpo() entry-point which is wrapped in an
    # if __name__ == '__main__' guard -- required because ART internally
    # uses multiprocessing.spawn.
    try:
        from kubeflow.common.types import KubernetesBackendConfig
        from kubeflow.trainer import TrainerClient
        from kubeflow.trainer.options.kubernetes import (
            ContainerPatch,
            JobSetSpecPatch,
            JobSetTemplatePatch,
            JobSpecPatch,
            JobTemplatePatch,
            PodSpecPatch,
            PodTemplatePatch,
            ReplicatedJobPatch,
            RuntimePatch,
            TrainingRuntimeSpecPatch,
        )
        from kubeflow.trainer.rhai import TrainingHubAlgorithms, TrainingHubTrainer

        if _api is None:
            raise RuntimeError("K8s API not initialized")

        client = TrainerClient(KubernetesBackendConfig(client_configuration=_api.configuration))

        runtime = select_runtime(client, log, runtime_name=training_runtime)

        # Build training parameters (same schema as the reference notebook)
        params: Dict = {
            "model_path": resolved,
            "backend": "art",
            "ckpt_output_dir": ckpt_dir,
            "num_iterations": int(training_num_iterations),
            "group_size": int(training_group_size),
            "prompt_batch_size": int(training_prompt_batch_size),
            "n_train": int(training_n_train),
            "learning_rate": float(training_learning_rate if training_learning_rate is not None else 1e-5),
            "gpu_memory_utilization": float(training_gpu_memory_utilization),
            "enforce_eager": bool(training_enforce_eager),
            "lora_r": int(training_lora_r),
            "lora_alpha": int(training_lora_alpha),
        }
        if training_data_config:
            params["data_config"] = training_data_config
        if data_path:
            params["data_path"] = data_path

        # Build pod-spec patch for /dev/shm and workspace PVC
        vols, vmts = [], []
        if kubernetes_config and getattr(kubernetes_config, "volumes", None):
            vols.extend(kubernetes_config.volumes)
        if kubernetes_config and getattr(kubernetes_config, "volume_mounts", None):
            vmts.extend(kubernetes_config.volume_mounts)

        tlbl = parse_kv(training_metadata_labels)
        tann = parse_kv(training_metadata_annotations)

        def _pod_spec():
            shm_vol = {"name": "dshm", "emptyDir": {"medium": "Memory"}}
            shm_mount = {"name": "dshm", "mountPath": "/dev/shm"}
            all_vols = [shm_vol] + vols
            all_mounts = [shm_mount] + vmts
            if training_pvc_name:
                all_vols.append({"name": "workspace", "persistentVolumeClaim": {"claimName": training_pvc_name}})
                all_mounts.append({"name": "workspace", "mountPath": pvc_path})
            return PodSpecPatch(
                volumes=all_vols,
                containers=[ContainerPatch(name="node", volume_mounts=all_mounts)],
            )

        resources = {
            "nvidia.com/gpu": training_resource_gpu_per_worker,
            "memory": training_resource_memory_per_worker,
            "cpu": int(training_resource_cpu_per_worker),
        }

        options = []
        if tlbl:
            from kubeflow.trainer.options.kubernetes import Labels

            options.append(Labels(labels=tlbl))
        if tann:
            from kubeflow.trainer.options.kubernetes import Annotations

            options.append(Annotations(annotations=tann))
        options.append(
            RuntimePatch(
                training_runtime_spec=TrainingRuntimeSpecPatch(
                    template=JobSetTemplatePatch(
                        spec=JobSetSpecPatch(
                            replicated_jobs=[
                                ReplicatedJobPatch(
                                    name="node",
                                    template=JobTemplatePatch(
                                        spec=JobSpecPatch(
                                            template=PodTemplatePatch(
                                                spec=_pod_spec(),
                                            )
                                        )
                                    ),
                                )
                            ]
                        )
                    )
                )
            )
        )

        # No func= passed -- Training Hub handles the lora_grpo() call
        # internally with a proper __main__ guard for multiprocessing.spawn.
        job = client.train(
            trainer=TrainingHubTrainer(
                algorithm=TrainingHubAlgorithms.LORA_GRPO,
                func_args=params,
                packages_to_install=[],
                env=dict(merged_env),
                resources_per_node=resources,
            ),
            options=options,
            runtime=runtime,
        )
        log.info(f"Job: {job}")
        wait_for_training_job(client, job, log)
    except Exception as e:
        log.exception(f"Training failed: {e}")
        raise

    # -- Post-training: locate adapter checkpoint ------------------------
    adapter_dirs = sorted(
        glob.glob(os.path.join(ckpt_dir, ".art", "*", "models", "*", "checkpoints", "*")),
        key=os.path.getmtime,
    )
    if not adapter_dirs:
        raise RuntimeError(
            f"No adapter checkpoints found under {ckpt_dir}/.art/. Training Hub lora_grpo() may have failed silently."
        )

    adapter_ckpt_path = adapter_dirs[-1]
    log.info(f"Latest adapter checkpoint: {adapter_ckpt_path}")

    # -- Post-training: merge LoRA adapters on CPU ----------------------
    log.info("Merging LoRA adapters into base model (PEFT merge_and_unload on CPU)...")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_model = AutoModelForCausalLM.from_pretrained(resolved, torch_dtype="auto", low_cpu_mem_usage=True)
    peft_model = PeftModel.from_pretrained(base_model, adapter_ckpt_path)
    merged_model = peft_model.merge_and_unload()

    merged_dir = os.path.join(ckpt_dir, "grpo-merged")
    os.makedirs(merged_dir, exist_ok=True)
    merged_model.save_pretrained(merged_dir)

    tokenizer = AutoTokenizer.from_pretrained(resolved)
    tokenizer.save_pretrained(merged_dir)
    log.info(f"Merged model saved to {merged_dir}")

    # Free memory after merge
    del merged_model, peft_model, base_model

    # -- Metrics and output artifacts ------------------------------------
    def log_training_metrics():
        output_metrics.log_metric("num_iterations", float(params.get("num_iterations", 5)))
        output_metrics.log_metric("group_size", float(params.get("group_size", 4)))
        output_metrics.log_metric("prompt_batch_size", float(params.get("prompt_batch_size", 50)))
        output_metrics.log_metric("n_train", float(params.get("n_train", 200)))
        output_metrics.log_metric("learning_rate", float(params.get("learning_rate", 1e-5)))
        output_metrics.log_metric("gpu_memory_utilization", float(params.get("gpu_memory_utilization", 0.45)))
        output_metrics.log_metric("lora_r", float(params.get("lora_r", 16)))
        output_metrics.log_metric("lora_alpha", float(params.get("lora_alpha", 8)))
        if adapter_ckpt_path:
            output_metrics.metadata["adapter_checkpoint"] = adapter_ckpt_path

    log_training_metrics()

    # persist_model targets the merged model directory directly so that
    # find_model_dir cannot accidentally pick an ART adapter's config.json.
    persist_model(merged_dir, pvc_path, training_base_model, output_model, log)

    if adapter_ckpt_path:
        output_model.metadata["adapter_checkpoint"] = adapter_ckpt_path

    return "training completed"


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(train_model, package_path=__file__.replace(".py", "_component.yaml"))
