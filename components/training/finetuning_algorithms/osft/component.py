"""OSFT Training Component.

Reusable inline OSFT (Orthogonal Subspace Fine-Tuning) training component.
- Configurable logging
- Optional Kubernetes connection (remote or in-cluster)
- PVC-based caches/checkpoints
- Dataset resolution (HF repo id, or local path)
- Basic metrics logging and checkpoint export
- Hardcoded to use mini-trainer backend and OSFT algorithm
"""

import os
from typing import Optional

from kfp import dsl

_SHARED_DIR = os.path.join(os.path.dirname(__file__), "..", "shared")


@dsl.component(
    base_image="quay.io/opendatahub/odh-training-th04-cpu-torch29-py312-rhel9:cpu-3.3",
    packages_to_install=[
        "kubernetes",
        "olot",
        "matplotlib",
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
    output_loss_chart: dsl.Output[dsl.HTML],
    dataset: dsl.Input[dsl.Dataset] = None,
    training_base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    training_effective_batch_size: int = 128,
    training_max_tokens_per_gpu: int = 64000,
    training_max_seq_len: int = 8192,
    training_learning_rate: Optional[float] = None,
    training_lr_warmup_steps: Optional[int] = None,
    training_checkpoint_at_epoch: Optional[bool] = None,
    training_num_epochs: Optional[int] = None,
    training_data_output_dir: Optional[str] = None,
    training_envs: str = "",
    training_resource_cpu_per_worker: str = "8",
    training_resource_gpu_per_worker: int = 1,
    training_resource_memory_per_worker: str = "32Gi",
    training_resource_num_procs_per_worker: str = "auto",
    training_resource_num_workers: int = 1,
    training_metadata_labels: str = "",
    training_metadata_annotations: str = "",
    training_unfreeze_rank_ratio: float = 0.25,
    training_osft_memory_efficient_init: bool = True,
    training_target_patterns: str = "",
    training_seed: Optional[int] = None,
    training_use_liger: Optional[bool] = None,
    training_use_processed_dataset: Optional[bool] = None,
    training_unmask_messages: Optional[bool] = None,
    training_lr_scheduler: Optional[str] = None,
    training_lr_scheduler_kwargs: str = "",
    training_save_final_checkpoint: Optional[bool] = None,
    training_fsdp_sharding_strategy: Optional[str] = None,
    training_runtime: str = "training-hub",
    kubernetes_config: dsl.TaskConfig = None,
) -> str:
    """Train model using OSFT (Orthogonal Subspace Fine-Tuning). Outputs model artifact and metrics.

    Args:
        pvc_path: Workspace PVC root path (use dsl.WORKSPACE_PATH_PLACEHOLDER).
        output_model: Output model artifact.
        output_metrics: Output training metrics artifact.
        output_loss_chart: Output HTML artifact with training loss chart.
        dataset: Input training dataset artifact.
        training_base_model: Base model (HuggingFace ID or local path).
        training_effective_batch_size: Effective batch size per optimizer step.
        training_max_tokens_per_gpu: Max tokens per GPU (memory cap).
        training_max_seq_len: Max sequence length in tokens.
        training_learning_rate: Learning rate (default: 5e-6).
        training_lr_warmup_steps: Learning rate warmup steps.
        training_checkpoint_at_epoch: Save checkpoint at each epoch.
        training_num_epochs: Number of training epochs.
        training_data_output_dir: Directory for processed training data.
        training_envs: Environment overrides as KEY=VAL,KEY=VAL.
        training_resource_cpu_per_worker: CPU cores per worker.
        training_resource_gpu_per_worker: GPUs per worker.
        training_resource_memory_per_worker: Memory per worker (e.g., 32Gi).
        training_resource_num_procs_per_worker: Processes per worker (auto or int).
        training_resource_num_workers: Number of training pods.
        training_metadata_labels: Pod labels as key=value,key=value.
        training_metadata_annotations: Pod annotations as key=value,key=value.
        training_unfreeze_rank_ratio: Fraction of parameters to unfreeze.
        training_osft_memory_efficient_init: Use memory-efficient initialization.
        training_target_patterns: Target layer patterns (comma-separated).
        training_seed: Random seed for reproducibility.
        training_use_liger: Enable Liger kernel optimizations.
        training_use_processed_dataset: Use pre-processed dataset.
        training_unmask_messages: Unmask assistant messages during training.
        training_lr_scheduler: LR scheduler type (cosine, linear, etc.).
        training_lr_scheduler_kwargs: LR scheduler kwargs as key=val,key=val.
        training_save_final_checkpoint: Save final checkpoint after training.
        training_fsdp_sharding_strategy: FSDP sharding strategy.
        training_runtime: Name of the ClusterTrainingRuntime to use.
        kubernetes_config: KFP TaskConfig for volumes/env/resources passthrough.

    Environment:
        HF_TOKEN: HuggingFace token for gated models (read from environment).
        OCI_PULL_SECRET_MODEL_DOWNLOAD: Docker config.json content for pulling OCI model images.
    """
    import os
    from typing import Dict

    from data import download_oci_model, prepare_jsonl, resolve_dataset
    from output import extract_metrics_from_jsonl, persist_model, plot_training_loss
    from setup import configure_env, create_logger, init_k8s, parse_kv, setup_hf_token
    from training import compute_nproc, select_runtime, wait_for_training_job

    log = create_logger("train_model")
    log.info(f"Initializing OSFT training component with: pvc={pvc_path}, model={training_base_model}")

    _api = init_k8s(log)

    cache = os.path.join(pvc_path, ".cache", "huggingface")
    default_env: Dict[str, str] = {
        "XDG_CACHE_HOME": "/tmp",
        "TRITON_CACHE_DIR": "/tmp/.triton",
        "HF_HOME": "/tmp/.cache/huggingface",
        "HF_DATASETS_CACHE": os.path.join(cache, "datasets"),
        "TRANSFORMERS_CACHE": os.path.join(cache, "transformers"),
        "NCCL_DEBUG": "INFO",
        "PYTHONUNBUFFERED": "1",
    }

    merged_env = configure_env(training_envs, default_env, log)
    setup_hf_token(merged_env, training_base_model, log)

    ds_dir = os.path.join(pvc_path, "dataset", "train")
    os.makedirs(ds_dir, exist_ok=True)

    resolve_dataset(dataset, ds_dir, log)

    jsonl = os.path.join(ds_dir, "train.jsonl")
    prepare_jsonl(ds_dir, jsonl, log)

    resolved = training_base_model

    if isinstance(training_base_model, str) and training_base_model.startswith("oci://"):
        resolved = download_oci_model(training_base_model, pvc_path, log)

    ckpt_dir = os.path.join(pvc_path, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    try:
        from kubeflow.common.types import KubernetesBackendConfig
        from kubeflow.trainer import TrainerClient
        from kubeflow.trainer.options.kubernetes import (
            ContainerOverride,
            PodSpecOverride,
            PodTemplateOverride,
            PodTemplateOverrides,
        )
        from kubeflow.trainer.rhai import TrainingHubAlgorithms, TrainingHubTrainer

        if _api is None:
            raise RuntimeError("K8s API not initialized")

        client = TrainerClient(KubernetesBackendConfig(client_configuration=_api.configuration))

        runtime = select_runtime(client, log, runtime_name=training_runtime)

        tgt = (
            [p.strip() for p in training_target_patterns.split(",") if p.strip()] if training_target_patterns else None
        )
        lr_kw = None
        if training_lr_scheduler_kwargs:
            kv = {}
            for it in [s.strip() for s in training_lr_scheduler_kwargs.split(",") if s.strip()]:
                if "=" not in it:
                    raise ValueError(f"Invalid lr kwargs: {it}")
                k, v = it.split("=", 1)
                kv[k.strip()] = v.strip()
            lr_kw = kv

        def _params() -> Dict:
            np, nn = compute_nproc(
                training_resource_gpu_per_worker,
                training_resource_num_procs_per_worker,
                num_workers=training_resource_num_workers,
            )
            b = {
                "model_path": resolved,
                "data_path": jsonl if os.path.exists(jsonl) else ds_dir,
                "effective_batch_size": int(training_effective_batch_size or 128),
                "max_tokens_per_gpu": int(training_max_tokens_per_gpu),
                "max_seq_len": int(training_max_seq_len or 8192),
                "learning_rate": float(training_learning_rate or 5e-6),
                "backend": "mini-trainer",
                "ckpt_output_dir": ckpt_dir,
                "data_output_dir": training_data_output_dir or os.path.join(ckpt_dir, "_internal_data_processing"),
                "warmup_steps": int(training_lr_warmup_steps) if training_lr_warmup_steps else 0,
                "checkpoint_at_epoch": bool(training_checkpoint_at_epoch)
                if training_checkpoint_at_epoch is not None
                else False,
                "num_epochs": int(training_num_epochs) if training_num_epochs else 1,
                "nproc_per_node": np,
                "nnodes": nn,
            }
            # OSFT-specific parameters
            b["unfreeze_rank_ratio"] = float(training_unfreeze_rank_ratio)
            b["osft_memory_efficient_init"] = bool(training_osft_memory_efficient_init)
            b["target_patterns"] = tgt or []
            if training_seed is not None:
                b["seed"] = int(training_seed)
            if training_use_liger is not None:
                b["use_liger"] = bool(training_use_liger)
            if training_use_processed_dataset is not None:
                b["use_processed_dataset"] = bool(training_use_processed_dataset)
            if training_unmask_messages is not None:
                b["unmask_messages"] = bool(training_unmask_messages)
            if training_lr_scheduler:
                b["lr_scheduler"] = training_lr_scheduler
            if lr_kw:
                b["lr_scheduler_kwargs"] = lr_kw
            if training_save_final_checkpoint is not None:
                b["save_final_checkpoint"] = bool(training_save_final_checkpoint)
            if training_fsdp_sharding_strategy:
                b["fsdp_sharding_strategy"] = training_fsdp_sharding_strategy.upper().strip()
            return b

        params = _params()

        def _train_func(p):
            a = dict(p or {})
            fsdp = a.pop("fsdp_sharding_strategy", None)
            from training_hub import osft as tr

            print("[PY] Launching OSFT training...", flush=True)

            if fsdp:
                try:
                    from instructlab.training.config import FSDPOptions, ShardingStrategies

                    sm = {
                        "FULL_SHARD": ShardingStrategies.FULL_SHARD,
                        "HYBRID_SHARD": ShardingStrategies.HYBRID_SHARD,
                        "NO_SHARD": ShardingStrategies.NO_SHARD,
                    }
                    if fsdp.upper() in sm:
                        a["fsdp_options"] = FSDPOptions(sharding_strategy=sm[fsdp.upper()])
                except ImportError as exc:
                    raise RuntimeError(
                        "FSDP support is not available. Required package 'instructlab.training.config' is missing."
                    ) from exc
            return tr(**a)

        vols, vmts = [], []
        if kubernetes_config and getattr(kubernetes_config, "volumes", None):
            vols.extend(kubernetes_config.volumes)
        if kubernetes_config and getattr(kubernetes_config, "volume_mounts", None):
            vmts.extend(kubernetes_config.volume_mounts)

        tlbl = parse_kv(training_metadata_labels)
        tann = parse_kv(training_metadata_annotations)

        def _pod_spec():
            return PodSpecOverride(volumes=vols, containers=[ContainerOverride(name="node", volume_mounts=vmts)])

        resources = {
            "nvidia.com/gpu": training_resource_gpu_per_worker,
            "memory": training_resource_memory_per_worker,
            "cpu": int(training_resource_cpu_per_worker),
        }

        job = client.train(
            trainer=TrainingHubTrainer(
                func=_train_func,
                func_args=params,
                algorithm=TrainingHubAlgorithms.OSFT,
                packages_to_install=[],
                env=dict(merged_env),
                resources_per_node=resources,
            ),
            options=[
                PodTemplateOverrides(
                    PodTemplateOverride(
                        target_jobs=["node"],
                        metadata={"labels": tlbl, "annotations": tann} if (tlbl or tann) else None,
                        spec=_pod_spec(),
                    )
                )
            ],
            runtime=runtime,
        )
        log.info(f"Job: {job}")
        wait_for_training_job(client, job, log)
    except Exception as e:
        log.error(f"Training failed: {e}")
        raise

    def get_training_metrics(root: str):
        # OSFT prioritizes training_metrics_0.jsonl
        pats = ["training_metrics_0.jsonl", "training_params_and_metrics_global0.jsonl"]
        mf = None
        for r, _, fs in os.walk(root):
            for p in pats:
                if p in fs:
                    mf = os.path.join(r, p)
                    break
            if mf:
                break
        if not mf or not os.path.exists(mf):
            return {}, []
        log.info(f"Metrics: {mf}")
        return extract_metrics_from_jsonl(mf)

    def log_training_metrics():
        output_metrics.log_metric("num_epochs", float(params.get("num_epochs") or 1))
        output_metrics.log_metric("effective_batch_size", float(params.get("effective_batch_size") or 128))
        output_metrics.log_metric("learning_rate", float(params.get("learning_rate") or 5e-6))
        output_metrics.log_metric("max_seq_len", float(params.get("max_seq_len") or 8192))
        output_metrics.log_metric("max_tokens_per_gpu", float(params.get("max_tokens_per_gpu") or 0))
        output_metrics.log_metric("unfreeze_rank_ratio", float(params.get("unfreeze_rank_ratio") or 0))
        tm, loss = get_training_metrics(ckpt_dir)
        for k, v in tm.items():
            output_metrics.log_metric(f"training_{k}", v)
        plot_training_loss(loss, output_loss_chart.path)

    log_training_metrics()

    persist_model(ckpt_dir, pvc_path, training_base_model, output_model, log)
    return "training completed"


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(train_model, package_path=__file__.replace(".py", "_component.yaml"))
