"""LoRA Training Component.

Reusable inline LoRA (Low-Rank Adaptation) training component.
- Configurable logging
- Optional Kubernetes connection (remote or in-cluster)
- PVC-based caches/checkpoints
- Dataset resolution (HF repo id, or local path)
- Basic metrics logging and checkpoint export
- Hardcoded to use unsloth backend and LoRA algorithm
"""

from typing import Optional

from kfp import dsl


@dsl.component(
    base_image="quay.io/opendatahub/odh-training-th04-cpu-torch29-py312-rhel9:cpu-3.3",
    packages_to_install=[
        "kubernetes",
        "olot",
        "matplotlib",
        # Shared utilities (logging, K8s init, model persistence, etc.) from the kfp-components package
        "kfp-components@git+https://github.com/red-hat-data-services/pipelines-components.git@main",
    ],
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
    training_max_tokens_per_gpu: int = 32000,
    training_max_seq_len: int = 8192,
    training_learning_rate: Optional[float] = None,
    training_lr_warmup_steps: Optional[int] = None,
    training_checkpoint_at_epoch: Optional[bool] = None,
    training_num_epochs: Optional[int] = None,
    training_data_output_dir: Optional[str] = None,
    training_envs: str = "",
    training_resource_cpu_per_worker: str = "4",
    training_resource_gpu_per_worker: int = 1,
    training_resource_memory_per_worker: str = "32Gi",
    training_resource_num_procs_per_worker: str = "auto",
    training_resource_num_workers: int = 1,
    training_metadata_labels: str = "",
    training_metadata_annotations: str = "",
    training_lora_r: int = 16,
    training_lora_alpha: int = 32,
    training_lora_dropout: float = 0.0,
    training_lora_target_modules: str = "",
    training_lora_use_rslora: Optional[bool] = None,
    training_lora_use_dora: Optional[bool] = None,
    training_lora_load_in_4bit: Optional[bool] = None,
    training_lora_load_in_8bit: Optional[bool] = None,
    training_lora_bnb_4bit_quant_type: Optional[str] = None,
    training_lora_bnb_4bit_compute_dtype: Optional[str] = None,
    training_lora_bnb_4bit_use_double_quant: Optional[bool] = None,
    training_lora_sample_packing: Optional[bool] = None,
    training_seed: Optional[int] = None,
    training_use_liger: Optional[bool] = None,
    training_lr_scheduler: Optional[str] = None,
    kubernetes_config: dsl.TaskConfig = None,
) -> str:
    """Train model using LoRA (Low-Rank Adaptation). Outputs model artifact and metrics.

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
        training_learning_rate: Learning rate (default: 2e-4).
        training_lr_warmup_steps: Learning rate warmup steps.
        training_checkpoint_at_epoch: Save checkpoint at each epoch.
        training_num_epochs: Number of training epochs (default: 3).
        training_data_output_dir: Directory for processed training data.
        training_envs: Environment overrides as KEY=VAL,KEY=VAL.
        training_resource_cpu_per_worker: CPU cores per worker.
        training_resource_gpu_per_worker: GPUs per worker.
        training_resource_memory_per_worker: Memory per worker (e.g., 32Gi).
        training_resource_num_procs_per_worker: Processes per worker (auto or int).
        training_resource_num_workers: Number of training pods.
        training_metadata_labels: Pod labels as key=value,key=value.
        training_metadata_annotations: Pod annotations as key=value,key=value.
        training_lora_r: LoRA rank (controls model capacity).
        training_lora_alpha: LoRA scaling factor.
        training_lora_dropout: Dropout rate for LoRA layers.
        training_lora_target_modules: Comma-separated list of modules to apply LoRA (empty = auto-detect).
        training_lora_use_rslora: Use Rank-Stabilized LoRA variant.
        training_lora_use_dora: Use Weight-Decomposed LoRA (DoRA).
        training_lora_load_in_4bit: Enable 4-bit quantization (QLoRA).
        training_lora_load_in_8bit: Enable 8-bit quantization.
        training_lora_bnb_4bit_quant_type: Quantization type (e.g., "nf4"). Reserved for future training_hub support.
        training_lora_bnb_4bit_compute_dtype: Compute dtype (e.g., "bfloat16"). Reserved for future support.
        training_lora_bnb_4bit_use_double_quant: Enable double quantization. Reserved for future training_hub support.
        training_lora_sample_packing: Pack multiple samples for efficiency (default: True in training_hub).
        training_seed: Random seed for reproducibility.
        training_use_liger: Enable Liger kernel optimizations.
        training_lr_scheduler: LR scheduler type (cosine, linear, etc.). Training_hub default: linear.
        kubernetes_config: KFP TaskConfig for volumes/env/resources passthrough.

    Environment:
        HF_TOKEN: HuggingFace token for gated models (read from environment).
        OCI_PULL_SECRET_MODEL_DOWNLOAD: Docker config.json content for pulling OCI model images.
    """
    import os
    from typing import Dict

    from kfp_components.components.training.shared import (
        configure_env,
        create_logger,
        download_oci_model,
        init_k8s,
        parse_kv,
        persist_model,
        plot_training_loss,
        prepare_jsonl,
        resolve_dataset,
        setup_hf_token,
    )

    log = create_logger("train_model")
    log.info(f"Initializing LoRA training component with: pvc={pvc_path}, model={training_base_model}")

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

        def _select_runtime(c):
            for r in c.list_runtimes():
                if getattr(r, "name", "") == "training-hub":
                    log.info(f"Runtime: {r}")
                    return r
            raise RuntimeError("Runtime 'training-hub' not found")

        runtime = _select_runtime(client)

        def _int(v, d: int) -> int:
            if v is None:
                return d
            if isinstance(v, int):
                return v
            s = str(v).strip()
            return int(s) if s else d

        def _nproc():
            auto = str(training_resource_num_procs_per_worker).strip().lower() == "auto"
            np = training_resource_gpu_per_worker if auto else _int(training_resource_num_procs_per_worker, 1)
            # TODO: LoRA (unsloth backend) only supports single-node training.
            # Hardcoded to 1 until unsloth/training_hub add multi-node LoRA support.
            nn = 1
            return max(np, 1), nn

        def _params() -> Dict:
            np, nn = _nproc()
            b = {
                "model_path": resolved,
                "data_path": jsonl if os.path.exists(jsonl) else ds_dir,
                "effective_batch_size": int(training_effective_batch_size or 128),
                "max_tokens_per_gpu": int(training_max_tokens_per_gpu),
                "max_seq_len": int(training_max_seq_len or 8192),
                "learning_rate": float(training_learning_rate or 2e-4),
                "backend": "unsloth",
                "ckpt_output_dir": ckpt_dir,
                "data_output_dir": training_data_output_dir or os.path.join(ckpt_dir, "_internal_data_processing"),
                "warmup_steps": int(training_lr_warmup_steps) if training_lr_warmup_steps is not None else 10,
                "checkpoint_at_epoch": bool(training_checkpoint_at_epoch)
                if training_checkpoint_at_epoch is not None
                else False,
                "num_epochs": int(training_num_epochs) if training_num_epochs else 3,
                "nproc_per_node": np,
                "nnodes": nn,
            }

            # LoRA-specific parameters
            b["lora_r"] = int(training_lora_r)
            b["lora_alpha"] = int(training_lora_alpha)
            b["lora_dropout"] = float(training_lora_dropout)

            if training_lora_target_modules:
                b["target_modules"] = [m.strip() for m in training_lora_target_modules.split(",") if m.strip()]

            if training_lora_use_rslora is not None:
                b["use_rslora"] = bool(training_lora_use_rslora)
            if training_lora_use_dora is not None:
                b["use_dora"] = bool(training_lora_use_dora)

            # QLoRA parameters
            if training_lora_load_in_4bit is not None:
                b["load_in_4bit"] = bool(training_lora_load_in_4bit)
            if training_lora_load_in_8bit is not None:
                b["load_in_8bit"] = bool(training_lora_load_in_8bit)
            # NOTE: bnb_4bit_quant_type, bnb_4bit_compute_dtype, bnb_4bit_use_double_quant
            # are accepted by training_hub's lora_sft() API but _load_unsloth_model() passes
            # them as raw kwargs to FastLanguageModel.from_pretrained() which causes a
            # TypeError. Omitted until training_hub wraps them in BitsAndBytesConfig.
            # Unsloth defaults to nf4 quantization when load_in_4bit=True.

            # Other optional parameters
            if training_lora_sample_packing is not None:
                b["sample_packing"] = bool(training_lora_sample_packing)
            if training_lr_scheduler:
                b["lr_scheduler"] = training_lr_scheduler
            if training_use_liger is not None:
                b["use_liger"] = bool(training_use_liger)
            if training_seed is not None:
                b["seed"] = int(training_seed)

            return b

        params = _params()

        def _train_func(p):
            import os

            from training_hub import lora_sft as tr

            print("[PY] Launching LoRA training...", flush=True)
            result = tr(**(p or {}))

            # Merge LoRA adapter weights into base model for eval/deployment compatibility.
            # LoRA training saves adapter-only files (adapter_config.json, adapter_model.safetensors)
            # but downstream components (lm-eval, vLLM) expect a full model with config.json.
            #
            # Strategy: Use Unsloth's save_pretrained_merged() which properly dequantizes
            # 4-bit QLoRA weights. Then post-process the saved files to:
            # 1. Strip base_model. prefix from safetensors weight keys
            # 2. Fix sharded index file if present
            # 3. Remove quantization_config from config.json
            ckpt_dir = p.get("ckpt_output_dir")
            if ckpt_dir and result and "model" in result:
                import glob as _glob
                import json

                from safetensors.torch import load_file, save_file

                # Remove any existing files from trainer.save_model()
                for _f in (
                    _glob.glob(ckpt_dir + "/*.safetensors")
                    + _glob.glob(ckpt_dir + "/model.safetensors.index.json")
                    + _glob.glob(ckpt_dir + "/adapter_config.json")
                ):
                    if os.path.exists(_f):
                        os.remove(_f)

                # Let Unsloth handle dequantization and saving
                print("[PY] Merging and saving model (Unsloth merged_16bit)...", flush=True)
                result["model"].save_pretrained_merged(ckpt_dir, result["tokenizer"], save_method="merged_16bit")

                # Post-process: fix weight key prefixes in safetensors files
                for sf_path in sorted(_glob.glob(ckpt_dir + "/*.safetensors")):
                    tensors = load_file(sf_path)
                    clean = {}
                    needs_fix = False
                    for k, v in tensors.items():
                        if k.startswith("base_model.model."):
                            clean[k[len("base_model.model.") :]] = v
                            needs_fix = True
                        elif k.startswith("base_model."):
                            clean[k[len("base_model.") :]] = v
                            needs_fix = True
                        else:
                            clean[k] = v
                    if needs_fix:
                        save_file(clean, sf_path)

                # Post-process: fix weight key prefixes in index file if sharded
                idx_path = ckpt_dir + "/model.safetensors.index.json"
                if os.path.exists(idx_path):
                    with open(idx_path) as f:
                        idx = json.load(f)
                    if "weight_map" in idx:
                        new_map = {}
                        for k, v in idx["weight_map"].items():
                            if k.startswith("base_model.model."):
                                new_map[k[len("base_model.model.") :]] = v
                            elif k.startswith("base_model."):
                                new_map[k[len("base_model.") :]] = v
                            else:
                                new_map[k] = v
                        idx["weight_map"] = new_map
                        with open(idx_path, "w") as f:
                            json.dump(idx, f, indent=2)

                # Post-process: remove quantization_config from config.json
                cfg_path = ckpt_dir + "/config.json"
                if os.path.exists(cfg_path):
                    with open(cfg_path) as f:
                        cfg = json.load(f)
                    if "quantization_config" in cfg:
                        del cfg["quantization_config"]
                        with open(cfg_path, "w") as f:
                            json.dump(cfg, f, indent=2)

                print("[PY] Merged model saved.", flush=True)

            return result

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
                algorithm=TrainingHubAlgorithms.LORA_SFT,
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
        client.wait_for_job_status(name=job, status={"Running"}, timeout=900)
        client.wait_for_job_status(name=job, status={"Complete", "Failed"}, timeout=1800)
        j = client.get_job(name=job)
        if getattr(j, "status", None) == "Failed":
            log.error(f"Job failed: {j.status}")
            raise RuntimeError(f"Job failed: {j.status}")
        elif getattr(j, "status", None) != "Complete":
            log.error(f"Unexpected status: {j.status}")
            raise RuntimeError(f"Unexpected status: {j.status}")
        log.info("Training completed successfully")
    except Exception as e:
        log.error(f"Training failed: {e}")
        raise

    def get_training_metrics(root: str):
        # TODO: Add loss chart support for LoRA once training_hub exposes a metrics/logging
        # file for the unsloth backend. Blocked on:
        # https://github.com/Red-Hat-AI-Innovation-Team/training_hub/pull/40
        return {}, []

    def log_training_metrics():
        output_metrics.log_metric("num_epochs", float(params.get("num_epochs") or 3))
        output_metrics.log_metric("effective_batch_size", float(params.get("effective_batch_size") or 128))
        output_metrics.log_metric("learning_rate", float(params.get("learning_rate") or 2e-4))
        output_metrics.log_metric("max_seq_len", float(params.get("max_seq_len") or 8192))
        output_metrics.log_metric("max_tokens_per_gpu", float(params.get("max_tokens_per_gpu") or 0))
        output_metrics.log_metric("lora_r", float(params.get("lora_r") or 16))
        output_metrics.log_metric("lora_alpha", float(params.get("lora_alpha") or 32))
        output_metrics.log_metric("lora_dropout", float(params.get("lora_dropout") or 0.0))
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
