"""LoRA Training Component.

Reusable inline LoRA (Low-Rank Adaptation) training component.
- Configurable logging
- Optional Kubernetes connection (remote or in-cluster)
- PVC-based caches/checkpoints
- Dataset resolution (HF repo id, or local path)
- Basic metrics logging and checkpoint export
- Hardcoded to use unsloth backend and LoRA algorithm
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
    # Batch params
    training_micro_batch_size: Optional[int] = None,
    training_gradient_accumulation_steps: Optional[int] = None,
    # Optimization params
    training_flash_attention: Optional[bool] = None,
    training_bf16: Optional[bool] = None,
    training_fp16: Optional[bool] = None,
    training_tf32: Optional[bool] = None,
    # Saving/Logging params
    training_save_steps: Optional[int] = None,
    training_eval_steps: Optional[int] = None,
    training_logging_steps: Optional[int] = None,
    training_save_total_limit: Optional[int] = None,
    # Logging integration params
    training_wandb_project: Optional[str] = None,
    training_wandb_entity: Optional[str] = None,
    training_wandb_run_name: Optional[str] = None,
    training_tensorboard_log_dir: Optional[str] = None,
    training_mlflow_tracking_uri: Optional[str] = None,
    training_mlflow_experiment_name: Optional[str] = None,
    training_mlflow_run_name: Optional[str] = None,
    # Dataset format params
    training_dataset_type: Optional[str] = None,
    training_field_messages: Optional[str] = None,
    training_field_instruction: Optional[str] = None,
    training_field_input: Optional[str] = None,
    training_field_output: Optional[str] = None,
    # Multi-GPU params
    training_enable_model_splitting: Optional[bool] = None,
    training_runtime: str = "training-hub",
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
        training_micro_batch_size: Micro batch size per GPU.
        training_gradient_accumulation_steps: Gradient accumulation steps.
        training_flash_attention: Enable flash attention.
        training_bf16: Use bfloat16 precision.
        training_fp16: Use float16 precision.
        training_tf32: Enable TF32 on Ampere+ GPUs.
        training_save_steps: Save checkpoint every N steps.
        training_eval_steps: Run evaluation every N steps.
        training_logging_steps: Log metrics every N steps.
        training_save_total_limit: Max checkpoints to keep.
        training_wandb_project: Weights & Biases project name.
        training_wandb_entity: Weights & Biases entity/team.
        training_wandb_run_name: Weights & Biases run name.
        training_tensorboard_log_dir: TensorBoard log directory.
        training_mlflow_tracking_uri: MLflow tracking server URI.
        training_mlflow_experiment_name: MLflow experiment name.
        training_mlflow_run_name: MLflow run name.
        training_dataset_type: Dataset format type.
        training_field_messages: Field name for messages in dataset.
        training_field_instruction: Field name for instruction in dataset.
        training_field_input: Field name for input in dataset.
        training_field_output: Field name for output in dataset.
        training_enable_model_splitting: Enable model splitting across GPUs.
        training_runtime: Name of the ClusterTrainingRuntime to use.
        kubernetes_config: KFP TaskConfig for volumes/env/resources passthrough.

    Environment:
        HF_TOKEN: HuggingFace token for gated models (read from environment).
        OCI_PULL_SECRET_MODEL_DOWNLOAD: Docker config.json content for pulling OCI model images.
    """
    import os
    from typing import Dict

    from data import download_oci_model, prepare_jsonl, resolve_dataset
    from output import persist_model, plot_training_loss
    from setup import configure_env, create_logger, init_k8s, parse_kv, setup_hf_token
    from training import compute_nproc, select_runtime, wait_for_training_job

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

    # Convert non-chat datasets to chat template format.
    # training_hub always applies format_chat_template() which expects a
    # "messages" column.  When the user supplies field mappings for
    # instruction/input/output style datasets we rewrite the JSONL here
    # so training_hub sees the chat structure it expects.
    if os.path.exists(jsonl) and (training_field_instruction or training_field_output):
        import json as _json

        f_inst = (training_field_instruction or "").strip()
        f_inp = (training_field_input or "").strip()
        f_out = (training_field_output or "").strip()

        converted_rows: list = []
        needs_convert = False

        with open(jsonl) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = _json.loads(line)
                if "messages" in row or "conversations" in row:
                    converted_rows.append(row)
                    continue
                needs_convert = True
                user_parts: list[str] = []
                if f_inst and f_inst in row:
                    user_parts.append(str(row[f_inst]))
                if f_inp and f_inp in row:
                    user_parts.append(str(row[f_inp]))
                assistant_text = str(row.get(f_out, "")) if f_out else ""
                converted_rows.append(
                    {
                        "messages": [
                            {"role": "user", "content": "\n".join(user_parts)},
                            {"role": "assistant", "content": assistant_text},
                        ]
                    }
                )

        if needs_convert:
            with open(jsonl, "w") as fh:
                for row in converted_rows:
                    fh.write(_json.dumps(row, ensure_ascii=False) + "\n")
            log.info(
                f"Converted {len(converted_rows)} rows from instruction format "
                f"(fields: {f_inst}/{f_inp}/{f_out}) to chat template format"
            )

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

        def _params() -> Dict:
            # LoRA (unsloth backend) only supports single-node training.
            np, nn = compute_nproc(
                training_resource_gpu_per_worker,
                training_resource_num_procs_per_worker,
                single_node=True,
            )
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
            if training_lora_load_in_4bit and training_lora_load_in_8bit:
                raise ValueError("Cannot enable both 4-bit and 8-bit quantization. Choose one or neither.")
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

            # Batch params
            if training_micro_batch_size is not None:
                b["micro_batch_size"] = int(training_micro_batch_size)
            if training_gradient_accumulation_steps is not None:
                b["gradient_accumulation_steps"] = int(training_gradient_accumulation_steps)

            # Optimization params
            if training_flash_attention is not None:
                b["flash_attention"] = bool(training_flash_attention)
            if training_bf16 is not None:
                b["bf16"] = bool(training_bf16)
            if training_fp16 is not None:
                b["fp16"] = bool(training_fp16)
            if training_tf32 is not None:
                b["tf32"] = bool(training_tf32)

            # Saving/Logging params
            if training_save_steps is not None:
                b["save_steps"] = int(training_save_steps)
            if training_eval_steps is not None:
                b["eval_steps"] = int(training_eval_steps)
            if training_logging_steps is not None:
                b["logging_steps"] = int(training_logging_steps)
            if training_save_total_limit is not None:
                b["save_total_limit"] = int(training_save_total_limit)

            # Logging integration params
            if training_wandb_project:
                b["wandb_project"] = training_wandb_project
            if training_wandb_entity:
                b["wandb_entity"] = training_wandb_entity
            if training_wandb_run_name:
                b["wandb_run_name"] = training_wandb_run_name
            if training_tensorboard_log_dir:
                b["tensorboard_log_dir"] = training_tensorboard_log_dir
            if training_mlflow_tracking_uri:
                b["mlflow_tracking_uri"] = training_mlflow_tracking_uri
            if training_mlflow_experiment_name:
                b["mlflow_experiment_name"] = training_mlflow_experiment_name
            if training_mlflow_run_name:
                b["mlflow_run_name"] = training_mlflow_run_name

            # Dataset format params
            if training_dataset_type:
                b["dataset_type"] = training_dataset_type
            if training_field_messages:
                b["field_messages"] = training_field_messages
            if training_field_instruction:
                b["field_instruction"] = training_field_instruction
            if training_field_input:
                b["field_input"] = training_field_input
            if training_field_output:
                b["field_output"] = training_field_output

            # Multi-GPU params
            if training_enable_model_splitting is not None:
                b["enable_model_splitting"] = bool(training_enable_model_splitting)

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
        wait_for_training_job(client, job, log)
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
