"""Training Component.

Reusable inline training component modeled after the OSFT notebook flow.
- Configurable logging
- Optional Kubernetes connection (remote or in-cluster)
- PVC-based caches/checkpoints
- Dataset resolution (HF repo id, or local path)
- Basic metrics logging and checkpoint export
"""

from typing import Optional

from kfp import dsl


@dsl.component(
    base_image="quay.io/opendatahub/odh-training-th04-cpu-torch29-py312-rhel9:cpu-3.3",
    packages_to_install=["kubernetes", "olot", "matplotlib"],
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
    training_algorithm: str = "OSFT",
    training_effective_batch_size: int = 128,
    training_max_tokens_per_gpu: int = 64000,
    training_max_seq_len: int = 8192,
    training_learning_rate: Optional[float] = None,
    training_backend: str = "mini-trainer",
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
    training_save_samples: Optional[int] = None,
    training_accelerate_full_state_at_epoch: Optional[bool] = None,
    training_fsdp_sharding_strategy: Optional[str] = None,
    kubernetes_config: dsl.TaskConfig = None,
) -> str:
    """Train model using TrainingHub (OSFT/SFT). Outputs model artifact and metrics.

    Args:
        pvc_path: Workspace PVC root path (use dsl.WORKSPACE_PATH_PLACEHOLDER).
        output_model: Output model artifact.
        output_metrics: Output training metrics artifact.
        output_loss_chart: Output HTML artifact with training loss chart.
        dataset: Input training dataset artifact.
        training_base_model: Base model (HuggingFace ID or local path).
        training_algorithm: Training algorithm: OSFT or SFT.
        training_effective_batch_size: Effective batch size per optimizer step.
        training_max_tokens_per_gpu: Max tokens per GPU (memory cap).
        training_max_seq_len: Max sequence length in tokens.
        training_learning_rate: Learning rate (default: 5e-6).
        training_backend: Backend: mini-trainer (OSFT) or instructlab-training (SFT).
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
        training_unfreeze_rank_ratio: [OSFT] Fraction of parameters to unfreeze.
        training_osft_memory_efficient_init: [OSFT] Use memory-efficient initialization.
        training_target_patterns: [OSFT] Target layer patterns (comma-separated).
        training_seed: Random seed for reproducibility.
        training_use_liger: Enable Liger kernel optimizations.
        training_use_processed_dataset: Use pre-processed dataset.
        training_unmask_messages: Unmask assistant messages during training.
        training_lr_scheduler: LR scheduler type (cosine, linear, etc.).
        training_lr_scheduler_kwargs: LR scheduler kwargs as key=val,key=val.
        training_save_final_checkpoint: Save final checkpoint after training.
        training_save_samples: [SFT] Number of samples to save.
        training_accelerate_full_state_at_epoch: [SFT] Save full accelerate state.
        training_fsdp_sharding_strategy: [SFT] FSDP sharding strategy.
        kubernetes_config: KFP TaskConfig for volumes/env/resources passthrough.

    Environment:
        HF_TOKEN: HuggingFace token for gated models (read from environment).
        OCI_PULL_SECRET_MODEL_DOWNLOAD: Docker config.json content for pulling OCI model images.
    """
    import json
    import logging
    import os
    import shutil
    import subprocess
    import sys
    from typing import Dict, List
    from typing import Optional as _Opt

    def _log():
        lg = logging.getLogger("train_model")
        lg.setLevel(logging.INFO)
        if not lg.handlers:
            h = logging.StreamHandler(sys.stdout)
            h.setLevel(logging.INFO)
            h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            lg.addHandler(h)
        return lg

    log = _log()
    log.info(f"Init: pvc={pvc_path}, model={training_base_model}")

    def find_model_dir(root: str) -> _Opt[str]:
        if not os.path.isdir(root):
            return None
        cands = []
        for r, _, fs in os.walk(root):
            if "config.json" in fs:
                try:
                    cands.append((os.path.getmtime(os.path.join(r, "config.json")), r))
                except OSError:
                    pass
        if not cands:
            latest = None
            for e in os.listdir(root):
                p = os.path.join(root, e)
                if os.path.isdir(p):
                    try:
                        m = os.path.getmtime(p)
                        if latest is None or m > latest[0]:
                            latest = (m, p)
                    except OSError:
                        pass
            return latest[1] if latest else None
        cands.sort(reverse=True)
        return cands[0][1]

    def _init_k8s() -> _Opt[object]:
        try:
            from kubernetes import client as k8s

            srv = os.environ.get("KUBERNETES_SERVER_URL", "").strip()
            tok = os.environ.get("KUBERNETES_AUTH_TOKEN", "").strip()
            if not srv or not tok:
                raise RuntimeError("K8s creds missing: KUBERNETES_SERVER_URL and KUBERNETES_AUTH_TOKEN required")
            log.info("K8s client from env")
            cfg = k8s.Configuration()
            cfg.host, cfg.verify_ssl = srv, False
            cfg.api_key = {"authorization": f"Bearer {tok}"}
            k8s.Configuration.set_default(cfg)
            return k8s.ApiClient(cfg)
        except Exception as e:
            log.warning(f"K8s client init failed: {e}")
            return None

    _api = _init_k8s()

    cache = os.path.join(pvc_path, ".cache", "huggingface")
    denv: Dict[str, str] = {
        "XDG_CACHE_HOME": "/tmp",
        "TRITON_CACHE_DIR": "/tmp/.triton",
        "HF_HOME": "/tmp/.cache/huggingface",
        "HF_DATASETS_CACHE": os.path.join(cache, "datasets"),
        "TRANSFORMERS_CACHE": os.path.join(cache, "transformers"),
        "NCCL_DEBUG": "INFO",
    }

    def parse_kv(s: str) -> Dict[str, str]:
        out = {}
        if not s:
            return out
        for it in s.split(","):
            it = it.strip()
            if not it:
                continue
            if "=" not in it:
                raise ValueError(f"Invalid kv: {it}")
            k, v = it.split("=", 1)
            k, v = k.strip(), v.strip()
            if not k:
                raise ValueError(f"Empty key: {it}")
            out[k] = v
        return out

    def _cfg_env(csv: str, base: Dict[str, str]) -> Dict[str, str]:
        m = {**base, **parse_kv(csv)}
        for k, v in m.items():
            os.environ[k] = v
        log.info(f"Env: {sorted(m.keys())}")
        return m

    menv = _cfg_env(training_envs, denv)

    hf_tok = os.environ.get("HF_TOKEN", "").strip()
    if hf_tok:
        menv["HF_TOKEN"] = hf_tok
        os.environ["HF_TOKEN"] = hf_tok
        log.info("HF_TOKEN propagated")
    elif isinstance(training_base_model, str):
        b = training_base_model.strip()
        if b.startswith("hf://") or ("/" in b and not b.startswith("oci://") and not os.path.exists(b)):
            log.warning(f"HF_TOKEN not set; only public models accessible for '{training_base_model}'")

    from datasets import load_dataset, load_from_disk

    ds_dir = os.path.join(pvc_path, "dataset", "train")
    os.makedirs(ds_dir, exist_ok=True)

    def _resolve_ds(inp, out_dir: str):
        if os.path.isdir(out_dir) and any(os.scandir(out_dir)):
            log.info(f"Using existing ds: {out_dir}")
            return
        if inp and getattr(inp, "path", None) and os.path.exists(inp.path):
            src = inp.path
            if os.path.isdir(src):
                log.info(f"Copy ds dir: {src}")
                shutil.copytree(src, out_dir, dirs_exist_ok=True)
            else:
                log.info(f"Copy ds file: {src}")
                dst = os.path.join(out_dir, os.path.basename(src))
                if not os.path.splitext(dst)[1]:
                    dst = os.path.join(out_dir, "train.jsonl")
                shutil.copy2(src, dst)
            return
        rp = ""
        try:
            if inp and hasattr(inp, "metadata") and isinstance(inp.metadata, dict):
                pvc_m = (inp.metadata.get("pvc_path") or inp.metadata.get("pvc_dir") or "").strip()
                if pvc_m and os.path.exists(pvc_m):
                    if os.path.isdir(pvc_m) and any(os.scandir(pvc_m)):
                        log.info(f"PVC ds dir: {pvc_m}")
                        shutil.copytree(pvc_m, out_dir, dirs_exist_ok=True)
                        return
                    elif os.path.isfile(pvc_m):
                        log.info(f"PVC ds file: {pvc_m}")
                        dst = os.path.join(out_dir, os.path.basename(pvc_m))
                        if not os.path.splitext(dst)[1]:
                            dst = os.path.join(out_dir, "train.jsonl")
                        shutil.copy2(pvc_m, dst)
                        return
                rp = (inp.metadata.get("artifact_path") or "").strip()
        except Exception:
            rp = ""
        if rp:
            if rp.startswith("s3://") or rp.startswith("http://") or rp.startswith("https://"):
                log.info(f"Remote ds: {rp}")
                ext = rp.lower()
                if ext.endswith(".json") or ext.endswith(".jsonl"):
                    ds = load_dataset("json", data_files=rp, split="train")
                elif ext.endswith(".parquet"):
                    ds = load_dataset("parquet", data_files=rp, split="train")
                else:
                    raise ValueError("Unsupported remote format")
                ds.save_to_disk(out_dir)
                return
            else:
                log.info(f"HF ds: {rp}")
                load_dataset(rp, split="train").save_to_disk(out_dir)
                return
        raise ValueError("No dataset provided")

    _resolve_ds(dataset, ds_dir)

    jsonl = os.path.join(ds_dir, "train.jsonl")
    try:
        dsk = load_from_disk(ds_dir)
        tr = dsk["train"] if isinstance(dsk, dict) else dsk
        try:
            tr.to_json(jsonl, lines=True)
            log.info(f"JSONL: {jsonl}")
        except AttributeError:
            with open(jsonl, "w") as f:
                for r in tr:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            log.info(f"JSONL manual: {jsonl}")
    except Exception as e:
        log.warning(f"JSONL export failed: {e}")

    def _skopeo(ref: str, dest: str, auth: str | None = None):
        os.makedirs(dest, exist_ok=True)
        af = None
        if auth:
            af = "/tmp/skopeo-auth.json"
            with open(af, "w") as f:
                f.write(auth)
        cmd = ["skopeo", "copy"]
        if af:
            cmd.extend(["--authfile", af])
        cmd.extend([f"docker://{ref}", f"dir:{dest}"])
        log.info(f"Run: {' '.join(cmd)}")
        r = subprocess.run(cmd, text=True, capture_output=True)
        if r.returncode != 0:
            log.error(f"skopeo fail: {r.stderr}")
            r.check_returncode()

    def _extract(img_dir: str, out: str) -> List[str]:
        import tarfile

        os.makedirs(out, exist_ok=True)
        ext = []
        for fn in os.listdir(img_dir):
            fp = os.path.join(img_dir, fn)
            if not os.path.isfile(fp) or fn.endswith(".json") or fn in {"manifest", "index.json"}:
                continue
            try:
                with tarfile.open(fp, mode="r:*") as tf:
                    for m in tf.getmembers():
                        if m.isfile() and m.name.startswith("models/"):
                            tf.extract(m, path=out)
                            ext.append(m.name)
            except Exception:
                pass
        log.info(f"Extracted: {len(ext)}")
        return ext

    def _find_hf(root: str) -> _Opt[str]:
        wt = {"pytorch_model.bin", "pytorch_model.bin.index.json", "model.safetensors", "model.safetensors.index.json"}
        tk = {"tokenizer.json", "tokenizer.model"}
        for dp, _, fns in os.walk(root):
            fn = set(fns)
            if "config.json" in fn and (fn & wt) and (fn & tk):
                return dp
        return None

    resolved = training_base_model

    def _oci_auth() -> str | None:
        raw = os.environ.get("OCI_PULL_SECRET_MODEL_DOWNLOAD", "").strip()
        if not raw:
            log.warning("OCI_PULL_SECRET_MODEL_DOWNLOAD not set")
            return None
        p = json.loads(raw)
        if not isinstance(p, dict) or not p.get("auths"):
            raise ValueError("Invalid OCI auth")
        return raw

    if isinstance(training_base_model, str) and training_base_model.startswith("oci://"):
        ref = training_base_model[6:]
        img_dir = os.path.join(pvc_path, "model-dir")
        mod_out = os.path.join(pvc_path, "model")
        if os.path.isdir(mod_out):
            shutil.rmtree(mod_out, ignore_errors=True)
        _skopeo(ref, img_dir, _oci_auth())
        _extract(img_dir, mod_out)
        cand = os.path.join(mod_out, "models")
        hfd = _find_hf(cand if os.path.isdir(cand) else mod_out)
        resolved = hfd if hfd else mod_out

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

        def _rt(c):
            for r in c.list_runtimes():
                if getattr(r, "name", "") == "training-hub":
                    log.info(f"Runtime: {r}")
                    return r
            raise RuntimeError("Runtime 'training-hub' not found")

        runtime = _rt(client)

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
            nn = _int(training_resource_num_workers, 1)
            return max(np, 1), max(nn, 1)

        def _params() -> Dict:
            np, nn = _nproc()
            b = {
                "model_path": resolved,
                "data_path": jsonl if os.path.exists(jsonl) else ds_dir,
                "effective_batch_size": int(training_effective_batch_size or 128),
                "max_tokens_per_gpu": int(training_max_tokens_per_gpu),
                "max_seq_len": int(training_max_seq_len or 8192),
                "learning_rate": float(training_learning_rate or 5e-6),
                "backend": training_backend,
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
            algo = (training_algorithm or "").strip().upper()
            if algo == "OSFT":
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
            elif algo == "SFT":
                if training_save_samples is not None:
                    b["save_samples"] = int(training_save_samples)
                if training_accelerate_full_state_at_epoch is not None:
                    b["accelerate_full_state_at_epoch"] = bool(training_accelerate_full_state_at_epoch)
                if training_lr_scheduler:
                    b["lr_scheduler"] = training_lr_scheduler
                if training_use_liger is not None:
                    b["use_liger"] = bool(training_use_liger)
            if training_fsdp_sharding_strategy:
                b["fsdp_sharding_strategy"] = training_fsdp_sharding_strategy.upper().strip()
            return b

        params = _params()
        algo_str = (training_algorithm or "").strip().upper()
        use_sft = algo_str == "SFT"
        algo_val = TrainingHubAlgorithms.SFT if use_sft else TrainingHubAlgorithms.OSFT
        if algo_val == TrainingHubAlgorithms.OSFT:
            params["unfreeze_rank_ratio"] = float(training_unfreeze_rank_ratio)
        params["_algorithm"] = "sft" if use_sft else "osft"

        def _train_func(p):
            a = dict(p or {})
            algo = a.pop("_algorithm", "sft")
            fsdp = a.pop("fsdp_sharding_strategy", None)
            if algo == "sft":
                from training_hub import sft as tr
            else:
                from training_hub import osft as tr
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
                except ImportError:
                    pass
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
                algorithm=algo_val,
                packages_to_install=[],
                env=dict(menv),
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
            raise RuntimeError(f"Job failed: {j.status}")
        elif getattr(j, "status", None) != "Complete":
            raise RuntimeError(f"Unexpected status: {j.status}")
        log.info("Job complete")
    except Exception as e:
        log.error(f"Training failed: {e}")
        raise

    def _metrics(root: str, algo: str = "osft"):
        pats = ["training_metrics_0.jsonl", "training_params_and_metrics_global0.jsonl"]
        if algo.lower() == "sft":
            pats = pats[::-1]
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
        met, loss = {}, []
        with open(mf) as f:
            for ln in f:
                if ln.strip():
                    try:
                        e = json.loads(ln)
                        for s, d in [
                            ("loss", "loss"),
                            ("avg_loss", "loss"),
                            ("lr", "learning_rate"),
                            ("grad_norm", "grad_norm"),
                            ("gradnorm", "grad_norm"),
                            ("val_loss", "eval_loss"),
                            ("epoch", "epoch"),
                            ("step", "step"),
                        ]:
                            if s in e and d not in met:
                                try:
                                    met[d] = float(e[s])
                                except (ValueError, TypeError):
                                    pass
                        lv = e.get("loss") or e.get("avg_loss")
                        if lv:
                            try:
                                loss.append(float(lv))
                            except (ValueError, TypeError):
                                pass
                    except Exception:
                        pass
        if loss:
            met["final_loss"], met["min_loss"] = loss[-1], min(loss)
            met["final_perplexity"] = __import__("math").exp(min(loss[-1], 10))
        return met, loss

    def _chart(loss: list, path: str):
        import base64
        from io import BytesIO

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not loss:
            with open(path, "w") as f:
                f.write("<html><body><p>No loss data.</p></body></html>")
            return
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(range(1, len(loss) + 1), loss, "b-", linewidth=2, label="Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ml, mi = min(loss), loss.index(min(loss))
        ax.annotate(
            f"Min:{ml:.4f}",
            xy=(mi + 1, ml),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="green"),
            color="green",
        )
        ax.annotate(
            f"Final:{loss[-1]:.4f}",
            xy=(len(loss), loss[-1]),
            xytext=(-60, 10),
            textcoords="offset points",
            fontsize=9,
            color="red",
        )
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        img = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        html = (
            f"<!DOCTYPE html><html><head><style>*{{margin:0;padding:0}}"
            f"body{{font-family:sans-serif;padding:5px}}.s{{font-size:10px;margin-bottom:5px}}"
            f".c{{width:100%}}</style></head><body>"
            f'<div class="s">Loss:{loss[0]:.4f}-{loss[-1]:.4f}(min {ml:.4f})|{len(loss)} steps</div>'
            f'<img class="c" src="data:image/png;base64,{img}"/></body></html>'
        )
        with open(path, "w") as f:
            f.write(html)

    def _log_met():
        output_metrics.log_metric("num_epochs", float(params.get("num_epochs") or 1))
        output_metrics.log_metric("effective_batch_size", float(params.get("effective_batch_size") or 128))
        output_metrics.log_metric("learning_rate", float(params.get("learning_rate") or 5e-6))
        output_metrics.log_metric("max_seq_len", float(params.get("max_seq_len") or 8192))
        output_metrics.log_metric("max_tokens_per_gpu", float(params.get("max_tokens_per_gpu") or 0))
        output_metrics.log_metric("unfreeze_rank_ratio", float(params.get("unfreeze_rank_ratio") or 0))
        algo = (training_algorithm or "osft").strip().lower()
        tm, loss = _metrics(ckpt_dir, algo)
        for k, v in tm.items():
            output_metrics.log_metric(f"training_{k}", v)
        _chart(loss, output_loss_chart.path)

    _log_met()

    def _persist():
        latest = find_model_dir(ckpt_dir)
        if not latest:
            raise RuntimeError(f"No model found in {ckpt_dir}")
        log.info(f"Model: {latest}")
        pvc_out = os.path.join(pvc_path, "final_model")
        if os.path.exists(pvc_out):
            shutil.rmtree(pvc_out, ignore_errors=True)
        shutil.copytree(latest, pvc_out, dirs_exist_ok=True)
        log.info(f"Copied to {pvc_out}")
        output_model.name = f"{training_base_model}-checkpoint"
        shutil.copytree(latest, output_model.path, dirs_exist_ok=True)
        log.info(f"Artifact: {output_model.path}")
        output_model.metadata["model_name"] = training_base_model
        output_model.metadata["artifact_path"] = output_model.path
        output_model.metadata["pvc_model_dir"] = pvc_out

    _persist()
    return "training completed"


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(train_model, package_path=__file__.replace(".py", "_component.yaml"))
