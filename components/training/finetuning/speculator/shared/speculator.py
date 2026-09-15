"""Runtime shared by the mode-specific Speculator KFP components."""

import os
from typing import Any


def run_speculator(mode: str, values: dict[str, Any]) -> str:
    """Run one fixed Speculator mode from a KFP launcher."""
    try:
        from output import persist_model
        from setup import configure_env, create_logger, init_k8s, setup_hf_token
        from training import select_runtime, wait_for_training_job
    except ModuleNotFoundError:
        from shared.output import persist_model
        from shared.setup import configure_env, create_logger, init_k8s, setup_hf_token
        from shared.training import select_runtime, wait_for_training_job

    log = create_logger(f"speculator_{mode}")
    pvc_path = values["pvc_path"]
    persistent_pvc = values.get("persistent_pvc") or None
    persistent_mount_path = values.get("persistent_mount_path", "/mnt/persistent")
    defaults = {
        "data_extraction_concurrency": 4,
        "data_extraction_max_samples": None,
        "data_extraction_regenerate_responses": False,
        "training_num_layers": 1,
        "training_ttt_steps": 1,
        "training_norm_before_residual": True,
        "training_norm_before_fc": False,
        "training_embed_requires_grad": False,
        "training_hidden_states_dtype": "bfloat16",
        "training_scheduler_type": "linear",
        "training_scheduler_num_cosine_cycles": 0.5,
        "training_checkpoint_freq": 1.0,
        "training_save_best": False,
        "training_log_freq": 1,
        "training_resume_from_checkpoint": False,
        "training_resource_cpu": "4",
        "training_resource_gpu": 1,
        "training_resource_memory": "64Gi",
        "vllm_resource_gpu": 1,
        "vllm_resource_memory": "96Gi",
        "vllm_gpu_memory_utilization": 0.9,
        "vllm_readiness_timeout_minutes": 60,
        "enable_progression_tracking": True,
        "metrics_port": 28080,
        "metrics_poll_interval_seconds": 30,
        "training_epochs": 3,
        "training_lr": 1e-4,
        "speculator_type": "eagle3",
        "total_seq_len": 2048,
        "training_envs": "",
        "training_runtime": "speculator-model-opt-cuda",
    }
    for key, default in defaults.items():
        values.setdefault(key, default)

    def pvc_parts(uri: str) -> tuple[str | None, str]:
        if uri.startswith("pvc://"):
            claim, _, path = uri.removeprefix("pvc://").partition("/")
            return claim, path
        return None, uri.lstrip("/")

    from kubernetes import client as k8s

    k8s_config = values.get("kubernetes_config")
    if not k8s_config:
        raise RuntimeError("KFP Kubernetes task configuration is required to resolve the workspace PVC")
    api_client = k8s.ApiClient()
    volumes = [api_client.sanitize_for_serialization(x) for x in (getattr(k8s_config, "volumes", None) or [])]
    mounts = [api_client.sanitize_for_serialization(x) for x in (getattr(k8s_config, "volume_mounts", None) or [])]
    workspace_mount = next((x for x in mounts if x.get("mountPath") == pvc_path), None)
    if not workspace_mount:
        raise RuntimeError(f"Could not find the KFP workspace mount at {pvc_path}")
    workspace_name = workspace_mount.get("name")
    workspace_volume = next((x for x in volumes if x.get("name") == workspace_name), None)
    workspace_claim = (workspace_volume or {}).get("persistentVolumeClaim", {}).get("claimName")
    if not workspace_claim:
        raise RuntimeError(f"KFP workspace volume {workspace_name!r} is not backed by a PVC")

    def trainer_path(uri: str | None) -> str | None:
        if not uri:
            return uri
        claim, path = pvc_parts(uri)
        return uri if claim else f"pvc://{persistent_pvc or workspace_claim}/{path}"

    def local_path(uri: str) -> str:
        if not uri:
            return persistent_mount_path if persistent_pvc else pvc_path
        claim, path = pvc_parts(uri)
        if persistent_pvc and not claim:
            return os.path.join(persistent_mount_path, path)
        if persistent_pvc and claim == persistent_pvc:
            return os.path.join(persistent_mount_path, path)
        if claim and claim != workspace_claim:
            raise ValueError(
                f"PVC {claim!r} is not the KFP workspace PVC {workspace_claim!r}; "
                "mount external PVCs explicitly before using them"
            )
        return os.path.join(pvc_path, path)

    for key in (
        "dataset_name",
        "hidden_states_path",
        "training_data_path",
        "vllm_endpoint",
        "verifier_model_pvc",
        "target_layer_ids",
        "training_from_pretrained",
        "packages_to_install",
        "pip_index_urls",
    ):
        if not values.get(key):
            values[key] = None
    for key in (
        "data_extraction_max_samples",
        "training_draft_vocab_size",
        "training_scheduler_warmup_steps",
        "training_scheduler_total_steps",
    ):
        if values.get(key) == 0:
            values[key] = None

    if mode == "data_only" and values.get("vllm_source") == "remote":
        values["verifier_model"] = values.get("verifier_model_pvc") or values["verifier_model"]
    elif mode in ("train_only", "online") and values.get("verifier_model_pvc"):
        values["verifier_model"] = values["verifier_model_pvc"]
    external = values.get("vllm_endpoint") is not None
    trainer_verifier_model = (
        trainer_path(values["verifier_model"])
        if external
        or values.get("vllm_source") == "remote"
        or mode == "offline"
        or values.get("verifier_model_pvc")
        or values["verifier_model"].startswith("pvc://")
        else values["verifier_model"]
    )
    if values.get("download_model") and not external:
        model_dir_name = values["verifier_model"].replace("/", "--")
        trainer_verifier_model = trainer_path(f"models/{model_dir_name}")
    trainer_output_dir = trainer_path(values["output_dir"])
    trainer_hidden_states_path = trainer_path(values.get("hidden_states_path"))
    trainer_training_data_path = trainer_path(values.get("training_data_path"))
    if external:
        if not trainer_hidden_states_path or not trainer_verifier_model.startswith("pvc://"):
            raise ValueError("External vLLM requires a pvc:// verifier_model and hidden_states_path")
        if not values.get("target_layer_ids"):
            raise ValueError("target_layer_ids is required with an external vLLM endpoint")

    log.info("Initializing Speculator %s mode", mode)
    api = init_k8s(log)
    cache_root = os.path.join(pvc_path, ".cache", "huggingface")
    default_env = {
        "XDG_CACHE_HOME": "/tmp",
        "TRITON_CACHE_DIR": "/tmp/.triton",
        "HF_HOME": "/tmp/.cache/huggingface",
        "HF_DATASETS_CACHE": os.path.join(cache_root, "datasets"),
        "TRANSFORMERS_CACHE": os.path.join(cache_root, "transformers"),
        "NCCL_DEBUG": "INFO",
        "PYTHONUNBUFFERED": "1",
    }
    env = configure_env(values["training_envs"], default_env, log)
    setup_hf_token(env, values["verifier_model"], log)

    try:
        from kubeflow.common.types import KubernetesBackendConfig
        from kubeflow.trainer import TrainerClient
        from kubeflow.trainer.rhai import SpeculativeDecodingTrainer, SpeculatorMode, SpeculatorType
        from kubeflow.trainer.rhai.speculator import SpeculatorConfig

        if api is None:
            raise RuntimeError("Kubernetes API not initialized")
        client = TrainerClient(KubernetesBackendConfig(client_configuration=api.configuration))
        runtime = select_runtime(client, log, runtime_name=values["training_runtime"].strip())
        modes = {
            "data_only": SpeculatorMode.DATA_ONLY,
            "train_only": SpeculatorMode.TRAIN_ONLY,
            "online": SpeculatorMode.ONLINE,
        }
        types = {
            "eagle3": SpeculatorType.EAGLE3,
            "dflash": SpeculatorType.DFLASH,
            "mtp": SpeculatorType.MTP,
            "peagle": SpeculatorType.PEAGLE,
        }
        speculator_type = values["speculator_type"].lower()
        if speculator_type not in types:
            raise ValueError(f"Unknown speculator_type {values['speculator_type']!r}; expected one of {sorted(types)}")
        config_values = {
            "num_layers": values["training_num_layers"],
            "ttt_steps": values["training_ttt_steps"],
            "norm_before_residual": values["training_norm_before_residual"],
            "norm_before_fc": values["training_norm_before_fc"],
            "embed_requires_grad": values["training_embed_requires_grad"],
            "hidden_states_dtype": values["training_hidden_states_dtype"],
            "scheduler_type": values["training_scheduler_type"],
            "scheduler_num_cosine_cycles": values["training_scheduler_num_cosine_cycles"],
            "checkpoint_freq": values["training_checkpoint_freq"],
            "save_best": values["training_save_best"],
            "log_freq": values["training_log_freq"],
            "resume_from_checkpoint": values["training_resume_from_checkpoint"],
            "datagen_concurrency": values["data_extraction_concurrency"],
        }
        for name in ("training_scheduler_warmup_steps", "training_scheduler_total_steps"):
            if values.get(name) is not None:
                config_values[name.removeprefix("training_")] = values[name]
        if values.get("target_layer_ids"):
            config_values["target_layer_ids"] = [int(x.strip()) for x in values["target_layer_ids"].split(",")]
        if values.get("training_from_pretrained"):
            config_values["from_pretrained"] = values["training_from_pretrained"]
        config = SpeculatorConfig(**config_values)
        log.info("SpeculatorConfig target_layer_ids=%s", config.target_layer_ids)
        training_resources = {
            "nvidia.com/gpu": values["training_resource_gpu"],
            "memory": values["training_resource_memory"],
            "cpu": values["training_resource_cpu"],
        }
        vllm_resources = {
            "nvidia.com/gpu": values["vllm_resource_gpu"],
            "memory": values["vllm_resource_memory"],
            "cpu": 4,
        }
        params = {
            "verifier_model": trainer_verifier_model,
            "mode": modes[mode],
            "output_dir": trainer_output_dir,
            "speculator_type": types[speculator_type],
            "total_seq_len": values["total_seq_len"],
            "vllm_gpu_memory_utilization": values["vllm_gpu_memory_utilization"],
            "vllm_readiness_timeout_minutes": values["vllm_readiness_timeout_minutes"],
            "enable_progression_tracking": values["enable_progression_tracking"],
            "metrics_port": values["metrics_port"],
            "metrics_poll_interval_seconds": values["metrics_poll_interval_seconds"],
        }
        if values.get("packages_to_install"):
            params["packages_to_install"] = [x.strip() for x in values["packages_to_install"].split(",")]
        if values.get("pip_index_urls"):
            params["pip_index_urls"] = [x.strip() for x in values["pip_index_urls"].split(",")]
        if mode == "data_only":
            params["dataset_name"] = values["dataset_name"]
            if values.get("data_extraction_max_samples") is not None:
                params["max_samples"] = values["data_extraction_max_samples"]
            if values["data_extraction_regenerate_responses"]:
                params["regenerate_responses"] = True
            if external:
                params.update(
                    vllm_endpoint=values["vllm_endpoint"],
                    hidden_states_path=trainer_hidden_states_path,
                    config=config,
                )
            else:
                params["vllm_resources"] = vllm_resources
        elif mode == "online":
            params.update(
                dataset_name=values["dataset_name"],
                vllm_resources=vllm_resources,
                training_resources=training_resources,
                epochs=values["training_epochs"],
                lr=values["training_lr"],
                config=config,
            )
            if values.get("training_draft_vocab_size") is not None:
                params["draft_vocab_size"] = values["training_draft_vocab_size"]
            if values.get("data_extraction_max_samples") is not None:
                params["max_samples"] = values["data_extraction_max_samples"]
            if values["data_extraction_regenerate_responses"]:
                params["regenerate_responses"] = True
        else:
            params.update(
                hidden_states_path=trainer_hidden_states_path,
                data_path=trainer_training_data_path or trainer_output_dir,
                training_resources=training_resources,
                epochs=values["training_epochs"],
                lr=values["training_lr"],
                config=config,
            )
            if values.get("training_draft_vocab_size") is not None:
                params["draft_vocab_size"] = values["training_draft_vocab_size"]
        trainer = SpeculativeDecodingTrainer(**params)

        volumes, mounts = [], []
        k8s_config = values.get("kubernetes_config")
        if k8s_config:
            volumes = list(getattr(k8s_config, "volumes", None) or [])
            mounts = list(getattr(k8s_config, "volume_mounts", None) or [])
        if volumes or mounts:
            from kubernetes import client as k8s

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

            api_client = k8s.ApiClient()
            volumes = [api_client.sanitize_for_serialization(x) if not isinstance(x, dict) else x for x in volumes]
            mounts = [api_client.sanitize_for_serialization(x) if not isinstance(x, dict) else x for x in mounts]
            # The Trainer SDK mounts every PVC URI at its checkpoint path. Do not
            # add the same user PVC again at the launcher mount path when the
            # output/model storage uses that PVC.
            storage_claims = set()
            for storage_uri in (trainer_output_dir, trainer_hidden_states_path, trainer_training_data_path):
                claim, _ = pvc_parts(storage_uri or "")
                if claim:
                    storage_claims.add(claim)
            duplicate_volume_names = {
                volume.get("name")
                for volume in volumes
                if volume.get("persistentVolumeClaim", {}).get("claimName") in storage_claims
            }
            if duplicate_volume_names:
                mounts = [mount for mount in mounts if mount.get("name") not in duplicate_volume_names]
                volumes = [volume for volume in volumes if volume.get("name") not in duplicate_volume_names]
            workspace_mount = next((x for x in mounts if x.get("mountPath") == pvc_path), None)
            if workspace_mount:
                # KFP already mounts the workspace in the TrainJob pod. Do not
                # patch that mount a second time. The Trainer SDK owns its
                # /mnt/kubeflow-checkpoints mount and adds it separately.
                workspace_name = workspace_mount.get("name", "kfp-workspace")
                mounts = [x for x in mounts if x.get("name") != workspace_name]
                volumes = [x for x in volumes if x.get("name") != workspace_name]
            mounts = [x for x in mounts if x.get("mountPath") != "/mnt/kubeflow-checkpoints"]
            if volumes or mounts:
                pod_spec = PodSpecPatch(
                    volumes=volumes,
                    containers=[ContainerPatch(name="node", volume_mounts=mounts)],
                )
                options = [
                    RuntimePatch(
                        training_runtime_spec=TrainingRuntimeSpecPatch(
                            template=JobSetTemplatePatch(
                                spec=JobSetSpecPatch(
                                    replicated_jobs=[
                                        ReplicatedJobPatch(
                                            name="node",
                                            template=JobTemplatePatch(
                                                spec=JobSpecPatch(template=PodTemplatePatch(spec=pod_spec))
                                            ),
                                        )
                                    ]
                                )
                            )
                        )
                    )
                ]
            else:
                options = None
        else:
            options = None
        job = client.train(trainer=trainer, options=options, runtime=runtime)
        wait_for_training_job(
            client,
            job,
            log,
        )
    except Exception:
        log.error("Speculator %s failed", mode)
        raise

    if mode == "data_only":
        hidden_states_dir = local_path(
            values["hidden_states_path"] if external else values["output_dir"] + "/hidden_states"
        )
        output = values.get("output_hidden_states")
        if output:
            output.uri = hidden_states_dir
            output.metadata["pvc_path"] = hidden_states_dir
        return "data_only completed - hidden states saved"
    model_dir = os.path.join(local_path(values["output_dir"]), "checkpoint_best")
    if values.get("output_model"):
        persist_model(
            model_dir,
            persistent_mount_path if persistent_pvc else pvc_path,
            values["verifier_model"],
            values["output_model"],
            log,
            prefer_best=True,
            strict_output=True,
        )
    if values.get("output_metrics"):
        values["output_metrics"].log_metric("mode", mode)
        values["output_metrics"].log_metric("training_epochs", float(values["training_epochs"]))
        values["output_metrics"].log_metric("learning_rate", float(values["training_lr"]))
    return f"{mode} completed - model trained"
