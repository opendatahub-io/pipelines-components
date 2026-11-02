"""Runtime shared by the mode-specific Speculator KFP components."""

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def _pvc_parts(uri: str) -> tuple[str | None, str]:
    """Split a Trainer PVC URI into its claim and relative path."""
    if uri.startswith("pvc://"):
        claim, separator, path = uri.removeprefix("pvc://").partition("/")
        if not claim or not separator:
            raise ValueError("PVC paths must use pvc://<claim>/<path>")
        return claim, path
    return None, uri


def _validate_relative_pvc_path(path: str, field_name: str) -> None:
    """Reject PVC paths that are empty, absolute, or contain traversal segments."""
    if not path or path.startswith("/") or any(part in {"", ".", ".."} for part in path.split("/")):
        raise ValueError(f"{field_name} must be a non-empty relative PVC path without '.' or '..' segments")


def _trainer_path(uri: str | None, persistent_pvc: str | None, workspace_claim: str | None) -> str | None:
    """Convert a bare PVC-relative path to a Trainer URI; preserve explicit claims."""
    if not uri:
        return uri
    if "://" in uri and not uri.startswith("pvc://"):
        raise ValueError("Trainer storage paths must be relative paths or pvc:// URIs")
    claim, path = _pvc_parts(uri)
    _validate_relative_pvc_path(path, "Trainer storage path")
    claim = claim or persistent_pvc or workspace_claim
    if not claim:
        raise RuntimeError("A PVC claim is required to resolve Trainer storage paths")
    return f"pvc://{claim}/{path}"


def _local_path(
    uri: str | None,
    persistent_pvc: str | None,
    persistent_mount_path: str,
    workspace_claim: str | None,
    workspace_path: str,
) -> str:
    """Resolve a bare or claim-based PVC path to its mounted local path."""
    if not uri:
        return persistent_mount_path if persistent_pvc else workspace_path
    if "://" in uri and not uri.startswith("pvc://"):
        raise ValueError("Local storage paths must be relative paths or pvc:// URIs")

    claim, path = _pvc_parts(uri)
    claim = claim or persistent_pvc or workspace_claim
    if persistent_pvc and claim == persistent_pvc:
        root = persistent_mount_path
    elif workspace_claim and claim == workspace_claim:
        root = workspace_path
    else:
        raise ValueError(f"PVC {claim!r} is not mounted in this task")

    _validate_relative_pvc_path(path, "PVC storage path")
    root_path = Path(root).resolve()
    resolved = (root_path / path).resolve()
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"PVC storage path must remain within its mounted root: {uri!r}") from exc
    return str(resolved)


def _validate_external_pvc_paths(
    raw_verifier_model: str,
    verifier_model_pvc: str | None,
    trainer_verifier_model: str,
    trainer_hidden_states_path: str | None,
) -> None:
    """Ensure external vLLM receives verifier and hidden states from PVC storage."""
    verifier_is_pvc_backed = bool(verifier_model_pvc) or raw_verifier_model.startswith("pvc://")
    if (
        not verifier_is_pvc_backed
        or not trainer_verifier_model.startswith("pvc://")
        or not trainer_hidden_states_path
        or not trainer_hidden_states_path.startswith("pvc://")
    ):
        raise ValueError("External vLLM requires PVC-backed verifier_model_pvc and hidden_states_path values")


def _validate_vllm_configuration(vllm_source: str | None, vllm_endpoint: str | None) -> str | None:
    """Require a remote endpoint only for remote vLLM and normalize blank values."""
    source = (vllm_source or "managed").strip().lower()
    if source not in {"managed", "remote"}:
        raise ValueError("vllm_source must be either 'managed' or 'remote'")

    endpoint = (vllm_endpoint or "").strip() or None
    if source == "remote" and endpoint is None:
        raise ValueError("vllm_endpoint is required when vllm_source='remote'")
    if source == "managed" and endpoint is not None:
        raise ValueError("vllm_endpoint must be omitted when vllm_source='managed'")
    return endpoint


def _resolve_pip_index_urls(configured_urls: str | None, env: dict[str, str]) -> list[str] | None:
    """Resolve pip indexes while rejecting URLs that embed credentials."""
    if configured_urls:
        urls = [url.strip() for url in configured_urls.split(",") if url.strip()]
    else:
        index_url = env.get("PIP_INDEX_URL", "").strip()
        urls = [index_url] if index_url else []

    extra_index_urls = env.get("PIP_EXTRA_INDEX_URL", "").split()
    for url in [*urls, *extra_index_urls]:
        parsed_url = urlsplit(url)
        if parsed_url.username or parsed_url.password or parsed_url.query or parsed_url.fragment:
            raise ValueError("Pip index URLs must not embed credentials or query tokens; use Secret-backed auth.")

    return urls or None


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
    # persistent_pvc is used verbatim as a Kubernetes PVC name, so reject stray
    # whitespace with a clear error instead of silently fixing it.
    pvc_value = values.get("persistent_pvc")
    if isinstance(pvc_value, str) and pvc_value != pvc_value.strip():
        raise ValueError(f"persistent_pvc must not contain leading or trailing whitespace (got {pvc_value!r}).")
    # Strip leading/trailing whitespace from every other string input before use;
    # this covers all string params for all modes (data_only, train_only, online).
    values = {
        key: value.strip() if isinstance(value, str) and key != "persistent_pvc" else value
        for key, value in values.items()
    }
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
    values["vllm_endpoint"] = _validate_vllm_configuration(values.get("vllm_source"), values.get("vllm_endpoint"))

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
        return _trainer_path(uri, persistent_pvc, workspace_claim)

    def local_path(uri: str) -> str:
        return _local_path(uri, persistent_pvc, persistent_mount_path, workspace_claim, pvc_path)

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

    raw_verifier_model = values["verifier_model"]
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
        _validate_external_pvc_paths(
            raw_verifier_model,
            values.get("verifier_model_pvc"),
            trainer_verifier_model,
            trainer_hidden_states_path,
        )
        if not values.get("target_layer_ids"):
            raise ValueError("target_layer_ids is required with an external vLLM endpoint")

    log.info("Initializing Speculator %s mode", mode)
    api = init_k8s(log)
    # KFP's workspace mount is not available in Trainer workers; use writable temp storage.
    cache_root = "/tmp/.cache/huggingface"
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
        params["env"] = env
        if values.get("packages_to_install"):
            params["packages_to_install"] = [x.strip() for x in values["packages_to_install"].split(",")]
        pip_index_urls = _resolve_pip_index_urls(values.get("pip_index_urls"), env)
        if pip_index_urls:
            params["pip_index_urls"] = pip_index_urls
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
                params.update(vllm_resources=vllm_resources, config=config)
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
                claim, _ = _pvc_parts(storage_uri or "")
                if claim:
                    storage_claims.add(claim)
            duplicate_mount_pairs = {(claim, persistent_mount_path) for claim in storage_claims}
            volume_claims = {
                volume.get("name"): volume.get("persistentVolumeClaim", {}).get("claimName") for volume in volumes
            }
            removed_mounts = [
                mount
                for mount in mounts
                if (volume_claims.get(mount.get("name")), mount.get("mountPath")) in duplicate_mount_pairs
            ]
            if removed_mounts:
                log.info(
                    "Removing duplicate PVC mounts: %s",
                    [
                        (volume_claims.get(mount.get("name")), mount.get("mountPath"), mount.get("name"))
                        for mount in removed_mounts
                    ],
                )
                mounts = [mount for mount in mounts if mount not in removed_mounts]
                remaining_mount_names = {mount.get("name") for mount in mounts}
                removed_volume_names = {
                    mount.get("name") for mount in removed_mounts if mount.get("name") not in remaining_mount_names
                }
                if removed_volume_names:
                    log.info("Removing duplicate PVC volumes: %s", sorted(removed_volume_names))
                    volumes = [volume for volume in volumes if volume.get("name") not in removed_volume_names]
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
        log.exception("Speculator %s failed", mode)
        raise

    if mode == "data_only":
        hidden_states_dir = local_path(
            values["hidden_states_path"] if external else values["output_dir"] + "/hidden_states"
        )
        if not os.path.isdir(hidden_states_dir):
            raise RuntimeError(
                f"Speculator data extraction completed without creating hidden states at "
                f"{hidden_states_dir!r}. Check the TrainJob logs for the extraction failure."
            )
        output = values.get("output_hidden_states")
        if output:
            output.uri = hidden_states_dir
            output.metadata["pvc_path"] = hidden_states_dir
        else:
            log.warning(
                "Hidden states were saved to PVC path %s; no KFP output artifact is wired for this task",
                hidden_states_dir,
            )
        return "data_only completed - hidden states saved"
    output_root = local_path(values["output_dir"])
    # persist_model selects "checkpoint_best" when it exists, then falls back to the latest checkpoint.
    if values.get("output_model"):
        persist_model(
            output_root,
            persistent_mount_path if persistent_pvc else pvc_path,
            values["verifier_model"],
            values["output_model"],
            log,
            prefer_best=True,
            strict_output=True,
        )
    if values.get("output_metrics"):
        values["output_metrics"].log_metric("training_epochs", float(values["training_epochs"]))
        values["output_metrics"].log_metric("learning_rate", float(values["training_lr"]))
    return f"{mode} completed - model trained"
