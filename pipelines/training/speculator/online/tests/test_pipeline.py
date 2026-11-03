"""Tests for the Speculator online pipeline."""

from kfp import compiler

from ..pipeline import speculator_online_pipeline

PUBLIC_INPUTS = [
    "p01_persistent_pvc",
    "p05_target_layer_ids",
    "p06_dataset_name",
    "p07_evaluation_dataset_uri",
    "p08_serving_namespace",
    "p02_verifier_model",
    "p03_verifier_model_pvc",
    "p04_download_model",
    "p09_data_extraction_max_samples",
    "p10_data_extraction_regenerate_responses",
    "p11_data_extraction_concurrency",
    "p12_output_dir",
    "p13_total_seq_len",
    "p14_training_epochs",
    "p15_training_lr",
    "p16_training_draft_vocab_size",
    "p17_training_num_layers",
    "p18_training_test_time_steps",
    "p19_training_norm_before_residual",
    "p20_training_norm_before_fc",
    "p21_training_embed_requires_grad",
    "p22_training_hidden_states_dtype",
    "p23_training_scheduler_type",
    "p24_training_scheduler_warmup_steps",
    "p25_training_scheduler_total_steps",
    "p26_training_save_best",
    "p27_training_resume_from_checkpoint",
    "p28_training_resource_cpu",
    "p29_training_resource_gpu",
    "p30_training_resource_memory",
    "p31_vllm_resource_memory",
    "p32_vllm_gpu_memory_utilization",
    "p33_vllm_readiness_timeout_minutes",
    "p34_job_runtime",
    "p35_packages_to_install",
    "p36_job_envs",
    "p37_evaluation_max_samples",
    "p38_evaluation_max_tokens",
    "p39_evaluation_temperature",
    "p40_evaluation_num_speculative_tokens",
    "p41_evaluation_gpu_memory_utilization",
    "p42_evaluation_startup_timeout_seconds",
    "p43_model_registry_address",
    "p44_model_registry_port",
    "p45_model_registry_name",
    "p46_model_registry_version",
    "p47_serving_endpoint_name",
    "p48_serving_runtime_image",
]


def _compile_pipeline(path):
    compiler.Compiler().compile(speculator_online_pipeline, package_path=str(path))


def _load_pipeline_documents(path):
    import yaml

    return list(yaml.safe_load_all(path.read_text(encoding="utf-8")))


def _executor_config(pipeline_spec, platform_spec, task):
    component_name = pipeline_spec["root"]["dag"]["tasks"][task]["componentRef"]["name"]
    executor_label = pipeline_spec["components"][component_name]["executorLabel"]
    component_executor = pipeline_spec["deploymentSpec"]["executors"][executor_label]
    platform_executor = platform_spec["platforms"]["kubernetes"]["deploymentSpec"]["executors"][executor_label]
    return {**component_executor, **platform_executor}


def test_pipeline_exposes_stage_ordered_inputs():
    """Numeric prefixes preserve the online workflow input order in the KFP UI."""
    inputs = speculator_online_pipeline.component_spec.inputs

    assert list(inputs) == PUBLIC_INPUTS
    assert inputs["p01_persistent_pvc"].default is None
    assert inputs["p02_verifier_model"].default == ""
    assert inputs["p03_verifier_model_pvc"].default == ""
    assert inputs["p04_download_model"].default is False
    assert inputs["p05_target_layer_ids"].default is None
    assert inputs["p06_dataset_name"].default is None
    assert inputs["p07_evaluation_dataset_uri"].default is None
    assert inputs["p08_serving_namespace"].default is None
    assert inputs["p09_data_extraction_max_samples"].default == 0
    assert inputs["p11_data_extraction_concurrency"].default == 1
    assert inputs["p34_job_runtime"].default == "vllm-extract-cuda"
    assert inputs["p43_model_registry_address"].default == ""
    assert inputs["p44_model_registry_port"].default == 8080
    assert inputs["p46_model_registry_version"].default == "1.0.0"
    assert inputs["p48_serving_runtime_image"].default.endswith(
        "@sha256:97c7d6f66f3f0897b3ff10ddd135de930d45bdd116cf41a43e3ad7270029116a"
    )

    removed_inputs = {
        "speculator_type",
        "vllm_resource_gpu",
        "training_scheduler_num_cosine_cycles",
        "training_checkpoint_freq",
        "training_log_freq",
        "training_from_pretrained",
        "pip_index_urls",
        "evaluation_sample_seed",
    }
    assert removed_inputs.isdisjoint(inputs)


def test_pipeline_compiles_as_a_single_online_trainer_job(tmp_path):
    """The online workflow uses one fused Trainer job rather than separate stages."""
    output_path = tmp_path / "speculator_online.yaml"
    _compile_pipeline(output_path)
    pipeline_spec, _ = _load_pipeline_documents(output_path)

    assert sorted(pipeline_spec["root"]["inputDefinitions"]["parameters"]) == sorted(PUBLIC_INPUTS)
    tasks = pipeline_spec["root"]["dag"]["tasks"]
    assert "download-model" in tasks
    assert "online-draft-model-training" in tasks
    assert "evaluate-draft-model" in tasks
    assert "kubeflow-model-registry" in tasks
    assert "speculative-decoding-serving" in tasks
    assert "extract-hidden-states" not in tasks
    assert "train-draft-model" not in tasks

    assert "download-model" in tasks["online-draft-model-training"]["dependentTasks"]
    assert "online-draft-model-training" in tasks["evaluate-draft-model"]["dependentTasks"]
    assert "evaluate-draft-model" in tasks["kubeflow-model-registry"]["dependentTasks"]
    assert "kubeflow-model-registry" in tasks["speculative-decoding-serving"]["dependentTasks"]

    download_inputs = tasks["download-model"]["inputs"]["parameters"]
    assert download_inputs["download_enabled"]["componentInputParameter"] == "p04_download_model"
    assert download_inputs["existing_model_path"]["componentInputParameter"] == "p03_verifier_model_pvc"

    online_inputs = tasks["online-draft-model-training"]["inputs"]["parameters"]
    assert online_inputs["verifier_model"]["componentInputParameter"] == "p02_verifier_model"
    assert online_inputs["verifier_model_pvc"]["componentInputParameter"] == "p03_verifier_model_pvc"
    assert online_inputs["download_model"]["componentInputParameter"] == "p04_download_model"
    assert online_inputs["training_envs"]["componentInputParameter"] == "p36_job_envs"
    assert online_inputs["training_runtime"]["componentInputParameter"] == "p34_job_runtime"
    assert online_inputs["target_layer_ids"]["componentInputParameter"] == "p05_target_layer_ids"
    assert online_inputs["vllm_resource_gpu"]["runtimeValue"]["constant"] == 1
    assert online_inputs["speculator_type"]["runtimeValue"]["constant"] == "eagle3"
    assert online_inputs["training_scheduler_num_cosine_cycles"]["runtimeValue"]["constant"] == 0.5
    assert online_inputs["training_checkpoint_freq"]["runtimeValue"]["constant"] == 1.0
    assert online_inputs["training_log_freq"]["runtimeValue"]["constant"] == 1
    assert "pip_index_urls" not in online_inputs
    assert "training_from_pretrained" not in online_inputs


def test_evaluation_and_serving_share_performance_inputs_and_fixed_seed(tmp_path):
    """Evaluation and serving use the same online-mode performance settings."""
    output_path = tmp_path / "speculator_online.yaml"
    _compile_pipeline(output_path)
    pipeline_spec, _ = _load_pipeline_documents(output_path)
    tasks = pipeline_spec["root"]["dag"]["tasks"]

    evaluation_inputs = tasks["evaluate-draft-model"]["inputs"]["parameters"]
    assert evaluation_inputs["evaluation_num_speculative_tokens"]["componentInputParameter"] == (
        "p40_evaluation_num_speculative_tokens"
    )
    assert evaluation_inputs["evaluation_gpu_memory_utilization"]["componentInputParameter"] == (
        "p41_evaluation_gpu_memory_utilization"
    )
    assert evaluation_inputs["evaluation_sample_seed"]["runtimeValue"]["constant"] == 42

    serving_inputs = tasks["speculative-decoding-serving"]["inputs"]["parameters"]
    assert serving_inputs["num_speculative_tokens"]["componentInputParameter"] == (
        "p40_evaluation_num_speculative_tokens"
    )
    assert serving_inputs["gpu_memory_utilization"]["componentInputParameter"] == (
        "p41_evaluation_gpu_memory_utilization"
    )


def test_pipeline_scopes_secrets_to_tasks_that_need_them(tmp_path):
    """HF token is scoped to download, training, and evaluation; registry/serving get none."""
    output_path = tmp_path / "speculator_online.yaml"
    _compile_pipeline(output_path)
    pipeline_spec, platform_spec = _load_pipeline_documents(output_path)

    for task_name in ("download-model", "online-draft-model-training", "evaluate-draft-model"):
        secrets = _executor_config(pipeline_spec, platform_spec, task_name)["secretAsEnv"]
        assert "hf-token" in {secret["secretName"] for secret in secrets}

    secrets = _executor_config(pipeline_spec, platform_spec, "online-draft-model-training")["secretAsEnv"]
    assert "kubernetes-credentials" in {secret["secretName"] for secret in secrets}

    for task_name in ("kubeflow-model-registry", "speculative-decoding-serving"):
        secrets = _executor_config(pipeline_spec, platform_spec, task_name).get("secretAsEnv", [])
        assert "hf-token" not in {secret.get("secretName") for secret in secrets}


def test_evaluation_requests_gpu_accelerator(tmp_path):
    """The evaluator is scheduled on an NVIDIA GPU in the compiled pipeline."""
    output_path = tmp_path / "speculator_online.yaml"
    _compile_pipeline(output_path)
    pipeline_spec, platform_spec = _load_pipeline_documents(output_path)
    executor = _executor_config(pipeline_spec, platform_spec, "evaluate-draft-model")

    accelerator = executor["container"]["resources"]["accelerator"]
    assert accelerator["resourceType"] == "nvidia.com/gpu"
    assert accelerator["count"] == "1"
