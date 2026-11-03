"""Tests for the Speculator offline pipeline."""

from kfp import compiler

from ..pipeline import speculator_offline_pipeline

PUBLIC_INPUTS = [
    "p01_persistent_pvc",
    "p02_verifier_model_pvc",
    "p03_vllm_endpoint",
    "p04_dataset_name",
    "p05_evaluation_dataset_uri",
    "p06_serving_namespace",
    "p07_data_extraction_max_samples",
    "p08_data_extraction_regenerate_responses",
    "p09_data_extraction_concurrency",
    "p10_output_dir",
    "p11_hidden_states_path",
    "p12_training_data_path",
    "p13_total_seq_len",
    "p14_speculator_type",
    "p15_target_layer_ids",
    "p16_training_epochs",
    "p17_training_lr",
    "p18_training_draft_vocab_size",
    "p19_training_num_layers",
    "p20_training_test_time_steps",
    "p21_training_norm_before_residual",
    "p22_training_norm_before_fc",
    "p23_training_embed_requires_grad",
    "p24_training_hidden_states_dtype",
    "p25_training_scheduler_type",
    "p26_training_scheduler_warmup_steps",
    "p27_training_scheduler_total_steps",
    "p30_training_save_best",
    "p32_training_resume_from_checkpoint",
    "p33_training_resource_cpu",
    "p34_training_resource_gpu",
    "p35_training_resource_memory",
    "p36_job_envs",
    "p37_job_runtime",
    "p38_packages_to_install",
    "p40_evaluation_max_samples",
    "p41_evaluation_max_tokens",
    "p42_evaluation_temperature",
    "p43_evaluation_num_speculative_tokens",
    "p44_evaluation_gpu_memory_utilization",
    "p45_evaluation_startup_timeout_seconds",
    "p47_model_registry_address",
    "p48_model_registry_port",
    "p49_model_registry_name",
    "p50_model_registry_version",
    "p51_serving_endpoint_name",
    "p52_serving_runtime_image",
]


def _compile_pipeline(path):
    compiler.Compiler().compile(speculator_offline_pipeline, package_path=str(path))


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
    """Numeric parameter prefixes preserve the UI's stage-oriented input order."""
    inputs = speculator_offline_pipeline.component_spec.inputs

    assert list(inputs) == PUBLIC_INPUTS
    assert inputs["p01_persistent_pvc"].default is None
    assert inputs["p02_verifier_model_pvc"].default is None
    assert inputs["p03_vllm_endpoint"].default is None
    assert inputs["p04_dataset_name"].default is None
    assert inputs["p05_evaluation_dataset_uri"].default is None
    assert inputs["p06_serving_namespace"].default is None
    assert inputs["p07_data_extraction_max_samples"].default == 0
    assert inputs["p08_data_extraction_regenerate_responses"].default is False
    assert inputs["p09_data_extraction_concurrency"].default == 1
    assert inputs["p15_target_layer_ids"].default == ""
    assert inputs["p16_training_epochs"].default == 3
    assert inputs["p37_job_runtime"].default == "speculator-model-opt-cuda"
    assert inputs["p47_model_registry_address"].default == ""
    assert "p28_training_scheduler_num_cosine_cycles" not in inputs
    assert "p29_training_checkpoint_freq" not in inputs
    assert "p31_training_log_freq" not in inputs
    assert "p39_pip_index_urls" not in inputs
    assert "p46_evaluation_sample_seed" not in inputs
    assert inputs["p52_serving_runtime_image"].default == (
        "registry.redhat.io/rhaii-fast/vllm-cuda-rhel9"
        "@sha256:97c7d6f66f3f0897b3ff10ddd135de930d45bdd116cf41a43e3ad7270029116a"
    )
    assert "training_from_pretrained" not in inputs


def test_pipeline_compiles_and_orders_all_stages(tmp_path):
    """Extraction, training, evaluation, registration, and serving run in order."""
    output_path = tmp_path / "speculator_offline.yaml"
    _compile_pipeline(output_path)
    pipeline_spec, _ = _load_pipeline_documents(output_path)

    compiled_pipeline = output_path.read_text(encoding="utf-8")
    assert "speculator-offline-pipeline" in compiled_pipeline
    assert "model-registry==0.3.9" in compiled_pipeline
    assert "model-registry==0.3.4" not in compiled_pipeline
    assert sorted(pipeline_spec["root"]["inputDefinitions"]["parameters"]) == PUBLIC_INPUTS

    tasks = pipeline_spec["root"]["dag"]["tasks"]
    assert "download-model" not in tasks
    assert "validate-required-offline-inputs" not in tasks
    assert "extract-hidden-states" in tasks["train-draft-model"]["dependentTasks"]
    assert "train-draft-model" in tasks["evaluate-draft-model"]["dependentTasks"]
    assert "evaluate-draft-model" in tasks["kubeflow-model-registry"]["dependentTasks"]
    assert "kubeflow-model-registry" in tasks["speculative-decoding-serving"]["dependentTasks"]

    assert tasks["train-draft-model"]["taskInfo"]["name"] == "train-draft-model"
    assert tasks["evaluate-draft-model"]["taskInfo"]["name"] == "evaluate-draft-model"
    assert tasks["kubeflow-model-registry"]["taskInfo"]["name"] == "kubeflow-model-registry"
    assert "displayName" not in tasks["train-draft-model"]["taskInfo"]
    assert "displayName" not in tasks["evaluate-draft-model"]["taskInfo"]
    assert "displayName" not in tasks["kubeflow-model-registry"]["taskInfo"]
    assert "displayName" not in tasks["train-draft-model"]["taskInfo"]
    assert "displayName" not in tasks["evaluate-draft-model"]["taskInfo"]


def test_pipeline_wires_shared_pvc_paths_and_stage_inputs(tmp_path):
    """The extraction output, training input, evaluator, Registry, and serving share the PVC."""
    output_path = tmp_path / "speculator_offline.yaml"
    _compile_pipeline(output_path)
    pipeline_spec, _ = _load_pipeline_documents(output_path)
    tasks = pipeline_spec["root"]["dag"]["tasks"]

    extraction_inputs = tasks["extract-hidden-states"]["inputs"]["parameters"]
    assert extraction_inputs["persistent_pvc"]["componentInputParameter"] == "p01_persistent_pvc"
    assert extraction_inputs["verifier_model_pvc"]["componentInputParameter"] == "p02_verifier_model_pvc"
    assert extraction_inputs["dataset_name"]["componentInputParameter"] == "p04_dataset_name"
    assert extraction_inputs["vllm_source"]["runtimeValue"]["constant"] == "remote"
    assert extraction_inputs["vllm_endpoint"]["componentInputParameter"] == "p03_vllm_endpoint"
    assert extraction_inputs["training_runtime"]["componentInputParameter"] == "p37_job_runtime"
    extraction_max_samples = extraction_inputs["data_extraction_max_samples"]
    assert extraction_max_samples["componentInputParameter"] == "p07_data_extraction_max_samples"
    assert extraction_inputs["hidden_states_path"]["componentInputParameter"] == "p11_hidden_states_path"
    assert (
        extraction_inputs["data_extraction_concurrency"]["componentInputParameter"] == "p09_data_extraction_concurrency"
    )

    training_inputs = tasks["train-draft-model"]["inputs"]["parameters"]
    assert training_inputs["persistent_pvc"]["componentInputParameter"] == "p01_persistent_pvc"
    assert training_inputs["verifier_model_pvc"]["componentInputParameter"] == "p02_verifier_model_pvc"
    assert training_inputs["training_data_path"]["componentInputParameter"] == "p12_training_data_path"
    assert training_inputs["hidden_states_path"]["componentInputParameter"] == "p11_hidden_states_path"
    assert training_inputs["training_ttt_steps"]["componentInputParameter"] == "p20_training_test_time_steps"
    assert (
        training_inputs["data_extraction_concurrency"]["componentInputParameter"] == "p09_data_extraction_concurrency"
    )
    assert training_inputs["training_runtime"]["componentInputParameter"] == "p37_job_runtime"
    assert training_inputs["training_envs"]["componentInputParameter"] == "p36_job_envs"
    assert training_inputs["training_scheduler_num_cosine_cycles"]["runtimeValue"]["constant"] == 0.5
    assert training_inputs["training_checkpoint_freq"]["runtimeValue"]["constant"] == 1.0
    assert training_inputs["training_log_freq"]["runtimeValue"]["constant"] == 1

    extraction_envs = tasks["extract-hidden-states"]["inputs"]["parameters"]["training_envs"]
    assert extraction_envs["componentInputParameter"] == "p36_job_envs"

    evaluation_inputs = tasks["evaluate-draft-model"]["inputs"]["parameters"]
    assert evaluation_inputs["evaluation_dataset_uri"]["componentInputParameter"] == "p05_evaluation_dataset_uri"
    assert evaluation_inputs["evaluation_dataset_pvc"]["componentInputParameter"] == "p01_persistent_pvc"
    assert (
        evaluation_inputs["evaluation_num_speculative_tokens"]["componentInputParameter"]
        == "p43_evaluation_num_speculative_tokens"
    )
    assert (
        evaluation_inputs["evaluation_gpu_memory_utilization"]["componentInputParameter"]
        == "p44_evaluation_gpu_memory_utilization"
    )
    assert evaluation_inputs["evaluation_sample_seed"]["runtimeValue"]["constant"] == 42

    registry_inputs = tasks["kubeflow-model-registry"]["inputs"]["parameters"]
    assert registry_inputs["model_pvc_name"]["componentInputParameter"] == "p01_persistent_pvc"
    assert registry_inputs["registry_address"]["componentInputParameter"] == "p47_model_registry_address"
    assert registry_inputs["registry_port"]["componentInputParameter"] == "p48_model_registry_port"
    assert registry_inputs["model_name"]["componentInputParameter"] == "p49_model_registry_name"
    assert registry_inputs["model_version"]["componentInputParameter"] == "p50_model_registry_version"

    serving_inputs = tasks["speculative-decoding-serving"]["inputs"]["parameters"]
    assert serving_inputs["model_cache_pvc"]["componentInputParameter"] == "p01_persistent_pvc"
    assert serving_inputs["namespace"]["componentInputParameter"] == "p06_serving_namespace"
    assert serving_inputs["endpoint_name"]["componentInputParameter"] == "p51_serving_endpoint_name"
    assert serving_inputs["verifier_model_dir"]["componentInputParameter"] == "p02_verifier_model_pvc"
    assert serving_inputs["runtime_image"]["componentInputParameter"] == "p52_serving_runtime_image"
    assert (
        serving_inputs["num_speculative_tokens"]["componentInputParameter"] == "p43_evaluation_num_speculative_tokens"
    )
    assert (
        serving_inputs["gpu_memory_utilization"]["componentInputParameter"] == "p44_evaluation_gpu_memory_utilization"
    )


def test_pipeline_scopes_secrets_to_tasks_that_need_them(tmp_path):
    """Trainer launchers receive Kubernetes credentials; only evaluation gets HF_TOKEN."""
    output_path = tmp_path / "speculator_offline.yaml"
    _compile_pipeline(output_path)
    pipeline_spec, platform_spec = _load_pipeline_documents(output_path)

    for task_name in ("extract-hidden-states", "train-draft-model"):
        secrets = _executor_config(pipeline_spec, platform_spec, task_name)["secretAsEnv"]
        assert {secret["secretName"] for secret in secrets} == {"kubernetes-credentials"}
        assert all(secret["optional"] is False for secret in secrets)

    secrets = _executor_config(pipeline_spec, platform_spec, "evaluate-draft-model")["secretAsEnv"]
    assert {secret["secretName"] for secret in secrets} == {"hf-token"}
    assert all(secret["optional"] is True for secret in secrets)

    for task_name in ("kubeflow-model-registry", "speculative-decoding-serving"):
        secrets = _executor_config(pipeline_spec, platform_spec, task_name).get("secretAsEnv", [])
        assert secrets == []


def test_evaluation_requests_gpu_accelerator(tmp_path):
    """The evaluation task requests one NVIDIA GPU in the compiled pipeline."""
    output_path = tmp_path / "speculator_offline.yaml"
    _compile_pipeline(output_path)
    pipeline_spec, platform_spec = _load_pipeline_documents(output_path)

    executor = _executor_config(pipeline_spec, platform_spec, "evaluate-draft-model")
    accelerator = executor["container"]["resources"]["accelerator"]
    assert accelerator["resourceType"] == "nvidia.com/gpu"
    assert accelerator["count"] == "1"
