"""Tests for the Speculator train-only pipeline."""

from kfp import compiler

from ..pipeline import speculator_training_pipeline

PUBLIC_INPUTS = [
    "p01_persistent_pvc",
    "p02_model_registry_address",
    "p03_model_registry_name",
    "p04_serving_namespace",
    "p05_serving_endpoint_name",
    "p06_verifier_model_pvc",
    "p07_evaluation_dataset_uri",
    "p08_training_data_path",
    "p09_hidden_states_path",
    "p10_output_dir",
    "p11_total_seq_len",
    "p12_speculator_type",
    "p13_target_layer_ids",
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
    "p26_training_scheduler_num_cosine_cycles",
    "p27_training_checkpoint_freq",
    "p28_training_save_best",
    "p29_training_log_freq",
    "p30_training_resume_from_checkpoint",
    "p31_training_resource_cpu",
    "p32_training_resource_gpu",
    "p33_training_resource_memory",
    "p34_job_envs",
    "p35_job_runtime",
    "p36_packages_to_install",
    "p37_pip_index_urls",
    "p38_evaluation_max_samples",
    "p39_evaluation_max_tokens",
    "p40_evaluation_temperature",
    "p41_num_speculative_tokens",
    "p42_gpu_memory_utilization",
    "p43_evaluation_startup_timeout_seconds",
    "p44_evaluation_sample_seed",
    "p45_model_registry_port",
    "p46_model_registry_version",
    "p47_serving_runtime_image",
]


def test_pipeline_exposes_training_registry_and_serving_inputs():
    """The pipeline exposes its PVC, training, Registry, and serving controls."""
    inputs = speculator_training_pipeline.component_spec.inputs
    assert list(inputs) == PUBLIC_INPUTS
    assert "verifier_model" not in inputs
    assert "download_model" not in inputs
    assert inputs["p13_target_layer_ids"].default == "2,14,25,28"
    assert inputs["p18_training_test_time_steps"].default == 1
    assert inputs["p41_num_speculative_tokens"].default == 5
    assert inputs["p42_gpu_memory_utilization"].default == 0.9
    assert inputs["p02_model_registry_address"].default is None
    assert inputs["p03_model_registry_name"].default is None
    assert inputs["p04_serving_namespace"].default is None
    assert inputs["p05_serving_endpoint_name"].default is None
    assert inputs["p06_verifier_model_pvc"].default is None
    assert inputs["p07_evaluation_dataset_uri"].default is None
    assert inputs["p08_training_data_path"].default == "speculator-output"
    assert inputs["p34_job_envs"].default == "NCCL_DEBUG=INFO,PYTHONUNBUFFERED=1"
    assert inputs["p35_job_runtime"].default == "speculator-model-opt-cuda"
    assert "training_envs" not in inputs
    assert "training_runtime" not in inputs


def test_progression_metrics_are_not_public_controls():
    """Progress tracking uses the training component's always-on defaults."""
    inputs = speculator_training_pipeline.component_spec.inputs

    assert "enable_progression_tracking" not in inputs
    assert "metrics_port" not in inputs
    assert "metrics_poll_interval_seconds" not in inputs


def test_pipeline_compiles_and_orders_training_evaluation_registry_serving(tmp_path):
    """Evaluation follows training, registration follows evaluation, then serving."""
    import yaml

    output_path = tmp_path / "speculator_training.yaml"
    compiler.Compiler().compile(speculator_training_pipeline, package_path=str(output_path))

    compiled_pipeline = output_path.read_text()
    assert "speculator-training-pipeline" in compiled_pipeline
    pipeline_documents = list(yaml.safe_load_all(compiled_pipeline))
    pipeline_spec = pipeline_documents[0]
    platform_spec = pipeline_documents[1]
    pipeline_inputs = pipeline_spec["root"]["inputDefinitions"]["parameters"]
    assert list(pipeline_inputs) == PUBLIC_INPUTS
    tasks = pipeline_spec["root"]["dag"]["tasks"]
    assert tasks["train-draft-model"]["taskInfo"]["name"] == "train-draft-model"
    assert tasks["evaluate-draft-model"]["taskInfo"]["name"] == "evaluate-draft-model"
    assert tasks["evaluate-draft-model"]["cachingOptions"].get("enableCache", False) is False
    assert "model-registry==0.3.9" in compiled_pipeline
    assert "model-registry==0.3.4" not in compiled_pipeline
    assert "train-draft-model" in tasks["evaluate-draft-model"]["dependentTasks"]
    assert "evaluate-draft-model" in tasks["kubeflow-model-registry"]["dependentTasks"]
    assert "kubeflow-model-registry" in tasks["speculative-decoding-serving"]["dependentTasks"]

    evaluation_component_name = tasks["evaluate-draft-model"]["componentRef"]["name"]
    evaluation_executor_label = pipeline_spec["components"][evaluation_component_name]["executorLabel"]
    evaluation_kubernetes_config = platform_spec["platforms"]["kubernetes"]["deploymentSpec"]["executors"][
        evaluation_executor_label
    ]
    assert evaluation_kubernetes_config["imagePullPolicy"] == "IfNotPresent"
    evaluation_secrets = evaluation_kubernetes_config["secretAsEnv"]
    assert {secret["secretName"] for secret in evaluation_secrets} == {"hf-token"}
    assert all(secret["optional"] is True for secret in evaluation_secrets)
    assert {key_to_env["envVar"] for secret in evaluation_secrets for key_to_env in secret["keyToEnv"]} == {"HF_TOKEN"}

    train_inputs = tasks["train-draft-model"]["inputs"]["parameters"]
    assert train_inputs["persistent_pvc"]["componentInputParameter"] == "p01_persistent_pvc"
    assert train_inputs["hidden_states_path"]["componentInputParameter"] == "p09_hidden_states_path"
    assert train_inputs["training_data_path"]["componentInputParameter"] == "p08_training_data_path"
    assert train_inputs["training_ttt_steps"]["componentInputParameter"] == "p18_training_test_time_steps"

    evaluation_inputs = tasks["evaluate-draft-model"]["inputs"]["parameters"]
    assert evaluation_inputs["evaluation_dataset_pvc"]["componentInputParameter"] == "p01_persistent_pvc"
    assert evaluation_inputs["evaluation_dataset_uri"]["componentInputParameter"] == "p07_evaluation_dataset_uri"
    assert (
        evaluation_inputs["evaluation_num_speculative_tokens"]["componentInputParameter"]
        == "p41_num_speculative_tokens"
    )
    assert (
        evaluation_inputs["evaluation_gpu_memory_utilization"]["componentInputParameter"]
        == "p42_gpu_memory_utilization"
    )

    registry_inputs = tasks["kubeflow-model-registry"]["inputs"]["parameters"]
    assert registry_inputs["model_pvc_name"]["componentInputParameter"] == "p01_persistent_pvc"

    serving_inputs = tasks["speculative-decoding-serving"]["inputs"]["parameters"]
    assert serving_inputs["model_cache_pvc"]["componentInputParameter"] == "p01_persistent_pvc"
    assert serving_inputs["verifier_model_dir"]["componentInputParameter"] == "p06_verifier_model_pvc"
    assert serving_inputs["num_speculative_tokens"]["componentInputParameter"] == "p41_num_speculative_tokens"
    assert serving_inputs["gpu_memory_utilization"]["componentInputParameter"] == "p42_gpu_memory_utilization"
