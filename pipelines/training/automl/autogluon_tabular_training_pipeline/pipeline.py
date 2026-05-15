from kfp import dsl
from kfp_components.components.data_processing.automl.tabular_data_loader import automl_data_loader
from kfp_components.components.training.automl.autogluon_leaderboard_evaluation import leaderboard_evaluation
from kfp_components.components.training.automl.autogluon_models_training import autogluon_models_training

MAX_CPUS = "32"
MAX_MEMORY = "64Gi"


@dsl.pipeline(
    name="autogluon-tabular-training-pipeline",
    description=(
        "End-to-end AutoGluon tabular training pipeline implementing a two-stage approach: "
        "first builds and selects top-performing models on sampled data, then refits them "
        "on the full dataset. The pipeline loads data from S3, splits it into train/test sets, "
        "trains multiple AutoGluon models using ensembling (stacking and bagging), selects the "
        "top N performers, refits each on the complete training data sequentially, and finally "
        "evaluates all refitted models to generate a comprehensive leaderboard with performance metrics."
    ),
    pipeline_config=dsl.PipelineConfig(
        workspace=dsl.WorkspaceConfig(
            size="12Gi",  # TODO: change to recommended size
            kubernetes=dsl.KubernetesWorkspaceConfig(
                pvcSpecPatch={
                    # use default storage class from the cluster
                    # "storageClassName": "gp3-csi",
                    "accessModes": ["ReadWriteOnce"],
                }
            ),
        ),
    ),
)
def autogluon_tabular_training_pipeline(
    train_data_secret_name: str,
    train_data_bucket_name: str,
    train_data_file_key: str,
    label_column: str,
    task_type: str,
    top_n: int = 3,
):
    """AutoGluon Tabular Training Pipeline.

    This pipeline implements an efficient two-stage training approach for AutoGluon tabular models
    that balances computational cost with model quality. The pipeline automates the complete
    machine learning workflow from data loading to final model evaluation.

    **Storage strategy:**

    Training datasets are stored on a PVC workspace (not S3 artifacts) so that all
    pipeline steps sharing the workspace can access them without extra downloads. Only
    the test dataset is written to an S3 artifact (for use by the leaderboard evaluation
    component). The workspace is provisioned via ``PipelineConfig.workspace``.

    **Pipeline Stages:**

    1. **Data Loading & Splitting**: Loads tabular (CSV) data from an S3-compatible
       object storage bucket using AWS credentials configured via Kubernetes secrets.
       The component samples the data (up to 1GB), then performs a two-stage split:
       *Primary split** (default 80/20): separates a *test set* (20%, written to an
         S3 artifact) from the *train portion* (80%).
         **Secondary split** (default 30/70 of the train portion): produces
         ``models_selection_train_dataset.csv`` (30%, used for model selection) and
         ``extra_train_dataset.csv`` (70%, passed to ``refit_full`` as extra data).
         Both train CSVs are written to the PVC workspace under
         ``{workspace_path}/datasets/``. For classification tasks the splits are
         stratified by the label column.

    2. **Model Training & Refitting**: Trains multiple AutoGluon models on the
       *selection train* data using stacking (1 level) and bagging (4 folds).
       All models are evaluated on the test set and ranked by performance. The top N
       models are selected and refitted sequentially on the full training data via
       ``refit_full``. Each refitted model is saved with a ``_FULL`` suffix and
       optimized for deployment. All model artifacts are stored under a single output
       artifact, avoiding a ``ParallelFor`` loop in the pipeline.

    3. **Leaderboard Evaluation**: Reads pre-computed metrics from the combined models
       artifact and generates an HTML-formatted leaderboard ranking models by their
       performance metrics for comparison and selection.

    **Two-Stage Training Benefits:**

    - **Efficient Exploration:** Initial model training uses a smaller selection-train
      split with efficient ensembling rather than expensive hyperparameter optimization.

    - **Optimal Performance:** Final models are refitted (``refit_full``) on the
      predictor's training and validation data plus the extra-train split, maximizing
      the amount of data seen during the final fit.

    - **Production-Ready:** Refitted models are AutoGluon Predictors optimized and ready
      for deployment.

    **AutoGluon Ensembling Approach:**

    The pipeline leverages AutoGluon's unique ensembling strategy that combines multiple
    model types using stacking and bagging rather than traditional hyperparameter optimization.
    This approach is more efficient and typically produces better results for tabular data
    by automatically:

    - Training diverse model families

    - Combining predictions using multi-level stacking

    - Using bootstrap aggregation (bagging) for robustness

    - Selecting optimal ensemble configurations

    Args:
        train_data_secret_name: Kubernetes secret name with S3 credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT, AWS_DEFAULT_REGION).
        train_data_bucket_name: S3-compatible bucket name containing the tabular data file.
        train_data_file_key: S3 object key of the CSV file (features and target column).
        label_column: Name of the target/label column in the dataset.
        task_type: "binary", "multiclass", or "regression"; drives metrics and model types.
        top_n: Number of top models to select and refit (default: 3); positive integer from range [1, 10].

    Returns:
        HTML artifact with leaderboard of refitted models ranked by task_type metric (e.g. accuracy, r2).

    Raises:
        FileNotFoundError: If the S3 file cannot be found or accessed.
        ValueError: If label_column missing, task_type invalid, top_n not positive, or split fails.
        KeyError: If AWS credentials missing in secret or required component outputs unavailable.

    Example:
        from kfp import dsl
        from pipelines.training.automl.autogluon_tabular_training_pipeline import (
            autogluon_tabular_training_pipeline
        )

        # Compile and run the pipeline
        pipeline = autogluon_tabular_training_pipeline(
            train_data_secret_name="my-s3-secret",
            train_data_bucket_name="my-data-bucket",
            train_data_file_key="datasets/housing_prices.csv",
            label_column="price",
            task_type="regression",
            top_n=3,
        )
    """  # noqa: E501
    from kfp.kubernetes import use_secret_as_env

    data_loader_task = automl_data_loader(
        bucket_name=train_data_bucket_name,
        file_key=train_data_file_key,
        workspace_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        label_column=label_column,
        task_type=task_type,
    )
    data_loader_task.set_caching_options(False)
    data_loader_task.set_cpu_request("2").set_memory_request("8Gi").set_cpu_limit(MAX_CPUS).set_memory_limit(MAX_MEMORY)

    use_secret_as_env(
        data_loader_task,
        secret_name=train_data_secret_name,
        secret_key_to_env={
            "AWS_ACCESS_KEY_ID": "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY": "AWS_SECRET_ACCESS_KEY",
            "AWS_S3_ENDPOINT": "AWS_S3_ENDPOINT",
            "AWS_DEFAULT_REGION": "AWS_DEFAULT_REGION",
        },
        optional=True,  # Mark as optional to not block the pipeline. If needed, error will be raised by component
    )

    # Stage 1 + 2: Model selection and sequential refit of top N models
    training_task = autogluon_models_training(
        label_column=label_column,
        task_type=task_type,
        top_n=top_n,
        train_data_path=data_loader_task.outputs["models_selection_train_data_path"],
        test_data=data_loader_task.outputs["sampled_test_dataset"],
        workspace_path=dsl.WORKSPACE_PATH_PLACEHOLDER,
        pipeline_name=dsl.PIPELINE_JOB_RESOURCE_NAME_PLACEHOLDER,
        run_id=dsl.PIPELINE_JOB_ID_PLACEHOLDER,
        sample_row=data_loader_task.outputs["sample_row"],
        sampling_config=data_loader_task.outputs["sample_config"],
        split_config=data_loader_task.outputs["split_config"],
        extra_train_data_path=data_loader_task.outputs["extra_train_data_path"],
    )
    training_task.set_caching_options(False)
    training_task.set_cpu_request("4").set_memory_request("16Gi").set_cpu_limit(MAX_CPUS).set_memory_limit(MAX_MEMORY)

    # Generate leaderboard
    leaderboard_evaluation_task = leaderboard_evaluation(
        models_artifact=training_task.outputs["models_artifact"],
        eval_metric=training_task.outputs["eval_metric"],
    )
    leaderboard_evaluation_task.set_caching_options(False)
    leaderboard_evaluation_task.set_cpu_request("1").set_memory_request("4Gi").set_cpu_limit(MAX_CPUS).set_memory_limit(
        MAX_MEMORY
    )


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        autogluon_tabular_training_pipeline,
        package_path=__file__.replace(".py", ".yaml"),
    )
