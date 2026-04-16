"""SDG Hub KFP Component.

Wraps the SDG Hub SDK to enable synthetic data generation
within Kubeflow Pipelines.
"""

import kfp.compiler
from kfp import dsl


@dsl.component(
    packages_to_install=["sdg-hub>=0.7.0,<1.0"],
)
def sdg(
    output_artifact: dsl.Output[dsl.Dataset],
    output_metrics: dsl.Output[dsl.Metrics],
    input_artifact: dsl.Input[dsl.Dataset] = None,
    input_pvc_path: str = "",
    flow_id: str = "",
    flow_yaml_path: str = "",
    model: str = "",
    max_concurrency: int = 10,
    checkpoint_pvc_path: str = "",
    save_freq: int = 100,
    log_level: str = "INFO",
    temperature: float = -1.0,
    max_tokens: int = -1,
    export_to_pvc: bool = False,
    export_path: str = "",
    runtime_params: dict = None,
):
    """Run an SDG Hub flow to generate synthetic data.

    Loads input data, selects and configures a flow, executes it,
    and writes the output as a JSONL artifact with execution metrics.

    Args:
        output_artifact: KFP Dataset artifact for downstream components.
        output_metrics: KFP Metrics artifact with execution stats.
        input_artifact: KFP Dataset artifact from upstream component (optional).
        input_pvc_path: Path to JSONL input file on a mounted PVC (optional).
        flow_id: Built-in flow ID from the SDG Hub registry.
        flow_yaml_path: Path to a custom flow YAML file.
        model: LiteLLM model identifier (e.g. 'openai/gpt-4o-mini').
        max_concurrency: Maximum concurrent LLM requests.
        checkpoint_pvc_path: PVC path for checkpoints (enables resume).
        save_freq: Checkpoint save frequency (number of samples).
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        temperature: LLM sampling temperature (0.0-2.0). Use -1 for flow default.
        max_tokens: Maximum response tokens. Use -1 for flow default.
        export_to_pvc: Whether to export output to PVC (in addition to KFP artifact).
        export_path: Base PVC path for exports (required if export_to_pvc is True).
        runtime_params: Per-block parameter overrides as a dict of {block_name: {param: value}}.
    """
    import logging
    import os
    import time

    import pandas as pd
    from sdg_hub.core.flow.base import Flow
    from sdg_hub.core.flow.registry import FlowRegistry
    from sdg_hub.core.utils.error_handling import FlowValidationError

    # Configure logging
    log_level_value = getattr(logging, log_level.upper(), None)
    if log_level_value is None:
        log_level_value = logging.INFO
    logging.basicConfig(
        level=log_level_value,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    start_time = time.time()

    logger.info("=" * 60)
    logger.info("SDG Hub KFP Component")
    logger.info("=" * 60)

    # Log configuration
    logger.info(f"Input Artifact: {'Provided' if input_artifact else 'Not provided'}")
    logger.info(f"Input PVC Path: {input_pvc_path or 'Not provided'}")
    logger.info(f"Flow ID: {flow_id or 'Not provided'}")
    logger.info(f"Flow YAML Path: {flow_yaml_path or 'Not provided'}")
    logger.info(f"Model: {model or 'Not provided'}")
    logger.info(f"Max Concurrency: {max_concurrency}")
    logger.info(f"Temperature: {'flow default' if temperature < 0 else temperature}")
    logger.info(f"Max Tokens: {'flow default' if max_tokens < 0 else max_tokens}")
    logger.info(f"Export to PVC: {export_to_pvc}")
    if export_to_pvc:
        logger.info(f"Export Path: {export_path or 'Not provided'}")
    runtime_params = runtime_params or {}
    if runtime_params:
        logger.info(f"Runtime params: {runtime_params}")

    # =========================================================================
    # INPUT HANDLING
    # =========================================================================
    if input_artifact:
        logger.info(f"Loading input from KFP artifact: {input_artifact.path}")
        if not os.path.exists(input_artifact.path):
            raise FileNotFoundError(f"Input artifact file not found: {input_artifact.path}")
        df = pd.read_json(input_artifact.path, lines=True)
        logger.info("Using input_artifact as data source")
    elif input_pvc_path:
        logger.info(f"Loading input from PVC: {input_pvc_path}")
        if not os.path.exists(input_pvc_path):
            raise FileNotFoundError(f"Input file not found: {input_pvc_path}")
        df = pd.read_json(input_pvc_path, lines=True)
        logger.info("Using input_pvc_path as data source")
    else:
        raise ValueError("No input provided. Supply 'input_artifact' or 'input_pvc_path'.")

    input_rows = len(df)
    logger.info(f"Loaded {input_rows} rows with columns: {list(df.columns)}")

    # =========================================================================
    # FLOW SELECTION
    # =========================================================================
    if not flow_id and not flow_yaml_path:
        raise ValueError(
            "Either 'flow_id' or 'flow_yaml_path' must be provided. "
            "Use 'flow_id' for built-in flows or 'flow_yaml_path' for custom YAML."
        )

    if flow_id and flow_yaml_path:
        logger.warning("Both 'flow_id' and 'flow_yaml_path' provided. Using 'flow_yaml_path' (takes precedence).")

    if flow_yaml_path:
        yaml_path = flow_yaml_path
        logger.info(f"Using custom flow YAML: {yaml_path}")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(
                f"Custom flow YAML not found: {yaml_path}. Ensure the file is mounted (e.g., via ConfigMap or PVC)."
            )
    else:
        logger.info(f"Looking up built-in flow: {flow_id}")
        try:
            yaml_path = FlowRegistry.get_flow_path_safe(flow_id)
        except ValueError as exc:
            raise ValueError(f"Flow lookup failed for '{flow_id}': {exc}") from exc
        logger.info(f"Found flow at: {yaml_path}")

    # =========================================================================
    # FLOW LOADING
    # =========================================================================
    logger.info(f"Loading flow from: {yaml_path}")
    try:
        flow = Flow.from_yaml(yaml_path)
    except FlowValidationError as exc:
        raise FlowValidationError(f"Failed to load flow from '{yaml_path}': {exc}") from exc

    logger.info(f"Flow loaded: '{flow.metadata.name}' v{flow.metadata.version} with {len(flow.blocks)} blocks")

    # =========================================================================
    # MODEL CONFIGURATION
    # =========================================================================
    if flow.is_model_config_required():
        if not model:
            raise ValueError(
                f"Flow '{flow.metadata.name}' contains LLM blocks and requires "
                "a 'model' parameter. Provide a LiteLLM model identifier "
                "(e.g., 'openai/gpt-4o-mini')."
            )

        api_key = os.environ.get("LLM_API_KEY", "")
        api_base = os.environ.get("LLM_API_BASE", "")

        model_kwargs = {}
        if temperature >= 0:
            model_kwargs["temperature"] = temperature
        if max_tokens > 0:
            model_kwargs["max_tokens"] = max_tokens

        logger.info(f"Configuring model: {model}")
        if api_base:
            logger.info(f"Using API base: {api_base}")

        flow.set_model_config(
            model=model,
            api_key=api_key if api_key else None,
            api_base=api_base if api_base else None,
            **model_kwargs,
        )
        logger.info("Model configuration applied to LLM blocks")
    else:
        logger.info("Flow has no LLM blocks - skipping model configuration")

    # =========================================================================
    # DATASET VALIDATION
    # =========================================================================
    validation_errors = flow.validate_dataset(df)
    if validation_errors:
        raise FlowValidationError(
            f"Dataset validation failed for flow '{flow.metadata.name}':\n"
            + "\n".join(f"  - {err}" for err in validation_errors)
        )
    logger.info("Dataset validation passed")

    # =========================================================================
    # FLOW EXECUTION
    # =========================================================================
    logger.info(f"Starting flow execution: {len(df)} samples, max_concurrency={max_concurrency}")

    generate_kwargs = {
        "max_concurrency": max_concurrency,
    }

    if checkpoint_pvc_path:
        generate_kwargs["checkpoint_dir"] = checkpoint_pvc_path
        generate_kwargs["save_freq"] = save_freq
        logger.info(f"Checkpointing enabled: dir={checkpoint_pvc_path}, save_freq={save_freq}")

    if runtime_params:
        generate_kwargs["runtime_params"] = runtime_params

    output_df = flow.generate(df, **generate_kwargs)
    output_rows = len(output_df)

    # =========================================================================
    # OUTPUT HANDLING
    # =========================================================================
    output_df.to_json(output_artifact.path, orient="records", lines=True)
    logger.info(f"Output written to: {output_artifact.path}")
    logger.info(f"Output: {output_rows} rows with columns: {list(output_df.columns)}")

    # =========================================================================
    # PVC EXPORT (OPTIONAL)
    # =========================================================================
    if export_to_pvc:
        if not export_path:
            raise ValueError(
                "export_to_pvc is True but export_path is not provided. "
                "Supply export_path (base PVC directory for exports)."
            )

        if flow_yaml_path:
            flow_name = os.path.splitext(os.path.basename(flow_yaml_path))[0] or "custom"
        elif flow_id:
            flow_name = flow_id.replace("/", "_").replace("\\", "_")
        else:
            flow_name = "custom"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        export_dir = os.path.join(export_path, flow_name, timestamp)

        os.makedirs(export_dir, exist_ok=True)
        export_file_path = os.path.join(export_dir, "generated.jsonl")

        output_df.to_json(export_file_path, orient="records", lines=True)
        logger.info(f"Output exported to PVC: {export_file_path}")

    # Write metrics
    execution_time = time.time() - start_time
    output_metrics.log_metric("input_rows", input_rows)
    output_metrics.log_metric("output_rows", output_rows)
    output_metrics.log_metric("execution_time_seconds", round(execution_time, 2))
    logger.info("Metrics logged")

    logger.info("=" * 60)
    logger.info(f"SDG Hub KFP Component completed in {execution_time:.2f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        sdg,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
