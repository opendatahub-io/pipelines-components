"""SDG LLM pipeline for OpenShift AI.

Creates sample input data, runs an LLM flow via the SDG Hub component,
and produces generated output as a KFP artifact.
"""

from kfp import compiler, dsl
from kfp.kubernetes import use_config_map_as_volume, use_secret_as_env
from kfp_components.components.data_processing.sdg.component import sdg


@dsl.component(packages_to_install=["pandas"])
def create_sample_data(output_data: dsl.Output[dsl.Dataset]) -> None:
    """Create sample input data with document and domain columns."""
    import pandas as pd

    data = [
        {"document": "Python is a programming language.", "domain": "technology"},
        {"document": "Machine learning is a subset of AI.", "domain": "technology"},
        {"document": "The Earth orbits the Sun.", "domain": "science"},
    ]
    df = pd.DataFrame(data)
    df.to_json(output_data.path, orient="records", lines=True)


@dsl.pipeline(
    name="sdg-llm-test-pipeline",
    description="Run SDG Hub LLM flow with sample data on OpenShift AI",
)
def sdg_llm_pipeline(
    model: str = "openai/gpt-4o-mini",
    max_concurrency: int = 1,
    temperature: float = 0.7,
    max_tokens: int = 256,
):
    """Run SDG LLM test flow end-to-end.

    Creates sample input data, runs the LLM test flow via the SDG Hub
    component, and outputs generated data as a KFP artifact.

    Args:
        model: LiteLLM model identifier.
        max_concurrency: Max concurrent LLM requests.
        temperature: LLM sampling temperature.
        max_tokens: Max response tokens.
    """
    # Step 1: Create sample data
    data_task = create_sample_data()

    # Step 2: Run SDG flow
    sdg_task = sdg(
        input_artifact=data_task.outputs["output_data"],
        flow_yaml_path="/etc/sdg/llm_test_flow.yaml",
        model=model,
        max_concurrency=max_concurrency,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Mount flow YAML from ConfigMap
    use_config_map_as_volume(
        task=sdg_task,
        config_map_name="sdg-llm-test-flow",
        mount_path="/etc/sdg",
    )

    # Mount prompt YAML from ConfigMap
    use_config_map_as_volume(
        task=sdg_task,
        config_map_name="sdg-llm-test-prompt",
        mount_path="/etc/sdg/prompts",
    )

    # Inject LLM API key from K8s Secret
    use_secret_as_env(
        task=sdg_task,
        secret_name="llm-credentials",
        secret_key_to_env={"api_key": "LLM_API_KEY"},
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        sdg_llm_pipeline,
        package_path="sdg_llm_pipeline.yaml",
    )
    print("Compiled: sdg_llm_pipeline.yaml")
