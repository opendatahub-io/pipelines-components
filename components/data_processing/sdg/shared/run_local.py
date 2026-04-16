"""Run the SDG component locally via KFP LocalRunner.

Patches KFP's SubprocessRunner to allow components with optional
Input[Dataset] artifacts to run locally. Without this patch, KFP
raises 'Input artifacts are not yet supported for local execution'
even when the artifact is not provided.
"""

import json
import os
import sys
import tempfile

# The component module lives in the parent sdg/ directory.
_COMPONENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _COMPONENT_DIR not in sys.path:
    sys.path.insert(0, _COMPONENT_DIR)

import kfp.local  # noqa: E402
import pandas as pd  # noqa: E402
from component import sdg  # noqa: E402
from kfp.local import executor_input_utils, task_dispatcher  # noqa: E402

# Paths
TEST_DATA = os.path.join(_COMPONENT_DIR, "tests", "test_data")
INPUT_PATH = os.path.abspath(os.path.join(TEST_DATA, "sample_input.jsonl"))
FLOW_PATH = os.path.abspath(os.path.join(TEST_DATA, "llm_test_flow.yaml"))


def _patched_construct_executor_input(component_spec, arguments, task_root, block_input_artifact):
    """Wrap construct_executor_input to skip the input artifact block.

    Removes input artifact keys from the component spec before calling
    the original function, so KFP doesn't reject or try to resolve them.
    """
    saved_artifacts = dict(component_spec.input_definitions.artifacts)
    component_spec.input_definitions.ClearField("artifacts")

    try:
        return _original_construct(
            component_spec=component_spec,
            arguments=arguments,
            task_root=task_root,
            block_input_artifact=False,
        )
    finally:
        for k, v in saved_artifacts.items():
            component_spec.input_definitions.artifacts[k].CopyFrom(v)


_original_construct = executor_input_utils.construct_executor_input
_original_run = task_dispatcher.run_single_task_implementation


def _patched_run(*args, **kwargs):
    """Pass through with block_input_artifact=False."""
    kwargs["block_input_artifact"] = False
    return _original_run(*args, **kwargs)


def main():
    """Run the SDG component with LLM test flow via patched LocalRunner."""
    executor_input_utils.construct_executor_input = _patched_construct_executor_input
    task_dispatcher.run_single_task_implementation = _patched_run

    with tempfile.TemporaryDirectory() as pipeline_root:
        kfp.local.init(
            runner=kfp.local.SubprocessRunner(use_venv=False),
            pipeline_root=pipeline_root,
        )

        print(f"Input:   {INPUT_PATH}")
        print(f"Flow:    {FLOW_PATH}")
        print(f"Output:  {pipeline_root}")
        print()

        task = sdg(
            input_pvc_path=INPUT_PATH,
            flow_yaml_path=FLOW_PATH,
            model="openai/gpt-4o-mini",
            max_concurrency=1,
            temperature=0.7,
            max_tokens=2048,
        )

        output_path = task.outputs["output_artifact"].path
        metrics_path = task.outputs["output_metrics"].path

        print("\n" + "=" * 60)
        print("GENERATED OUTPUT")
        print("=" * 60)
        df = pd.read_json(output_path, lines=True)
        pd.set_option("display.max_colwidth", 80)
        pd.set_option("display.width", 200)
        print(df.to_string(index=False))

        print("\n" + "=" * 60)
        print("METRICS")
        print("=" * 60)
        with open(metrics_path) as f:
            print(json.dumps(json.load(f), indent=2))


if __name__ == "__main__":
    main()
