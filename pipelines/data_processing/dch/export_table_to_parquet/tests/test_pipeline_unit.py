"""Check the deployment gateway and outputs in the standalone pipeline spec."""

import pytest
from kfp import compiler

from ..pipeline import create_export_table_to_parquet_pipeline


@pytest.mark.parametrize("gateway", ["", "gateway.example.com:443"])
def test_compiles_a_single_export_task(tmp_path, gateway):
    """Require a gateway at run time and preserve an optional compile-time restriction."""
    import yaml

    options = {"trusted_host": gateway} if gateway else {}
    pipeline = create_export_table_to_parquet_pipeline(**options)
    output = tmp_path / "pipeline.yaml"
    compiler.Compiler().compile(pipeline, str(output))

    spec = yaml.safe_load(output.read_text())
    assert spec["pipelineInfo"]["name"] == "dch-export-table-to-parquet"
    root = spec["root"]
    inputs = root["inputDefinitions"]["parameters"]
    assert set(inputs) == {
        "DCH_HOST",
        "connection_id",
        "source_table",
    }
    assert "defaultValue" not in inputs["source_table"]
    assert "defaultValue" not in inputs["DCH_HOST"]

    (task,) = root["dag"]["tasks"].values()
    assert task["taskInfo"]["name"] == "export-table-to-parquet"
    parameters = task["inputs"]["parameters"]
    assert task["cachingOptions"].get("enableCache", False) is False
    assert parameters["trusted_host"] == {"runtimeValue": {"constant": gateway}}
    assert parameters["dch_host"] == {"componentInputParameter": "DCH_HOST"}
    assert parameters["source_table"] == {"componentInputParameter": "source_table"}

    outputs = root["outputDefinitions"]
    assert outputs["artifacts"]["snapshot"]["artifactType"]["schemaTitle"] == "system.Dataset"
    assert outputs["artifacts"]["report"]["artifactType"]["schemaTitle"] == "system.HTML"
    assert outputs["parameters"]["snapshot_uri"]["parameterType"] == "STRING"
