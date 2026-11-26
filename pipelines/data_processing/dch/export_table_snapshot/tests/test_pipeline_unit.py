"""Check the deployment gateway and outputs in the standalone pipeline spec."""

from kfp import compiler

from ..pipeline import create_export_table_snapshot_pipeline


def test_compiles_a_single_export_task(tmp_path):
    """Require the fixed gateway Secret and check the run inputs and outputs."""
    import yaml

    pipeline = create_export_table_snapshot_pipeline()
    output = tmp_path / "pipeline.yaml"
    compiler.Compiler().compile(pipeline, str(output))

    spec, platforms = yaml.safe_load_all(output.read_text())
    assert spec["pipelineInfo"]["name"] == "dch-export-table-snapshot"
    assert spec["pipelineInfo"]["displayName"] == "DCH Table Export"
    root = spec["root"]
    inputs = root["inputDefinitions"]["parameters"]
    assert set(inputs) == {
        "connection_id",
        "source_table",
        "output_format",
    }
    assert "defaultValue" not in inputs["source_table"]
    assert "defaultValue" not in inputs["connection_id"]
    assert "dch-gateway Secret" in inputs["connection_id"]["description"]
    assert inputs["output_format"]["defaultValue"] == "parquet"

    (task,) = root["dag"]["tasks"].values()
    assert task["taskInfo"]["name"] == "export-table-snapshot"
    assert task["componentRef"]["name"] == "comp-export-table-snapshot"
    parameters = task["inputs"]["parameters"]
    assert task["cachingOptions"].get("enableCache", False) is False
    assert parameters["connection_id"] == {"componentInputParameter": "connection_id"}
    assert parameters["source_table"] == {"componentInputParameter": "source_table"}
    assert parameters["output_format"] == {"componentInputParameter": "output_format"}

    executors = platforms["platforms"]["kubernetes"]["deploymentSpec"]["executors"]
    (secret,) = executors["exec-export-table-snapshot"]["secretAsEnv"]
    assert secret["secretNameParameter"] == {"runtimeValue": {"constant": "dch-gateway"}}
    assert secret["keyToEnv"] == [{"secretKey": "host", "envVar": "DCH_TRUSTED_HOST"}]
    assert secret["optional"] is False

    outputs = root["outputDefinitions"]
    assert outputs["artifacts"]["snapshot"]["artifactType"]["schemaTitle"] == "system.Dataset"
    assert outputs["artifacts"]["report"]["artifactType"]["schemaTitle"] == "system.HTML"
    assert outputs["parameters"]["snapshot_uri"]["parameterType"] == "STRING"
