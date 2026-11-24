"""Check the deployment gateway and outputs in the standalone pipeline spec."""

from kfp import compiler

from ..pipeline import create_export_table_to_parquet_pipeline


def test_compiles_a_single_export_task(tmp_path):
    """Keep the trusted gateway fixed and expose the dataset and report to callers."""
    import yaml

    gateway = "gateway.example.com:443"
    pipeline = create_export_table_to_parquet_pipeline(gateway)
    output = tmp_path / "pipeline.yaml"
    compiler.Compiler().compile(pipeline, str(output))

    spec = yaml.safe_load(output.read_text())
    assert spec["pipelineInfo"]["name"] == "dch-export-table-to-parquet"
    root = spec["root"]
    inputs = root["inputDefinitions"]["parameters"]
    assert set(inputs) == {"DCH_HOST", "connection_id", "table_name", "table_schema"}
    assert inputs["table_schema"]["defaultValue"] == "public"

    (task,) = root["dag"]["tasks"].values()
    parameters = task["inputs"]["parameters"]
    assert task["cachingOptions"].get("enableCache", False) is False
    assert parameters["trusted_host"] == {"runtimeValue": {"constant": gateway}}
    assert parameters["dch_host"] == {"componentInputParameter": "DCH_HOST"}

    outputs = root["outputDefinitions"]
    assert outputs["artifacts"]["snapshot"]["artifactType"]["schemaTitle"] == "system.Dataset"
    assert outputs["artifacts"]["report"]["artifactType"]["schemaTitle"] == "system.HTML"
    assert outputs["parameters"]["snapshot_uri"]["parameterType"] == "STRING"
