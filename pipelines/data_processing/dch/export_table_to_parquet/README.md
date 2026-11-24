# Dch Export Table To Parquet ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Check DCH access, then export the selected table and an HTML report.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `DCH_HOST` | `str` | `None` | DCH gateway hostname and optional port for the run. |
| `connection_id` | `str` | `None` | ID of the saved PostgreSQL connection in the run namespace. |
| `table_name` | `str` | `None` | Exact PostgreSQL table name. |
| `table_schema` | `str` | `public` | Schema containing the table. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `SnapshotOutputs` | The Parquet Dataset, HTML report, and exact Parquet URI. |

## Metadata 🗂️

- **Name**: dch-export-table-to-parquet
- **Description**: Export a selected DCH PostgreSQL table as a Parquet artifact and publish an HTML report.
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.16.1
  - External Services:
    - Name: Data Connect Hub, Version: ===6d58bc9975a431c8f8a9040e610520d20c980f36
- **Tags**:
  - data_processing
  - data_connect_hub
  - postgres
  - parquet
- **Last Verified**: 2026-11-24 00:00:00+00:00
- **Owners**:
  - Approvers:
    - LukaszCmielowski
  - Reviewers:
    - LukaszCmielowski

## Additional Resources 📚

- **Documentation**: [https://github.com/opendatahub-io/data-connect-hub](https://github.com/opendatahub-io/data-connect-hub)

<!-- custom-content -->

## Prerequisites

Follow the [component setup steps](../../../../components/data_processing/dch/export_table_to_parquet/README.md#prerequisites)
for DSPA artifact storage, gateway TLS, the DCH connection, and ServiceAccount permissions.

## Compile and run

Use the gateway for the deployment where the pipeline will run. The compiler fixes the trusted
gateway in the task so a run parameter cannot redirect the pod's ServiceAccount token.

```python
from kfp import compiler
from kfp_components.pipelines.data_processing.dch.export_table_to_parquet import (
    create_export_table_to_parquet_pipeline,
)

pipeline = create_export_table_to_parquet_pipeline(trusted_host="dch-gateway.example.com:443")
compiler.Compiler().compile(pipeline, package_path="dch-export-table-to-parquet.yaml")
```

Upload the YAML to the DSPA in the connection's tenant namespace.

| Run input | Value |
| --- | --- |
| `DCH_HOST` | The same gateway hostname and optional port used during compilation. |
| `connection_id` | The saved DCH PostgreSQL connection ID. |
| `table_name` | The exact name of the table to export. |
| `table_schema` | The schema containing the table. Defaults to `public`. |

Open the task log for the connection checks. On success, KFP uploads the Parquet `snapshot`
and HTML `report` to the DSPA artifact store. The `snapshot_uri` output identifies the Parquet
object. Pass it and a label column to the
[separate AutoGluon pipeline](../../../../components/data_processing/dch/export_table_to_parquet/README.md#train-from-the-snapshot)
to train a tabular model.
