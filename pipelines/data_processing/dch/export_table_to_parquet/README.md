# Dch Export Table To Parquet ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Export a DCH table as Parquet and publish an HTML summary.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `DCH_HOST` | `str` | `None` | DCH gateway hostname and optional port for the run. |
| `connection_id` | `str` | `None` | ID of the saved SQL-capable relational DCH connection in the run namespace. |
| `source_table` | `str` | `None` | Source table in schema.table_name format, such as analytics.customers. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `SnapshotOutputs` | The Parquet Dataset, HTML report, and exact Parquet URI. |

## Metadata 🗂️

- **Name**: dch-export-table-to-parquet
- **Description**: Export a relational table through DCH Flight SQL and publish a Parquet artifact and HTML report.
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.16.1
  - External Services:
    - Name: Data Connect Hub, Version: ===6d58bc9975a431c8f8a9040e610520d20c980f36
- **Tags**:
  - data_processing
  - data_connect_hub
  - sql
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

Compile the pipeline once and enter the DCH gateway when you create a run.

```python
from kfp import compiler
from kfp_components.pipelines.data_processing.dch.export_table_to_parquet import (
    create_export_table_to_parquet_pipeline,
)

pipeline = create_export_table_to_parquet_pipeline()
compiler.Compiler().compile(pipeline, package_path="dch-export-table-to-parquet.yaml")
```

Upload the YAML to the DSPA in the connection's tenant namespace. Use the
deployment's DCH HTTPS gateway for `DCH_HOST`. The task sends its
ServiceAccount token to that endpoint. If pipeline runners can submit runs but
should not choose the token destination, the pipeline author can set
`trusted_host` when compiling a version restricted to one gateway.

| Run input | Value |
| --- | --- |
| `DCH_HOST` | The deployment's DCH HTTPS gateway hostname and optional port. |
| `connection_id` | The saved DCH connection ID for a relational connector that accepts SQL over Flight. |
| `source_table` | The table in `schema.table_name` format, for example `analytics.customers`. |

For the first run, choose a small table. Check for `[PASS] DCH connection lookup`
and `[PASS] Parquet snapshot` in the task log, then open the HTML `report` and read
the Parquet file at `snapshot_uri` to verify that KFP uploaded both artifacts to
the DSPA artifact store. Pass `snapshot_uri` and a label column to the
[separate AutoGluon pipeline](../../../../components/data_processing/dch/export_table_to_parquet/README.md#train-from-the-snapshot)
to train a tabular model.

## Export size limit

The task stages one compressed Parquet file on pod disk, with a fixed limit of
1 GiB including the footer. If the export would exceed the limit, the task
fails and deletes the partial file before publishing a snapshot. To export a
larger source, select a smaller table or a view with fewer rows or columns.
