# Dch Export Table To Parquet ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Export a DCH table as CSV or Parquet and publish an HTML summary.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `DCH_HOST` | `str` | `None` | DCH gateway hostname and optional port for the run. |
| `connection_id` | `str` | `None` | ID of the saved SQL-capable relational DCH connection in the run namespace. |
| `source_table` | `str` | `None` | Source table in schema.table_name format, such as analytics.customers. |
| `output_format` | `str` | `csv` | Snapshot format, either csv (default) or parquet. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `SnapshotOutputs` | The Dataset, HTML report, and exact snapshot URI. |

## Metadata 🗂️

- **Name**: dch-export-table-to-parquet
- **Description**: Export a relational table through DCH Flight SQL as CSV or Parquet and publish an HTML report.
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
  - csv
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
for pipeline server artifact storage, gateway TLS, the DCH connection, and ServiceAccount permissions.

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

Upload the YAML to the pipeline server in the connection's tenant namespace. Use the
deployment's DCH HTTPS gateway for `DCH_HOST`. The task sends its
ServiceAccount token to that endpoint. If pipeline runners can submit runs but
should not choose the token destination, the pipeline author can set
`trusted_host` when compiling a version restricted to one gateway.

| Run input | Value |
| --- | --- |
| `DCH_HOST` | The deployment's DCH HTTPS gateway hostname and optional port. |
| `connection_id` | The saved DCH connection ID for a relational connector that accepts SQL over Flight. |
| `source_table` | The table in `schema.table_name` format, for example `analytics.customers`. |
| `output_format` | `csv` (default) or `parquet`. |

For the first run, choose a small table. Check for `[PASS] DCH connection lookup`
and `[PASS] Table snapshot` in the task log, then open the HTML `report` and read
the file at `snapshot_uri` to verify that KFP uploaded both artifacts to
the pipeline server's artifact store.

## Train with AutoML

Start with a relational table that contains the training features and target
column. In this example, `analytics.customer_churn` contains a binary target
column named `churned`.

1. Run the DCH export with the saved connection, `source_table=analytics.customer_churn`,
   and `output_format=csv` (the default). The export reads the table over Flight SQL
   and publishes the CSV snapshot to the pipeline server's artifact store.
2. Copy `snapshot_uri` from the completed run. Use the snapshot as training data in the
   [AutoGluon tabular training pipeline](../../../training/automl/autogluon_tabular_training_pipeline/README.md).

Split the exported `snapshot_uri` into its bucket and object key. For example,
for `s3://pipeline-artifacts/exports/customer-churn.csv`, set the training inputs
as follows.

| AutoML input | Value |
| --- | --- |
| `train_data_bucket_name` | `pipeline-artifacts` |
| `train_data_file_key` | `exports/customer-churn.csv` |
| `train_data_secret_name` | A Kubernetes Secret with read access to the snapshot in the artifact store. |
| `label_column` | `churned` |
| `task_type` | `binary` |

When training completes, open the AutoML leaderboard and check that the trained
model artifacts are available. To use a Parquet snapshot, select an AutoML version
that includes the support tracked in [PR #375](https://github.com/opendatahub-io/pipelines-components/pull/375).

## Export size limit

The task stages one CSV or compressed Parquet file on pod disk, with a fixed
limit of 1 GiB including the CSV header or Parquet footer. If the export would
exceed the limit, the task fails and deletes the partial file. To export a larger
source, select a smaller table or a view with fewer rows or columns.
