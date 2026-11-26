# Data Connect Hub Export Table Snapshot ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Export a DCH table as CSV or Parquet and publish an HTML summary.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `DCH_HOST` | `str` | `None` | Hostname and optional port of the DCH gateway route exposed to the task pod. |
| `connection_id` | `str` | `None` | ID of the saved SQL-capable relational DCH connection in the run namespace. |
| `source_table` | `str` | `None` | Source table in schema.table_name format, such as analytics.customers. |
| `output_format` | `str` | `parquet` | Snapshot format, either parquet (default) or csv. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `SnapshotOutputs` | The Dataset, HTML report, and exact snapshot URI. |

## Metadata 🗂️

- **Name**: dch-export-table-snapshot
- **Description**: Export a relational table through DCH Flight SQL as CSV or Parquet and publish an HTML report.
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.16.1
- **Tags**:
  - data_processing
  - data_connect_hub
  - sql
  - csv
  - parquet
- **Last Verified**: 2026-11-24 00:00:00+00:00
- **Owners**:
  - Approvers:
    - mateusz-szewczyk-dev
    - LukaszCmielowski
    - Mateusz-Switala
    - jakub-walaszczyk
  - Reviewers:
    - mateusz-szewczyk-dev
    - Mateusz-Switala

## Additional Resources 📚

- **Documentation**: [https://github.com/opendatahub-io/data-connect-hub](https://github.com/opendatahub-io/data-connect-hub)

<!-- custom-content -->

## Prerequisites

Follow the [component setup steps](../../../../components/data_processing/dch/export_table_snapshot/README.md#prerequisites)
for pipeline server artifact storage, gateway TLS, the DCH connection, and ServiceAccount permissions.

## Compile and run

From the repository root with the project installed, run:

```sh
python pipelines/data_processing/dch/export_table_snapshot/pipeline.py
```

Upload the generated `dch-export-table-snapshot.yaml` from the same directory to the
pipeline server in the connection's tenant namespace. For `DCH_HOST`, use the hostname
and optional port of the route that serves the DCH REST and Flight SQL APIs. DCH can
share the OpenShift AI ingress hostname if the gateway routes its APIs there.
The task sends its ServiceAccount token to that endpoint. To restrict the token
destination, pass `trusted_host` to `create_export_table_snapshot_pipeline()` before
compiling the pipeline with the KFP SDK.

| Run input | Value |
| --- | --- |
| `DCH_HOST` | The deployment's DCH HTTPS gateway hostname and optional port. |
| `connection_id` | The saved DCH connection ID for a relational connector that accepts SQL over Flight. |
| `source_table` | The table in `schema.table_name` format, for example `analytics.customers`. |
| `output_format` | `parquet` (default) or `csv`. |

For the first run, choose a small table. Check for `[PASS] DCH connection lookup`
and `[PASS] Table snapshot` in the task log, then open the HTML `report` and read
the file at `snapshot_uri` to verify that KFP uploaded both artifacts to
the pipeline server's artifact store.

## Train with AutoML

Start with a relational table that contains the training features and target
column. In this example, `analytics.customer_churn` contains a binary target
column named `churned`.

1. Run the DCH export with the saved connection, `source_table=analytics.customer_churn`,
   and set `output_format=csv`. The export reads the table over Flight SQL
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
