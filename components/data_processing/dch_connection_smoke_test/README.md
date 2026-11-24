# Dch Connection Smoke Test ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Check DCH access and export the selected PostgreSQL table through Flight SQL.

Checks the gateway CA and task token, resolves the connection's credential Secret, and runs the same read-only catalog count through Flight and PostgreSQL. Publishes a Parquet dataset and HTML report through KFP's artifact store.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `dch_host` | `str` | `None` | DCH gateway hostname and optional port requested for the run. |
| `connection_id` | `str` | `None` | ID of a saved PostgreSQL DCH connection in the task namespace. |
| `trusted_host` | `str` | `None` | Gateway hostname and optional port fixed when composing the pipeline. The component sends the task ServiceAccount token only to this gateway. |
| `table_name` | `str` | `None` | Exact name of the source table. |
| `snapshot` | `dsl.Output[dsl.Dataset]` | `None` | Parquet dataset artifact. |
| `report` | `dsl.Output[dsl.HTML]` | `None` | HTML report with check results and snapshot statistics. |
| `table_schema` | `str` | `public` | Schema containing the source table. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `NamedTuple('Outputs', [('snapshot_uri', str)])` | The exact artifact URI of the Parquet file. |

## Usage Examples 🧪

```python
"""Train an AutoGluon tabular model from a published DCH snapshot."""

from kfp import dsl


@dsl.component(
    base_image="python:3.12-bookworm",
    packages_to_install=["autogluon.tabular==1.5.0"],
)
def train_tabular_snapshot(
    snapshot: dsl.Input[dsl.Dataset],
    model: dsl.Output[dsl.Model],
    label_column: str,
    time_limit_seconds: int,
):
    """Train and publish an AutoGluon tabular predictor."""
    import pandas as pd
    from autogluon.tabular import TabularPredictor

    data = pd.read_parquet(snapshot.path)
    if data.empty or label_column not in data.columns:
        raise ValueError("The snapshot must contain training rows and the requested label column.")
    TabularPredictor(label=label_column, path=model.path, verbosity=1).fit(
        data,
        time_limit=time_limit_seconds,
        hyperparameters={"RF": {}, "XT": {}},
    )


@dsl.pipeline(name="train-automl-from-dch-snapshot")
def train_automl_from_dch_snapshot(
    snapshot_uri: str,
    label_column: str,
    time_limit_seconds: int = 120,
) -> dsl.Model:
    """Import the snapshot from the DSPA artifact store and train a tabular model."""
    dataset = dsl.importer(artifact_uri=snapshot_uri, artifact_class=dsl.Dataset)
    training = train_tabular_snapshot(
        snapshot=dataset.output,
        label_column=label_column,
        time_limit_seconds=time_limit_seconds,
    )
    return training.outputs["model"]

```

## Metadata 🗂️

- **Name**: dch_connection_smoke_test
- **Description**: Verify a saved PostgreSQL Data Connect Hub connection, snapshot a selected table as Parquet, and publish an HTML report from one KFP task.
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
  - smoke_test
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

- Run the task in the tenant namespace that owns the saved, Flight-enabled PostgreSQL connection.
- The task uses its mounted Kubernetes ServiceAccount token and namespace. The token audience must
  be accepted by DCH. Grant the runner `dch-read` in the tenant for REST metadata and Flight SQL.
- The pod must trust the DCH gateway's TLS certificate. On OpenShift AI, configure the DSPA CA
  bundle when the gateway uses a private issuer. The component uses the launcher's
  `SSL_CERT_FILE` when set.
- Grant DCH Flight `get` on the connection's exact credential Secret. Grant the pipeline runner
  `get` on the same Secret for the direct PostgreSQL check. The Secret contains `URI` and
  optionally `CA_CERT` for the PostgreSQL server.
- The task pod needs network access to the DCH gateway, the Kubernetes API, and PostgreSQL.
  The task installs its dependencies at startup and needs access to GitHub and PyPI.
- Use the DSPA's S3-backed artifact store as the pipeline root. The PostgreSQL role needs
  read access to the selected table and the task needs enough ephemeral disk to stage the
  complete Parquet file.

## Use in a pipeline

Set the trusted host when composing the pipeline, using your own gateway address. The run
supplies the saved connection ID and exact table name. Set caching off so every run exercises
the current connection and captures fresh data.

```python
from kfp import dsl
from kfp_components.components.data_processing.dch_connection_smoke_test import (
    dch_connection_smoke_test,
)

DCH_GATEWAY = "dch-gateway.example.com:443"


@dsl.pipeline(name="check-dch-connection")
def smoke_test(connection_id: str, table_name: str, table_schema: str = "public") -> str:
    task = dch_connection_smoke_test(
        dch_host=DCH_GATEWAY,
        connection_id=connection_id,
        trusted_host=DCH_GATEWAY,
        table_name=table_name,
        table_schema=table_schema,
    ).set_caching_options(False)
    return task.outputs["snapshot_uri"]
```

The task logs each check and exits with status 1 when a check fails. Both connectivity queries
count rows in `pg_catalog.pg_namespace`. After those checks pass, DCH Flight reads the entire
selected table in Arrow batches and the task writes one Parquet file. KFP uploads the dataset
and HTML report using its configured artifact-store credentials. The report includes the
table, row and column counts, file size, check results, and the exact Parquet URI.

## Train from the snapshot

Compile the [AutoGluon example](example_pipelines.py) from the installed package:

```sh
python -c 'from kfp import compiler; from kfp_components.components.data_processing.dch_connection_smoke_test.example_pipelines import train_automl_from_dch_snapshot as pipeline; compiler.Compiler().compile(pipeline, "train-automl-from-dch-snapshot.yaml")'
```

Upload the YAML and run it on a pipeline server with read access to the same artifact store.
Pass the first run's `snapshot_uri` and the name of the target column as `label_column`.
AutoGluon infers classification or regression from the label and saves the fitted predictor
as a KFP Model artifact. For example, export a prepared subscription table and train a churn
classifier using its `churned` column. The existing tabular AutoML template takes CSV, while
this example reads the Parquet snapshot directly.
