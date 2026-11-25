# Export Table To Parquet ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Check DCH access and export the selected PostgreSQL table through Flight SQL.

Uses the gateway CA and task token to look up the saved connection and read the selected table through Flight SQL. Optional diagnostics count catalog rows over Flight and directly against PostgreSQL using the connection's credential Secret. Publishes a Parquet dataset and HTML report through KFP.

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
| `run_postgres_diagnostics` | `bool` | `False` | Count PostgreSQL catalog rows over Flight SQL and a direct connection. |
| `max_snapshot_size_bytes` | `int` | `1073741824` | Maximum size in bytes of the staged Parquet file, including its footer. |

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

- **Name**: export_table_to_parquet
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

Run in the DCH tenant namespace that owns the connection. Configure a DSPA with an S3-backed
artifact store and use its default pipeline root. The task pod needs access to the DCH gateway
over HTTPS and Flight SQL. The DCH Flight service uses the connection Secret to access
PostgreSQL. The task uses the Red Hat UBI 9 Python 3.12 image from `registry.access.redhat.com`.
It installs its dependencies from GitHub and PyPI and uses ephemeral disk to stage the complete
Parquet file before KFP uploads it.

The staged Parquet file is capped at 1 GiB by default. Set `max_snapshot_size_bytes`
to a positive byte limit that leaves space on the task pod for runtime files and the
HTML report. The cap applies to the compressed Parquet file, including its footer.
If the next write would exceed the limit, the task stops and deletes the staged file.

### 1. Configure gateway TLS trust

Use the public CA bundle that validates the DCH gateway when the gateway uses a private issuer.
The component uses the launcher's `SSL_CERT_FILE` when set. Set the names for your deployment
and check the existing DSPA CA reference.

```sh
TENANT_NAMESPACE=your-project
DSPA_NAME=your-dspa
RUNNER_SA="pipeline-runner-${DSPA_NAME}"

oc get dspa "$DSPA_NAME" -n "$TENANT_NAMESPACE" -o jsonpath='{.spec.apiServer.cABundle}{"\n"}'
```

If the reference is empty, create a ConfigMap from your public CA PEM file and configure the
DSPA to use it.

```sh
oc create configmap dch-ingress-ca -n "$TENANT_NAMESPACE" --from-file=ca-bundle.crt=/path/to/dch-ingress-ca.pem
oc patch dspa "$DSPA_NAME" -n "$TENANT_NAMESPACE" --type=merge \
  -p '{"spec":{"apiServer":{"cABundle":{"configMapName":"dch-ingress-ca","configMapKey":"ca-bundle.crt"}}}}'
oc rollout status "deployment/ds-pipeline-$DSPA_NAME" -n "$TENANT_NAMESPACE"
```

If DSPA already points to a CA ConfigMap, add the DCH issuer CA to its referenced key and
preserve the other certificates. Start a new run after DSPA reconciles and check for
`using launcher CA bundle` in the task log.
[OpenShift AI certificate configuration](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/3.5/html/installing_and_uninstalling_openshift_ai_self-managed/working-with-certificates_certs)

### 2. Grant the runner access to the DCH APIs

The component reads the Kubernetes ServiceAccount token and namespace mounted in the task pod.
DCH must accept the token's audience. If the task pod uses a different ServiceAccount, set
`RUNNER_SA` to its actual `.spec.serviceAccountName`. Bind `dch-read` in the tenant namespace
to allow the REST connection lookup and Flight query.
[DCH tenant authorization and authentication](https://github.com/opendatahub-io/data-connect-hub/blob/6d58bc9975a431c8f8a9040e610520d20c980f36/docs/user-guide/auth.md#24-grant-users-access-to-flight-and-rest-apis)

```sh
oc create rolebinding dch-pipeline-runner-read -n "$TENANT_NAMESPACE" \
  --clusterrole=dch-read --serviceaccount="$TENANT_NAMESPACE:$RUNNER_SA"
```

### 3. Create or select the PostgreSQL connection

Use a saved, Flight-enabled `Postgres` connection in the tenant. Save its ID and its
`credentials_ref.secret` name. The Secret must contain `URI` and can contain `CA_CERT`
if PostgreSQL uses a private CA. The database role must be able to read the
selected table.
[DCH PostgreSQL connector requirements](https://github.com/opendatahub-io/data-connect-hub/blob/6d58bc9975a431c8f8a9040e610520d20c980f36/docs/user-guide/connectors.md)
· [DCH connection creation](https://github.com/opendatahub-io/data-connect-hub/blob/6d58bc9975a431c8f8a9040e610520d20c980f36/docs/user-guide/getting-started-ocp.md#create-connection)

To create a connection through REST, the API caller needs `dch-read-write`. If you submit
inline credentials so DCH creates the Secret, grant the REST ServiceAccount
`dch-rest-service-secret-access` in the tenant. With an existing Secret reference, the
REST ServiceAccount needs `get` on that Secret if you run the connection readiness check.
[DCH REST Secret permissions](https://github.com/opendatahub-io/data-connect-hub/blob/6d58bc9975a431c8f8a9040e610520d20c980f36/docs/user-guide/auth.md#23-grant-rest-service-permission-to-manage-connection-secrets)
· [DCH readiness troubleshooting](https://github.com/opendatahub-io/data-connect-hub/blob/6d58bc9975a431c8f8a9040e610520d20c980f36/docs/user-guide/auth.md#connection-creation-succeeds-but-readiness-reports-secret-cannot-be-read)

```sh
DCH_REST_NAMESPACE=your-rest-namespace
DCH_REST_SA=your-rest-service-account

oc create rolebinding dch-rest-service-secret-access -n "$TENANT_NAMESPACE" \
  --clusterrole=dch-rest-service-secret-access --serviceaccount="$DCH_REST_NAMESPACE:$DCH_REST_SA"
```

### 4. Grant Flight access to the connection Secret

Find the Flight deployment that serves your gateway and use its namespace and ServiceAccount.
Grant Flight `get` on the exact Secret so it can query PostgreSQL.
[DCH Flight Secret access](https://github.com/opendatahub-io/data-connect-hub/blob/6d58bc9975a431c8f8a9040e610520d20c980f36/docs/user-guide/auth.md#22-grant-flight-service-permission-to-read-connection-secrets)

```sh
oc get deployments -A -l app.kubernetes.io/part-of=data-connect-hub \
  -o custom-columns='NAMESPACE:.metadata.namespace,DEPLOYMENT:.metadata.name,SA:.spec.template.spec.serviceAccountName'

CREDENTIAL_SECRET_NAME=your-connection-secret
FLIGHT_NAMESPACE=your-flight-namespace
FLIGHT_SA=your-flight-service-account

oc create role dch-connection-secret-reader -n "$TENANT_NAMESPACE" \
  --verb=get --resource=secrets --resource-name="$CREDENTIAL_SECRET_NAME"
oc create rolebinding dch-flight-source-secret -n "$TENANT_NAMESPACE" \
  --role=dch-connection-secret-reader --serviceaccount="$FLIGHT_NAMESPACE:$FLIGHT_SA"
```

For readiness checks on a connection that references an existing Secret, set the REST
ServiceAccount variables from step 3 and bind the scoped Role to that account as well.

```sh
oc create rolebinding dch-rest-source-secret -n "$TENANT_NAMESPACE" \
  --role=dch-connection-secret-reader --serviceaccount="$DCH_REST_NAMESPACE:$DCH_REST_SA"
```

Verify the Flight grant from an account that can impersonate its ServiceAccount.

```sh
oc auth can-i get "secret/$CREDENTIAL_SECRET_NAME" -n "$TENANT_NAMESPACE" \
  --as="system:serviceaccount:$FLIGHT_NAMESPACE:$FLIGHT_SA"
```

## Optional direct PostgreSQL diagnostics

Set `run_postgres_diagnostics` to `true` to count rows in `pg_catalog.pg_namespace`
over both Flight SQL and a direct PostgreSQL connection. The database role needs
read access to that catalog. The task pod then needs network access to the Kubernetes
API and PostgreSQL, and the runner needs `get` on the same Secret. A failed diagnostic
fails that run before it publishes a snapshot.

```sh
oc create rolebinding dch-runner-source-secret -n "$TENANT_NAMESPACE" \
  --role=dch-connection-secret-reader --serviceaccount="$TENANT_NAMESPACE:$RUNNER_SA"

oc auth can-i get "secret/$CREDENTIAL_SECRET_NAME" -n "$TENANT_NAMESPACE" \
  --as="system:serviceaccount:$TENANT_NAMESPACE:$RUNNER_SA"
```

## Use in a pipeline

To run the component as a standalone pipeline, use the [DCH table export pipeline](../../../../pipelines/data_processing/dch/export_table_to_parquet/README.md).

Set the trusted host when composing the pipeline, using your own gateway address. The run
supplies the saved connection ID and exact table name. Set caching off so every run exercises
the current connection and captures fresh data.

```python
from kfp import dsl
from kfp_components.components.data_processing.dch.export_table_to_parquet import (
    export_table_to_parquet,
)

DCH_GATEWAY = "dch-gateway.example.com:443"


@dsl.pipeline(name="check-dch-connection")
def smoke_test(connection_id: str, table_name: str, table_schema: str = "public") -> str:
    task = export_table_to_parquet(
        dch_host=DCH_GATEWAY,
        connection_id=connection_id,
        trusted_host=DCH_GATEWAY,
        table_name=table_name,
        table_schema=table_schema,
    ).set_caching_options(False)
    return task.outputs["snapshot_uri"]
```

The task logs each check and exits with status 1 when a requested check fails.
During a normal export, DCH Flight queries only the selected table. The task
reads it in Arrow batches and writes one Parquet file. KFP uploads the dataset
and HTML report using its configured artifact-store credentials. The report includes
the table, row and column counts, file size, check results, and the exact Parquet URI.

## Train from the snapshot

Compile the [AutoGluon example](example_pipelines.py) from the installed package:

```python
from kfp import compiler
from kfp_components.components.data_processing.dch.export_table_to_parquet.example_pipelines import (
    train_automl_from_dch_snapshot,
)

compiler.Compiler().compile(train_automl_from_dch_snapshot, "train-automl-from-dch-snapshot.yaml")
```

Upload the YAML and run it on a pipeline server with read access to the same artifact store.
Pass the first run's `snapshot_uri` and the name of the target column as `label_column`.
AutoGluon infers classification or regression from the label and saves the fitted predictor
as a KFP Model artifact. For example, export a prepared subscription table and train a churn
classifier using its `churned` column. The existing tabular AutoML template takes CSV, while
this example reads the Parquet snapshot directly.
