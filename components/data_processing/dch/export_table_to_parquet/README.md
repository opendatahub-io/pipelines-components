# Export Table To Parquet ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Export a relational table through a saved DCH Flight SQL connection.

Uses the gateway CA and task token to look up the saved connection and read the selected table through Flight SQL. Publishes a Parquet dataset and HTML report through KFP.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `dch_host` | `str` | `None` | DCH gateway hostname and optional port requested for the run. |
| `connection_id` | `str` | `None` | ID of a saved SQL-capable relational DCH connection in the task namespace. |
| `source_table` | `str` | `None` | Source table in schema.table_name format, such as analytics.customers. |
| `snapshot` | `dsl.Output[dsl.Dataset]` | `None` | Parquet dataset artifact. |
| `report` | `dsl.Output[dsl.HTML]` | `None` | HTML report with the source table, snapshot statistics, and artifact URI. |
| `max_snapshot_size_bytes` | `int` | `1073741824` | Maximum size in bytes of the staged Parquet file, including its footer. |
| `trusted_host` | `str` | `""` | Optional HTTPS gateway restriction set by the pipeline author. An empty value uses dch_host as the endpoint for the task ServiceAccount token. |

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
- **Description**: Export a relational table through a saved Data Connect Hub Flight SQL connection as Parquet and publish an HTML report from one KFP task.
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
over HTTPS and Flight SQL. The DCH Flight service uses the saved connection to access
the source database. The task uses the Red Hat UBI 9 Python 3.12 image from `registry.access.redhat.com`.
It installs its dependencies from GitHub and PyPI and uses ephemeral disk to stage the complete
Parquet file before KFP uploads it.

The staged Parquet file is capped at 1 GiB by default. Set `max_snapshot_size_bytes`
to a positive byte limit that leaves space on the task pod for runtime files and the
HTML report. The cap applies to the compressed Parquet file, including its footer.
If the next write would exceed the limit, the task stops and deletes the staged file.

### Check the DSPA artifact store

Set the names for your deployment. For an external S3 artifact store, check that the
DSPA host and region match the bucket's provider and location. The task pod must be
able to resolve and reach the resulting S3 endpoint and upload to the bucket.

```sh
TENANT_NAMESPACE=your-project
DSPA_NAME=your-dspa
RUNNER_SA="pipeline-runner-${DSPA_NAME}"

oc get dspa "$DSPA_NAME" -n "$TENANT_NAMESPACE" \
  -o custom-columns=BUCKET:.spec.objectStorage.externalStorage.bucket,HOST:.spec.objectStorage.externalStorage.host,REGION:.spec.objectStorage.externalStorage.region
```

For a launcher error such as `PutObject ... no such host`, compare the hostname
in the log with the provider's endpoint for that bucket. Correct the DSPA host or
region if they differ.

### 1. Configure gateway TLS trust

Use the public CA bundle that validates the DCH gateway when the gateway uses a private issuer.
The component uses the launcher's `SSL_CERT_FILE` when set. Check the existing DSPA CA reference.

```sh
oc get dspa "$DSPA_NAME" -n "$TENANT_NAMESPACE" -o jsonpath='{.spec.apiServer.cABundle}{"\n"}'
```

If the reference is empty, reuse `dch-ingress-ca` in the tenant namespace if it
already contains the gateway's issuer CA under `ca-bundle.crt`. Create it from
the public CA PEM file only if it is missing.

```sh
oc create configmap dch-ingress-ca -n "$TENANT_NAMESPACE" --from-file=ca-bundle.crt=/path/to/dch-ingress-ca.pem
```

Set the DSPA reference for either an existing or a newly created ConfigMap.

```sh
oc patch dspa "$DSPA_NAME" -n "$TENANT_NAMESPACE" --type=merge \
  -p '{"spec":{"apiServer":{"cABundle":{"configMapName":"dch-ingress-ca","configMapKey":"ca-bundle.crt"}}}}'
```

If DSPA already has a CA reference, add the gateway's issuer CA to the referenced
key if it is missing and preserve the other certificates. After the operator
reconciles, verify that the generated `dsp-trusted-ca-$DSPA_NAME` ConfigMap
includes the issuer and wait for the pipeline server rollout.

```sh
oc get configmap "dsp-trusted-ca-$DSPA_NAME" -n "$TENANT_NAMESPACE" \
  -o jsonpath='{.data.dsp-ca\.crt}' |
  openssl crl2pkcs7 -nocrl -certfile /dev/stdin |
  openssl pkcs7 -print_certs -noout
oc rollout status "deployment/ds-pipeline-$DSPA_NAME" -n "$TENANT_NAMESPACE"
```

Start a fresh run. A `[PASS] DCH connection lookup` line confirms that the task
completed the TLS handshake and looked up the saved connection.
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

### 3. Create or select a relational connection

Use a saved tabular DCH connection whose connector accepts SQL queries over Flight.
The export uses `SELECT * FROM "schema"."table_name"`, so the connector must accept
double-quoted, schema-qualified identifiers and the source credentials must have
read access to the table. Save the connection ID and follow the connector's
credential and TLS setup instructions.
[DCH connector requirements](https://github.com/opendatahub-io/data-connect-hub/blob/6d58bc9975a431c8f8a9040e610520d20c980f36/docs/user-guide/connectors.md)
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

If the connector uses a credential Secret, get its name from the saved
connection's `credentials_ref.secret` field.
Find the Flight deployment that serves your gateway and use its namespace and ServiceAccount.
Grant Flight `get` on that Secret so it can query the source database.
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

## Use in a pipeline

To run the component as a standalone pipeline, use the [DCH table export pipeline](../../../../pipelines/data_processing/dch/export_table_to_parquet/README.md).

Supply the DCH gateway, saved connection ID, and `source_table` when creating a run.
Use the `schema.table_name` format, for example `analytics.customers`.
The component sends the task ServiceAccount token to the selected HTTPS gateway.
Set an explicit `trusted_host` when composing the pipeline to restrict that destination.
Disable caching so each run captures fresh data.

```python
from kfp import dsl
from kfp_components.components.data_processing.dch.export_table_to_parquet import (
    export_table_to_parquet,
)

@dsl.pipeline(name="export-dch-table")
def export_dch_table(DCH_HOST: str, connection_id: str, source_table: str) -> str:
    task = export_table_to_parquet(
        dch_host=DCH_HOST,
        connection_id=connection_id,
        source_table=source_table,
    ).set_caching_options(False)
    return task.outputs["snapshot_uri"]
```

The task streams the selected table in Arrow batches through DCH Flight and writes one
Parquet file. KFP uploads the dataset and HTML report using its artifact-store credentials.
The report includes the table, row and column counts, file size, and the Parquet URI.
Connection checks and errors appear in the task log.

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
