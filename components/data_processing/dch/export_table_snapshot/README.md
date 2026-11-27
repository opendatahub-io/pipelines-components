# Export Table Snapshot ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Export a relational table through a saved DCH Flight SQL connection.

Uses the gateway CA and task token to look up the saved connection and read the selected table through Flight SQL. Publishes a CSV or Parquet dataset and HTML report through KFP.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `connection_id` | `str` | `None` | ID of a saved Flight SQL connection in the task namespace. Requires the dch-gateway Secret with a host key for the Data Connect Hub gateway. |
| `source_table` | `str` | `None` | Source table in schema.table_name format, such as analytics.customers. |
| `snapshot` | `dsl.Output[dsl.Dataset]` | `None` | CSV or Parquet dataset artifact. |
| `report` | `dsl.Output[dsl.HTML]` | `None` | HTML report with the source table, snapshot statistics, and artifact URI. |
| `output_format` | `str` | `parquet` | Snapshot format, either csv or parquet. |

## Outputs 📤

| Name | Type | Description |
| ---- | ---- | ----------- |
| Output | `NamedTuple('Outputs', [('snapshot_uri', str)])` | The exact artifact URI of the exported file. |

## Metadata 🗂️

- **Name**: export_table_snapshot
- **Description**: Export a relational table through a saved Data Connect Hub Flight SQL connection as CSV or Parquet and publish an HTML report from one KFP task.
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.16.1
  - External Services:
    - Name: Data Connect Hub Python SDK, Version: ==0.2.1
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

The component installs `data-connect-hub[flight]==0.2.1` at run time.
The task needs access to PyPI or a package mirror that also supplies the Flight SQL
dependencies, including `adbc-driver-flightsql`. [DCH Python SDK on PyPI](https://pypi.org/project/data-connect-hub/0.2.1/)

Run in the DCH tenant namespace that owns the connection. Configure a pipeline server with an S3-backed
artifact store and use its default pipeline root. The task pod needs access to the DCH gateway
over HTTPS and Flight SQL. The DCH Flight service uses the saved connection to access
the source database. The task uses the Red Hat UBI 9 Python 3.12 image from `registry.access.redhat.com`.
The task stages the full file on pod disk before KFP uploads it.

### Export size limit

Each export can stage up to 1 GiB of CSV or compressed Parquet, including the CSV
header or Parquet footer.
Allow pod disk space for the snapshot, runtime files, and HTML report. If the
next write would exceed 1 GiB, the task fails and deletes the staged file.
For larger sources, export a smaller table or a view with fewer rows or columns.

### Check the pipeline server's artifact store

Use the name of the pipeline server's `DataSciencePipelinesApplication` resource
in the commands below. For external S3 storage, check that the configured host
and region match the bucket's provider and location. The task pod must be able
to resolve and reach the S3 endpoint and upload to the bucket.

```sh
TENANT_NAMESPACE=your-project
PIPELINE_SERVER_NAME=your-pipeline-server
RUNNER_SA="pipeline-runner-${PIPELINE_SERVER_NAME}"

oc get dspa "$PIPELINE_SERVER_NAME" -n "$TENANT_NAMESPACE" \
  -o custom-columns=BUCKET:.spec.objectStorage.externalStorage.bucket,HOST:.spec.objectStorage.externalStorage.host,REGION:.spec.objectStorage.externalStorage.region
```

For a launcher error such as `PutObject ... no such host`, compare the hostname
in the log with the provider's endpoint for that bucket. Correct the pipeline
server's artifact store host or region if they differ.

### Create the required gateway Secret

An administrator must create a Secret named `dch-gateway` in the pipeline run's
tenant namespace. Set its `host` key to the approved gateway hostname and optional
port for the DCH REST and Flight SQL APIs.

```sh
oc create secret generic dch-gateway -n "$TENANT_NAMESPACE" \
  --from-literal=host=dch.example.com:443
```

**Trust requirement.** The task sends its ServiceAccount token to the host
configured in `dch-gateway`. Anyone who can create, update, patch, or delete and
replace this Secret must be trusted to act as the pipeline runner. If users can
launch the pipeline but cannot act as the runner, restrict those Secret
operations to trusted administrators before making the pipeline available to them.

The Secret is required even when the gateway uses a certificate the task already trusts.

### 1. Configure gateway TLS trust

Use the public CA bundle that validates the DCH gateway when the gateway uses a private issuer.
The component uses the launcher's `SSL_CERT_FILE` when set. Check the pipeline
server's CA bundle reference.

```sh
oc get dspa "$PIPELINE_SERVER_NAME" -n "$TENANT_NAMESPACE" -o jsonpath='{.spec.apiServer.cABundle}{"\n"}'
```

If the reference is empty, reuse `dch-ingress-ca` in the tenant namespace if it
already contains the gateway's issuer CA under `ca-bundle.crt`. Create it from
the public CA PEM file only if it is missing.

```sh
oc create configmap dch-ingress-ca -n "$TENANT_NAMESPACE" --from-file=ca-bundle.crt=/path/to/dch-ingress-ca.pem
```

Set the pipeline server's CA bundle reference to the ConfigMap.

```sh
oc patch dspa "$PIPELINE_SERVER_NAME" -n "$TENANT_NAMESPACE" --type=merge \
  -p '{"spec":{"apiServer":{"cABundle":{"configMapName":"dch-ingress-ca","configMapKey":"ca-bundle.crt"}}}}'
```

If the pipeline server already has a CA reference, add the gateway's issuer CA to the referenced
key if it is missing and preserve the other certificates. After the operator
reconciles, verify that the generated `dsp-trusted-ca-$PIPELINE_SERVER_NAME` ConfigMap
includes the issuer and wait for the pipeline server rollout.

```sh
oc get configmap "dsp-trusted-ca-$PIPELINE_SERVER_NAME" -n "$TENANT_NAMESPACE" \
  -o jsonpath='{.data.dsp-ca\.crt}' |
  openssl crl2pkcs7 -nocrl -certfile /dev/stdin |
  openssl pkcs7 -print_certs -noout
oc rollout status "deployment/ds-pipeline-$PIPELINE_SERVER_NAME" -n "$TENANT_NAMESPACE"
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

To run the component as a standalone pipeline, use the [DCH table export pipeline](../../../../pipelines/data_processing/dch/export_table_snapshot/README.md).

Supply the saved connection ID and `source_table` when creating a run.
Use the `schema.table_name` format, for example `analytics.customers`.
Parquet is the default and preserves types such as nested lists. Set
`output_format="csv"` to export a CSV file.
When composing your own pipeline, inject the `host` key from the required
`dch-gateway` Secret as `DCH_TRUSTED_HOST`.
Disable caching so each run captures fresh data.

```python
from kfp import dsl
from kfp.kubernetes import use_secret_as_env
from kfp_components.components.data_processing.dch.export_table_snapshot import (
    export_table_snapshot,
)

@dsl.pipeline(name="export-dch-table")
def export_dch_table(connection_id: str, source_table: str, output_format: str = "parquet") -> str:
    task = export_table_snapshot(
        connection_id=connection_id,
        source_table=source_table,
        output_format=output_format,
    ).set_caching_options(False)
    use_secret_as_env(task, secret_name="dch-gateway", secret_key_to_env={"host": "DCH_TRUSTED_HOST"})
    return task.outputs["snapshot_uri"]
```

The task streams the selected table in Arrow batches through DCH Flight and writes one
CSV or Parquet file. KFP uploads the dataset and HTML report using its artifact-store credentials.
The report includes the table, row and column counts, format, file size, and snapshot URI.
Both files are required for a successful run. If writing the report fails, the task fails
even if it has finished writing the snapshot.

Connection checks and errors appear in the task log. The connection lookup uses a
20-second REST timeout and allows up to two retries for transient failures. The Flight SQL
read uses a 300-second timeout and a failed stream ends the task.

## Train with AutoML

Use the exported CSV snapshot as input to the managed tabular AutoML pipeline.
Parquet snapshots require an AutoML version with Parquet input support. Follow the
[AutoML handoff instructions](../../../../pipelines/data_processing/dch/export_table_snapshot/README.md#train-with-automl).
