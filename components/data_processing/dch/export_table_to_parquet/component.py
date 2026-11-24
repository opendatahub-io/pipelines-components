"""Validate a saved DCH PostgreSQL connection and export a Parquet snapshot."""

from typing import NamedTuple

from kfp import dsl

DCH_SDK = (
    "data-connect-hub[flight] @ git+https://github.com/opendatahub-io/data-connect-hub.git"
    "@6d58bc9975a431c8f8a9040e610520d20c980f36#subdirectory=sdk/python"
)


@dsl.component(
    base_image="python:3.12-bookworm",
    packages_to_install=[DCH_SDK, "psycopg[binary]==3.3.4"],
)
def export_table_to_parquet(
    dch_host: str,
    connection_id: str,
    trusted_host: str,
    table_name: str,
    snapshot: dsl.Output[dsl.Dataset],
    report: dsl.Output[dsl.HTML],
    table_schema: str = "public",
) -> NamedTuple("Outputs", [("snapshot_uri", str)]):
    """Check DCH access and export the selected PostgreSQL table through Flight SQL.

    Checks the gateway CA and task token, resolves the connection's credential
    Secret, and runs the same read-only catalog count through Flight and PostgreSQL.
    Publishes a Parquet dataset and HTML report through KFP's artifact store.

    Args:
        dch_host: DCH gateway hostname and optional port requested for the run.
        connection_id: ID of a saved PostgreSQL DCH connection in the task namespace.
        trusted_host: Gateway hostname and optional port fixed when composing the pipeline.
            The component sends the task ServiceAccount token only to this gateway.
        table_name: Exact name of the source table.
        snapshot: Parquet dataset artifact.
        report: HTML report with check results and snapshot statistics.
        table_schema: Schema containing the source table.

    Returns:
        The exact artifact URI of the Parquet file.
    """
    import base64
    import json
    import os
    import re
    import ssl
    from contextlib import closing
    from html import escape
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from urllib.error import HTTPError
    from urllib.parse import quote, urlsplit
    from urllib.request import Request, urlopen

    import psycopg
    import pyarrow as pa
    import pyarrow.parquet as pq
    from data_connect_hub import DataConnectClient, DCHError

    sql = "SELECT COUNT(*) AS row_count FROM pg_catalog.pg_namespace"
    ca = os.getenv("SSL_CERT_FILE") or None
    account = Path("/var/run/secrets/kubernetes.io/serviceaccount")
    checks = dict.fromkeys(
        (
            "Gateway and input",
            "Task ServiceAccount token",
            "DCH connection lookup",
            "DCH Flight SQL",
            "Kubernetes Secret read",
            "Direct PostgreSQL SQL",
            "Parquet snapshot",
        ),
        "SKIP",
    )
    details = {}

    def start(name, detail):
        print(f"[START] {name} | {detail}", flush=True)

    def record(name, status, detail):
        checks[name] = status
        details[name] = detail
        print(f"[{status}] {name} | {detail}", flush=True)

    def log_failure(name, exc):
        # Translate known failures without printing exception messages or response bodies.
        hints = (
            (
                "failed to obtain credentials",
                "Flight could not obtain the saved source credentials.",
            ),
            ("secret not found", "The saved source Secret is missing or inaccessible to Flight."),
            ("unauthenticated", "Check the runner token and the DCH token audience."),
            ("permission denied", "Check the runner's DCH tenant permissions."),
            ("certificate verify failed", "Check the relevant server certificate and CA trust."),
            ("password authentication failed", "PostgreSQL rejected the saved credentials."),
            ("could not translate host name", "The server hostname could not be resolved."),
            ("connection refused", "The server refused the connection."),
            ("timed out", "The connection or query timed out."),
        )
        raw_message = str(exc).lower()
        hint = next((detail for text, detail in hints if text in raw_message), "")
        http_status = getattr(exc, "status_code", None)
        sqlstate = getattr(exc, "sqlstate", None)
        details = [type(exc).__name__]
        if isinstance(http_status, int):
            details.append(f"HTTP {http_status}")
        if sqlstate and re.fullmatch(r"[0-9A-Z]{5}", sqlstate):
            details.append(f"SQLSTATE {sqlstate}")
        if http_status == 401:
            hint = "Check the runner token and the DCH token audience."
        elif http_status == 403:
            hint = "Check the runner's DCH tenant permissions."
        elif http_status == 404:
            hint = "Check the connection ID and the DCH tenant namespace."
        if hint:
            details.append(hint)
        record(name, "FAIL", " | ".join(details))

    def authority(value):
        value = value.strip()
        parsed = urlsplit(value if "://" in value else f"https://{value}")
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
            or parsed.port == 0
        ):
            raise ValueError("Expected an HTTPS gateway hostname and optional port")
        return parsed.hostname, parsed.port or 443

    def read_token():
        return (account / "token").read_text().strip()

    def dch_client(tenant, flight_timeout):
        return DataConnectClient(
            endpoint=trusted_host.strip().rstrip("/"),
            tenant_id=tenant,
            token_provider=read_token,
            ca_cert=ca,
            rest_timeout=20,
            flight_timeout=flight_timeout,
            max_retries=0,
        )

    def run_checks():
        name = "Gateway and input"
        start(name, "Validating DCH_HOST before sending the ServiceAccount token.")
        try:
            if authority(dch_host) != authority(trusted_host):
                record(name, "FAIL", f"DCH_HOST must match the compiled gateway {trusted_host}.")
                return
        except ValueError:
            record(name, "FAIL", "Expected an HTTPS gateway hostname with an optional port.")
            return
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", connection_id):
            record(name, "FAIL", "The connection ID is empty or has an invalid format.")
            return
        if any(not identifier or "\0" in identifier for identifier in (table_schema, table_name)):
            record(name, "FAIL", "Provide a valid source schema and table name.")
            return
        record(name, "PASS", f"Using trusted gateway {trusted_host}.")

        name = "Task ServiceAccount token"
        start(name, "Checking the task pod's mounted Kubernetes token and namespace.")
        try:
            tenant = (account / "namespace").read_text().strip()
            if not tenant or not read_token():
                raise ValueError("Empty namespace or token")
        except (OSError, ValueError):
            record(name, "FAIL", "The default Kubernetes token or namespace file is unavailable.")
            return
        record(name, "PASS", f"Using the run namespace as DCH tenant: {tenant}.")

        name = "DCH connection lookup"
        trust = "launcher CA bundle" if ca else "default CA trust"
        start(name, f"Looking up the saved connection via DCH REST using {trust}.")
        try:
            with dch_client(tenant, flight_timeout=20) as client:
                connection = client.get_connection(connection_id)
                if connection.format != "tabular":
                    record(name, "FAIL", "The saved connection is not tabular.")
                    return
                record(name, "PASS", f"DCH REST returned the connection using {trust}.")

                name = "DCH Flight SQL"
                start(name, sql)
                count = client.read(sql, connection_id).column("row_count")[0].as_py()
                record(name, "PASS", f"COUNT(*) returned {count}.")
        except (DCHError, OSError, ValueError, KeyError, IndexError) as exc:
            log_failure(name, exc)
            if name == "DCH connection lookup":
                return

        name = "Kubernetes Secret read"
        secret_name = connection.credentials_ref.secret
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?", secret_name):
            record(name, "FAIL", "DCH returned an invalid Secret name.")
            return
        start(name, f"Reading the referenced Secret {secret_name} in namespace {tenant}.")
        try:
            namespaced_url = (
                f"https://kubernetes.default.svc/api/v1/namespaces/{quote(tenant, safe='')}"
                f"/secrets/{quote(secret_name, safe='')}"
            )
            request = Request(namespaced_url, headers={"Authorization": f"Bearer {read_token()}"})
            context = ssl.create_default_context(cafile=str(account / "ca.crt"))
            with urlopen(request, context=context, timeout=10) as response:
                secret = json.load(response)["data"]
            uri = base64.b64decode(secret["URI"], validate=True).decode().strip()
            ca_pem = base64.b64decode(secret.get("CA_CERT", ""), validate=True)
            if not uri.startswith(("postgres://", "postgresql://")):
                raise ValueError("Expected a PostgreSQL URI")
        except HTTPError as exc:
            hint = {
                401: "Kubernetes rejected the mounted token.",
                403: "Grant the runner get on this Secret in the run namespace.",
                404: "Check that the referenced Secret exists in the run namespace.",
            }.get(exc.code, "The Kubernetes API rejected the Secret read.")
            record(name, "FAIL", f"HTTP {exc.code} | {hint}")
            return
        except OSError:
            record(name, "FAIL", "Could not reach the Kubernetes API or load its CA certificate.")
            return
        except (ValueError, TypeError, KeyError):
            record(name, "FAIL", "Expected a valid PostgreSQL URI and optional CA_CERT in the Secret.")
            return
        record(name, "PASS", "Retrieved the PostgreSQL URI and checked the optional CA key.")

        name = "Direct PostgreSQL SQL"
        start(name, f"Executing with the retrieved credentials. {sql}")
        try:
            with TemporaryDirectory() as tmp:
                options = {}
                if ca_pem:
                    ca_file = Path(tmp) / "postgres-ca.pem"
                    ca_file.write_bytes(ca_pem)
                    options["sslrootcert"] = str(ca_file)
                with psycopg.connect(uri, connect_timeout=10, **options) as conn:
                    conn.read_only = True
                    conn.execute("SET LOCAL statement_timeout = '20s'")
                    row = conn.execute(sql).fetchone()
                    if row is None:
                        raise ValueError("COUNT(*) returned no row")
                    (count,) = row
            record(name, "PASS", f"COUNT(*) returned {count}.")
        except (psycopg.Error, OSError, ValueError) as exc:
            log_failure(name, exc)
        return tenant

    def quote_identifier(identifier):
        return '"' + identifier.replace('"', '""') + '"'

    def write_snapshot(tenant):
        name = "Parquet snapshot"
        source = f"{quote_identifier(table_schema)}.{quote_identifier(table_name)}"
        start(name, f"Reading {source} through DCH Flight SQL.")
        query = f"SELECT * FROM {source}"
        if not snapshot.uri:
            raise ValueError("KFP did not assign a snapshot artifact URI")
        snapshot.uri += ".parquet"
        destination = Path(snapshot.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        row_count = 0
        with (
            TemporaryDirectory(dir=destination.parent) as tmp,
            dch_client(tenant, flight_timeout=300) as client,
            closing(client.read_batches(query, connection_id)) as batches,
        ):
            temporary_path = Path(tmp) / "snapshot.parquet"
            first_batch = next(batches, None)
            if first_batch is None:
                empty_table = client.read(f"{query} LIMIT 0", connection_id)
                schema = empty_table.schema
                pq.write_table(empty_table, temporary_path)
            else:
                schema = first_batch.schema
                with pq.ParquetWriter(temporary_path, schema) as writer:
                    writer.write_batch(first_batch)
                    row_count = first_batch.num_rows
                    for batch in batches:
                        writer.write_batch(batch)
                        row_count += batch.num_rows
            temporary_path.replace(destination)

        snapshot.metadata.update(
            format="parquet",
            row_count=row_count,
            column_count=len(schema),
            size_bytes=destination.stat().st_size,
        )
        record(name, "PASS", f"Wrote {row_count:,} rows and {len(schema):,} columns to {snapshot.uri}.")

    def write_report():
        check_rows = "".join(
            f"<tr><th scope='row'>{escape(name)}</th><td>{status}</td><td>{escape(details[name])}</td></tr>"
            for name, status in checks.items()
        )
        metadata = snapshot.metadata
        report_path = Path(report.path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            "<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<title>DCH table snapshot</title><body><h1>DCH table snapshot</h1>"
            "<p>Overall result: <strong>PASS</strong></p><dl>"
            f"<dt>Table</dt><dd><code>{escape(table_schema)}.{escape(table_name)}</code></dd>"
            f"<dt>Rows</dt><dd>{metadata['row_count']:,}</dd>"
            f"<dt>Columns</dt><dd>{metadata['column_count']:,}</dd>"
            f"<dt>Parquet size</dt><dd>{metadata['size_bytes']:,} bytes</dd>"
            f"<dt>Location</dt><dd><code>{escape(snapshot.uri)}</code></dd></dl>"
            "<h2>Checks</h2><table><thead><tr><th>Check</th><th>Status</th>"
            f"<th>Detail</th></tr></thead><tbody>{check_rows}</tbody></table></body></html>",
            encoding="utf-8",
        )

    print("[START] DCH connection smoke test", flush=True)
    tenant = run_checks()
    if all(status == "PASS" for name, status in checks.items() if name != "Parquet snapshot"):
        try:
            write_snapshot(tenant)
            write_report()
        except (DCHError, pa.ArrowException, OSError, ValueError) as exc:
            log_failure("Parquet snapshot", exc)
    for name, status in checks.items():
        if status == "SKIP":
            print(f"[SKIP] {name} | A prerequisite check did not pass.", flush=True)

    failed = [name for name, status in checks.items() if status == "FAIL"]
    if failed:
        print(f"[FAIL] Overall result | Failed checks: {', '.join(failed)}.", flush=True)
        raise SystemExit(1)
    print("[PASS] Overall result | All checks passed.", flush=True)
    return (snapshot.uri,)
