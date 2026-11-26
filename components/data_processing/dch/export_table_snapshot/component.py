"""Export a relational table from a saved DCH connection as CSV or Parquet."""

from typing import NamedTuple

from kfp import dsl


@dsl.component(
    base_image="registry.access.redhat.com/ubi9/python-312:latest",
    packages_to_install=["data-connect-hub[flight]==0.2.1"],
)
def export_table_snapshot(
    connection_id: str,
    source_table: str,
    snapshot: dsl.Output[dsl.Dataset],
    report: dsl.Output[dsl.HTML],
    output_format: str = "parquet",
) -> NamedTuple("Outputs", [("snapshot_uri", str)]):
    """Export a relational table through a saved DCH Flight SQL connection.

    Uses the gateway CA and task token to look up the saved connection and read
    the selected table through Flight SQL. Publishes a CSV or Parquet dataset and
    HTML report through KFP.

    Args:
        connection_id: ID of a saved Flight SQL connection in the task namespace.
            Requires the dch-gateway Secret with a host key for the Data Connect Hub gateway.
        source_table: Source table in schema.table_name format, such as analytics.customers.
        snapshot: CSV or Parquet dataset artifact.
        report: HTML report with the source table, snapshot statistics, and artifact URI.
        output_format: Snapshot format, either csv or parquet.

    Returns:
        The exact artifact URI of the exported file.
    """
    import os
    import re
    from contextlib import closing
    from html import escape
    from io import BufferedWriter
    from itertools import chain
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from urllib.parse import urlsplit

    import pyarrow as pa
    import pyarrow.parquet as pq
    from data_connect_hub import DataConnectClient
    from pyarrow import csv

    gateway = os.getenv("DCH_TRUSTED_HOST", "").strip().rstrip("/")
    table_parts = [identifier.strip() for identifier in source_table.split(".", 2)]
    output_format = output_format.strip().lower()
    snapshot_writers = {"csv": csv.CSVWriter, "parquet": pq.ParquetWriter}
    ca = os.getenv("SSL_CERT_FILE") or None
    account = Path("/var/run/secrets/kubernetes.io/serviceaccount")
    snapshot_limit_bytes = 1_073_741_824
    checks = dict.fromkeys(
        (
            "Gateway and input",
            "Task ServiceAccount token",
            "DCH connection lookup",
            "Table snapshot",
            "HTML report",
        ),
        "SKIP",
    )

    class SnapshotSizeLimitExceeded(ValueError):
        """A write would exceed the snapshot size limit."""

    class BoundedSnapshotWriter(BufferedWriter):
        """Check the file position before passing each write to disk."""

        def write(self, data):
            if self.tell() + len(data) > snapshot_limit_bytes:
                raise SnapshotSizeLimitExceeded
            return super().write(data)

    def start(name, detail):
        print(f"[START] {name} | {detail}", flush=True)

    def record(name, status, detail):
        checks[name] = status
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
            ("password authentication failed", "The source database rejected the saved credentials."),
            ("could not translate host name", "The server hostname could not be resolved."),
            ("connection refused", "The server refused the connection."),
            ("timed out", "The connection or query timed out."),
        )
        raw_message = str(exc).lower()
        hint = next((detail for text, detail in hints if text in raw_message), "")
        if output_format == "csv" and isinstance(exc, pa.ArrowException) and "unsupported type" in raw_message:
            hint = "Set output_format=parquet to export columns that CSV cannot represent."
        http_status = getattr(exc, "status_code", None)
        details = [type(exc).__name__]
        if isinstance(http_status, int):
            details.append(f"HTTP {http_status}")
        if http_status == 401:
            hint = "Check the runner token and the DCH token audience."
        elif http_status == 403:
            hint = "Check the runner's DCH tenant permissions."
        elif http_status == 404:
            hint = "Check the connection ID and the DCH tenant namespace."
        if hint:
            details.append(hint)
        record(name, "FAIL", " | ".join(details))

    def validate_gateway(value):
        if "\\" in value or any(character.isspace() or ord(character) < 32 for character in value):
            raise ValueError("Expected an HTTPS gateway hostname and optional port")
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

    def read_token():
        return (account / "token").read_text().strip()

    def dch_client(tenant, flight_timeout, max_retries=0):
        return DataConnectClient(
            endpoint=gateway,
            tenant_id=tenant,
            token_provider=read_token,
            ca_cert=ca,
            rest_timeout=20,
            flight_timeout=flight_timeout,
            max_retries=max_retries,
        )

    def run_checks():
        name = "Gateway and input"
        start(name, "Validating the gateway from the dch-gateway Secret before using the ServiceAccount token.")
        try:
            validate_gateway(gateway)
        except ValueError:
            record(name, "FAIL", "Set the dch-gateway Secret's host key to an HTTPS hostname and optional port.")
            return
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", connection_id):
            record(name, "FAIL", "The connection ID is empty or has an invalid format.")
            return
        if len(table_parts) != 2 or not all(table_parts) or "\0" in source_table:
            record(name, "FAIL", "Provide source_table in schema.table_name format, such as analytics.customers.")
            return
        if output_format not in snapshot_writers:
            record(name, "FAIL", "Set output_format to csv or parquet.")
            return
        record(name, "PASS", f"Using DCH gateway {gateway}.")

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
            with dch_client(tenant, flight_timeout=20, max_retries=2) as client:
                connection = client.get_connection(connection_id)
                if connection.format != "tabular":
                    record(name, "FAIL", "The saved connection is not tabular.")
                    return
                record(name, "PASS", f"DCH REST returned the connection using {trust}.")
        except Exception as exc:
            log_failure(name, exc)
            return
        return tenant

    def quote_identifier(identifier):
        return '"' + identifier.replace('"', '""') + '"'

    def write_snapshot(tenant):
        name = "Table snapshot"
        source = ".".join(quote_identifier(identifier) for identifier in table_parts)
        start(name, f"Reading {source} through DCH Flight SQL.")
        query = f"SELECT * FROM {source}"
        if not snapshot.uri:
            raise ValueError("KFP did not assign a snapshot artifact URI")
        snapshot.uri += f".{output_format}"
        destination = Path(snapshot.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        row_count = 0
        with (
            TemporaryDirectory(dir=destination.parent) as tmp,
            dch_client(tenant, flight_timeout=300) as client,
            closing(client.read_batches(query, connection_id)) as batches,
        ):
            temporary_path = Path(tmp) / f"snapshot.{output_format}"
            with BoundedSnapshotWriter(temporary_path.open("wb", buffering=0)) as sink:
                first_batch = next(batches, None)
                if first_batch is None:
                    schema = client.read(f"{query} WHERE 1 = 0", connection_id).schema
                else:
                    schema = first_batch.schema
                    batches = chain((first_batch,), batches)
                with snapshot_writers[output_format](sink, schema) as writer:
                    for batch in batches:
                        writer.write_batch(batch)
                        row_count += batch.num_rows
            temporary_path.replace(destination)

        snapshot.metadata.update(
            format=output_format,
            row_count=row_count,
            column_count=len(schema),
            size_bytes=destination.stat().st_size,
        )
        record(name, "PASS", f"Wrote {row_count:,} rows and {len(schema):,} columns to {snapshot.uri}.")

    def write_report():
        metadata = snapshot.metadata
        report_path = Path(report.path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<title>DCH table snapshot</title></head><body><h1>DCH table snapshot</h1><dl>"
            f"<dt>Table</dt><dd><code>{escape('.'.join(table_parts))}</code></dd>"
            f"<dt>Rows</dt><dd>{metadata['row_count']:,}</dd>"
            f"<dt>Columns</dt><dd>{metadata['column_count']:,}</dd>"
            f"<dt>Format</dt><dd>{output_format.upper()}</dd>"
            f"<dt>File size</dt><dd>{metadata['size_bytes']:,} bytes</dd>"
            f"<dt>Location</dt><dd><code>{escape(snapshot.uri)}</code></dd></dl></body></html>",
            encoding="utf-8",
        )
        record("HTML report", "PASS", "Wrote the snapshot summary.")

    print("[START] DCH table export", flush=True)
    tenant = run_checks()
    if tenant is not None:
        try:
            write_snapshot(tenant)
        except SnapshotSizeLimitExceeded:
            record(
                "Table snapshot",
                "FAIL",
                f"Snapshot file exceeds the 1 GiB limit ({snapshot_limit_bytes:,} bytes). "
                "Export a smaller table or view.",
            )
        except Exception as exc:
            log_failure("Table snapshot", exc)
        else:
            try:
                write_report()
            except Exception as exc:
                record("HTML report", "FAIL", f"Could not write the snapshot summary ({type(exc).__name__}).")
    for name, status in checks.items():
        if status == "SKIP":
            print(f"[SKIP] {name} | A prerequisite check did not pass.", flush=True)

    failed = [name for name, status in checks.items() if status == "FAIL"]
    if failed:
        summary = f"Failed checks: {', '.join(failed)}."
        print(f"[FAIL] Overall result | {summary}", flush=True)
        raise RuntimeError(summary)
    print("[PASS] Overall result | All checks passed.", flush=True)
    return (snapshot.uri,)
