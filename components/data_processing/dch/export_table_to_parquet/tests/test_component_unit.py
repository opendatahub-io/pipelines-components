"""Unit tests for DCH table export and connection diagnostics."""

import base64
import io
import json
import sys
from pathlib import Path
from random import Random
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from kfp import compiler, dsl

from ..component import export_table_to_parquet

GATEWAY = "dch-gateway.example.com:443"
CONNECTION_ID = "saved-connection"
TOKEN = "test-runner-token"
URI = "postgresql://test-user:test-password@source.example.test/db"


@pytest.fixture
def services(monkeypatch):
    """Provide a pod identity and mock DCH, Kubernetes, and PostgreSQL calls."""
    import pyarrow as pa

    account = Path("/var/run/secrets/kubernetes.io/serviceaccount")
    original_read = Path.read_text

    def read_text(path, *args, **kwargs):
        if path.parent == account:
            return {"token": TOKEN, "namespace": "tenant"}[path.name]
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setenv("SSL_CERT_FILE", "/kfp/certs/ca.crt")

    class DCHError(Exception):
        """Mock SDK error with a REST status."""

        status_code = 403

    dch = MagicMock()
    dch.__enter__.return_value = dch
    dch.get_connection.return_value = SimpleNamespace(
        format="tabular",
        credentials_ref=SimpleNamespace(secret="postgres-credentials"),
    )
    dch.read.return_value.column.return_value = [SimpleNamespace(as_py=lambda: 4)]
    dch.read_batches.side_effect = lambda *_: (
        pa.record_batch({"id": ids, "label": labels}) for ids, labels in (([1, 2], ["a", "b"]), ([3], ["a"]))
    )
    client = MagicMock(return_value=dch)
    monkeypatch.setitem(sys.modules, "data_connect_hub", SimpleNamespace(DataConnectClient=client, DCHError=DCHError))

    data = {"URI": base64.b64encode(URI.encode()).decode()}
    secret_read = MagicMock(return_value=io.BytesIO(json.dumps({"data": data}).encode()))
    monkeypatch.setattr("ssl.create_default_context", MagicMock())
    monkeypatch.setattr("urllib.request.urlopen", secret_read)

    postgres = MagicMock()
    database = postgres.connect.return_value.__enter__.return_value
    database.execute.return_value.fetchone.return_value = (4,)
    monkeypatch.setitem(sys.modules, "psycopg", postgres)
    return SimpleNamespace(client=client, dch=dch, secret_read=secret_read, postgres=postgres, error=DCHError)


def test_component_compiles(tmp_path):
    """Compile the component and check its artifact and parameter interfaces."""
    output = tmp_path / "component.yaml"
    compiler.Compiler().compile(export_table_to_parquet, str(output))
    assert output.stat().st_size > 0
    spec = export_table_to_parquet.component_spec
    assert set(spec.inputs) == {
        "dch_host",
        "connection_id",
        "trusted_host",
        "table_name",
        "table_schema",
        "run_postgres_diagnostics",
        "max_snapshot_size_bytes",
    }
    assert spec.inputs["run_postgres_diagnostics"].default is False
    assert spec.inputs["max_snapshot_size_bytes"].default == 1_073_741_824
    assert set(spec.outputs) == {"snapshot", "report", "snapshot_uri"}


@pytest.fixture
def outputs(tmp_path):
    """Supply local paths for the two KFP artifacts."""
    return dsl.Dataset(uri=str(tmp_path / "snapshot")), dsl.HTML(uri=str(tmp_path / "report"))


def test_rejects_an_untrusted_host_before_using_the_token(services, outputs, capsys):
    """Keep the Kubernetes bearer token within the configured gateway."""
    with pytest.raises(SystemExit) as failure:
        export_table_to_parquet.python_func("untrusted.example.com", CONNECTION_ID, GATEWAY, "training", *outputs)
    assert failure.value.code == 1
    services.client.assert_not_called()
    services.secret_read.assert_not_called()
    assert "[FAIL] Gateway and input" in capsys.readouterr().out


@pytest.mark.parametrize("run_postgres_diagnostics", [False, True])
def test_exports_through_flight_with_optional_postgres_diagnostics(services, outputs, capsys, run_postgres_diagnostics):
    """Export the selected table and query the catalog only when diagnostics are requested."""
    import pyarrow.parquet as pq

    extra_inputs = {}
    if run_postgres_diagnostics:
        extra_inputs["run_postgres_diagnostics"] = True
    else:
        del services.dch.get_connection.return_value.credentials_ref
    result = export_table_to_parquet.python_func(
        GATEWAY, CONNECTION_ID, GATEWAY, 'training"<view>', *outputs, table_schema="ml", **extra_inputs
    )
    snapshot, report = outputs
    assert result == (snapshot.uri,)
    assert snapshot.uri.endswith(".parquet")
    assert pq.read_table(snapshot.path).to_pydict() == {"id": [1, 2, 3], "label": ["a", "b", "a"]}
    assert snapshot.metadata == {
        "format": "parquet",
        "row_count": 3,
        "column_count": 2,
        "size_bytes": Path(snapshot.path).stat().st_size,
    }
    html = Path(report.path).read_text()
    assert "ml.training&quot;&lt;view&gt;" in html
    assert snapshot.uri in html and "<dt>Rows</dt><dd>3</dd>" in html
    assert TOKEN not in html and URI not in html
    options = services.client.call_args.kwargs
    assert options["endpoint"] == GATEWAY
    assert options["tenant_id"] == "tenant"
    assert options["token_provider"]() == TOKEN
    assert options["ca_cert"] == "/kfp/certs/ca.crt"
    services.dch.get_connection.assert_called_once_with(CONNECTION_ID)
    services.dch.read_batches.assert_called_once_with('SELECT * FROM "ml"."training""<view>"', CONNECTION_ID)
    output = capsys.readouterr().out
    if run_postgres_diagnostics:
        sql = "SELECT COUNT(*) AS row_count FROM pg_catalog.pg_namespace"
        services.dch.read.assert_called_once_with(sql, CONNECTION_ID)
        request = services.secret_read.call_args.args[0]
        assert request.full_url.endswith("/namespaces/tenant/secrets/postgres-credentials")
        services.postgres.connect.assert_called_once_with(URI, connect_timeout=10)
        database = services.postgres.connect.return_value.__enter__.return_value
        database.execute.assert_any_call(sql)
        assert "[PASS] DCH Flight SQL catalog count" in output
        assert "[PASS] Direct PostgreSQL SQL" in output
    else:
        services.dch.read.assert_not_called()
        services.secret_read.assert_not_called()
        services.postgres.connect.assert_not_called()
        assert "DCH Flight SQL catalog count</th><td>SKIP</td>" in html
        assert "Direct PostgreSQL SQL</th><td>SKIP</td>" in html
        assert "[SKIP] Direct PostgreSQL SQL | PostgreSQL diagnostics were not requested." in output
    assert "[PASS] Overall result" in output
    assert TOKEN not in output and URI not in output


def test_failed_rest_lookup_skips_secret_access(services, outputs, capsys):
    """Report authorization errors without logging sensitive response details."""
    services.dch.get_connection.side_effect = services.error(f"Failure with {TOKEN} and {URI}")
    with pytest.raises(SystemExit) as failure:
        export_table_to_parquet.python_func(
            GATEWAY, CONNECTION_ID, GATEWAY, "training", *outputs, run_postgres_diagnostics=True
        )
    assert failure.value.code == 1
    services.dch.read.assert_not_called()
    services.secret_read.assert_not_called()
    services.dch.read_batches.assert_not_called()
    assert not any(Path(artifact.path).exists() for artifact in outputs)
    output = capsys.readouterr().out
    assert "[FAIL] DCH connection lookup" in output and "HTTP 403" in output
    assert "[SKIP] Kubernetes Secret read" in output
    assert TOKEN not in output and URI not in output


def test_requested_diagnostics_fail_when_secret_access_is_denied(services, outputs, capsys):
    """Keep diagnostic failures visible and prevent publishing a successful snapshot."""
    from urllib.error import HTTPError

    services.secret_read.side_effect = HTTPError("https://kubernetes.default.svc", 403, "Forbidden", {}, None)
    with pytest.raises(SystemExit) as failure:
        export_table_to_parquet.python_func(
            GATEWAY, CONNECTION_ID, GATEWAY, "training", *outputs, run_postgres_diagnostics=True
        )

    assert failure.value.code == 1
    services.postgres.connect.assert_not_called()
    services.dch.read_batches.assert_not_called()
    assert "[FAIL] Kubernetes Secret read | HTTP 403" in capsys.readouterr().out


def test_writes_valid_parquet_for_an_empty_table(services, outputs):
    """Recover the schema when Flight emits no batches for an empty table."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([("id", pa.int64()), ("label", pa.string())])
    services.dch.read_batches.side_effect = lambda *_: (batch for batch in [])
    services.dch.read.return_value = schema.empty_table()
    export_table_to_parquet.python_func(GATEWAY, CONNECTION_ID, GATEWAY, "empty_training", *outputs)
    snapshot, _ = outputs
    assert pq.read_schema(snapshot.path) == schema
    assert pq.ParquetFile(snapshot.path).metadata.num_rows == 0
    services.dch.read.assert_called_once_with('SELECT * FROM "public"."empty_training" LIMIT 0', CONNECTION_ID)


def test_removes_partial_snapshot_if_flight_fails(services, outputs, capsys):
    """Clean up the staged parquet file when a query breaks mid-stream."""
    import pyarrow as pa

    def broken_batches(*_):
        yield pa.record_batch({"id": [1]})
        raise services.error(f"Failure with {TOKEN} and {URI}")

    services.dch.read_batches.side_effect = broken_batches
    with pytest.raises(SystemExit) as failure:
        export_table_to_parquet.python_func(GATEWAY, CONNECTION_ID, GATEWAY, "training", *outputs)
    assert failure.value.code == 1
    assert not any(Path(artifact.path).exists() for artifact in outputs)
    assert not list(Path(outputs[0].path).parent.iterdir())
    output = capsys.readouterr().out
    assert "[FAIL] Parquet snapshot" in output
    assert TOKEN not in output and URI not in output


def test_rejects_nonpositive_snapshot_limit(services, outputs, capsys):
    """Reject an invalid size limit before contacting DCH."""
    with pytest.raises(SystemExit) as failure:
        export_table_to_parquet.python_func(
            GATEWAY, CONNECTION_ID, GATEWAY, "training", *outputs, max_snapshot_size_bytes=0
        )
    assert failure.value.code == 1
    services.client.assert_not_called()
    assert "max_snapshot_size_bytes must be a positive integer" in capsys.readouterr().out


def test_stops_reading_and_removes_snapshot_when_a_batch_exceeds_limit(services, outputs, capsys):
    """Abort the stream on an oversized write and clean up the staged file."""
    import pyarrow as pa

    def oversized_batches(*_):
        yield pa.record_batch({"blob": [Random(0).randbytes(2048)]})
        pytest.fail("The export requested another batch after exceeding its size limit")

    services.dch.read_batches.side_effect = oversized_batches

    with pytest.raises(SystemExit) as failure:
        export_table_to_parquet.python_func(
            GATEWAY, CONNECTION_ID, GATEWAY, "training", *outputs, max_snapshot_size_bytes=128
        )
    assert failure.value.code == 1
    assert not list(Path(outputs[0].path).parent.iterdir())
    output = capsys.readouterr().out
    assert "[FAIL] Parquet snapshot | Parquet file exceeds max_snapshot_size_bytes (128 bytes)" in output


@pytest.mark.parametrize("empty", [False, True])
def test_enforces_limit_including_parquet_footer(services, outputs, tmp_path, capsys, empty):
    """Allow an exact fit and reject a one-byte overflow, even for an empty table."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if empty:
        services.dch.read_batches.side_effect = lambda *_: (batch for batch in [])
        services.dch.read.return_value = pa.schema([("id", pa.int64())]).empty_table()
    run = export_table_to_parquet.python_func
    run(GATEWAY, CONNECTION_ID, GATEWAY, "training", *outputs)
    file_size = Path(outputs[0].path).stat().st_size

    exact = dsl.Dataset(uri=str(tmp_path / "exact")), dsl.HTML(uri=str(tmp_path / "exact-report"))
    run(GATEWAY, CONNECTION_ID, GATEWAY, "training", *exact, max_snapshot_size_bytes=file_size)
    assert Path(exact[0].path).stat().st_size == file_size
    assert pq.ParquetFile(exact[0].path).metadata.num_rows == (0 if empty else 3)

    rejected = tmp_path / "rejected"
    capped = dsl.Dataset(uri=str(rejected / "snapshot")), dsl.HTML(uri=str(rejected / "report"))
    with pytest.raises(SystemExit) as failure:
        run(GATEWAY, CONNECTION_ID, GATEWAY, "training", *capped, max_snapshot_size_bytes=file_size - 1)
    assert failure.value.code == 1
    assert not list(rejected.iterdir())
    assert "[FAIL] Parquet snapshot | Parquet file exceeds max_snapshot_size_bytes" in capsys.readouterr().out
