"""Unit tests for the DCH connection smoke test."""

import base64
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from kfp import compiler, dsl

from ..component import db_table_2_parquet

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
    compiler.Compiler().compile(db_table_2_parquet, str(output))
    assert output.stat().st_size > 0
    spec = db_table_2_parquet.component_spec
    assert set(spec.inputs) == {"dch_host", "connection_id", "trusted_host", "table_name", "table_schema"}
    assert set(spec.outputs) == {"snapshot", "report", "snapshot_uri"}


@pytest.fixture
def outputs(tmp_path):
    """Supply local paths for the two KFP artifacts."""
    return dsl.Dataset(uri=str(tmp_path / "snapshot")), dsl.HTML(uri=str(tmp_path / "report"))


def test_rejects_an_untrusted_host_before_using_the_token(services, outputs, capsys):
    """Keep the Kubernetes bearer token within the configured gateway."""
    with pytest.raises(SystemExit) as failure:
        db_table_2_parquet.python_func("untrusted.example.com", CONNECTION_ID, GATEWAY, "training", *outputs)
    assert failure.value.code == 1
    services.client.assert_not_called()
    services.secret_read.assert_not_called()
    assert "[FAIL] Gateway and input" in capsys.readouterr().out


def test_checks_flight_and_direct_postgres(services, outputs, capsys):
    """Exercise both routes and omit the source credentials from check logs."""
    import pyarrow.parquet as pq

    result = db_table_2_parquet.python_func(
        GATEWAY, CONNECTION_ID, GATEWAY, 'training"<view>', *outputs, table_schema="ml"
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
    sql = "SELECT COUNT(*) AS row_count FROM pg_catalog.pg_namespace"
    services.dch.read.assert_called_once_with(sql, CONNECTION_ID)
    services.dch.read_batches.assert_called_once_with('SELECT * FROM "ml"."training""<view>"', CONNECTION_ID)
    request = services.secret_read.call_args.args[0]
    assert request.full_url.endswith("/namespaces/tenant/secrets/postgres-credentials")
    services.postgres.connect.assert_called_once_with(URI, connect_timeout=10)
    output = capsys.readouterr().out
    assert "[PASS] DCH Flight SQL" in output
    assert "[PASS] Direct PostgreSQL SQL" in output
    assert "[PASS] Overall result" in output
    assert TOKEN not in output and URI not in output


def test_failed_rest_lookup_skips_secret_access(services, outputs, capsys):
    """Report authorization errors without logging sensitive response details."""
    services.dch.get_connection.side_effect = services.error(f"Failure with {TOKEN} and {URI}")
    with pytest.raises(SystemExit) as failure:
        db_table_2_parquet.python_func(GATEWAY, CONNECTION_ID, GATEWAY, "training", *outputs)
    assert failure.value.code == 1
    services.dch.read.assert_not_called()
    services.secret_read.assert_not_called()
    services.dch.read_batches.assert_not_called()
    assert not any(Path(artifact.path).exists() for artifact in outputs)
    output = capsys.readouterr().out
    assert "[FAIL] DCH connection lookup" in output and "HTTP 403" in output
    assert "[SKIP] Kubernetes Secret read" in output
    assert TOKEN not in output and URI not in output


def test_writes_valid_parquet_for_an_empty_table(services, outputs):
    """Recover the schema when Flight emits no batches for an empty table."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([("id", pa.int64()), ("label", pa.string())])
    services.dch.read_batches.side_effect = lambda *_: (batch for batch in [])
    services.dch.read.side_effect = [services.dch.read.return_value, schema.empty_table()]
    db_table_2_parquet.python_func(GATEWAY, CONNECTION_ID, GATEWAY, "empty_training", *outputs)
    snapshot, _ = outputs
    assert pq.read_schema(snapshot.path) == schema
    assert pq.ParquetFile(snapshot.path).metadata.num_rows == 0
    services.dch.read.assert_called_with('SELECT * FROM "public"."empty_training" LIMIT 0', CONNECTION_ID)


def test_removes_partial_snapshot_if_flight_fails(services, outputs, capsys):
    """Clean up the staged parquet file when a query breaks mid-stream."""
    import pyarrow as pa

    def broken_batches(*_):
        yield pa.record_batch({"id": [1]})
        raise services.error(f"Failure with {TOKEN} and {URI}")

    services.dch.read_batches.side_effect = broken_batches
    with pytest.raises(SystemExit) as failure:
        db_table_2_parquet.python_func(GATEWAY, CONNECTION_ID, GATEWAY, "training", *outputs)
    assert failure.value.code == 1
    assert not any(Path(artifact.path).exists() for artifact in outputs)
    assert not list(Path(outputs[0].path).parent.iterdir())
    output = capsys.readouterr().out
    assert "[FAIL] Parquet snapshot" in output
    assert TOKEN not in output and URI not in output
