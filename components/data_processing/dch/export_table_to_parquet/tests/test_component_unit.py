"""Unit tests for DCH table export."""

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
SOURCE_TABLE = "public.training"
TOKEN = "test-runner-token"


@pytest.fixture
def services(monkeypatch):
    """Provide a pod identity and mock the DCH client."""
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
    dch.get_connection.return_value = SimpleNamespace(format="tabular")
    dch.read_batches.side_effect = lambda *_: (
        pa.record_batch({"id": ids, "label": labels}) for ids, labels in (([1, 2], ["a", "b"]), ([3], ["a"]))
    )
    client = MagicMock(return_value=dch)
    monkeypatch.setitem(sys.modules, "data_connect_hub", SimpleNamespace(DataConnectClient=client, DCHError=DCHError))

    return SimpleNamespace(client=client, dch=dch, error=DCHError)


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
        "source_table",
        "max_snapshot_size_bytes",
    }
    assert spec.inputs["max_snapshot_size_bytes"].default == 1_073_741_824
    assert set(spec.outputs) == {"snapshot", "report", "snapshot_uri"}


@pytest.fixture
def outputs(tmp_path):
    """Supply local paths for the two KFP artifacts."""
    return dsl.Dataset(uri=str(tmp_path / "snapshot")), dsl.HTML(uri=str(tmp_path / "report"))


def test_rejects_an_untrusted_host_before_using_the_token(services, outputs, capsys):
    """Keep the bearer token within the gateway when the pipeline author pins one."""
    with pytest.raises(SystemExit) as failure:
        export_table_to_parquet.python_func("untrusted.example.com", CONNECTION_ID, GATEWAY, SOURCE_TABLE, *outputs)
    assert failure.value.code == 1
    services.client.assert_not_called()
    assert "[FAIL] Gateway and input" in capsys.readouterr().out


def test_uses_runtime_gateway_when_no_gateway_is_pinned(services, outputs):
    """Use the HTTPS gateway selected at run time for a generic pipeline."""
    result = export_table_to_parquet.python_func(f"https://{GATEWAY}/", CONNECTION_ID, "", SOURCE_TABLE, *outputs)

    assert result == (outputs[0].uri,)
    assert services.client.call_args.kwargs["endpoint"] == f"https://{GATEWAY}"


@pytest.mark.parametrize("gateway", ["", "http://gateway.example.com", "https://gateway.example.com/api", "user@host"])
def test_rejects_invalid_runtime_gateway(services, outputs, capsys, gateway):
    """Reject missing or malformed gateways before opening a DCH client."""
    with pytest.raises(SystemExit) as failure:
        export_table_to_parquet.python_func(gateway, CONNECTION_ID, "", SOURCE_TABLE, *outputs)

    assert failure.value.code == 1
    services.client.assert_not_called()
    assert "Expected an HTTPS gateway hostname with an optional port." in capsys.readouterr().out


@pytest.mark.parametrize("source_table", ["training", ".training", "public.", "public.training.extra", "public.bad\0"])
def test_rejects_invalid_qualified_table(services, outputs, capsys, source_table):
    """Require a nonempty schema and table before contacting DCH."""
    with pytest.raises(SystemExit) as failure:
        export_table_to_parquet.python_func(GATEWAY, CONNECTION_ID, "", source_table, *outputs)

    assert failure.value.code == 1
    services.client.assert_not_called()
    assert "Provide source_table in schema.table_name format" in capsys.readouterr().out


def test_exports_selected_table_through_flight(services, outputs, capsys):
    """Quote each identifier and publish the selected table and export report."""
    import pyarrow.parquet as pq

    result = export_table_to_parquet.python_func(GATEWAY, CONNECTION_ID, GATEWAY, 'ml.training"<view>', *outputs)
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
    assert TOKEN not in html
    options = services.client.call_args.kwargs
    assert options["endpoint"] == GATEWAY
    assert options["tenant_id"] == "tenant"
    assert options["token_provider"]() == TOKEN
    assert options["ca_cert"] == "/kfp/certs/ca.crt"
    services.dch.get_connection.assert_called_once_with(CONNECTION_ID)
    services.dch.read_batches.assert_called_once_with('SELECT * FROM "ml"."training""<view>"', CONNECTION_ID)
    services.dch.read.assert_not_called()
    output = capsys.readouterr().out
    assert "[PASS] Overall result" in output
    assert TOKEN not in output


def test_failed_rest_lookup_skips_export(services, outputs, capsys):
    """Report authorization errors without logging sensitive response details."""
    services.dch.get_connection.side_effect = services.error(f"Failure with {TOKEN}")
    with pytest.raises(SystemExit) as failure:
        export_table_to_parquet.python_func(GATEWAY, CONNECTION_ID, GATEWAY, SOURCE_TABLE, *outputs)
    assert failure.value.code == 1
    services.dch.read.assert_not_called()
    services.dch.read_batches.assert_not_called()
    assert not any(Path(artifact.path).exists() for artifact in outputs)
    output = capsys.readouterr().out
    assert "[FAIL] DCH connection lookup" in output and "HTTP 403" in output
    assert "[SKIP] Parquet snapshot" in output
    assert TOKEN not in output


def test_writes_valid_parquet_for_an_empty_table(services, outputs):
    """Recover the schema when Flight emits no batches for an empty table."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([("id", pa.int64()), ("label", pa.string())])
    services.dch.read_batches.side_effect = lambda *_: (batch for batch in [])
    services.dch.read.return_value = schema.empty_table()
    export_table_to_parquet.python_func(GATEWAY, CONNECTION_ID, GATEWAY, "public.empty_training", *outputs)
    snapshot, _ = outputs
    assert pq.read_schema(snapshot.path) == schema
    assert pq.ParquetFile(snapshot.path).metadata.num_rows == 0
    services.dch.read.assert_called_once_with('SELECT * FROM "public"."empty_training" LIMIT 0', CONNECTION_ID)


def test_removes_partial_snapshot_if_flight_fails(services, outputs, capsys):
    """Clean up the staged parquet file when a query breaks mid-stream."""
    import pyarrow as pa

    def broken_batches(*_):
        yield pa.record_batch({"id": [1]})
        raise services.error(f"Failure with {TOKEN}")

    services.dch.read_batches.side_effect = broken_batches
    with pytest.raises(SystemExit) as failure:
        export_table_to_parquet.python_func(GATEWAY, CONNECTION_ID, GATEWAY, SOURCE_TABLE, *outputs)
    assert failure.value.code == 1
    assert not any(Path(artifact.path).exists() for artifact in outputs)
    assert not list(Path(outputs[0].path).parent.iterdir())
    output = capsys.readouterr().out
    assert "[FAIL] Parquet snapshot" in output
    assert TOKEN not in output


def test_rejects_nonpositive_snapshot_limit(services, outputs, capsys):
    """Reject an invalid size limit before contacting DCH."""
    with pytest.raises(SystemExit) as failure:
        export_table_to_parquet.python_func(
            GATEWAY, CONNECTION_ID, GATEWAY, SOURCE_TABLE, *outputs, max_snapshot_size_bytes=0
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
            GATEWAY, CONNECTION_ID, GATEWAY, SOURCE_TABLE, *outputs, max_snapshot_size_bytes=128
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
    run(GATEWAY, CONNECTION_ID, GATEWAY, SOURCE_TABLE, *outputs)
    file_size = Path(outputs[0].path).stat().st_size

    exact = dsl.Dataset(uri=str(tmp_path / "exact")), dsl.HTML(uri=str(tmp_path / "exact-report"))
    run(GATEWAY, CONNECTION_ID, GATEWAY, SOURCE_TABLE, *exact, max_snapshot_size_bytes=file_size)
    assert Path(exact[0].path).stat().st_size == file_size
    assert pq.ParquetFile(exact[0].path).metadata.num_rows == (0 if empty else 3)

    rejected = tmp_path / "rejected"
    capped = dsl.Dataset(uri=str(rejected / "snapshot")), dsl.HTML(uri=str(rejected / "report"))
    with pytest.raises(SystemExit) as failure:
        run(GATEWAY, CONNECTION_ID, GATEWAY, SOURCE_TABLE, *capped, max_snapshot_size_bytes=file_size - 1)
    assert failure.value.code == 1
    assert not list(rejected.iterdir())
    assert "[FAIL] Parquet snapshot | Parquet file exceeds max_snapshot_size_bytes" in capsys.readouterr().out
