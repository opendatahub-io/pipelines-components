"""Unit tests for DCH table export."""

import csv
import sys
from io import BufferedWriter
from pathlib import Path
from random import Random
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from kfp import compiler, dsl

from ..component import export_table_snapshot

GATEWAY = "dch-gateway.example.com:443"
CONNECTION_ID = "saved-connection"
SOURCE_TABLE = "public.training"
TOKEN = "test-runner-token"
SNAPSHOT_LIMIT_BYTES = 1_073_741_824


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
    monkeypatch.setenv("DCH_TRUSTED_HOST", GATEWAY)

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
    compiler.Compiler().compile(export_table_snapshot, str(output))
    assert output.stat().st_size > 0
    spec = export_table_snapshot.component_spec
    assert set(spec.inputs) == {
        "connection_id",
        "source_table",
        "output_format",
    }
    assert spec.inputs["output_format"].default == "parquet"
    assert set(spec.outputs) == {"snapshot", "report", "snapshot_uri"}


@pytest.fixture
def outputs(tmp_path):
    """Supply local paths for the two KFP artifacts."""
    return dsl.Dataset(uri=str(tmp_path / "snapshot")), dsl.HTML(uri=str(tmp_path / "report"))


def test_uses_gateway_from_secret(services, outputs, monkeypatch):
    """Use the HTTPS gateway injected from the Secret."""
    monkeypatch.setenv("DCH_TRUSTED_HOST", f"https://{GATEWAY}/")
    result = export_table_snapshot.python_func(CONNECTION_ID, SOURCE_TABLE, *outputs)

    assert result == (outputs[0].uri,)
    assert services.client.call_args.kwargs["endpoint"] == f"https://{GATEWAY}"


@pytest.mark.parametrize(
    "gateway", [None, "", "http://gateway.example.com", "https://gateway.example.com/api", "user@host"]
)
def test_rejects_missing_or_invalid_gateway(services, outputs, capsys, monkeypatch, gateway):
    """Reject missing or malformed Secret values before reading the token."""
    if gateway is None:
        monkeypatch.delenv("DCH_TRUSTED_HOST")
    else:
        monkeypatch.setenv("DCH_TRUSTED_HOST", gateway)
    with pytest.raises(RuntimeError, match="Failed checks: Gateway and input"):
        export_table_snapshot.python_func(CONNECTION_ID, SOURCE_TABLE, *outputs)

    services.client.assert_not_called()
    log = capsys.readouterr().out
    assert "Set the dch-gateway Secret's host key" in log
    assert "[START] Task ServiceAccount token" not in log


@pytest.mark.parametrize("source_table", ["training", ".training", "public.", "public.training.extra", "public.bad\0"])
def test_rejects_invalid_qualified_table(services, outputs, capsys, source_table):
    """Require a nonempty schema and table before contacting DCH."""
    with pytest.raises(RuntimeError, match="Failed checks: Gateway and input"):
        export_table_snapshot.python_func(CONNECTION_ID, source_table, *outputs)

    services.client.assert_not_called()
    assert "Provide source_table in schema.table_name format" in capsys.readouterr().out


def test_exports_selected_table_through_flight(services, outputs, capsys):
    """Quote each identifier and publish the selected table and export report."""
    import pyarrow.parquet as pq

    result = export_table_snapshot.python_func(CONNECTION_ID, 'ml.training"<view>', *outputs)
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
    assert "<head><meta charset='utf-8'>" in html
    assert "</title></head><body>" in html
    assert "ml.training&quot;&lt;view&gt;" in html
    assert snapshot.uri in html and "<dt>Rows</dt><dd>3</dd>" in html
    assert "<table>" not in html
    assert TOKEN not in html
    options = services.client.call_args.kwargs
    assert options["endpoint"] == GATEWAY
    assert options["tenant_id"] == "tenant"
    assert options["token_provider"]() == TOKEN
    assert options["ca_cert"] == "/kfp/certs/ca.crt"
    assert [call.kwargs["max_retries"] for call in services.client.call_args_list] == [2, 0]
    services.dch.get_connection.assert_called_once_with(CONNECTION_ID)
    services.dch.read_batches.assert_called_once_with('SELECT * FROM "ml"."training""<view>"', CONNECTION_ID)
    services.dch.read.assert_not_called()
    output = capsys.readouterr().out
    assert "[PASS] Overall result" in output
    assert TOKEN not in output


def test_exports_csv_with_one_header_and_quoted_fields(services, outputs):
    """Write multiple CSV batches using one header and preserve punctuation and newlines."""
    import pyarrow as pa

    values = ['quoted "value"', "contains,comma", "two\nlines"]
    services.dch.read_batches.side_effect = lambda *_: (
        pa.record_batch({"label": batch}) for batch in (values[:2], values[2:])
    )
    result = export_table_snapshot.python_func(CONNECTION_ID, SOURCE_TABLE, *outputs, output_format="csv")

    snapshot, report = outputs
    assert result == (snapshot.uri,)
    assert snapshot.uri.endswith(".csv")
    with Path(snapshot.path).open(encoding="utf-8", newline="") as stream:
        assert list(csv.reader(stream)) == [["label"], *[[value] for value in values]]
    assert snapshot.metadata == {
        "format": "csv",
        "row_count": 3,
        "column_count": 1,
        "size_bytes": Path(snapshot.path).stat().st_size,
    }
    assert "<dt>Format</dt><dd>CSV</dd>" in Path(report.path).read_text()


def test_rejects_unknown_output_format(services, outputs, capsys):
    """Reject an unsupported format before contacting DCH."""
    with pytest.raises(RuntimeError, match="Failed checks: Gateway and input"):
        export_table_snapshot.python_func(CONNECTION_ID, SOURCE_TABLE, *outputs, output_format="json")

    services.client.assert_not_called()
    assert "Set output_format to csv or parquet." in capsys.readouterr().out


def test_csv_reports_unsupported_column_type(services, outputs, capsys):
    """Point to Parquet and clean up when Arrow CSV cannot write a nested column."""
    import pyarrow as pa

    services.dch.read_batches.side_effect = lambda *_: (batch for batch in [pa.record_batch({"values": [[1, 2]]})])
    with pytest.raises(RuntimeError, match="Failed checks: Table snapshot"):
        export_table_snapshot.python_func(CONNECTION_ID, SOURCE_TABLE, *outputs, output_format="csv")

    assert not list(Path(outputs[0].path).parent.iterdir())
    assert "Set output_format=parquet" in capsys.readouterr().out


@pytest.mark.parametrize("sdk_error", [True, False])
def test_failed_rest_lookup_skips_export(services, outputs, capsys, sdk_error):
    """Summarize SDK and unexpected REST errors without logging sensitive response details."""
    error_type = services.error if sdk_error else RuntimeError
    services.dch.get_connection.side_effect = error_type(f"Failure with {TOKEN}")
    with pytest.raises(RuntimeError, match="Failed checks: DCH connection lookup"):
        export_table_snapshot.python_func(CONNECTION_ID, SOURCE_TABLE, *outputs)
    services.dch.read.assert_not_called()
    services.dch.read_batches.assert_not_called()
    assert not any(Path(artifact.path).exists() for artifact in outputs)
    output = capsys.readouterr().out
    assert f"[FAIL] DCH connection lookup | {error_type.__name__}" in output
    if sdk_error:
        assert "HTTP 403" in output
    assert "[SKIP] Table snapshot" in output
    assert TOKEN not in output


@pytest.mark.parametrize("output_format", ["csv", "parquet"])
def test_writes_valid_snapshot_for_an_empty_table(services, outputs, output_format):
    """Recover the schema when Flight emits no batches for an empty table."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([("id", pa.int64()), ("label", pa.string())])
    services.dch.read_batches.side_effect = lambda *_: (batch for batch in [])
    services.dch.read.return_value = schema.empty_table()
    export_table_snapshot.python_func(CONNECTION_ID, "public.empty_training", *outputs, output_format=output_format)
    snapshot, _ = outputs
    assert snapshot.uri.endswith(f".{output_format}")
    assert snapshot.metadata["row_count"] == 0
    if output_format == "csv":
        with Path(snapshot.path).open(encoding="utf-8", newline="") as stream:
            assert list(csv.reader(stream)) == [["id", "label"]]
    else:
        assert pq.read_schema(snapshot.path) == schema
        assert pq.ParquetFile(snapshot.path).metadata.num_rows == 0
    services.dch.read.assert_called_once_with('SELECT * FROM "public"."empty_training" WHERE 1 = 0', CONNECTION_ID)


def test_removes_partial_snapshot_if_flight_fails(services, outputs, capsys):
    """Clean up the staged parquet file when a query breaks mid-stream."""
    import pyarrow as pa

    def broken_batches(*_):
        yield pa.record_batch({"id": [1]})
        raise RuntimeError(f"Failure with {TOKEN}")

    services.dch.read_batches.side_effect = broken_batches
    with pytest.raises(RuntimeError, match="Failed checks: Table snapshot"):
        export_table_snapshot.python_func(CONNECTION_ID, SOURCE_TABLE, *outputs)
    assert not any(Path(artifact.path).exists() for artifact in outputs)
    assert not list(Path(outputs[0].path).parent.iterdir())
    output = capsys.readouterr().out
    assert "[FAIL] Table snapshot" in output
    assert TOKEN not in output


def test_logs_report_failure_separately_from_snapshot(services, outputs, capsys, monkeypatch):
    """Attribute a report write failure to the report after the snapshot succeeds."""
    snapshot, report = outputs
    original_write = Path.write_text

    def write_text(path, *args, **kwargs):
        if path == Path(report.path):
            raise PermissionError("Permission denied")
        return original_write(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write_text)
    with pytest.raises(RuntimeError, match="Failed checks: HTML report"):
        export_table_snapshot.python_func(CONNECTION_ID, SOURCE_TABLE, *outputs)

    assert Path(snapshot.path).is_file()
    output = capsys.readouterr().out
    assert "[PASS] Table snapshot" in output
    assert "[FAIL] HTML report | Could not write the snapshot summary (PermissionError)." in output
    assert "[FAIL] Table snapshot" not in output


def _simulate_written_bytes(monkeypatch, written_bytes):
    """Exercise the 1 GiB limit without writing a large test file."""

    class NearCapacityWriter(BufferedWriter):
        def tell(self):
            return super().tell() + written_bytes

    monkeypatch.setattr("io.BufferedWriter", NearCapacityWriter)


@pytest.mark.parametrize("output_format", ["csv", "parquet"])
def test_stops_reading_and_removes_snapshot_when_a_batch_exceeds_limit(
    services, outputs, capsys, monkeypatch, output_format
):
    """Abort the stream on an oversized write and clean up the staged file."""
    import pyarrow as pa

    def oversized_batches(*_):
        yield pa.record_batch({"value": [Random(0).randbytes(2048).hex()]})
        pytest.fail("The export requested another batch after exceeding its size limit")

    services.dch.read_batches.side_effect = oversized_batches
    _simulate_written_bytes(monkeypatch, SNAPSHOT_LIMIT_BYTES - 128)

    with pytest.raises(RuntimeError, match="Failed checks: Table snapshot"):
        export_table_snapshot.python_func(CONNECTION_ID, SOURCE_TABLE, *outputs, output_format=output_format)
    assert not list(Path(outputs[0].path).parent.iterdir())
    output = capsys.readouterr().out
    assert "[FAIL] Table snapshot | Snapshot file exceeds the 1 GiB limit (1,073,741,824 bytes)" in output


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("output_format", ["csv", "parquet"])
def test_enforces_limit_for_complete_file(services, outputs, tmp_path, capsys, monkeypatch, empty, output_format):
    """Allow an exact fit and reject a one-byte overflow, even for an empty table."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if empty:
        services.dch.read_batches.side_effect = lambda *_: (batch for batch in [])
        services.dch.read.return_value = pa.schema([("id", pa.int64())]).empty_table()
    run = export_table_snapshot.python_func
    run(CONNECTION_ID, SOURCE_TABLE, *outputs, output_format=output_format)
    file_size = Path(outputs[0].path).stat().st_size

    _simulate_written_bytes(monkeypatch, SNAPSHOT_LIMIT_BYTES - file_size)
    exact = dsl.Dataset(uri=str(tmp_path / "exact")), dsl.HTML(uri=str(tmp_path / "exact-report"))
    run(CONNECTION_ID, SOURCE_TABLE, *exact, output_format=output_format)
    assert Path(exact[0].path).stat().st_size == file_size
    if output_format == "csv":
        assert Path(exact[0].path).read_bytes() == Path(outputs[0].path).read_bytes()
    else:
        assert pq.ParquetFile(exact[0].path).metadata.num_rows == (0 if empty else 3)

    rejected = tmp_path / "rejected"
    capped = dsl.Dataset(uri=str(rejected / "snapshot")), dsl.HTML(uri=str(rejected / "report"))
    _simulate_written_bytes(monkeypatch, SNAPSHOT_LIMIT_BYTES - file_size + 1)
    with pytest.raises(RuntimeError, match="Failed checks: Table snapshot"):
        run(CONNECTION_ID, SOURCE_TABLE, *capped, output_format=output_format)
    assert not list(rejected.iterdir())
    assert "[FAIL] Table snapshot | Snapshot file exceeds the 1 GiB limit" in capsys.readouterr().out
