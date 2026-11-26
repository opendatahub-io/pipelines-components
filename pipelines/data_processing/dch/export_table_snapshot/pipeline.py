"""Create an uploadable KFP pipeline for exporting a relational table through DCH."""

from typing import NamedTuple

from kfp import dsl
from kfp.dsl import graph_component
from kfp.kubernetes import use_secret_as_env
from kfp_components.components.data_processing.dch.export_table_snapshot import export_table_snapshot


class SnapshotOutputs(NamedTuple):
    """Artifacts and the exact snapshot URI published by a successful run."""

    snapshot: dsl.Dataset
    report: dsl.HTML
    snapshot_uri: str


def create_export_table_snapshot_pipeline() -> graph_component.GraphComponent:
    """Build the export pipeline with its required gateway Secret."""

    @dsl.pipeline(name="dch-export-table-snapshot", display_name="DCH Table Export")
    def pipeline(
        connection_id: str,
        source_table: str,
        output_format: str = "parquet",
    ) -> SnapshotOutputs:
        """Export a DCH table as CSV or Parquet and publish an HTML summary.

        Args:
            connection_id: ID of the saved Flight SQL connection in the run namespace.
                Requires the dch-gateway Secret with a host key in that namespace.
            source_table: Source table in schema.table_name format, such as analytics.customers.
            output_format: Snapshot format, either parquet (default) or csv.

        Returns:
            The Dataset, HTML report, and exact snapshot URI.
        """
        task = export_table_snapshot(
            connection_id=connection_id,
            source_table=source_table,
            output_format=output_format,
        )
        task.set_display_name("export-table-snapshot").set_caching_options(False)
        use_secret_as_env(task, secret_name="dch-gateway", secret_key_to_env={"host": "DCH_TRUSTED_HOST"})
        return SnapshotOutputs(
            snapshot=task.outputs["snapshot"],
            report=task.outputs["report"],
            snapshot_uri=task.outputs["snapshot_uri"],
        )

    return pipeline


dch_export_table_pipeline = create_export_table_snapshot_pipeline()


if __name__ == "__main__":
    from pathlib import Path

    from kfp.compiler import Compiler

    Compiler().compile(
        dch_export_table_pipeline,
        package_path=str(Path(__file__).with_name("dch-export-table-snapshot.yaml")),
    )
