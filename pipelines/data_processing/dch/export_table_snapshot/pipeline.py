"""Create an uploadable KFP pipeline for exporting a relational table through DCH."""

from typing import NamedTuple

from kfp import dsl
from kfp.dsl import graph_component
from kfp_components.components.data_processing.dch.export_table_snapshot import export_table_snapshot


class SnapshotOutputs(NamedTuple):
    """Artifacts and the exact snapshot URI published by a successful run."""

    snapshot: dsl.Dataset
    report: dsl.HTML
    snapshot_uri: str


def create_export_table_snapshot_pipeline(trusted_host: str = "") -> graph_component.GraphComponent:
    """Build the one-task export pipeline.

    Args:
        trusted_host: Optional gateway restriction. By default the endpoint is supplied at run time.

    Returns:
        The KFP graph to compile for the deployment.
    """

    @dsl.pipeline(name="dch-export-table-snapshot", display_name="DCH Table Export")
    def pipeline(
        DCH_HOST: str,
        connection_id: str,
        source_table: str,
        output_format: str = "parquet",
    ) -> SnapshotOutputs:
        """Export a DCH table as CSV or Parquet and publish an HTML summary.

        Args:
            DCH_HOST: Hostname and optional port of the DCH gateway route exposed to the task pod.
            connection_id: ID of the saved SQL-capable relational DCH connection in the run namespace.
            source_table: Source table in schema.table_name format, such as analytics.customers.
            output_format: Snapshot format, either parquet (default) or csv.

        Returns:
            The Dataset, HTML report, and exact snapshot URI.
        """
        task = export_table_snapshot(
            dch_host=DCH_HOST,
            connection_id=connection_id,
            trusted_host=trusted_host,
            source_table=source_table,
            output_format=output_format,
        )
        task.set_display_name("export-table-snapshot").set_caching_options(False)
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
