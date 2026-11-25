"""Create an uploadable KFP pipeline for exporting a relational table through DCH."""

from typing import NamedTuple

from kfp import dsl
from kfp.dsl.graph_component import GraphComponent
from kfp_components.components.data_processing.dch.export_table_to_parquet import export_table_to_parquet


class SnapshotOutputs(NamedTuple):
    """Artifacts and the exact Parquet URI published by a successful run."""

    snapshot: dsl.Dataset
    report: dsl.HTML
    snapshot_uri: str


def create_export_table_to_parquet_pipeline(trusted_host: str = "") -> GraphComponent:
    """Build the one-task export pipeline.

    Args:
        trusted_host: Optional gateway restriction. By default the endpoint is supplied at run time.

    Returns:
        The KFP graph to compile for the deployment.
    """

    @dsl.pipeline(name="dch-export-table-to-parquet")
    def pipeline(
        DCH_HOST: str,
        connection_id: str,
        source_table: str,
    ) -> SnapshotOutputs:
        """Export a DCH table as Parquet and publish an HTML summary.

        Args:
            DCH_HOST: DCH gateway hostname and optional port for the run.
            connection_id: ID of the saved SQL-capable relational DCH connection in the run namespace.
            source_table: Source table in schema.table_name format, such as analytics.customers.

        Returns:
            The Parquet Dataset, HTML report, and exact Parquet URI.
        """
        task = export_table_to_parquet(
            dch_host=DCH_HOST,
            connection_id=connection_id,
            trusted_host=trusted_host,
            source_table=source_table,
        )
        task.set_caching_options(False)
        return SnapshotOutputs(
            snapshot=task.outputs["snapshot"],
            report=task.outputs["report"],
            snapshot_uri=task.outputs["snapshot_uri"],
        )

    return pipeline


def create_pipeline_for_validation() -> GraphComponent:
    """Build the default pipeline for CI validation."""
    return create_export_table_to_parquet_pipeline()
