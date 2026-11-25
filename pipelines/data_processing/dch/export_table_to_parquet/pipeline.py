"""Create an uploadable KFP pipeline for exporting a DCH PostgreSQL table."""

from typing import NamedTuple

from kfp import dsl
from kfp.dsl.graph_component import GraphComponent
from kfp_components.components.data_processing.dch.export_table_to_parquet import export_table_to_parquet


class SnapshotOutputs(NamedTuple):
    """Artifacts and the exact Parquet URI published by a successful run."""

    snapshot: dsl.Dataset
    report: dsl.HTML
    snapshot_uri: str


def create_export_table_to_parquet_pipeline(trusted_host: str) -> GraphComponent:
    """Build the one-task export pipeline for a deployment's trusted DCH gateway.

    Args:
        trusted_host: Gateway hostname and optional port fixed in the compiled pipeline.

    Returns:
        The KFP graph to compile for the deployment.
    """

    @dsl.pipeline(name="dch-export-table-to-parquet")
    def pipeline(
        DCH_HOST: str,
        connection_id: str,
        table_name: str,
        table_schema: str = "public",
        run_postgres_diagnostics: bool = False,
        max_snapshot_size_bytes: int = 1_073_741_824,
    ) -> SnapshotOutputs:
        """Check DCH access, then export the selected table and an HTML report.

        Args:
            DCH_HOST: DCH gateway hostname and optional port for the run.
            connection_id: ID of the saved PostgreSQL connection in the run namespace.
            table_name: Exact PostgreSQL table name.
            table_schema: Schema containing the table.
            run_postgres_diagnostics: Count PostgreSQL catalog rows over Flight SQL and a direct connection.
            max_snapshot_size_bytes: Maximum size in bytes of the staged Parquet file, including its footer.

        Returns:
            The Parquet Dataset, HTML report, and exact Parquet URI.
        """
        task = export_table_to_parquet(
            dch_host=DCH_HOST,
            connection_id=connection_id,
            trusted_host=trusted_host,
            table_name=table_name,
            table_schema=table_schema,
            run_postgres_diagnostics=run_postgres_diagnostics,
            max_snapshot_size_bytes=max_snapshot_size_bytes,
        )
        task.set_display_name("check-connection").set_caching_options(False)
        return SnapshotOutputs(
            snapshot=task.outputs["snapshot"],
            report=task.outputs["report"],
            snapshot_uri=task.outputs["snapshot_uri"],
        )

    return pipeline


def create_pipeline_for_validation() -> GraphComponent:
    """Build the pipeline for CI using a reserved, non-deployable gateway."""
    return create_export_table_to_parquet_pipeline(trusted_host="dch-gateway.invalid:443")
