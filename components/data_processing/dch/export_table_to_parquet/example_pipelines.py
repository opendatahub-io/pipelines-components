"""Train an AutoGluon tabular model from a published DCH snapshot."""

from kfp import dsl


@dsl.component(
    base_image="python:3.12-bookworm",
    packages_to_install=["autogluon.tabular==1.5.0"],
)
def train_tabular_snapshot(
    snapshot: dsl.Input[dsl.Dataset],
    model: dsl.Output[dsl.Model],
    label_column: str,
    time_limit_seconds: int,
):
    """Train and publish an AutoGluon tabular predictor."""
    import pandas as pd
    from autogluon.tabular import TabularPredictor

    data = pd.read_parquet(snapshot.path)
    if data.empty or label_column not in data.columns:
        raise ValueError("The snapshot must contain training rows and the requested label column.")
    TabularPredictor(label=label_column, path=model.path, verbosity=1).fit(
        data,
        time_limit=time_limit_seconds,
        hyperparameters={"RF": {}, "XT": {}},
    )


@dsl.pipeline(name="train-automl-from-dch-snapshot")
def train_automl_from_dch_snapshot(
    snapshot_uri: str,
    label_column: str,
    time_limit_seconds: int = 120,
) -> dsl.Model:
    """Import the snapshot from the DSPA artifact store and train a tabular model."""
    dataset = dsl.importer(artifact_uri=snapshot_uri, artifact_class=dsl.Dataset)
    training = train_tabular_snapshot(
        snapshot=dataset.output,
        label_column=label_column,
        time_limit_seconds=time_limit_seconds,
    )
    return training.outputs["model"]
