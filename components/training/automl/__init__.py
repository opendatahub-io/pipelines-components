import os

if os.environ.get("_KFP_RUNTIME", "false") != "true":
    from . import (
        autogluon_leaderboard_evaluation,
        autogluon_models_training,
        autogluon_timeseries_models_training,
    )

    __all__ = [
        "autogluon_leaderboard_evaluation",
        "autogluon_models_training",
        "autogluon_timeseries_models_training",
    ]
