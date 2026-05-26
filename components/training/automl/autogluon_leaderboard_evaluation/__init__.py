import os

if os.environ.get('_KFP_RUNTIME', 'false') != 'true':
    from .component import leaderboard_evaluation

    __all__ = ["leaderboard_evaluation"]
