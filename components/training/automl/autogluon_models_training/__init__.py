import os

if os.environ.get('_KFP_RUNTIME', 'false') != 'true':
    from .component import autogluon_models_training

    __all__ = ["autogluon_models_training"]
