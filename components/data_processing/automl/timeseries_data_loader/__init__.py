"""Export the component for authoring without loading its decorator in the executor."""

import os

if os.environ.get("_KFP_RUNTIME", "false") != "true":
    from .component import timeseries_data_loader

    __all__ = ["timeseries_data_loader"]
