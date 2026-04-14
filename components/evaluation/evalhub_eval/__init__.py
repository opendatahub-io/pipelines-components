"""EvalHub Evaluation Component.

Serves a model on KServe, evaluates via EvalHub, then removes created
serving resources.
"""

from .component import evalhub_evaluate

__all__ = ["evalhub_evaluate"]
