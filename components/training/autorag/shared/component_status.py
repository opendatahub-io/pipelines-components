"""Re-export for package imports.

Implementation lives in ``runtime_embed/`` so that directory can be used as the
single KFP ``embedded_artifact_path`` without duplicating sources. Components that
embed only this helper should set
``embedded_artifact_path`` to ``runtime_embed/component_status.py``.
"""

from .runtime_embed.component_status import *  # noqa: F403
