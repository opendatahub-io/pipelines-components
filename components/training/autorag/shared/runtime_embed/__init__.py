"""KFP-embedable AutoRAG helpers (status, MLflow, and database connections).

Database-consuming components embed this directory as a single
``embedded_artifact_path``. Keep modules here self-contained: the odh-autorag
image does not install ``kfp_components``.
"""
