import os

DEFAULT_AUTOML_IMAGE = "quay.io/opendatahub/odh-automl:odh-stable"
DEFAULT_AUTORAG_IMAGE = "quay.io/opendatahub/odh-autorag:odh-stable"

AUTOML_IMAGE = os.getenv("RELATED_IMAGE_ODH_AUTOML_IMAGE", DEFAULT_AUTOML_IMAGE)
AUTORAG_IMAGE = os.getenv("RELATED_IMAGE_ODH_AUTORAG_IMAGE", DEFAULT_AUTORAG_IMAGE)

DEFAULT_RAY_RAG_BASE_IMAGE = (
    "registry.redhat.io/rhoai/odh-pipeline-runtime-datascience-cpu-py312-rhel9"
    "@sha256:f9844dc150592a9f196283b3645dda92bd80dfdb3d467fa8725b10267ea5bdbc"
)
RAY_RAG_BASE_IMAGE = os.getenv("RELATED_IMAGE_RAG_BASE_RUNTIME", DEFAULT_RAY_RAG_BASE_IMAGE)
