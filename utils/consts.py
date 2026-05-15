import os

DEFAULT_AUTOML_IMAGE = "quay.io/opendatahub/odh-automl:odh-stable"
DEFAULT_AUTORAG_IMAGE = "quay.io/opendatahub/odh-autorag:odh-stable"

AUTOML_IMAGE = os.getenv("RELATED_IMAGE_ODH_AUTOML_IMAGE", DEFAULT_AUTOML_IMAGE)
AUTORAG_IMAGE = os.getenv("RELATED_IMAGE_ODH_AUTORAG_IMAGE", DEFAULT_AUTORAG_IMAGE)
