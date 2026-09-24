"""Shared AutoRAG text-extraction preset helpers.

Both the optimization pipeline and the indexing pipeline expose a single
``preset`` parameter with the same vocabulary:

* ``speed`` -- CPU extraction, no Docling table parsing (fastest).
* ``balanced`` -- CPU extraction with TableFormer table reconstruction.
* ``gpu_accelerated`` -- the ``balanced`` quality tier run on an NVIDIA GPU.

The hardware dimension (CPU vs GPU) is folded into ``gpu_accelerated`` so the two
pipelines share one knob. Quality-only components (chunking, model pre-selection,
optimization) treat ``gpu_accelerated`` as ``balanced``.
"""

from typing import Callable, Optional

from kfp import dsl
from kfp_components.components.data_processing.autorag.text_extraction.component import (
    text_extraction,
)
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]

GPU_RESOURCE = "nvidia.com/gpu"
GPU_PRESET = "gpu_accelerated"
DEFAULT_EXTRACTION_PRESET = "speed"
VALID_EXTRACTION_PRESETS = ("speed", "balanced", "gpu_accelerated")


@dsl.component(base_image=AUTORAG_IMAGE, install_kfp_package=False)
def normalize_extraction_preset(preset: Optional[str] = None) -> str:
    """Validate the unified extraction preset and return its canonical value.

    Omitted, ``null``, or empty values fall back to ``speed`` so existing runs
    stay CPU-based at the speed quality tier.
    """
    if preset is None or not preset.strip():
        return "speed"
    valid = ("speed", "balanced", "gpu_accelerated")
    if preset not in valid:
        raise ValueError(f"preset must be one of {valid}; got {preset!r}.")
    return preset


def gpu_aware_text_extraction(
    *,
    documents_descriptor: "dsl.pipeline_channel.PipelineArtifactChannel",
    normalized_preset: "dsl.pipeline_channel.PipelineParameterChannel",
    configure: Callable[[dsl.PipelineTask], None],
    ocr_lang=None,
):
    """Build the CPU/GPU text-extraction branch and return its extracted-text artifact.

    ``normalized_preset`` must be the output of :func:`normalize_extraction_preset`.
    ``configure`` is called with each branch task so callers can attach resources,
    caching options, and object-storage secrets. Only the ``gpu_accelerated``
    branch requests an NVIDIA GPU; the CPU branch forwards the normalized
    ``speed``/``balanced`` preset to the extraction quality tier.

    ``ocr_lang`` (a pipeline parameter/channel or ``None``) is forwarded to both
    branches so the RapidOCR model bundle matches the corpus language regardless of
    whether extraction runs on CPU or GPU.
    """
    with dsl.If(normalized_preset == GPU_PRESET):
        gpu_task = text_extraction(
            documents_descriptor=documents_descriptor,
            preset=GPU_PRESET,
            ocr_lang=ocr_lang,
        )
        configure(gpu_task)
        gpu_task.set_accelerator_type(GPU_RESOURCE).set_accelerator_limit(1)

    with dsl.Else():
        cpu_task = text_extraction(
            documents_descriptor=documents_descriptor,
            preset=normalized_preset,
            ocr_lang=ocr_lang,
        )
        configure(cpu_task)

    return dsl.OneOf(
        gpu_task.outputs["extracted_text"],
        cpu_task.outputs["extracted_text"],
    )
