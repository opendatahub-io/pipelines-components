"""Shared AutoRAG text-extraction preset helpers.

Quality and hardware are two orthogonal knobs shared by the optimization pipeline
and the indexing pipeline:

* ``preset`` -- the quality tier of extraction:
    * ``speed`` -- no Docling table parsing (fastest).
    * ``balanced`` -- TableFormer table reconstruction.
* ``gpu_acceleration`` -- whether Docling extraction runs on an NVIDIA GPU. The
  quality tier is unchanged; the GPU only speeds up extraction. Any preset can run
  with or without GPU acceleration.

Keeping the two dimensions separate lets callers request, for example, speed-quality
extraction on a GPU, and keeps the generated indexing configuration transparent.
Quality-only components (chunking, model pre-selection, optimization) only ever see
``preset`` and are unaffected by ``gpu_acceleration``.
"""

from typing import Callable, Optional

from kfp import dsl
from kfp_components.components.data_processing.autorag.text_extraction.component import (
    text_extraction,
)
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]

GPU_RESOURCE = "nvidia.com/gpu"
DEFAULT_EXTRACTION_PRESET = "speed"
VALID_EXTRACTION_PRESETS = ("speed", "balanced")


@dsl.component(base_image=AUTORAG_IMAGE, install_kfp_package=False)
def normalize_extraction_preset(preset: Optional[str] = None) -> str:
    """Validate the extraction quality preset and return its canonical value.

    Omitted, ``null``, or empty values fall back to ``speed`` so existing runs
    stay at the speed quality tier.
    """
    if preset is None or not preset.strip():
        return "speed"
    valid = ("speed", "balanced")
    if preset not in valid:
        raise ValueError(f"preset must be one of {valid}; got {preset!r}.")
    return preset


def gpu_aware_text_extraction(
    *,
    documents_descriptor: "dsl.pipeline_channel.PipelineArtifactChannel",
    normalized_preset: "dsl.pipeline_channel.PipelineParameterChannel",
    gpu_acceleration: "dsl.pipeline_channel.PipelineParameterChannel",
    configure: Callable[[dsl.PipelineTask], None],
    ocr_lang=None,
):
    """Build the CPU/GPU text-extraction branch and return its extracted-text artifact.

    ``normalized_preset`` must be the output of :func:`normalize_extraction_preset`
    and selects the quality tier. ``gpu_acceleration`` is a boolean pipeline
    parameter/channel that selects the hardware: only the GPU branch requests an
    NVIDIA GPU and runs extraction with ``gpu_acceleration=True``. Both branches use
    the same ``preset``, so quality and hardware stay independent.

    ``configure`` is called with each branch task so callers can attach resources,
    caching options, and object-storage secrets.

    ``ocr_lang`` (a pipeline parameter/channel or ``None``) is forwarded to both
    branches so the RapidOCR model bundle matches the corpus language regardless of
    whether extraction runs on CPU or GPU.
    """
    with dsl.If(gpu_acceleration == True):  # noqa: E712 -- KFP channels require ``==`` equality.
        gpu_task = text_extraction(
            documents_descriptor=documents_descriptor,
            preset=normalized_preset,
            gpu_acceleration=True,
            ocr_lang=ocr_lang,
        )
        gpu_task.set_display_name("text-extraction-gpu")
        configure(gpu_task)
        gpu_task.set_accelerator_type(GPU_RESOURCE).set_accelerator_limit(1)

    with dsl.Else():
        cpu_task = text_extraction(
            documents_descriptor=documents_descriptor,
            preset=normalized_preset,
            gpu_acceleration=False,
            ocr_lang=ocr_lang,
        )
        cpu_task.set_display_name("text-extraction-cpu")
        configure(cpu_task)

    return dsl.OneOf(
        gpu_task.outputs["extracted_text"],
        cpu_task.outputs["extracted_text"],
    )
