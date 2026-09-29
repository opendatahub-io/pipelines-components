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

Both dimensions are applied to a single ``text_extraction`` task: the quality tier is
forwarded as ``preset`` and the hardware choice is turned into an accelerator count
(``1`` when GPU-accelerated, ``0`` otherwise) that drives the task's GPU request. This
avoids a CPU/GPU conditional branch, so the compiled DAG stays a single extraction
step with clean input/output names.
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


@dsl.component(base_image=AUTORAG_IMAGE, install_kfp_package=False)
def gpu_accelerator_count(gpu_acceleration: bool = False) -> int:
    """Translate the ``gpu_acceleration`` toggle into an NVIDIA GPU count.

    Returns ``1`` when GPU acceleration is requested and ``0`` otherwise. The value
    drives ``text_extraction``'s accelerator limit so a single task can run on either
    CPU or GPU without a conditional branch.
    """
    return 1 if gpu_acceleration else 0


def configurable_text_extraction(
    *,
    documents_descriptor: "dsl.pipeline_channel.PipelineArtifactChannel",
    normalized_preset: "dsl.pipeline_channel.PipelineParameterChannel",
    gpu_acceleration: "dsl.pipeline_channel.PipelineParameterChannel",
    configure: Callable[[dsl.PipelineTask], None],
    ocr_lang=None,
):
    """Build a single text-extraction task and return its extracted-text artifact.

    ``normalized_preset`` must be the output of :func:`normalize_extraction_preset`
    and selects the quality tier. ``gpu_acceleration`` is a boolean pipeline
    parameter/channel that selects the hardware: it is forwarded to the component (so
    the CUDA runtime is only used when requested) and converted into the task's
    NVIDIA GPU count via :func:`gpu_accelerator_count`. Quality and hardware stay
    independent, and there is no CPU/GPU conditional branch.

    ``configure`` is called with the extraction task so callers can attach resources,
    caching options, and object-storage secrets.

    ``ocr_lang`` (a pipeline parameter/channel or ``None``) is forwarded so the
    RapidOCR model bundle matches the corpus language.
    """
    accelerator_count_task = gpu_accelerator_count(gpu_acceleration=gpu_acceleration)
    accelerator_count_task.set_caching_options(False)

    extraction_task = text_extraction(
        documents_descriptor=documents_descriptor,
        preset=normalized_preset,
        gpu_acceleration=gpu_acceleration,
        ocr_lang=ocr_lang,
    )
    configure(extraction_task)
    # A single task requests 0 or 1 GPUs at runtime, so CPU runs compile to
    # ``nvidia.com/gpu: 0`` and GPU runs to ``nvidia.com/gpu: 1``.
    extraction_task.set_accelerator_type(GPU_RESOURCE).set_accelerator_limit(accelerator_count_task.output)
    # Consuming the count only through the accelerator resource field does not register
    # a DAG ordering edge, so make the dependency explicit: the count must be resolved
    # before the extraction pod is created.
    extraction_task.after(accelerator_count_task)

    return extraction_task.outputs["extracted_text"]
