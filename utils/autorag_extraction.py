"""Shared AutoRAG text-extraction inputs.

Extraction has two independent knobs used by both the optimization and indexing
pipelines: ``preset`` (quality tier -- ``speed`` skips Docling table parsing,
``balanced`` runs TableFormer) and ``gpu_acceleration`` (run Docling on an NVIDIA
GPU). Both feed a single ``text_extraction`` task, so the compiled DAG stays one
extraction step with no CPU/GPU branch.
"""

from typing import NamedTuple, Optional

from kfp import dsl
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]

GPU_RESOURCE = "nvidia.com/gpu"


@dsl.component(base_image=AUTORAG_IMAGE, install_kfp_package=False)
def prepare_extraction_inputs(
    preset: Optional[str] = None,
    gpu_acceleration: bool = False,
) -> NamedTuple("ExtractionInputs", [("preset", str), ("gpu_count", int)]):  # type: ignore[valid-type]
    """Resolve the runtime inputs for the text-extraction task.

    A blank or ``null`` preset falls back to ``speed``; any other unsupported value
    is rejected. ``gpu_count`` is ``1`` when GPU acceleration is requested and ``0``
    otherwise: ``set_accelerator_limit`` needs an integer channel, and KFP cannot
    derive one from the boolean at compile time.
    """
    from typing import NamedTuple

    resolved_preset = preset.strip() if preset and preset.strip() else "speed"
    valid = ("speed", "balanced")
    if resolved_preset not in valid:
        raise ValueError(f"preset must be one of {valid}; got {resolved_preset!r}.")

    extraction_inputs = NamedTuple("ExtractionInputs", [("preset", str), ("gpu_count", int)])
    return extraction_inputs(resolved_preset, 1 if gpu_acceleration else 0)
