# Models Pre Selector ✨

> ⚠️ **Stability: alpha** — This asset is not yet stable and may change.

## Overview 🧾

Model pre-selection for AutoRAG experiments.

Trims the candidate model set when the search space carries more models than the optimizer should explore. This component **always runs** and decides internally: when the report holds more than ``DEFAULT_N_FOUNDATION_MODELS`` foundation models or more than ``DEFAULT_N_EMBEDDING_MODELS`` embedding
models, it runs ``ai4rag.core.experiment.mps.ModelsPreSelector`` over a benchmark sample and writes a *limited* report; otherwise it passes the full report through unchanged (a cheap copy — no MaaS calls, no document loading).

Running as a distinct step keeps the (potentially expensive) pre-selection evaluation observable in the pipeline and produces the reduced search-space report consumed by ``rag_templates_optimization``.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `search_space_report` | `dsl.Input[dsl.Artifact]` | `None` | Input artifact with the full search space report produced by ``search_space_preparation``. |
| `extracted_text` | `dsl.Input[dsl.Artifact]` | `None` | Input artifact with extracted text documents. Only read when pre-selection actually runs. |
| `test_data` | `dsl.Input[dsl.Artifact]` | `None` | Input artifact with benchmark questions and expected answers. Only read when pre-selection actually runs. |
| `search_space_mps_report` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact for the (possibly reduced) JSON search space report. |
| `embedded_artifact` | `dsl.EmbeddedInput[dsl.Dataset]` | `None` | Embedded ``autorag.shared`` helpers injected by KFP at runtime. |
| `component_status` | `dsl.Output[dsl.Artifact]` | `None` | Output artifact containing stage-level progress tracking. |
| `preset` | `str` | `speed` | Pipeline quality tier. "speed" (default) uses 10 benchmark query threads during pre-selection; "balanced" uses 4 (reduced due to larger per-request context). |

## Usage Examples 🧪

```python
"""Example pipelines demonstrating usage of models_pre_selector."""

from kfp import dsl
from kfp_components.components.training.autorag.models_pre_selector import models_pre_selector


@dsl.pipeline(name="models-pre-selector-example")
def example_pipeline():
    """Example pipeline using models_pre_selector."""
    search_space_report = dsl.importer(
        artifact_uri="gs://placeholder/search_space_report",
        artifact_class=dsl.Artifact,
    )
    extracted_text = dsl.importer(
        artifact_uri="gs://placeholder/extracted_text",
        artifact_class=dsl.Artifact,
    )
    test_data = dsl.importer(
        artifact_uri="gs://placeholder/test_data",
        artifact_class=dsl.Artifact,
    )
    models_pre_selector(
        search_space_report=search_space_report.output,
        extracted_text=extracted_text.output,
        test_data=test_data.output,
    )

```

## Metadata 🗂️

- **Name**: models_pre_selector
- **Stability**: alpha
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: ai4rag, Version: ~=0.12.0
    - Name: MaaS, Version: >=1.0.0
    - Name: pandas, Version: >=2.0.0
- **Tags**:
  - training
  - autorag
  - model-selection
  - optimization
- **Last Verified**: 2026-08-24 00:00:00+00:00
- **Owners**:
  - No Parent Owners: Yes
  - Approvers:
    - LukaszCmielowski
    - DorotaDR
  - Reviewers:
    - filip-komarzyniec
    - jakub-walaszczyk
    - MichalSteczko
