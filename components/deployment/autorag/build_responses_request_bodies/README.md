# Prepare Responses API Requests ✨

> ⚠️ **Stability: experimental** — This asset is not yet stable and may change.

## Overview 🧾

Emit one Llama Stack ``POST /v1/responses`` JSON body per RAG pattern directory.

Expects the ``rag_patterns`` layout from ``rag_templates_optimization``: each subdirectory contains ``pattern.json``. For each pattern, writes ``v1_responses_body.json``, ``create_model_response.py``, and ``README.md`` (how to run the script) under a matching output subdirectory. The helper script
embeds the Llama Stack base URL from environment variable ``LLAMA_STACK_CLIENT_BASE_URL`` at pipeline run time (default ``http://localhost:8321`` if unset); each per-pattern ``README.md`` documents how to run the script and override that URL if needed. The generated ``create_model_response.py``
resolves the API key from ``LLAMA_STACK_CLIENT_API_KEY`` or ``LLAMA_STACK_API_KEY`` (or a one-time prompt), sets ``os.environ`` for the process when you type a key at the prompt, then loops on questions until an empty line. No secret file is written. For TLS, the script honors ``REQUESTS_CA_BUNDLE``
/ ``SSL_CERT_FILE`` for custom CA bundles (e.g. corporate/private PKI), and ``LLAMA_STACK_TLS_INSECURE=1`` as a dev-only opt-out that disables certificate verification with a stderr warning.

Request-body construction is defined inside this function so Kubeflow embeds it in ``ephemeral_component.py`` (module-level helpers in this file are not shipped to the executor).

The generated body matches OpenAI-compatible ``POST /v1/responses`` (see OpenAI's `Migrate to the Responses API <https://developers.openai.com/api/docs/guides/migrate-to-responses>`__ and Llama Stack's ``POST /v1/responses``): ``model``, ``input``, ``stream``, ``store``, ``metadata`` (string values
only), optional ``instructions``, and optional ``tools`` / ``file_search`` when a collection name is set, plus ``tool_choice`` forcing file search and ``include: ["file_search_call.results"]`` to return file-search hits in the response. Replace ``vector_store_ids`` with Llama Stack--registered
vector store identifiers if your deployment does not use the collection name as the store id.

## Inputs 📥

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `rag_patterns` | `dsl.InputPath(dsl.Artifact)` | `None` | Local path to the ``rag_patterns`` directory (same layout as ``leaderboard_evaluation``). |
| `responses_api_artifacts` | `dsl.Output[dsl.Artifact]` | `None` | Output directory artifact; mirrors pattern folder names (JSON body, helper script, and README per pattern). Request bodies use ``RESPONSES_BODY_DEFAULT_QUESTION`` as the placeholder user message; the interactive script replaces it when you run it locally. |

## Usage Examples 🧪

```python
"""Example pipelines demonstrating usage of prepare_responses_api_requests."""

from kfp import dsl
from kfp_components.components.deployment.autorag.build_responses_request_bodies import (
    prepare_responses_api_requests,
)


@dsl.pipeline(name="build-responses-request-bodies-example")
def example_pipeline():
    """Example pipeline using prepare_responses_api_requests."""
    rag_patterns = dsl.importer(
        artifact_uri="gs://placeholder/rag_patterns",
        artifact_class=dsl.Artifact,
    )
    prepare_responses_api_requests(rag_patterns=rag_patterns.output)

```

## Metadata 🗂️

- **Name**: prepare_responses_api_requests
- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: Llama Stack, Version: >=0.1.0
- **Tags**:
  - deployment
  - autorag
  - llama-stack
  - responses-api
- **Last Verified**: 2026-03-30 00:00:00+00:00
- **Owners**:
  - Approvers:
    - LukaszCmielowski
  - Reviewers:
    - filip-komarzyniec
    - witold-nowogorski

## Additional Resources 📚

- **Documentation**: [https://llamastack.github.io/docs/api/create-openai-response-v-1-responses-post](https://llamastack.github.io/docs/api/create-openai-response-v-1-responses-post)
