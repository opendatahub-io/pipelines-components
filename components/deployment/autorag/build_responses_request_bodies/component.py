"""Create Llama Stack /v1/responses JSON bodies from RAG pattern artifacts."""

from kfp import dsl
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=AUTORAG_IMAGE,  # noqa: E501
)
def prepare_responses_api_requests(
    rag_patterns: dsl.InputPath(dsl.Artifact),
    responses_api_artifacts: dsl.Output[dsl.Artifact],
):
    """Emit one Llama Stack ``POST /v1/responses`` JSON body per RAG pattern directory.

    Expects the ``rag_patterns`` layout from ``rag_templates_optimization``: each subdirectory
    contains ``pattern.json``. For each pattern, writes ``v1_responses_body.json``,
    ``create_model_response.py``, and ``README.md`` (how to run the script) under a matching
    output subdirectory.
    The helper script embeds the Llama Stack base URL from environment variable
    ``LLAMA_STACK_CLIENT_BASE_URL`` at pipeline run time (default ``http://localhost:8321`` if unset);
    each per-pattern ``README.md`` documents how to run the script and override that URL if needed.
    The generated ``create_model_response.py`` resolves the API key from ``LLAMA_STACK_CLIENT_API_KEY``
    or ``LLAMA_STACK_API_KEY`` (or a one-time prompt), sets ``os.environ`` for the process when you
    type a key at the prompt, then loops on questions until an empty line. No secret file is written.
    For TLS, the script honors ``REQUESTS_CA_BUNDLE`` / ``SSL_CERT_FILE`` for custom CA bundles
    (e.g. corporate/private PKI), and ``LLAMA_STACK_TLS_INSECURE=1`` as a dev-only opt-out that
    disables certificate verification with a stderr warning.

    Request-body construction is defined inside this function so Kubeflow embeds it in
    ``ephemeral_component.py`` (module-level helpers in this file are not shipped to the executor).

    The generated body matches OpenAI-compatible ``POST /v1/responses`` (see OpenAI's
    `Migrate to the Responses API <https://developers.openai.com/api/docs/guides/migrate-to-responses>`__
    and Llama Stack's ``POST /v1/responses``): ``model``, ``input``, ``stream``, ``store``,
    ``metadata`` (string values only), optional ``instructions``, and optional ``tools`` /
    ``file_search`` when a collection name is set, plus ``tool_choice`` forcing file search
    and ``include: ["file_search_call.results"]`` to return file-search hits in the response.
    Replace
    ``vector_store_ids`` with Llama Stack--registered vector store identifiers if your deployment
    does not use the collection name as the store id.

    Args:
        rag_patterns: Local path to the ``rag_patterns`` directory (same layout as ``leaderboard_evaluation``).
        responses_api_artifacts: Output directory artifact; mirrors pattern folder names (JSON body,
            helper script, and README per pattern).
            Request bodies use ``RESPONSES_BODY_DEFAULT_QUESTION`` as the placeholder user message;
            the interactive script replaces it when you run it locally.
    """
    import json
    import os
    from pathlib import Path

    # Fixed placeholder question embedded in generated ``POST /v1/responses`` bodies (overridden at
    # runtime by ``create_model_response.py`` when the user enters a question).
    RESPONSES_BODY_DEFAULT_QUESTION = "What information is available in the indexed knowledge base?"

    output_filename = "v1_responses_body.json"
    script_filename = "create_model_response.py"
    readme_filename = "README.md"
    llama_stack_base_url = (
        (os.environ.get("LLAMA_STACK_CLIENT_BASE_URL") or "http://localhost:8321").strip().rstrip("/")
    )
    grounding_instruction = (
        "Use only information found in file_search results. "
        'If file_search results are insufficient, reply exactly: "I cannot answer based on the provided context."'
    )
    default_language_instruction = "Respond in the same language as the user question."
    default_instructions = f"{grounding_instruction} {default_language_instruction}"

    def _metadata_string_values(pattern: dict) -> dict[str, str]:
        """Build compact metadata dict with string values only (Llama Stack API constraint)."""
        settings = pattern.get("settings") or {}
        vs = settings.get("vector_store") or {}
        emb = settings.get("embedding") or {}
        return {
            "rag_pattern_name": str(pattern.get("name") or ""),
            "rag_pattern_iteration": str(pattern.get("iteration", "")),
            "vector_datasource_type": str(vs.get("datasource_type") or ""),
            "embedding_model_id": str(emb.get("model_id") or ""),
        }

    def _ranking_options_from_retrieval(retrieval: dict) -> dict | None:
        """Map hybrid retrieval settings to file_search ranking_options when possible."""
        search_mode = str(retrieval.get("search_mode") or "").strip().lower()
        if search_mode and search_mode != "hybrid":
            return None
        ranker = retrieval.get("ranker_strategy")
        if ranker is None and retrieval.get("ranker_alpha") is None and retrieval.get("ranker_k") is None:
            return None
        out: dict = {}
        if ranker is not None:
            out["ranker"] = str(ranker)
        if retrieval.get("ranker_alpha") is not None:
            out["alpha"] = float(retrieval["ranker_alpha"])
        if retrieval.get("ranker_k") is not None:
            out["impact_factor"] = float(retrieval["ranker_k"])
        return out or None

    def _tools_from_pattern(pattern: dict) -> list[dict]:
        """Build Responses API `tools` list; file_search uses vector_store_ids from the pattern collection."""
        settings = pattern.get("settings") or {}
        vs = settings.get("vector_store") or {}
        collection = (vs.get("collection_name") or "").strip()
        if not collection:
            return []
        retrieval = settings.get("retrieval") or {}
        try:
            max_results = int(retrieval.get("number_of_chunks") or 5)
        except (TypeError, ValueError):
            max_results = 5
        tool: dict = {
            "type": "file_search",
            "vector_store_ids": [collection],
            "max_num_results": max_results,
        }
        ro = _ranking_options_from_retrieval(retrieval) if isinstance(retrieval, dict) else None
        if ro:
            tool["ranking_options"] = ro
        return [tool]

    def _user_message_placeholder_text() -> str:
        """Placeholder user message in the JSON; interactive script overwrites at runtime."""
        return RESPONSES_BODY_DEFAULT_QUESTION

    def _language_instruction_from_user_message(user_message_text: object) -> str:
        """Extract language guidance from generation.user_message_text when present."""
        if not isinstance(user_message_text, str):
            return ""
        text = user_message_text.strip()
        if not text:
            return ""
        lowered = text.lower()
        if "language of the question" in lowered:
            return default_language_instruction
        if "same language" in lowered and "question" in lowered:
            return default_language_instruction
        return ""

    def _normalize_system_message_for_file_search(system_message_text: object) -> str:
        """Rewrite legacy context-section wording into file_search-grounded Responses wording."""
        if not isinstance(system_message_text, str):
            return ""
        text = system_message_text.strip()
        if not text:
            return ""
        replacements = {
            "based solely on the information i provide in the context section": (
                "using only information found in file_search results"
            ),
            "based on the context provided only": "using only information found in file_search results",
            "context section": "file_search results",
            "question section": "user question",
        }
        normalized = text
        lowered = normalized.lower()
        for source, target in replacements.items():
            idx = lowered.find(source)
            if idx >= 0:
                normalized = normalized[:idx] + target + normalized[idx + len(source) :]
                lowered = normalized.lower()
        return normalized.strip()

    def _compose_instructions(generation: dict) -> str:
        """Compose Responses API instructions from system/user generation fields."""
        system_text = _normalize_system_message_for_file_search(generation.get("system_message_text"))
        language_hint = _language_instruction_from_user_message(generation.get("user_message_text"))
        if not system_text:
            return default_instructions

        parts = [system_text]
        lowered = system_text.lower()
        if "file_search" not in lowered and "retriev" not in lowered:
            parts.append(grounding_instruction)
        if language_hint:
            if "same language as the user question" not in lowered and "language of the question" not in lowered:
                parts.append(language_hint)
        elif "same language as the user question" not in lowered and "language of the question" not in lowered:
            parts.append(default_language_instruction)
        return " ".join(p.strip() for p in parts if p and p.strip())

    def _build_llama_stack_v1_responses_body(pattern: dict) -> dict:
        """Build the JSON object sent as the HTTP body for Llama Stack ``POST /v1/responses``."""
        settings = pattern.get("settings") or {}
        generation = settings.get("generation") or {}
        model_id = str(generation.get("model_id") or "").strip()
        instructions = _compose_instructions(generation)

        tools = _tools_from_pattern(pattern)
        body: dict = {
            "model": model_id,
            "stream": False,
            "store": True,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": _user_message_placeholder_text(),
                        }
                    ],
                }
            ],
            "metadata": _metadata_string_values(pattern),
        }
        body["instructions"] = instructions.strip()
        if tools:
            body["tools"] = tools
            body["tool_choice"] = {"type": "file_search"}
            body["include"] = ["file_search_call.results"]
        return body

    def _request_script_contents(base_url: str) -> str:
        """Build a runnable script that loads local JSON and posts to ``/v1/responses``."""
        return f'''#!/usr/bin/env python3
"""Interactive helper: load v1_responses_body.json and call POST /v1/responses."""

import copy
import json
import os
from getpass import getpass
from pathlib import Path
from urllib import request
from urllib.error import HTTPError, URLError
import ssl
import sys


LLAMA_STACK_BASE_URL = "{base_url.rstrip("/")}"
BODY_PATH = Path(__file__).with_name("{output_filename}")


def _set_question(body: dict, question: str) -> None:
    """Replace first user input_text payload with an interactive question."""
    try:
        body["input"][0]["content"][0]["text"] = question
    except (KeyError, IndexError, TypeError):
        pass


def _api_key_from_env() -> str:
    return (
        os.environ.get("LLAMA_STACK_CLIENT_API_KEY", "").strip()
        or os.environ.get("LLAMA_STACK_API_KEY", "").strip()
    )


def _apply_api_key_to_env(key: str) -> None:
    """Store key only in this process environment (for the rest of this run)."""
    if not key:
        return
    os.environ["LLAMA_STACK_CLIENT_API_KEY"] = key
    os.environ["LLAMA_STACK_API_KEY"] = key


def _resolve_api_key_once() -> str:
    """Use ``LLAMA_STACK_CLIENT_API_KEY`` / ``LLAMA_STACK_API_KEY``, else one interactive prompt."""
    key = _api_key_from_env()
    if key:
        return key
    key = getpass("Llama Stack API key (press Enter if not required): ").strip()
    if key:
        _apply_api_key_to_env(key)
    return key


def _build_ssl_context():
    """Build an SSLContext honoring corporate CA bundles or a dev-only insecure flag.

    Precedence:
      1. LLAMA_STACK_TLS_INSECURE=1  -> verification disabled (dev only, prints warning)
      2. REQUESTS_CA_BUNDLE          -> custom CA bundle path
      3. SSL_CERT_FILE               -> custom CA bundle path
      4. None                        -> stdlib default (system trust store)
    """
    insecure = os.environ.get("LLAMA_STACK_TLS_INSECURE", "").strip().lower() in ("1", "true", "yes", "on")
    if insecure:
        print(
            "WARNING: TLS verification is DISABLED (LLAMA_STACK_TLS_INSECURE=1). "
            "Do not use this against production Llama Stack deployments.",
            file=sys.stderr,
        )
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    cafile = (
        os.environ.get("REQUESTS_CA_BUNDLE", "").strip()
        or os.environ.get("SSL_CERT_FILE", "").strip()
    )
    if cafile:
        return ssl.create_default_context(cafile=cafile)
    return None


def _post_responses(body: dict, api_key: str, ssl_context) -> int:
    url = f"{{LLAMA_STACK_BASE_URL}}/v1/responses"
    headers = {{
        "Content-Type": "application/json",
        "Accept": "application/json",
    }}
    if api_key:
        headers["Authorization"] = f"Bearer {{api_key}}"

    req = request.Request(
        url=url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=120, context=ssl_context) as resp:
            payload = resp.read().decode("utf-8")
            print(payload)
            return 0
    except HTTPError as e:
        print(f"HTTP {{e.code}}: {{e.reason}}")
        detail = e.read().decode("utf-8", errors="replace")
        if detail:
            print(detail)
        return 2
    except URLError as e:
        print(f"Connection error: {{e}}")
        return 3


def main() -> int:
    if not BODY_PATH.is_file():
        print(f"Missing JSON body file: {{BODY_PATH}}")
        return 1

    api_key = _resolve_api_key_once()
    ssl_context = _build_ssl_context()

    with BODY_PATH.open(encoding="utf-8") as f:
        template = json.load(f)

    print("Enter questions; an empty line exits.")
    while True:
        question = input("Question (empty line to exit): ").strip()
        if not question:
            return 0
        body = copy.deepcopy(template)
        _set_question(body, question)
        rc = _post_responses(body, api_key, ssl_context)
        if rc != 0:
            return rc


if __name__ == "__main__":
    raise SystemExit(main())
'''

    def _pattern_readme_contents(base_url: str) -> str:
        """Short README next to each pattern's JSON and script."""
        url = base_url.rstrip("/")
        ofn = output_filename
        sfn = script_filename
        return (
            f"# Call Llama Stack ``POST /v1/responses`` for this RAG pattern\n\n"
            f"This folder contains:\n\n"
            f"- ``{ofn}`` — request body for the OpenAI-compatible Responses API "
            f"(Llama Stack ``/v1/responses``).\n"
            f"- ``{sfn}`` — interactive helper that loads the JSON, resolves the API key from "
            f"environment variables or one prompt (then sets ``os.environ`` for this run only), "
            f"then loops: ask a question, POST, repeat until you enter an empty question.\n\n"
            "## Prerequisites\n\n"
            "- Python 3 available on your PATH as ``python3`` "
            "(or run with your Python interpreter).\n"
            "- Network reachability to Llama Stack.\n"
            f"- Keep ``{ofn}`` in the **same directory** as ``{sfn}`` "
            "(the script loads the JSON from alongside itself).\n\n"
            "## Server URL\n\n"
            "The script was generated with:\n\n"
            "```text\n"
            f'LLAMA_STACK_BASE_URL = "{url}"\n'
            "```\n\n"
            "That value came from the pipeline run (environment variable "
            "``LLAMA_STACK_CLIENT_BASE_URL``, or ``http://localhost:8321`` if unset). "
            f"To use a different URL, edit ``LLAMA_STACK_BASE_URL`` inside ``{sfn}``, or re-run "
            "the pipeline step with ``LLAMA_STACK_CLIENT_BASE_URL`` set as needed.\n\n"
            "## Run\n\n"
            "From this directory:\n\n"
            "```bash\n"
            f"python3 {sfn}\n"
            "```\n\n"
            "Optional:\n\n"
            "```bash\n"
            f"chmod +x {sfn}\n"
            f"./{sfn}\n"
            "```\n\n"
            "Flow:\n\n"
            "1. **API key** — use ``LLAMA_STACK_CLIENT_API_KEY`` or ``LLAMA_STACK_API_KEY`` "
            "(recommended: export in your shell before running). If unset, you are prompted "
            "**once**; a typed key is applied only to this process (``os.environ`` for the rest "
            "of this run). Press Enter if your server needs no bearer token.\n"
            "2. **Questions** — repeatedly enter a question; each request uses a fresh copy of "
            "the JSON body. An **empty line** exits.\n\n"
            "Each response body is printed to stdout (JSON).\n\n"
            "To reuse the same key in a **new** shell session, export "
            "``LLAMA_STACK_CLIENT_API_KEY`` (or ``LLAMA_STACK_API_KEY``) again; nothing is "
            "written to disk by this script.\n\n"
            "## TLS / certificates\n\n"
            "The script uses Python's standard library (``urllib`` + ``ssl``). It honors these "
            "environment variables, in order of precedence:\n\n"
            "1. **``LLAMA_STACK_TLS_INSECURE=1``** — **dev only.** Disables TLS verification "
            "entirely (``check_hostname=False``, ``verify_mode=CERT_NONE``) and prints a warning "
            "to stderr. **Never** use this against a production Llama Stack deployment; it "
            "exposes you to man-in-the-middle attacks.\n"
            "2. **``REQUESTS_CA_BUNDLE=/path/to/ca-bundle.pem``** — path to a PEM file containing "
            "the CA(s) that signed your Llama Stack server certificate. Use this when your "
            "deployment is fronted by a private/corporate CA that is not in the OS default trust "
            "store.\n"
            "3. **``SSL_CERT_FILE=/path/to/ca-bundle.pem``** — same effect as "
            "``REQUESTS_CA_BUNDLE``; honored for compatibility with the Python stdlib "
            "convention.\n"
            "4. If none of the above are set, the script uses the system trust store.\n\n"
            "Example (private/corporate CA):\n\n"
            "```bash\n"
            "export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/corporate-ca.pem\n"
            f"python3 {sfn}\n"
            "```\n\n"
            "Example (dev cluster with self-signed cert, **non-production**):\n\n"
            "```bash\n"
            "export LLAMA_STACK_TLS_INSECURE=1\n"
            f"python3 {sfn}\n"
            "```\n\n"
            "## Troubleshooting\n\n"
            f"- **Missing JSON body file** — restore ``{ofn}`` next to the script or "
            "re-download the artifact folder.\n"
            "- **TLS / certificate errors** (e.g. ``CERTIFICATE_VERIFY_FAILED``) — your Llama "
            "Stack server certificate is signed by a CA not in the OS trust store. See the "
            "**TLS / certificates** section above for ``REQUESTS_CA_BUNDLE`` / ``SSL_CERT_FILE`` "
            "and the dev-only ``LLAMA_STACK_TLS_INSECURE`` opt-out.\n"
            "- **Connection or HTTP errors** — confirm the base URL, firewall, and that "
            "the model and ``file_search`` / vector store identifiers in the JSON match your "
            "deployment.\n"
        )

    root = Path(rag_patterns)
    out_root = Path(responses_api_artifacts.path)
    out_root.mkdir(parents=True, exist_ok=True)

    written: list[dict] = []
    if not root.is_dir():
        raise FileNotFoundError(f"rag_patterns path is not a directory: {root}")

    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        pattern_path = sub / "pattern.json"
        if not pattern_path.is_file():
            continue
        with pattern_path.open(encoding="utf-8") as f:
            pattern = json.load(f)
        body = _build_llama_stack_v1_responses_body(pattern)
        dest_dir = out_root / sub.name
        dest_dir.mkdir(parents=True, exist_ok=True)
        out_file = dest_dir / output_filename
        with out_file.open("w", encoding="utf-8") as out_f:
            json.dump(body, out_f, indent=2, ensure_ascii=False)
        script_file = dest_dir / script_filename
        script_file.write_text(_request_script_contents(llama_stack_base_url), encoding="utf-8")
        readme_file = dest_dir / readme_filename
        readme_file.write_text(_pattern_readme_contents(llama_stack_base_url), encoding="utf-8")
        written.append(
            {
                "pattern_dir": sub.name,
                "body_path": str(out_file),
                "script_path": str(script_file),
                "readme_path": str(readme_file),
            }
        )

    responses_api_artifacts.metadata["name"] = "responses_api_artifacts"
    responses_api_artifacts.metadata["metadata"] = {"patterns": written}


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        prepare_responses_api_requests,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
