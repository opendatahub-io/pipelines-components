"""Unit tests for prepare_responses_api_requests."""

import json
import os
from unittest import mock

from ..component import prepare_responses_api_requests

OUTPUT_FILENAME = "v1_responses_body.json"
SCRIPT_FILENAME = "create_model_response.py"
README_FILENAME = "README.md"
RESPONSES_BODY_DEFAULT_QUESTION = "What information is available in the indexed knowledge base?"


def _minimal_pattern(collection: str = "my_collection") -> dict:
    return {
        "name": "pattern_a",
        "iteration": 0,
        "settings": {
            "vector_store": {"datasource_type": "ls_milvus", "collection_name": collection},
            "chunking": {"method": "recursive", "chunk_size": 512, "chunk_overlap": 64},
            "embedding": {"model_id": "emb-1", "distance_metric": "cosine"},
            "retrieval": {
                "method": "simple",
                "number_of_chunks": 3,
                "search_mode": "hybrid",
                "ranker_strategy": "weighted",
                "ranker_alpha": 0.7,
                "ranker_k": 50,
            },
            "generation": {
                "model_id": "gen-1",
                "system_message_text": "Answer from context only.",
                "user_message_text": (
                    "Q: {question}\nCtx: {reference_documents}\nRespond exclusively in the language of the question."
                ),
            },
        },
    }


def _pattern_with_search_mode(search_mode: str) -> dict:
    """Return minimal pattern with selectable retrieval search mode."""
    p = _minimal_pattern()
    p["settings"]["retrieval"]["search_mode"] = search_mode
    return p


def _run_python_func(tmp_path, rag_subdirs: list[tuple[str, dict]]):
    """Create rag layout, run component python_func, return (out_path, out_art)."""
    rag = tmp_path / "in"
    for sub_name, pattern in rag_subdirs:
        (rag / sub_name).mkdir(parents=True)
        with (rag / sub_name / "pattern.json").open("w", encoding="utf-8") as f:
            json.dump(pattern, f)
    out = tmp_path / "out"
    out.mkdir()
    out_art = mock.Mock()
    out_art.path = str(out)
    out_art.metadata = {}
    with mock.patch.dict(os.environ, {"LLAMA_STACK_CLIENT_BASE_URL": "https://llama-stack.example.com"}, clear=False):
        prepare_responses_api_requests.python_func(
            rag_patterns=str(rag),
            responses_api_artifacts=out_art,
        )
    return out, out_art


class TestBuildLlamaStackV1ResponsesBody:
    """Assertions on JSON bodies produced via ``python_func`` (logic lives inside the component)."""

    def test_maps_model_instructions_input_and_file_search(self, tmp_path):
        """Body includes model, user message, file_search tool, and ranking options."""
        out, _ = _run_python_func(tmp_path, [("p1", _minimal_pattern())])
        body = json.loads((out / "p1" / OUTPUT_FILENAME).read_text(encoding="utf-8"))
        assert body["model"] == "gen-1"
        assert body["stream"] is False
        assert body["store"] is True
        assert body["instructions"] == (
            "Answer from context only. "
            "Use only information found in file_search results. "
            'If file_search results are insufficient, reply exactly: "I cannot answer based on the provided context." '
            "Respond in the same language as the user question."
        )
        assert len(body["input"]) == 1
        msg = body["input"][0]
        assert msg["role"] == "user"
        assert msg["type"] == "message"
        assert msg["content"][0]["text"] == RESPONSES_BODY_DEFAULT_QUESTION
        assert len(body["tools"]) == 1
        assert body["tool_choice"] == {"type": "file_search"}
        assert body["include"] == ["file_search_call.results"]
        assert body["tools"][0]["type"] == "file_search"
        assert body["tools"][0]["vector_store_ids"] == ["my_collection"]
        assert body["tools"][0]["max_num_results"] == 3
        assert body["tools"][0]["ranking_options"]["ranker"] == "weighted"
        assert body["tools"][0]["ranking_options"]["alpha"] == 0.7
        assert body["metadata"] == {
            "rag_pattern_name": "pattern_a",
            "rag_pattern_iteration": "0",
            "vector_datasource_type": "ls_milvus",
            "embedding_model_id": "emb-1",
        }
        script = (out / "p1" / SCRIPT_FILENAME).read_text(encoding="utf-8")
        assert 'LLAMA_STACK_BASE_URL = "https://llama-stack.example.com"' in script
        assert "Question (empty line to exit):" in script
        assert "_apply_api_key_to_env" in script
        assert "copy.deepcopy" in script
        assert "while True" in script
        assert "API key" in script
        readme = (out / "p1" / README_FILENAME).read_text(encoding="utf-8")
        assert readme.startswith("# Call Llama Stack")
        assert f"python3 {SCRIPT_FILENAME}" in readme
        assert OUTPUT_FILENAME in readme
        assert 'LLAMA_STACK_BASE_URL = "https://llama-stack.example.com"' in readme
        assert "LLAMA_STACK_CLIENT_API_KEY" in readme
        assert ".llama_stack_client_api_key" not in readme

    def test_body_aligns_with_llama_stack_v1_responses_openapi_shape(self, tmp_path):
        """Top-level fields and nested message/tool shapes match Responses create contract."""
        out, _ = _run_python_func(tmp_path, [("p1", _minimal_pattern())])
        body = json.loads((out / "p1" / OUTPUT_FILENAME).read_text(encoding="utf-8"))
        assert isinstance(body["model"], str)
        assert body["stream"] is False
        assert body["store"] is True
        assert isinstance(body["input"], list) and len(body["input"]) >= 1
        msg = body["input"][0]
        assert msg.get("type") == "message"
        assert msg.get("role") in ("system", "developer", "user", "assistant")
        assert isinstance(msg.get("content"), list)
        part = msg["content"][0]
        assert part.get("type") == "input_text"
        assert isinstance(part.get("text"), str)
        assert isinstance(body["metadata"], dict)
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in body["metadata"].items())
        assert len(body["tools"]) == 1
        assert body["tool_choice"] == {"type": "file_search"}
        assert body["include"] == ["file_search_call.results"]
        fs = body["tools"][0]
        assert fs.get("type") == "file_search"
        assert isinstance(fs.get("vector_store_ids"), list)
        assert isinstance(fs.get("max_num_results"), int)

    def test_no_tools_when_collection_missing(self, tmp_path):
        """Omit file_search when vector collection name is empty."""
        out, _ = _run_python_func(tmp_path, [("p1", _minimal_pattern(collection=""))])
        body = json.loads((out / "p1" / OUTPUT_FILENAME).read_text(encoding="utf-8"))
        assert "tools" not in body
        assert "tool_choice" not in body
        assert "include" not in body
        assert body["store"] is True

    def test_hybrid_search_includes_ranking_options(self, tmp_path):
        """Hybrid mode maps ranker details into file_search ranking_options."""
        out, _ = _run_python_func(tmp_path, [("p1", _pattern_with_search_mode("hybrid"))])
        body = json.loads((out / "p1" / OUTPUT_FILENAME).read_text(encoding="utf-8"))
        ranking = body["tools"][0]["ranking_options"]
        assert ranking["ranker"] == "weighted"
        assert ranking["alpha"] == 0.7
        assert ranking["impact_factor"] == 50.0

    def test_non_hybrid_search_omits_ranking_options(self, tmp_path):
        """Non-hybrid mode should not inject hybrid ranking_options."""
        out, _ = _run_python_func(tmp_path, [("p1", _pattern_with_search_mode("vector"))])
        body = json.loads((out / "p1" / OUTPUT_FILENAME).read_text(encoding="utf-8"))
        assert "ranking_options" not in body["tools"][0]

    def test_input_uses_placeholder_not_generation_template(self, tmp_path):
        """``input`` uses the fixed placeholder, not ``user_message_text`` template expansion."""
        pattern = _minimal_pattern()
        pattern["settings"]["generation"]["user_message_text"] = "Question: {question}\\nContext: {reference_documents}"
        out, _ = _run_python_func(tmp_path, [("p1", pattern)])
        body = json.loads((out / "p1" / OUTPUT_FILENAME).read_text(encoding="utf-8"))
        assert body["input"][0]["content"][0]["text"] == RESPONSES_BODY_DEFAULT_QUESTION

    def test_rewrites_legacy_context_section_wording_for_file_search(self, tmp_path):
        """Legacy system_message_text wording is normalized for file_search usage."""
        pattern = _minimal_pattern()
        pattern["settings"]["generation"]["system_message_text"] = (
            "Please answer the question I provide in the Question section below, "
            "based solely on the information I provide in the Context section."
        )
        out, _ = _run_python_func(tmp_path, [("p1", pattern)])
        body = json.loads((out / "p1" / OUTPUT_FILENAME).read_text(encoding="utf-8"))
        assert "Question section" not in body["instructions"]
        assert "Context section" not in body["instructions"]
        assert "file_search results" in body["instructions"]
        assert "Respond in the same language as the user question." in body["instructions"]

    def test_script_supports_custom_ca_bundle_and_insecure_tls(self, tmp_path):
        """Generated helper exposes CA-bundle env vars and a dev-only insecure flag.

        Covers the corporate-PKI / dev-cluster TLS gap: the script must build an
        explicit ``ssl.SSLContext`` that honors ``REQUESTS_CA_BUNDLE`` /
        ``SSL_CERT_FILE`` for private CAs, and ``LLAMA_STACK_TLS_INSECURE`` as a
        dev-only opt-out, then pass that context into ``urlopen``.
        """
        out, _ = _run_python_func(tmp_path, [("p1", _minimal_pattern())])
        script = (out / "p1" / SCRIPT_FILENAME).read_text(encoding="utf-8")
        # The helper builds an SSLContext from env vars.
        assert "import ssl" in script
        assert "_build_ssl_context" in script
        assert "REQUESTS_CA_BUNDLE" in script
        assert "SSL_CERT_FILE" in script
        assert "LLAMA_STACK_TLS_INSECURE" in script
        assert "create_default_context" in script
        # Insecure mode actually disables verification and warns on stderr.
        assert "ctx.check_hostname = False" in script
        assert "ssl.CERT_NONE" in script
        assert "file=sys.stderr" in script
        # The context is plumbed into urlopen (not just constructed and ignored).
        assert "context=ssl_context" in script
        # README documents the new env vars in a dedicated section.
        readme = (out / "p1" / README_FILENAME).read_text(encoding="utf-8")
        assert "## TLS / certificates" in readme
        assert "REQUESTS_CA_BUNDLE" in readme
        assert "SSL_CERT_FILE" in readme
        assert "LLAMA_STACK_TLS_INSECURE" in readme


class TestBuildResponsesRequestBodiesPythonFunc:
    """Tests for prepare_responses_api_requests.python_func."""

    def test_writes_one_json_per_pattern_subdir(self, tmp_path):
        """Writes v1_responses_body.json under each pattern subdirectory."""
        out, out_art = _run_python_func(
            tmp_path,
            [("p1", _minimal_pattern("c1")), ("p2", _minimal_pattern("c2"))],
        )
        for name, coll in (("p1", "c1"), ("p2", "c2")):
            fp = out / name / OUTPUT_FILENAME
            sp = out / name / SCRIPT_FILENAME
            rp = out / name / README_FILENAME
            assert fp.is_file()
            assert sp.is_file()
            assert rp.is_file()
            body = json.loads(fp.read_text(encoding="utf-8"))
            assert body["tools"][0]["vector_store_ids"] == [coll]
        assert len(out_art.metadata["metadata"]["patterns"]) == 2
        for entry in out_art.metadata["metadata"]["patterns"]:
            assert "readme_path" in entry

    def test_skips_dirs_without_pattern_json(self, tmp_path):
        """Ignore subdirectories that do not contain pattern.json."""
        rag = tmp_path / "in"
        (rag / "empty").mkdir(parents=True)
        out = tmp_path / "out"
        out.mkdir()
        out_art = mock.Mock()
        out_art.path = str(out)
        out_art.metadata = {}
        prepare_responses_api_requests.python_func(
            rag_patterns=str(rag),
            responses_api_artifacts=out_art,
        )
        assert list(out.iterdir()) == []


class TestBuildResponsesRequestBodiesDefinition:
    """Sanity checks on the KFP component wrapper."""

    def test_callable_and_python_func(self):
        """Component is a KFP task factory with a python_func entrypoint."""
        assert callable(prepare_responses_api_requests)
        assert hasattr(prepare_responses_api_requests, "python_func")

    def test_signature_has_artifact_io(self):
        """python_func accepts rag_patterns and responses_api_artifacts artifacts."""
        import inspect

        sig = inspect.signature(prepare_responses_api_requests.python_func)
        assert "rag_patterns" in sig.parameters
        assert "responses_api_artifacts" in sig.parameters
        assert "default_question" not in sig.parameters
