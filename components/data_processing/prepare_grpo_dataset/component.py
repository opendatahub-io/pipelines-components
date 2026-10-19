"""Adapt GRPO dataset sources to Training Hub's built-in input contract."""

from typing import NamedTuple

from kfp import dsl


@dsl.component(
    base_image="quay.io/opendatahub/odh-th06-cpu-torch291-py312:odh-3.4",
    packages_to_install=["datasets>=2.14.0", "s3fs>=2023.1.0"],
)
def prepare_grpo_dataset(
    dataset_uri: str,
    pvc_mount_path: str,
    dataset_config: str = "",
    max_http_download_bytes: int = 2_147_483_648,
    max_invalid_row_ratio: float = 0.10,
) -> NamedTuple(
    "PrepareDatasetOutputs",
    [
        ("resolved_data_path", str),
        ("resolved_data_config", str),
        ("use_hf_direct", bool),
    ],
):
    """Resolve any supported source for Training Hub's GRPO loader.

    Hugging Face references are returned without downloading or rewriting them.
    Local/PVC JSON or JSONL files, S3 JSON/JSONL objects, and HTTP(S) JSON/JSONL
    resources are converted to Training Hub's local canonical format on the
    shared PVC. The component intentionally does not subset, split, or sample the
    data; Training Hub owns ``n_train`` and ``n_val``.

    Private S3 sources use ``AWS_ACCESS_KEY_ID`` and
    ``AWS_SECRET_ACCESS_KEY``. ``AWS_S3_ENDPOINT`` and
    ``AWS_DEFAULT_REGION`` are optional for S3-compatible services. The
    consuming pipeline should inject these variables from its RHOAI
    Connection Secret with ``kfp.kubernetes.use_secret_as_env``. The secret
    name is a pipeline-task setting, not a component runtime input.

    See the Training Hub GRPO loader contract:
    https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/src/training_hub/algorithms/lora_grpo.py

    The normalized output always uses ``messages`` and ``tools`` so Training
    Hub sees one local format for every row. Single-turn source rows are
    represented as system, user, and assistant-with-tool-call messages.
    Multi-turn rows retain their conversation history and are normalized to
    modern OpenAI ``tool_calls``/``tool`` messages. Malformed rows are skipped
    only when they raise an expected ``ValueError``; the component fails if the
    invalid-row ratio exceeds ``max_invalid_row_ratio``.

    Args:
        dataset_uri: Hugging Face URI/repository ID, ``s3://`` object,
            ``http://`` or ``https://`` JSON/JSONL URL, or local JSON/JSONL
            path visible in the component container. Bare local/PVC paths are
            preferred; ``pvc://`` is retained as a legacy alias for the same
            local/PVC handling and validation.
        pvc_mount_path: Mount point of the shared PVC used for caches and
            normalized local data.
        dataset_config: Optional Hugging Face dataset configuration. An inline
            ``:config`` suffix on an HF URI takes precedence.
        max_http_download_bytes: Maximum HTTP(S) response size in bytes. The
            default is 2 GiB; the limit is enforced before and during the
            download.
        max_invalid_row_ratio: Maximum fraction of invalid rows to skip before
            failing validation. The default is 10%; use ``0.0`` for strict
            validation.

    Returns:
        A Training Hub data path, config, and flag indicating whether the path
        should be passed through the direct Hugging Face loader.
    """
    import hashlib
    import json
    import os
    import urllib.parse
    import urllib.request
    import uuid
    from pathlib import Path

    from datasets import load_dataset

    def _parse_json_value(value, field_name):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{field_name} must contain valid JSON") from exc
        return value

    def _normalise_tools(value):
        tools = _parse_json_value(value, "tools")
        if isinstance(tools, dict):
            tools = [tools]
        if not isinstance(tools, list) or not tools:
            raise ValueError("tools must be a non-empty list")

        normalized = []
        for tool in tools:
            if not isinstance(tool, dict):
                raise ValueError("each tool must be an object")
            if "function" in tool:
                function = tool["function"]
                if not isinstance(function, dict):
                    raise ValueError("tool.function must be an object")
                normalized_tool = dict(tool)
                normalized_tool.setdefault("type", "function")
            else:
                # Accept a bare function definition and emit the OpenAI tool
                # envelope required by the rollout API.
                function = dict(tool)
                normalized_tool = {"type": "function", "function": function}
            name = normalized_tool.get("function", {}).get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("tool function name must be a non-empty string")
            normalized.append(normalized_tool)
        return normalized

    def _message_tool_calls(message):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            return []
        if isinstance(message.get("function_call"), dict):
            function_call = message["function_call"]
            return [
                {
                    "name": function_call.get("name", ""),
                    "arguments": function_call.get("arguments", "{}"),
                }
            ]
        tool_calls = message.get("tool_calls") or []
        if not isinstance(tool_calls, list) or not tool_calls:
            return []
        calls = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            calls.append(
                {
                    "name": function.get("name", ""),
                    "arguments": function.get("arguments", "{}"),
                }
            )
        return calls

    def _message_tool_call(message):
        calls = _message_tool_calls(message)
        return calls[0] if calls else None

    def _normalise_messages(messages, row_index):
        messages = _parse_json_value(messages, "messages")
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")

        normalized = []
        pending_tool_call_ids = []

        def _set_tool_response_id(message, fallback_id):
            explicit_id = message.get("tool_call_id")
            if explicit_id:
                if explicit_id in pending_tool_call_ids:
                    pending_tool_call_ids.remove(explicit_id)
                return
            message["tool_call_id"] = pending_tool_call_ids.pop(0) if pending_tool_call_ids else fallback_id

        for message_index, original in enumerate(messages):
            if not isinstance(original, dict):
                raise ValueError(f"messages[{message_index}] must be an object")
            message = dict(original)
            role = message.get("role")
            if role == "assistant" and isinstance(message.get("function_call"), dict):
                function_call = message.pop("function_call")
                tool_call_id = f"call_{row_index}_{message_index}"
                arguments = function_call.get("arguments", "{}")
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, separators=(",", ":"))
                message["tool_calls"] = [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": function_call.get("name", ""),
                            "arguments": arguments,
                        },
                    }
                ]
                pending_tool_call_ids.append(tool_call_id)
            elif role == "assistant" and message.get("tool_calls"):
                if not isinstance(message["tool_calls"], list):
                    raise ValueError("assistant tool_calls must be a list")
                tool_calls = []
                for call_index, original_call in enumerate(message["tool_calls"]):
                    if not isinstance(original_call, dict):
                        raise ValueError("assistant tool calls must be objects")
                    call = dict(original_call)
                    function = dict(call.get("function") or {})
                    call["id"] = call.get("id") or f"call_{row_index}_{message_index}_{call_index}"
                    call.setdefault("type", "function")
                    arguments = function.get("arguments", "{}")
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments, separators=(",", ":"))
                    function["arguments"] = arguments
                    call["function"] = function
                    tool_calls.append(call)
                    pending_tool_call_ids.append(call["id"])
                message["tool_calls"] = tool_calls
            elif role == "function":
                message["role"] = "tool"
                _set_tool_response_id(message, f"call_{row_index}")
                message.setdefault("content", "")
                message.pop("name", None)
            elif role == "tool":
                _set_tool_response_id(message, f"call_{row_index}")

            for call in _message_tool_calls(message):
                if not str(call.get("name", "")).strip():
                    raise ValueError("assistant tool-call function name must be non-empty")
            normalized.append(message)
        return normalized

    def _tool_call_count(messages):
        return sum(len(_message_tool_calls(message)) for message in messages)

    def _single_turn_messages(question, system_prompt, target_name, target_arguments, row_index):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{row_index}_0",
                        "type": "function",
                        "function": {
                            "name": target_name,
                            "arguments": json.dumps(target_arguments, separators=(",", ":")),
                        },
                    }
                ],
            },
        ]

    def _canonicalize_row(row, row_index):
        if not isinstance(row, dict):
            raise ValueError("row must be an object")

        messages = row.get("messages")
        if messages is not None:
            normalized_messages = _normalise_messages(messages, row_index)
            tools = row.get("tools") or row.get("available_tools") or []
            tools = _normalise_tools(tools)
            tool_call_count = _tool_call_count(normalized_messages)
            if tool_call_count == 0:
                raise ValueError("messages contain no assistant tool call")
            # Preserve every trace as messages. Training Hub will decompose
            # each assistant tool-call turn into a training sample.
            return {
                "id": row.get("id", row.get("uuid", str(row_index))),
                "messages": normalized_messages,
                "tools": tools,
            }

        tools = row.get("tools") or row.get("available_tools")
        target_name = row.get("target_tool_name") or row.get("tool_name")
        target_arguments = row.get("target_arguments")
        if target_arguments is None:
            target_arguments = row.get("arguments", {})
        if tools is None or target_name is None:
            raise ValueError("row is missing tool-call fields")
        target_name = str(target_name).strip()
        if not target_name:
            raise ValueError("target_tool_name must be a non-empty string")
        target_arguments = _parse_json_value(target_arguments, "target_arguments")
        question = row.get("question") or ""
        system_prompt = row.get("system_prompt") or ""
        return {
            "id": row.get("id", row.get("uuid", str(row_index))),
            "messages": _single_turn_messages(
                question,
                system_prompt,
                target_name,
                target_arguments,
                row_index,
            ),
            "tools": _normalise_tools(tools),
        }

    def _is_huggingface_reference(uri):
        parsed = urllib.parse.urlparse(uri)
        if uri.startswith("hf://"):
            return True
        if parsed.scheme:
            return False
        return not uri.lower().endswith((".json", ".jsonl")) and not uri.startswith(("/", "./", "../"))

    def _normalise_huggingface_reference(uri, config):
        reference = uri.removeprefix("hf://")
        if ":" in reference:
            reference, inline_config = reference.split(":", 1)
            config = inline_config or config
        if not reference or reference.startswith(("/", ".")):
            raise ValueError("invalid Hugging Face dataset reference")
        return reference, config

    if not dataset_uri or not dataset_uri.strip():
        raise ValueError("dataset_uri must not be empty")
    if not pvc_mount_path or not pvc_mount_path.strip():
        raise ValueError("pvc_mount_path must not be empty")
    if not 0.0 <= max_invalid_row_ratio <= 1.0:
        raise ValueError("max_invalid_row_ratio must be between 0.0 and 1.0")

    dataset_uri = dataset_uri.strip()
    if _is_huggingface_reference(dataset_uri):
        reference, resolved_config = _normalise_huggingface_reference(dataset_uri, dataset_config.strip())
        print("Using Training Hub Hugging Face loader for the supplied dataset reference")
        return reference, resolved_config, True

    pvc_root = Path(pvc_mount_path).resolve()
    cache_root = pvc_root / ".cache" / "prepare_grpo_dataset"
    cache_root.mkdir(parents=True, exist_ok=True)
    hf_cache = cache_root / "huggingface"
    tmp_root = cache_root / "tmp"
    for path in (hf_cache, tmp_root):
        path.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(cache_root / "home")
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache / "datasets")
    os.environ["HF_HUB_CACHE"] = str(hf_cache / "hub")
    os.environ["TMPDIR"] = str(tmp_root)
    os.environ["TEMP"] = str(tmp_root)
    os.environ["TMP"] = str(tmp_root)
    Path(os.environ["HOME"]).mkdir(parents=True, exist_ok=True)

    parsed = urllib.parse.urlparse(dataset_uri)
    supported_schemes = {"", "pvc", "http", "https", "s3"}
    if parsed.scheme not in supported_schemes:
        raise ValueError(f"unsupported dataset URI scheme: {parsed.scheme}")

    def _resolve_pvc_path(relative_path):
        candidate = (pvc_root / relative_path).resolve()
        try:
            candidate.relative_to(pvc_root)
        except ValueError as exc:
            raise ValueError("dataset path must remain within pvc_mount_path") from exc
        return candidate

    source_uri = dataset_uri
    if dataset_uri.startswith("pvc://"):
        relative_path = dataset_uri.removeprefix("pvc://").lstrip("/")
        source_uri = str(_resolve_pvc_path(relative_path))
    elif not parsed.scheme and not os.path.isabs(dataset_uri):
        source_uri = str(_resolve_pvc_path(dataset_uri))

    if parsed.scheme in {"http", "https"}:
        if max_http_download_bytes <= 0:
            raise ValueError("max_http_download_bytes must be positive")
        suffix = Path(urllib.parse.urlparse(dataset_uri).path).suffix.lower()
        if suffix not in {".json", ".jsonl"}:
            raise ValueError("HTTP(S) dataset URIs must point to a .json or .jsonl resource")
        filename = hashlib.sha256(dataset_uri.encode()).hexdigest() + suffix
        downloaded_path = cache_root / "downloads" / filename
        downloaded_path.parent.mkdir(parents=True, exist_ok=True)
        if downloaded_path.exists() and downloaded_path.stat().st_size > max_http_download_bytes:
            raise ValueError(f"cached HTTP dataset exceeds max_http_download_bytes ({max_http_download_bytes} bytes)")
        if not downloaded_path.exists():
            print("Downloading HTTP(S) dataset to the shared PVC")
            temporary_download_path = downloaded_path.with_name(f"{downloaded_path.name}.{uuid.uuid4().hex}.tmp")
            try:
                with (
                    urllib.request.urlopen(dataset_uri, timeout=300) as response,
                    temporary_download_path.open("wb") as output,
                ):
                    raw_content_length = getattr(response, "headers", {}).get("Content-Length")
                    content_length = None
                    if raw_content_length:
                        try:
                            content_length = int(raw_content_length)
                        except (TypeError, ValueError):
                            pass
                    if content_length is not None and content_length > max_http_download_bytes:
                        raise ValueError(
                            f"HTTP dataset exceeds max_http_download_bytes ({max_http_download_bytes} bytes)"
                        )

                    bytes_written = 0
                    while True:
                        chunk = response.read(min(1024 * 1024, max_http_download_bytes - bytes_written + 1))
                        if not chunk:
                            break
                        bytes_written += len(chunk)
                        if bytes_written > max_http_download_bytes:
                            raise ValueError(
                                f"HTTP dataset exceeds max_http_download_bytes ({max_http_download_bytes} bytes)"
                            )
                        output.write(chunk)
                os.replace(temporary_download_path, downloaded_path)
            finally:
                temporary_download_path.unlink(missing_ok=True)
        source_uri = str(downloaded_path)

    if parsed.scheme in {"", "pvc"} and not str(source_uri).lower().endswith((".json", ".jsonl")):
        raise ValueError("local/PVC dataset paths must end in .json or .jsonl")
    if parsed.scheme in {"", "pvc"} and not Path(source_uri).is_file():
        raise FileNotFoundError(f"dataset file not found: {source_uri}")

    normalized_name = hashlib.sha256(f"{dataset_uri}\n{dataset_config}".encode()).hexdigest() + ".jsonl"
    normalized_path = cache_root / "normalized" / normalized_name
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = normalized_path.with_name(f"{normalized_path.name}.{uuid.uuid4().hex}.tmp")

    data_files = source_uri if parsed.scheme == "s3" else str(source_uri)
    storage_options = {}
    if parsed.scheme == "s3":
        access_key = os.environ.get("AWS_ACCESS_KEY_ID", "").strip()
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip()
        if bool(access_key) != bool(secret_key):
            raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set together")
        if access_key and secret_key:
            storage_options.update({"key": access_key, "secret": secret_key})
        endpoint = os.environ.get("AWS_S3_ENDPOINT", "").strip()
        region = os.environ.get("AWS_DEFAULT_REGION", "").strip()
        client_kwargs = {}
        if endpoint:
            client_kwargs["endpoint_url"] = endpoint
        if region:
            client_kwargs["region_name"] = region
        if client_kwargs:
            storage_options["client_kwargs"] = client_kwargs
    print("Loading non-Hugging Face dataset for GRPO normalization")
    dataset = load_dataset(
        "json",
        data_files=data_files,
        split="train",
        streaming=True,
        cache_dir=str(hf_cache / "datasets"),
        **({"storage_options": storage_options} if parsed.scheme == "s3" else {}),
    )

    written = 0
    skipped = 0
    first_errors = []
    with temporary_path.open("w", encoding="utf-8") as output:
        for row_index, row in enumerate(dataset):
            try:
                normalized_row = _canonicalize_row(row, row_index)
            except ValueError as exc:
                skipped += 1
                if len(first_errors) < 5:
                    first_errors.append(f"row {row_index}: {exc}")
                continue
            output.write(json.dumps(normalized_row, ensure_ascii=False) + "\n")
            written += 1
    total_rows = written + skipped
    if written == 0:
        temporary_path.unlink(missing_ok=True)
        details = "; ".join(first_errors) or "no readable rows"
        raise ValueError(f"no valid GRPO tool-call rows were found ({details})")
    if skipped / total_rows > max_invalid_row_ratio:
        temporary_path.unlink(missing_ok=True)
        details = "; ".join(first_errors)
        raise ValueError(
            f"too many invalid GRPO rows: skipped {skipped} of {total_rows} "
            f"({skipped / total_rows:.1%}); maximum allowed is "
            f"{max_invalid_row_ratio:.1%}. {details}"
        )
    os.replace(temporary_path, normalized_path)
    print(f"Normalized {written} rows to the shared PVC; skipped {skipped} invalid rows")
    if first_errors:
        print("First skipped-row reasons: " + "; ".join(first_errors))
    return str(normalized_path), "", False


if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        prepare_grpo_dataset,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
