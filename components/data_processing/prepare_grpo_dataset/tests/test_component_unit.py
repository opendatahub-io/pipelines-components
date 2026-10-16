"""Unit tests for the prepare_grpo_dataset component."""

import io
import json
from unittest import mock

import pytest

from ..component import prepare_grpo_dataset


def _raw_tool_call_row():
    return {
        "question": "What is the weather in Boston?",
        "available_tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "messages": [
            {"role": "system", "content": "You call tools."},
            {"role": "user", "content": "What is the weather in Boston?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"Boston"}',
                        },
                    }
                ],
            },
        ],
    }


def _run_component(tmp_path, dataset_rows, **component_kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("\n".join(json.dumps(row) for row in dataset_rows) + "\n")
    with mock.patch("datasets.load_dataset", return_value=iter(dataset_rows)) as load_dataset:
        result = prepare_grpo_dataset.python_func(
            dataset_uri=str(input_path),
            pvc_mount_path=str(tmp_path / "pvc"),
            **component_kwargs,
        )
    return result, load_dataset


def test_component_has_expected_interface():
    """Expose the preparer inputs and dynamic resolver outputs."""
    assert callable(prepare_grpo_dataset)
    assert hasattr(prepare_grpo_dataset, "python_func")
    assert "dataset_uri" in prepare_grpo_dataset.component_spec.inputs
    assert "pvc_mount_path" in prepare_grpo_dataset.component_spec.inputs
    assert "max_http_download_bytes" in prepare_grpo_dataset.component_spec.inputs
    assert "max_invalid_row_ratio" in prepare_grpo_dataset.component_spec.inputs
    assert "resolved_data_path" in prepare_grpo_dataset.component_spec.outputs
    assert "resolved_data_config" in prepare_grpo_dataset.component_spec.outputs
    assert "use_hf_direct" in prepare_grpo_dataset.component_spec.outputs


def test_huggingface_reference_is_not_materialized(tmp_path, capsys):
    """Keep HF references direct and avoid creating a PVC cache."""
    result = prepare_grpo_dataset.python_func(
        dataset_uri="hf://Agent-Ark/Toucan-1.5M:Qwen3",
        pvc_mount_path=str(tmp_path / "pvc"),
        dataset_config="ignored",
    )

    assert result == ("Agent-Ark/Toucan-1.5M", "Qwen3", True)
    assert not (tmp_path / "pvc").exists()
    assert "Agent-Ark/Toucan-1.5M" not in capsys.readouterr().out


def test_local_rows_are_normalized_without_sampling_or_splitting(tmp_path):
    """Normalize all supplied local rows without component-side sampling."""
    result, load_dataset = _run_component(tmp_path, [_raw_tool_call_row()])

    normalized_path, config, use_hf_direct = result
    with open(normalized_path, encoding="utf-8") as normalized_file:
        normalized = [json.loads(line) for line in normalized_file]

    assert config == ""
    assert use_hf_direct is False
    assert len(normalized) == 1
    assert normalized[0]["target_tool_name"] == "get_weather"
    assert normalized[0]["target_arguments"] == {"city": "Boston"}
    assert normalized[0]["tools"][0]["type"] == "function"
    load_dataset.assert_called_once()
    assert load_dataset.call_args.kwargs["streaming"] is True


def test_local_jsonl_uses_the_datasets_reader(tmp_path):
    """Read a real local JSONL source before applying GRPO normalization."""
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(json.dumps(_raw_tool_call_row()) + "\n")

    result = prepare_grpo_dataset.python_func(
        dataset_uri=str(input_path),
        pvc_mount_path=str(tmp_path / "pvc"),
    )

    with open(result[0], encoding="utf-8") as normalized_file:
        normalized = [json.loads(line) for line in normalized_file]
    assert len(normalized) == 1
    assert normalized[0]["target_tool_name"] == "get_weather"


def test_real_jsonl_reader_null_fields_use_legacy_fallbacks(tmp_path):
    """Handle null fields produced when JSONL rows have different schemas."""
    tool_definitions = _raw_tool_call_row()["available_tools"]
    rows = [
        {
            "question": "first",
            "tools": tool_definitions,
            "target_tool_name": "get_weather",
            "target_arguments": {"city": "Boston"},
        },
        {
            "question": None,
            "available_tools": tool_definitions,
            "target_tool_name": None,
            "tool_name": "get_weather",
            "target_arguments": None,
            "arguments": {"city": "Boston"},
        },
    ]
    input_path = tmp_path / "input.jsonl"
    input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    result = prepare_grpo_dataset.python_func(
        dataset_uri=str(input_path),
        pvc_mount_path=str(tmp_path / "pvc"),
    )

    with open(result[0], encoding="utf-8") as normalized_file:
        normalized = [json.loads(line) for line in normalized_file]
    assert len(normalized) == 2
    assert normalized[1]["question"] == ""
    assert normalized[1]["target_tool_name"] == "get_weather"
    assert normalized[1]["target_arguments"] == {"city": "Boston"}


def test_existing_canonical_rows_are_preserved(tmp_path):
    """Keep already-canonical rows usable by the Training Hub loader."""
    row = {
        "question": "What is the weather in Boston?",
        "tools": [
            {
                "name": "get_weather",
                "description": "Get the weather.",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        "target_tool_name": "get_weather",
        "target_arguments": {"city": "Boston"},
    }
    result, _ = _run_component(tmp_path, [row])
    with open(result[0], encoding="utf-8") as normalized_file:
        normalized = [json.loads(line) for line in normalized_file]

    assert normalized[0]["target_tool_name"] == "get_weather"
    assert normalized[0]["tools"][0]["function"]["name"] == "get_weather"


def test_none_fields_fall_back_to_legacy_names(tmp_path):
    """Handle null fields emitted by the datasets/PyArrow reader."""
    row = {
        "question": None,
        "tools": None,
        "available_tools": _raw_tool_call_row()["available_tools"],
        "target_tool_name": None,
        "tool_name": "get_weather",
        "target_arguments": None,
        "arguments": {"city": "Boston"},
    }

    result, _ = _run_component(tmp_path, [row])
    with open(result[0], encoding="utf-8") as normalized_file:
        normalized = [json.loads(line) for line in normalized_file]

    assert normalized[0]["question"] == ""
    assert normalized[0]["target_tool_name"] == "get_weather"
    assert normalized[0]["target_arguments"] == {"city": "Boston"}
    assert normalized[0]["tools"][0]["function"]["name"] == "get_weather"


def test_multiple_tool_calls_use_fifo_response_ids(tmp_path):
    """Assign missing tool-response IDs to the matching calls in order."""
    row = _raw_tool_call_row()
    row["available_tools"].append(
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get the time.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    )
    row["messages"][2]["tool_calls"].append(
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "get_time", "arguments": "{}"},
        }
    )
    row["messages"].extend(
        [
            {"role": "tool", "content": "Boston", "name": "get_weather"},
            {"role": "tool", "content": "12:00", "name": "get_time"},
        ]
    )

    result, _ = _run_component(tmp_path, [row])
    with open(result[0], encoding="utf-8") as normalized_file:
        normalized = [json.loads(line) for line in normalized_file]

    tool_messages = [message for message in normalized[0]["messages"] if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in tool_messages] == ["call_1", "call_2"]


def test_invalid_rows_are_skipped_and_reported(tmp_path):
    """Skip malformed rows while retaining valid tool-call examples."""
    bad_row = {"question": "missing tool target"}
    result, _ = _run_component(tmp_path, [bad_row, *([_raw_tool_call_row()] * 10)])
    with open(result[0], encoding="utf-8") as normalized_file:
        normalized = [json.loads(line) for line in normalized_file]

    assert len(normalized) == 10
    assert result[2] is False


def test_too_many_invalid_rows_fail(tmp_path):
    """Fail instead of silently training on a mostly invalid dataset."""
    with pytest.raises(ValueError, match="too many invalid GRPO rows"):
        _run_component(tmp_path, [{"question": "missing tool target"}, _raw_tool_call_row()])


def test_invalid_row_ratio_can_be_configured(tmp_path):
    """Allow callers to choose a stricter or more permissive invalid-row limit."""
    rows = [{"question": "missing tool target"}, *([_raw_tool_call_row()] * 9)]

    result, _ = _run_component(tmp_path, rows)
    assert result[2] is False

    with pytest.raises(ValueError, match="too many invalid GRPO rows"):
        _run_component(tmp_path / "strict", rows, max_invalid_row_ratio=0.09)


def test_invalid_row_ratio_must_be_between_zero_and_one(tmp_path):
    """Reject invalid threshold values before processing the dataset."""
    with pytest.raises(ValueError, match="max_invalid_row_ratio"):
        prepare_grpo_dataset.python_func(
            dataset_uri="hf://example/dataset",
            pvc_mount_path=str(tmp_path / "pvc"),
            max_invalid_row_ratio=1.1,
        )


def test_unexpected_normalization_errors_propagate(tmp_path):
    """Do not classify implementation errors as invalid dataset rows."""

    class BrokenRow(dict):
        def get(self, *args, **kwargs):
            raise TypeError("normalizer bug")

    with pytest.raises(TypeError, match="normalizer bug"):
        _run_component(tmp_path, [BrokenRow()])


def test_http_source_is_downloaded_to_pvc(tmp_path):
    """Download an HTTP JSONL resource into the shared PVC cache."""
    row = _raw_tool_call_row()
    with mock.patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(row).encode())) as urlopen:
        with mock.patch("datasets.load_dataset", return_value=iter([row])):
            result = prepare_grpo_dataset.python_func(
                dataset_uri="https://example.test/data.jsonl",
                pvc_mount_path=str(tmp_path / "pvc"),
            )

    assert result[2] is False
    urlopen.assert_called_once_with("https://example.test/data.jsonl", timeout=300)


@pytest.mark.parametrize("headers", [{}, {"Content-Length": "10"}])
def test_http_download_respects_maximum_size(tmp_path, headers):
    """Reject an HTTP response larger than the configured limit."""
    response = io.BytesIO(b"0123456789")
    response.headers = headers
    with mock.patch("urllib.request.urlopen", return_value=response):
        with pytest.raises(ValueError, match="max_http_download_bytes"):
            prepare_grpo_dataset.python_func(
                dataset_uri="https://example.test/data.jsonl",
                pvc_mount_path=str(tmp_path / "pvc"),
                max_http_download_bytes=9,
            )


def test_failed_http_download_does_not_leave_a_cache_file(tmp_path):
    """Do not publish or retain partial HTTP downloads."""

    class FailingResponse(io.BytesIO):
        def read(self, *args, **kwargs):
            raise RuntimeError("download failed")

    with mock.patch("urllib.request.urlopen", return_value=FailingResponse(b"partial")):
        with pytest.raises(RuntimeError, match="download failed"):
            prepare_grpo_dataset.python_func(
                dataset_uri="https://example.test/data.jsonl",
                pvc_mount_path=str(tmp_path / "pvc"),
            )

    download_dir = tmp_path / "pvc" / ".cache" / "prepare_grpo_dataset" / "downloads"
    assert not list(download_dir.iterdir())


@pytest.mark.parametrize("dataset_uri", ["pvc://../outside.jsonl", "../outside.jsonl"])
def test_pvc_relative_paths_cannot_escape_mount(tmp_path, dataset_uri):
    """Keep PVC-relative paths inside the configured mount."""
    with pytest.raises(ValueError, match="within pvc_mount_path"):
        prepare_grpo_dataset.python_func(
            dataset_uri=dataset_uri,
            pvc_mount_path=str(tmp_path / "pvc"),
        )


def test_missing_bare_local_path_fails_before_dataset_load(tmp_path):
    """Report missing local files before invoking the dataset reader."""
    missing_path = tmp_path / "missing.jsonl"
    with mock.patch("datasets.load_dataset") as load_dataset:
        with pytest.raises(FileNotFoundError, match="dataset file not found"):
            prepare_grpo_dataset.python_func(
                dataset_uri=str(missing_path),
                pvc_mount_path=str(tmp_path / "pvc"),
            )

    load_dataset.assert_not_called()
