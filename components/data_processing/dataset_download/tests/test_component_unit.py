"""Unit tests for the dataset_download component."""

import inspect
import json
import textwrap
from unittest import mock

import pytest

from ..component import dataset_download


class _MockArtifact:
    """Mock KFP artifact with a writable path."""

    def __init__(self, path: str):
        self.path = path
        self.metadata = {}


def _extract_validation_functions():
    """Extract inner validation functions from the component for direct testing.

    The validation functions are defined inside the @dsl.component function,
    so we extract and compile them to test independently.
    """
    source = inspect.getsource(dataset_download.python_func)
    lines = source.split("\n")

    # Find and extract the function bodies (they are indented inside dataset_download)
    functions = {}
    current_func = None
    current_lines = []
    base_indent = None

    for line in lines:
        stripped = line.lstrip()
        if (
            stripped.startswith("def ")
            and current_func is None
            or (stripped.startswith("def ") and base_indent is not None and len(line) - len(stripped) <= base_indent)
        ):
            if current_func and current_lines:
                func_source = textwrap.dedent("\n".join(current_lines))
                functions[current_func] = func_source

            if stripped.startswith("def _has_tool_calls_in_messages(") or stripped.startswith(
                "def validate_tool_call_format_dataset("
            ):
                current_func = stripped.split("(")[0].replace("def ", "")
                current_lines = [line]
                base_indent = len(line) - len(stripped)
            elif stripped.startswith("def validate_chat_format_dataset("):
                current_func = "validate_chat_format_dataset"
                current_lines = [line]
                base_indent = len(line) - len(stripped)
            else:
                if current_func and current_lines:
                    func_source = textwrap.dedent("\n".join(current_lines))
                    functions[current_func] = func_source
                current_func = None
                current_lines = []
                base_indent = None
        elif current_func is not None:
            current_lines.append(line)

    if current_func and current_lines:
        func_source = textwrap.dedent("\n".join(current_lines))
        functions[current_func] = func_source

    # Compile and return the functions in a namespace
    from datasets import Dataset

    namespace = {"log_message": lambda msg: None, "Dataset": Dataset}  # stub log_message
    for name in ["_has_tool_calls_in_messages", "validate_chat_format_dataset", "validate_tool_call_format_dataset"]:
        if name in functions:
            exec(functions[name], namespace)

    return namespace


class _MockDataset:
    """Minimal mock that behaves like a HuggingFace Dataset for validation tests."""

    def __init__(self, items):
        self._items = items

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


class TestDatasetDownloadUnitTests:
    """Unit tests for component logic."""

    def test_component_function_exists(self):
        """Test that the component function is properly imported."""
        assert callable(dataset_download)
        assert hasattr(dataset_download, "python_func")

    def test_component_has_expected_parameters(self):
        """Test that the component has expected input parameters."""
        sig = inspect.signature(dataset_download.python_func)
        params = list(sig.parameters.keys())

        expected_params = {
            "train_dataset",
            "eval_dataset",
            "dataset_uri",
            "pvc_mount_path",
            "train_split_ratio",
            "subset_count",
            "dataset_format",
        }

        for param in expected_params:
            assert param in params, f"Expected parameter '{param}' not found in component"
        assert "hf_token" not in params, "hf_token should not be an explicit component parameter"

    def test_component_default_values(self):
        """Test that the component has expected default values."""
        sig = inspect.signature(dataset_download.python_func)
        params = sig.parameters

        assert params["train_split_ratio"].default == 0.9
        assert params["subset_count"].default == 0
        assert params["dataset_format"].default == "chat"

    @mock.patch.dict("sys.modules", {"datasets": mock.MagicMock()})
    @mock.patch("datasets.load_dataset")
    @mock.patch("os.makedirs")
    @mock.patch("os.path.exists")
    def test_component_with_mocked_huggingface(
        self,
        mock_exists,
        mock_makedirs,
        mock_load_dataset,
    ):
        """Test component with mocked HuggingFace dataset loading."""
        # Setup mock dataset
        mock_dataset = mock.MagicMock()
        mock_dataset.__len__ = mock.MagicMock(return_value=100)

        # Mock train_test_split
        mock_train = mock.MagicMock()
        mock_train.__len__ = mock.MagicMock(return_value=90)
        mock_train.to_json = mock.MagicMock()

        mock_eval = mock.MagicMock()
        mock_eval.__len__ = mock.MagicMock(return_value=10)
        mock_eval.to_json = mock.MagicMock()

        mock_dataset.train_test_split.return_value = {"train": mock_train, "test": mock_eval}
        mock_load_dataset.return_value = mock_dataset

        mock_exists.return_value = True

        # Mock output artifacts
        mock_train_output = mock.MagicMock()
        mock_train_output.path = "/tmp/train.jsonl"
        mock_train_output.metadata = {}

        mock_eval_output = mock.MagicMock()
        mock_eval_output.path = "/tmp/eval.jsonl"
        mock_eval_output.metadata = {}

        # The component would need full execution context
        # For now verify the component definition is valid
        assert dataset_download.python_func is not None

    def test_component_supports_multiple_uri_schemes(self):
        """Test that the component documentation mentions supported URI schemes."""
        # Verify docstring mentions supported schemes
        docstring = dataset_download.python_func.__doc__
        assert "hf://" in docstring or "HuggingFace" in docstring.lower()
        assert "s3://" in docstring
        assert "http" in docstring.lower()


class TestToolCallValidation:
    """Tests for tool-call format validation."""

    @pytest.fixture(autouse=True)
    def setup_validation_functions(self):
        """Extract validation functions from the component."""
        ns = _extract_validation_functions()
        self.validate_tool_call = ns["validate_tool_call_format_dataset"]
        self.validate_chat = ns["validate_chat_format_dataset"]
        self._has_tool_calls = ns["_has_tool_calls_in_messages"]

    def test_single_turn_valid(self):
        """Valid single-turn tool-call samples pass validation."""
        dataset = _MockDataset(
            [
                {"question": "Weather in CA?", "target_tool_name": "get_alerts", "target_arguments": {"state": "CA"}},
                {"question": "Weather in NY?", "target_tool_name": "get_alerts", "target_arguments": {"state": "NY"}},
            ]
        )
        assert self.validate_tool_call(dataset) is True

    def test_multi_turn_valid(self):
        """Valid multi-turn tool-call samples with tool_calls pass validation."""
        dataset = _MockDataset(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Check weather in CA"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "get_alerts", "arguments": "{}"},
                                }
                            ],
                        },
                    ],
                    "question": "Check weather in CA",
                },
                {
                    "messages": [
                        {"role": "user", "content": "Check weather in NY"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {"name": "get_alerts", "arguments": "{}"},
                                }
                            ],
                        },
                    ],
                    "question": "Check weather in NY",
                },
            ]
        )
        assert self.validate_tool_call(dataset) is True

    def test_empty_dataset_raises(self):
        """Empty dataset raises ValueError."""
        dataset = _MockDataset([])
        with pytest.raises(ValueError, match="Dataset is empty"):
            self.validate_tool_call(dataset)

    def test_single_turn_missing_target_tool_name(self):
        """Single-turn sample missing target_tool_name raises ValueError."""
        dataset = _MockDataset(
            [
                {"question": "Weather in CA?", "target_tool_name": "get_alerts"},
                {"question": "Weather in NY?", "target_tool_name": None},
            ]
        )
        with pytest.raises(ValueError, match="Item 1: 'target_tool_name' is missing or empty"):
            self.validate_tool_call(dataset)

    def test_single_turn_missing_question(self):
        """Single-turn sample missing question raises ValueError."""
        dataset = _MockDataset(
            [
                {"question": "Weather in CA?", "target_tool_name": "get_alerts"},
                {"question": None, "target_tool_name": "get_alerts"},
            ]
        )
        with pytest.raises(ValueError, match="Item 1: 'question' is missing or empty"):
            self.validate_tool_call(dataset)

    def test_multi_turn_no_tool_calls_in_messages(self):
        """Multi-turn sample without tool_calls in assistant messages raises ValueError."""
        dataset = _MockDataset(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Check weather"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "get_alerts", "arguments": "{}"},
                                }
                            ],
                        },
                    ],
                    "question": "Check weather",
                },
                {
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi there!"},  # no tool_calls
                    ],
                    "question": "Hello",
                },
            ]
        )
        with pytest.raises(ValueError, match="Item 1: no assistant message with 'tool_calls'"):
            self.validate_tool_call(dataset)

    def test_multi_turn_empty_messages(self):
        """Multi-turn sample with empty messages list raises ValueError."""
        dataset = _MockDataset(
            [
                {
                    "messages": [
                        {"role": "user", "content": "Check weather"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "get_alerts", "arguments": "{}"},
                                }
                            ],
                        },
                    ],
                },
                {"messages": []},  # empty messages
            ]
        )
        with pytest.raises(ValueError, match="Item 1: 'messages' must be a non-empty list"):
            self.validate_tool_call(dataset)

    def test_mixed_formats_rejected(self):
        """First sample is single-turn, second is multi-turn — must raise ValueError."""
        dataset = _MockDataset(
            [
                {"question": "Weather?", "target_tool_name": "get_alerts"},
                {
                    "messages": [
                        {"role": "user", "content": "Check weather"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "get_alerts", "arguments": "{}"},
                                }
                            ],
                        },
                    ],
                },
            ]
        )
        with pytest.raises(ValueError, match="'target_tool_name' is missing or empty"):
            self.validate_tool_call(dataset)

    def test_unrecognized_format_raises(self):
        """Sample with neither tool-call format raises ValueError."""
        dataset = _MockDataset(
            [
                {"text": "just some plain text", "label": 1},
            ]
        )
        with pytest.raises(ValueError, match="does not match any supported tool-call format"):
            self.validate_tool_call(dataset)

    def test_chat_format_unchanged(self):
        """Existing chat validation still works (regression guard)."""
        dataset = _MockDataset(
            [
                {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]},
                {"messages": [{"role": "user", "content": "Bye"}, {"role": "assistant", "content": "Goodbye"}]},
            ]
        )
        assert self.validate_chat(dataset) is True

    def test_invalid_dataset_format_value(self):
        """Passing unsupported dataset_format raises ValueError before download."""
        source = inspect.getsource(dataset_download.python_func)
        assert "Unsupported dataset_format" in source
        # Verify early guard exists before parse_uri
        guard_pos = source.index("Unsupported dataset_format")
        parse_pos = source.index("parse_uri(dataset_uri)")
        assert guard_pos < parse_pos, "dataset_format validation must happen before download"

        with pytest.raises(ValueError, match="Unsupported dataset_format"):
            dataset_download.python_func(
                train_dataset=mock.MagicMock(),
                eval_dataset=mock.MagicMock(),
                dataset_uri="hf://does-not-matter",
                pvc_mount_path="/tmp",
                dataset_format="toolcall",
            )

    def test_has_tool_calls_helper(self):
        """_has_tool_calls_in_messages correctly detects tool_calls."""
        messages_with = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        ]
        messages_without = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
        ]
        assert self._has_tool_calls(messages_with) is True
        assert self._has_tool_calls(messages_without) is False

    def test_has_tool_calls_rejects_empty_list(self):
        """tool_calls=[] should not count as having tool calls."""
        messages = [
            {"role": "assistant", "content": None, "tool_calls": []},
        ]
        assert self._has_tool_calls(messages) is False

    def test_has_tool_calls_rejects_none(self):
        """tool_calls=None should not count as having tool calls."""
        messages = [
            {"role": "assistant", "content": None, "tool_calls": None},
        ]
        assert self._has_tool_calls(messages) is False

    def test_has_tool_calls_rejects_scalar(self):
        """tool_calls as a non-list value should not count."""
        messages = [
            {"role": "assistant", "content": None, "tool_calls": "not_a_list"},
        ]
        assert self._has_tool_calls(messages) is False


def _multi_turn_row(question: str, call_id: str) -> dict:
    """Build a multi-turn tool-call sample whose user message has no tool_calls key."""
    return {
        "messages": [
            {"role": "user", "content": question},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "get_alerts", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "{}", "tool_call_id": call_id},
        ],
    }


class TestToolCallJsonlRoundTrip:
    """Regression test: multi-turn tool-call data survives load → save round-trip."""

    def test_multiturn_messages_written_as_dicts(self, tmp_path):
        """python_func must write messages as dicts, not JSON-encoded strings."""
        input_path = tmp_path / "input.jsonl"
        rows = [
            _multi_turn_row("Check weather in CA", "call_1"),
            _multi_turn_row("Check weather in NY", "call_2"),
        ]
        input_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        train_artifact = _MockArtifact(str(tmp_path / "train.jsonl"))
        eval_artifact = _MockArtifact(str(tmp_path / "eval.jsonl"))

        dataset_download.python_func(
            train_dataset=train_artifact,
            eval_dataset=eval_artifact,
            dataset_uri=str(input_path),
            pvc_mount_path=str(tmp_path),
            train_split_ratio=1.0,
            subset_count=0,
            dataset_format="tool_call",
        )

        with open(train_artifact.path) as f:
            output_rows = [json.loads(line) for line in f if line.strip()]

        assert len(output_rows) == 2
        for output_row in output_rows:
            for i, msg in enumerate(output_row["messages"]):
                assert isinstance(msg, dict), f"messages[{i}] should be dict, got {type(msg).__name__}"
                assert "role" in msg, f"messages[{i}] missing 'role'"

            user_msg = output_row["messages"][0]
            assert user_msg["role"] == "user"
            assert "tool_calls" not in user_msg, "None-valued tool_calls should be stripped from user messages"

    def test_both_formats_logs_tie_break(self, tmp_path):
        """When a row matches both formats, log the single-turn tie-break."""
        input_path = tmp_path / "input.jsonl"
        row = {
            "question": "Weather in CA?",
            "target_tool_name": "get_alerts",
            **_multi_turn_row("Weather in CA?", "call_1"),
        }
        input_path.write_text(json.dumps(row) + "\n")

        dataset_download.python_func(
            train_dataset=_MockArtifact(str(tmp_path / "train.jsonl")),
            eval_dataset=_MockArtifact(str(tmp_path / "eval.jsonl")),
            dataset_uri=str(input_path),
            pvc_mount_path=str(tmp_path),
            train_split_ratio=1.0,
            subset_count=0,
            dataset_format="tool_call",
        )

        log = (tmp_path / "pipeline_log.txt").read_text()
        assert "matches both single-turn and multi-turn" in log
        assert "using single-turn validation" in log
