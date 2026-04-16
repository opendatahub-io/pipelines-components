"""Unit tests for the shared training utilities."""

import logging
from dataclasses import dataclass
from datetime import datetime
from unittest import mock

import pytest

from ..training import _log_job_details, wait_for_training_job


@dataclass
class FakeStep:
    """Fake TrainJob step for testing."""

    name: str
    pod_name: str
    status: str = "Running"


@dataclass
class FakeTrainJob:
    """Fake TrainJob object for testing."""

    name: str
    status: str
    steps: list
    creation_timestamp: datetime = None


@pytest.fixture
def log():
    """Create a test logger."""
    return logging.getLogger("test_training")


@pytest.fixture
def mock_client():
    """Create a mock TrainerClient with a test namespace."""
    client = mock.MagicMock()
    client.backend.namespace = "test-ns"
    return client


def _make_train_job(status="Running", steps=None, name="test-job"):
    if steps is None:
        steps = [FakeStep(name="node-0", pod_name="test-job-node-0-abc")]
    return FakeTrainJob(
        name=name,
        status=status,
        steps=steps,
        creation_timestamp=datetime(2026, 3, 11, 10, 0, 0),
    )


class TestSharedTrainingUnitTests:
    """Unit tests for shared training utility functions."""

    def test_log_job_details_logs_job_info(self, mock_client, log):
        """Test that job name, namespace, and status are logged."""
        train_job = _make_train_job()
        with mock.patch.object(log, "info") as mock_info:
            _log_job_details(mock_client, train_job, log, has_node_0=True)
            messages = [str(c) for c in mock_info.call_args_list]
            assert any("test-job" in m and "test-ns" in m and "Running" in m for m in messages)
            assert any("2026" in m for m in messages)

    def test_log_job_details_logs_pod_names_and_kubectl(self, mock_client, log):
        """Test that pod names and kubectl commands are logged."""
        train_job = _make_train_job(
            steps=[
                FakeStep(name="node-0", pod_name="job-node-0-abc"),
                FakeStep(name="node-1", pod_name="job-node-1-def"),
            ]
        )
        with mock.patch.object(log, "info") as mock_info:
            _log_job_details(mock_client, train_job, log, has_node_0=True)
            messages = [str(c) for c in mock_info.call_args_list]
            assert any("job-node-0-abc" in m for m in messages)
            assert any("job-node-1-def" in m for m in messages)
            assert any("kubectl -n test-ns logs job-node-0-abc -f" in m for m in messages)

    def test_log_job_details_logs_no_pods_when_steps_empty(self, mock_client, log):
        """Test that a 'no pods' message is logged when steps are empty."""
        train_job = _make_train_job(steps=[])
        with mock.patch.object(log, "info") as mock_info:
            _log_job_details(mock_client, train_job, log, has_node_0=False)
            messages = [str(c) for c in mock_info.call_args_list]
            assert any("No training pods found yet" in m for m in messages)

    def test_log_job_details_shows_streaming_message_when_has_node_0(self, mock_client, log):
        """Test that streaming message appears when node-0 exists."""
        train_job = _make_train_job()
        with mock.patch.object(log, "info") as mock_info:
            _log_job_details(mock_client, train_job, log, has_node_0=True)
            messages = [str(c) for c in mock_info.call_args_list]
            assert any("Streaming logs for node-0" in m for m in messages)

    def test_log_job_details_hides_streaming_message_when_no_node_0(self, mock_client, log):
        """Test that streaming message is absent when node-0 is missing."""
        train_job = _make_train_job(steps=[FakeStep(name="worker-0", pod_name="w-pod")])
        with mock.patch.object(log, "info") as mock_info:
            _log_job_details(mock_client, train_job, log, has_node_0=False)
            messages = [str(c) for c in mock_info.call_args_list]
            assert not any("Streaming logs for node-0" in m for m in messages)

    def test_wait_happy_path_streams_logs_and_completes(self, mock_client, log):
        """Test that logs are streamed and job completes successfully."""
        running_job = _make_train_job(status="Running")
        complete_job = _make_train_job(status="Complete")
        mock_client.get_job.side_effect = [running_job, complete_job]
        mock_client.get_job_logs.return_value = iter(["epoch 1", "epoch 2", "done"])

        wait_for_training_job(mock_client, "test-job", log)

        assert mock_client.wait_for_job_status.call_count == 2
        mock_client.wait_for_job_status.assert_any_call(name="test-job", status={"Running"}, timeout=900)
        mock_client.wait_for_job_status.assert_any_call(name="test-job", status={"Complete", "Failed"}, timeout=1800)
        mock_client.get_job_logs.assert_called_once_with(name="test-job", step="node-0", follow=True)

    def test_wait_streaming_retries_then_succeeds(self, mock_client, log):
        """Test that log streaming retries on failure and succeeds."""
        running_job = _make_train_job(status="Running")
        complete_job = _make_train_job(status="Complete")
        mock_client.get_job.side_effect = [running_job, complete_job]
        mock_client.get_job_logs.side_effect = [
            RuntimeError("container creating"),
            iter(["log line"]),
        ]

        with mock.patch("components.training.finetuning.shared.training.time.sleep"):
            wait_for_training_job(mock_client, "test-job", log)

        assert mock_client.get_job_logs.call_count == 2
        # Should always wait for final status
        assert mock_client.wait_for_job_status.call_count == 2

    def test_wait_streaming_fails_all_retries_falls_back_to_polling(self, mock_client, log):
        """Test that polling is used when all streaming retries fail."""
        running_job = _make_train_job(status="Running")
        complete_job = _make_train_job(status="Complete")
        mock_client.get_job.side_effect = [running_job, complete_job]
        mock_client.get_job_logs.side_effect = RuntimeError("container not ready")

        with mock.patch("components.training.finetuning.shared.training.time.sleep"):
            wait_for_training_job(mock_client, "test-job", log)

        assert mock_client.get_job_logs.call_count == 3
        # Should have fallen back to polling for Complete/Failed
        assert mock_client.wait_for_job_status.call_count == 2
        mock_client.wait_for_job_status.assert_any_call(name="test-job", status={"Complete", "Failed"}, timeout=1800)

    def test_wait_node_0_not_found_skips_streaming(self, mock_client, log):
        """Test that streaming is skipped when node-0 step is absent."""
        job_no_node0 = _make_train_job(
            status="Running",
            steps=[FakeStep(name="worker-0", pod_name="w-pod")],
        )
        complete_job = _make_train_job(status="Complete")
        mock_client.get_job.side_effect = [job_no_node0, complete_job]

        wait_for_training_job(mock_client, "test-job", log)

        mock_client.get_job_logs.assert_not_called()
        # Should fall back to polling
        assert mock_client.wait_for_job_status.call_count == 2

    def test_wait_get_job_fails_skips_logging_and_streaming(self, mock_client, log):
        """Test that a get_job failure skips logging and streaming."""
        complete_job = _make_train_job(status="Complete")
        mock_client.get_job.side_effect = [
            RuntimeError("API error"),
            complete_job,
        ]

        with mock.patch.object(log, "warning") as mock_warning:
            wait_for_training_job(mock_client, "test-job", log)
            warnings = [str(c) for c in mock_warning.call_args_list]
            # Should warn about the API error, not about missing node-0
            assert any("Could not retrieve TrainJob details" in w for w in warnings)
            assert not any("node-0 step not found" in w for w in warnings)

        mock_client.get_job_logs.assert_not_called()
        # Should fall back to polling
        assert mock_client.wait_for_job_status.call_count == 2

    @pytest.mark.parametrize(
        "status,match",
        [
            ("Failed", "Job failed"),
            ("Suspended", "Unexpected status"),
        ],
    )
    def test_wait_job_bad_status_raises_runtime_error(self, mock_client, log, status, match):
        """Test that Failed or unexpected status raises RuntimeError."""
        running_job = _make_train_job(status="Running")
        bad_job = _make_train_job(status=status)
        mock_client.get_job.side_effect = [running_job, bad_job]
        mock_client.get_job_logs.return_value = iter([])

        with pytest.raises(RuntimeError, match=match):
            wait_for_training_job(mock_client, "test-job", log)
