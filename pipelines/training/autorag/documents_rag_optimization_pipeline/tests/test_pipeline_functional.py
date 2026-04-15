"""Parametrized functional tests for Documents RAG Optimization pipeline on RHOAI.

These tests require a Red Hat OpenShift AI (RHOAI) cluster with Data Science Pipelines
enabled, and environment variables set for cluster URL, credentials, and pipeline
parameters. When not set, tests are skipped. See .env.example for required variables.

Test scenarios are defined in test_configs.json and loaded via test_configs.py. Each
scenario specifies pipeline parameter overrides and an expected result (pass or fail).
Filter scenarios by tags with RHOAI_TEST_CONFIG_TAGS (e.g. smoke, milvus-lite).

Passing criteria for expected-pass tests (from RHAIENG-4142):
- Pipeline run finishes with status success
- At least 1 pattern is generated
- All desired artifacts exist (indexing notebook, inference notebook, evaluation_results.json, pattern.json)
- Notebooks can be run and complete successfully (via papermill)
"""

import logging
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest


logger = logging.getLogger(__name__)


def _session_docrag_functional_config():
    """Lazy import so import-guard allows only stdlib at module scope."""
    from .integration_config import DOCRAG_FUNCTIONAL_CONFIG

    return DOCRAG_FUNCTIONAL_CONFIG


def _session_configs_for_run():
    """Lazy import to load test configs."""
    from .test_configs import get_test_configs_for_run

    return get_test_configs_for_run()


# Module-level constants for skipif and parametrize
DOCRAG_FUNCTIONAL_CONFIG = _session_docrag_functional_config()
CONFIGS_FOR_RUN = _session_configs_for_run()

# Pipeline display name in KFP (from pipeline decorator)
PIPELINE_DISPLAY_NAME = "documents-rag-optimization-pipeline"

# Shorter timeout for expected-fail tests (failures should surface quickly)
_EXPECTED_FAIL_TIMEOUT_CAP = 600


def _make_docrag_run_name():
    """Return a run name: docrag-func-<6 hex chars>-<YYYYMMDD-HHMMSS>."""
    hex_part = secrets.token_hex(3)
    time_part = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"docrag-func-{hex_part}-{time_part}"


def _run_pipeline_and_wait(client, compiled_path, arguments, timeout):
    """Submit pipeline run and wait for completion; return run_id and run detail."""
    run_name = _make_docrag_run_name()
    run = client.create_run_from_pipeline_package(
        compiled_path,
        arguments=arguments,
        run_name=run_name,
        enable_caching=False,
    )
    run_id = run.run_id
    detail = client.wait_for_run_completion(run_id, timeout=timeout)
    return run_id, detail


def _get_run_state(detail):
    """Extract the run state string from a run detail object."""
    run = getattr(detail, "run", detail)
    state = getattr(run, "state", None)
    if state is None and hasattr(run, "status"):
        state = getattr(run.status, "state", None)
    return state.upper() if isinstance(state, str) else None


def _run_succeeded(detail):
    """Return True if the run finished successfully."""
    return _get_run_state(detail) == "SUCCEEDED"


def _run_failed(detail):
    """Return True if the run finished with FAILED state (not timeout or running)."""
    return _get_run_state(detail) == "FAILED"


def _collect_failure_details(client, run_id, config=None):
    """Collect per-task failure details from a failed pipeline run.

    Queries the KFP v2 API for task-level status. task_details is a list of
    task objects. Pod names are in child_tasks (not on the task itself).
    Optionally fetches pod logs for failed tasks via the Kubernetes API,
    using the RHOAI token from config for authentication.

    Args:
        client: KFP client instance.
        run_id: The pipeline run ID.
        config: Optional functional config dict (from get_docrag_functional_config).
            If provided, its rhoai_token and rhoai_kfp_url are used for k8s auth.

    Returns:
        Formatted string with failure details.
    """
    lines = [f"\n{'=' * 80}", f"FAILURE DETAILS FOR RUN: {run_id}", "=" * 80]
    failed_pod_names = []

    # --- Run-level and task-level details from KFP v2 API ---
    try:
        run_detail = client.get_run(run_id)
        run_obj = getattr(run_detail, "run", run_detail)

        # Run-level error
        run_error = getattr(run_obj, "error", None)
        if run_error:
            error_msg = getattr(run_error, "message", str(run_error))
            lines.append(f"\nRUN ERROR: {error_msg}")

        # Task-level details (DSP v2: run_details.task_details is a list)
        rd = getattr(run_obj, "run_details", None)
        task_list = getattr(rd, "task_details", None) if rd else None

        if task_list:
            # Filter out internal tasks (drivers, executors, root) for cleaner output
            _INTERNAL_SUFFIXES = ("-driver",)
            _INTERNAL_NAMES = ("root", "executor")

            for task in task_list:
                name = getattr(task, "display_name", None) or getattr(task, "task_id", "?")
                state = getattr(task, "state", None)
                state_str = str(state).upper() if state else "NOT_STARTED"

                # Skip internal/infrastructure tasks for readability
                if name in _INTERNAL_NAMES or any(name.endswith(s) for s in _INTERNAL_SUFFIXES):
                    # Still collect pod names from failed internal tasks
                    if state_str in ("FAILED", "ERROR"):
                        failed_pod_names.extend(_get_child_pod_names(task))
                    continue

                # Collect pod names from child_tasks (pod_name on task itself is always None)
                child_pods = _get_child_pod_names(task)

                if state_str in ("FAILED", "ERROR", "SYSTEM_ERROR"):
                    lines.append(f"\nFAILED TASK: {name}")
                    lines.append(f"  State: {state_str}")

                    task_error = getattr(task, "error", None)
                    if task_error:
                        error_msg = getattr(task_error, "message", str(task_error))
                        lines.append(f"  Error: {error_msg}")

                    if child_pods:
                        lines.append(f"  Pods: {', '.join(child_pods)}")
                        failed_pod_names.extend(child_pods)

                    start = getattr(task, "start_time", None)
                    end = getattr(task, "end_time", None)
                    if start and end:
                        lines.append(f"  Duration: {start} -> {end}")
                else:
                    lines.append(f"  TASK: {name} — {state_str}")
        else:
            lines.append("\n[No task_details in run response]")
    except Exception as e:
        lines.append(f"\n[Could not fetch run details from KFP API: {e}]")

    # --- Pod logs via Kubernetes API ---
    if failed_pod_names:
        try:
            namespace = getattr(client, "_namespace", None)
            token = config.get("rhoai_token") if config else None
            kfp_url = config.get("rhoai_kfp_url") if config else None
            _append_pod_logs(namespace, failed_pod_names, lines, token=token, kfp_url=kfp_url)
        except Exception as e:
            lines.append(f"\n[Could not fetch pod logs: {e}]")
    else:
        lines.append("\n[No failed pod names found in task details]")

    lines.append("=" * 80)
    return "\n".join(lines)


def _get_child_pod_names(task):
    """Extract pod names from a task's child_tasks list."""
    child_tasks = getattr(task, "child_tasks", None)
    if not child_tasks:
        return []
    pods = []
    for child in child_tasks:
        pod = child.get("pod_name") if isinstance(child, dict) else getattr(child, "pod_name", None)
        if pod:
            pods.append(pod)
    return pods


def _derive_k8s_api_url(kfp_url):
    """Derive OpenShift API server URL from a KFP route URL.

    Standard OCP: https://<route>.apps.<cluster-domain> -> https://api.<cluster-domain>:6443
    ROSA:         https://<route>.apps.rosa.<cluster-domain> -> https://api.<cluster-domain>:6443
    """
    from urllib.parse import urlparse

    hostname = urlparse(kfp_url).hostname or ""
    apps_idx = hostname.find(".apps.")
    if apps_idx < 0:
        return None
    base_domain = hostname[apps_idx + len(".apps.") :]
    # ROSA clusters insert "rosa." between "apps." and the cluster domain
    if base_domain.startswith("rosa."):
        base_domain = base_domain[len("rosa.") :]
    return f"https://api.{base_domain}:6443"


def _append_pod_logs(namespace, pod_names, lines, token=None, kfp_url=None):
    """Fetch pod logs for specific pods and append to lines.

    Authentication priority:
    1. In-cluster config (ServiceAccount)
    2. RHOAI bearer token with API URL derived from RHOAI_KFP_URL (OpenShift)
    3. Local kubeconfig (~/.kube/config)
    """
    try:
        from kubernetes import client as k8s_client
        from kubernetes import config as k8s_config
    except ImportError:
        lines.append("\n[kubernetes package not installed; skipping pod log fetch]")
        return

    api_client = None

    # 1. Try in-cluster config
    try:
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        # 2. Try RHOAI token with API URL derived from KFP URL
        api_url = _derive_k8s_api_url(kfp_url) if kfp_url else None
        if token and api_url:
            verify_ssl = os.environ.get("KFP_VERIFY_SSL", "true").strip().lower()
            verify_ssl = verify_ssl not in ("0", "false", "no")

            configuration = k8s_client.Configuration()
            configuration.host = api_url
            configuration.api_key = {"authorization": f"Bearer {token}"}
            configuration.verify_ssl = verify_ssl
            if not verify_ssl:
                import urllib3

                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            api_client = k8s_client.ApiClient(configuration)
        else:
            # 3. Try local kubeconfig
            try:
                k8s_config.load_kube_config()
            except k8s_config.ConfigException:
                lines.append(
                    "\n[No kubernetes config found (not in-cluster, could not derive "
                    "API URL from KFP URL, no ~/.kube/config); skipping pod log fetch]"
                )
                return

    api = k8s_client.CoreV1Api(api_client=api_client)
    ns = namespace or "default"

    for pod_name in pod_names:
        lines.append(f"\n--- Pod logs: {pod_name} ---")
        try:
            pod = api.read_namespaced_pod(name=pod_name, namespace=ns)
            containers = [c.name for c in (pod.spec.containers or [])]
        except Exception as e:
            lines.append(f"  [Could not read pod: {e}]")
            continue

        for container_name in containers:
            try:
                log = api.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=ns,
                    container=container_name,
                    tail_lines=100,
                )
                lines.append(f"[container: {container_name}]")
                lines.append(log if log else "(empty)")
            except Exception as e:
                lines.append(f"[container: {container_name}] error: {e}")


def _validate_artifacts_in_s3(s3_client, bucket, prefix):
    """List and categorize S3 artifacts under prefix.

    Returns:
        Dict with keys: "pattern_keys", "indexing_notebook_keys", "inference_notebook_keys",
        "evaluation_results_keys", "leaderboard_keys", "responses_body_keys", "all_keys".
    """
    result = {
        "pattern_keys": [],
        "indexing_notebook_keys": [],
        "inference_notebook_keys": [],
        "evaluation_results_keys": [],
        "leaderboard_keys": [],
        "responses_body_keys": [],
        "all_keys": [],
    }
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents") or []:
                key = obj["Key"]
                result["all_keys"].append(key)
                lower_key = key.lower()
                if key.endswith("pattern.json") or "rag_patterns" in lower_key:
                    result["pattern_keys"].append(key)
                if key.endswith(".ipynb") and "indexing" in lower_key:
                    result["indexing_notebook_keys"].append(key)
                if key.endswith(".ipynb") and "inference" in lower_key:
                    result["inference_notebook_keys"].append(key)
                if "evaluation_results.json" in key:
                    result["evaluation_results_keys"].append(key)
                if "leaderboard" in lower_key or key.endswith(".html"):
                    result["leaderboard_keys"].append(key)
                if "v1_responses_body.json" in key:
                    result["responses_body_keys"].append(key)
    except Exception:
        pass
    return result


def _download_and_execute_notebooks(s3_client, bucket, notebook_keys):
    """Download notebooks from S3 and execute them via papermill.

    Args:
        s3_client: Boto3 S3 client.
        bucket: S3 bucket name.
        notebook_keys: List of S3 keys pointing to .ipynb files.

    Raises:
        AssertionError: If any notebook fails execution.
    """
    import papermill as pm

    errors = []
    with tempfile.TemporaryDirectory(prefix="docrag-nb-") as tmpdir:
        for key in notebook_keys:
            filename = Path(key).name
            input_path = Path(tmpdir) / f"input_{filename}"
            output_path = Path(tmpdir) / f"output_{filename}"

            s3_client.download_file(bucket, key, str(input_path))

            try:
                pm.execute_notebook(
                    str(input_path),
                    str(output_path),
                    kernel_name="python3",
                )
            except pm.PapermillExecutionError as e:
                errors.append(f"Notebook {filename} (key={key}) failed: {e}")
            except Exception as e:
                errors.append(f"Notebook {filename} (key={key}) execution error: {e}")

    if errors:
        raise AssertionError("Notebook execution failures:\n" + "\n".join(errors))


@pytest.mark.functional
@pytest.mark.skipif(
    DOCRAG_FUNCTIONAL_CONFIG is None,
    reason="RHOAI functional test env not set (set RHOAI_KFP_URL, RHOAI_TOKEN, pipeline params; see .env.example)",
)
@pytest.mark.parametrize("test_config", CONFIGS_FOR_RUN, ids=[c.id for c in CONFIGS_FOR_RUN])
class TestDocumentsRagOptimizationFunctional:
    """Functional tests running parametrized pipeline scenarios on RHOAI."""

    def test_docrag_pipeline_with_config(
        self,
        test_config,
        docrag_functional_config,
        functional_kfp_client,
        compiled_pipeline_path,
        pipeline_run_timeout,
        functional_s3_client,
    ):
        """Run pipeline for one test config; validate based on expected result.

        For expected-pass scenarios: assert success, validate artifacts, execute notebooks.
        For expected-fail scenarios: assert the pipeline run fails (not succeeds).
        """
        if not functional_kfp_client:
            pytest.skip("Functional test prerequisites not available")

        config = docrag_functional_config
        arguments = test_config.get_pipeline_arguments(config)

        timeout = pipeline_run_timeout
        if test_config.expected_result == "fail":
            timeout = min(timeout, _EXPECTED_FAIL_TIMEOUT_CAP)

        run_id, detail = _run_pipeline_and_wait(
            functional_kfp_client,
            compiled_pipeline_path,
            arguments,
            timeout,
        )

        if test_config.expected_result == "fail":
            state = _get_run_state(detail)
            assert not _run_succeeded(detail), (
                f"[{test_config.id}] Pipeline run {run_id} was expected to FAIL but succeeded"
            )
            assert _run_failed(detail), (
                f"[{test_config.id}] Pipeline run {run_id} expected state FAILED but got {state}"
            )
            # Log failure details for observability even on expected failures
            logger.info(_collect_failure_details(functional_kfp_client, run_id, config=config))
            return

        # Expected pass: validate success
        if not _run_succeeded(detail):
            failure_info = _collect_failure_details(functional_kfp_client, run_id, config=config)
            pytest.fail(
                f"[{test_config.id}] Pipeline run {run_id} was expected to PASS but failed; "
                f"state={_get_run_state(detail)}"
                f"{failure_info}"
            )

        # Artifact validation (requires S3 config)
        if not functional_s3_client or not config.get("s3_bucket_artifacts"):
            return

        artifact_bucket = config["s3_bucket_artifacts"]
        prefix = f"{PIPELINE_DISPLAY_NAME}/{run_id}"
        artifacts = _validate_artifacts_in_s3(functional_s3_client, artifact_bucket, prefix)

        assert len(artifacts["pattern_keys"]) >= 1, (
            f"[{test_config.id}] Expected at least 1 pattern artifact under {prefix}; "
            f"found {artifacts['pattern_keys']}"
        )
        assert len(artifacts["indexing_notebook_keys"]) >= 1, (
            f"[{test_config.id}] Expected at least 1 indexing notebook under {prefix}; "
            f"found {artifacts['indexing_notebook_keys']}"
        )
        assert len(artifacts["inference_notebook_keys"]) >= 1, (
            f"[{test_config.id}] Expected at least 1 inference notebook under {prefix}; "
            f"found {artifacts['inference_notebook_keys']}"
        )
        assert len(artifacts["evaluation_results_keys"]) >= 1, (
            f"[{test_config.id}] Expected evaluation_results.json under {prefix}; "
            f"found {artifacts['evaluation_results_keys']}"
        )

        # Notebook execution validation
        all_notebook_keys = artifacts["indexing_notebook_keys"] + artifacts["inference_notebook_keys"]
        _download_and_execute_notebooks(functional_s3_client, artifact_bucket, all_notebook_keys)
