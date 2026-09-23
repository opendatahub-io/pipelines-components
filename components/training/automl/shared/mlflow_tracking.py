"""MLflow tracking helpers for AutoML pipeline components.

Reads the platform-native ``KFP_MLFLOW_CONFIG`` JSON blob that Kubeflow/RHOAI injects
into every pipeline step. No custom connection secret is required: the tracking URI,
parent run, experiment, and workspace all come from that blob, and authentication uses
the pod's mounted Kubernetes service-account token.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

logger = logging.getLogger(__name__)

MlflowMode = Literal["disabled", "kfp"]

# Env var holding the platform-injected MLflow config (JSON). Set by the KFP MLflow
# integration on every step; absent when the platform integration is not enabled.
KFP_MLFLOW_CONFIG_ENV = "KFP_MLFLOW_CONFIG"

# Mounted service-account token used to authenticate to the MLflow server when
# ``authType`` is ``kubernetes``. MLflow sends it as an ``Authorization: Bearer`` header
# via ``MLFLOW_TRACKING_TOKEN`` (the runtime image's MLflow has no built-in kubernetes
# auth provider, so we populate the token ourselves).
SERVICE_ACCOUNT_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
KUBERNETES_AUTH_TYPE = "kubernetes"

OPTIONAL_METRIC_ARTIFACTS = (
    "feature_importance.json",
    "confusion_matrix.json",
    "curves.json",
    "back_testing.json",
)

# Redundant per-model metrics excluded from child-run logging.
# Leaving it empty for now, can be used to reduced noise
METRIC_EXCLUDE_KEYS: frozenset[str] = frozenset({})

RUN_TYPE_PIPELINE = "pipeline"
RUN_TYPE_MODEL = "model"
MLFLOW_PARENT_RUN_ID_TAG = "mlflow.parentRunId"

# Relative locations inside each ``<model>_FULL`` directory of the models artifact.
MODEL_PREDICTOR_SUBDIR = "predictor"
MODEL_NOTEBOOK_RELPATH = "notebooks/automl_predictor_notebook.ipynb"

# HTTP header RHOAI's multi-tenant MLflow requires to scope every request to a workspace
# (the project namespace, e.g. "ns-automl-benchmarking"). Sent from MLFLOW_WORKSPACE.
MLFLOW_WORKSPACE_HEADER = "x-mlflow-workspace"


@dataclass(frozen=True)
class MlflowConfig:
    """Resolved MLflow settings for the current pod, parsed from ``KFP_MLFLOW_CONFIG``."""

    mode: MlflowMode
    tracking_uri: str
    experiment_id: str = ""
    run_id: str = ""
    workspace: str = ""
    auth_type: str = ""
    timeout: str = ""


def resolve_mlflow_config() -> MlflowConfig | None:
    """Resolve MLflow config from the platform-injected ``KFP_MLFLOW_CONFIG`` blob.

    Returns ``None`` (tracking disabled) when the env var is absent, is not valid JSON,
    or lacks an ``endpoint``. The blob has the shape::

        {
            "endpoint": "...",
            "workspacesEnabled": true,
            "workspace": "ns-...",
            "parentRunId": "...",
            "experimentId": "2",
            "authType": "kubernetes",
            "timeout": "30s",
        }
    """
    raw = os.getenv(KFP_MLFLOW_CONFIG_ENV, "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("%s is set but is not valid JSON; MLflow logging disabled.", KFP_MLFLOW_CONFIG_ENV)
        return None
    if not isinstance(data, dict):
        logger.warning("%s is not a JSON object; MLflow logging disabled.", KFP_MLFLOW_CONFIG_ENV)
        return None

    tracking_uri = str(data.get("endpoint", "")).strip()
    if not tracking_uri:
        logger.warning("%s has no 'endpoint'; MLflow logging disabled.", KFP_MLFLOW_CONFIG_ENV)
        return None

    auth_type = str(data.get("authType", "")).strip()
    # Kubernetes auth attaches a ServiceAccount bearer token to every request. Refuse to send it
    # over a non-HTTPS endpoint, which would leak the token in cleartext (CWE-319). Disable
    # tracking rather than risk the credential.
    if auth_type == KUBERNETES_AUTH_TYPE and not tracking_uri.lower().startswith("https://"):
        logger.warning(
            "%s uses '%s' auth but 'endpoint' is not HTTPS; refusing to send bearer token over "
            "cleartext. MLflow logging disabled.",
            KFP_MLFLOW_CONFIG_ENV,
            KUBERNETES_AUTH_TYPE,
        )
        return None

    # Only scope requests to a workspace when the server has workspaces enabled.
    workspace = str(data.get("workspace", "")).strip() if data.get("workspacesEnabled") else ""
    return MlflowConfig(
        mode="kfp",
        tracking_uri=tracking_uri,
        experiment_id=str(data.get("experimentId", "")).strip(),
        run_id=str(data.get("parentRunId", "")).strip(),
        workspace=workspace,
        auth_type=auth_type,
        timeout=str(data.get("timeout", "")).strip(),
    )


def is_mlflow_enabled() -> bool:
    """Return True when a MLflow tracking URI is available."""
    return resolve_mlflow_config() is not None


def read_mlflow_env() -> dict[str, str]:
    """Collect MLflow-related environment variables from the pod."""
    config = resolve_mlflow_config()
    if config is None:
        return {
            "mlflow_tracking_uri": "",
            "mlflow_experiment_id": "",
            "mlflow_run_id": "",
            "mlflow_workspace": "",
            "mlflow_auth_type": "",
            "tracking_mode": "disabled",
        }
    return {
        "mlflow_tracking_uri": config.tracking_uri,
        "mlflow_experiment_id": config.experiment_id,
        "mlflow_run_id": config.run_id,
        "mlflow_workspace": config.workspace,
        "mlflow_auth_type": config.auth_type,
        "tracking_mode": config.mode,
    }


def build_mlflow_run_url(tracking_uri: str, experiment_id: str, run_id: str) -> str:
    """Build a deep-link URL to the MLflow UI parent run view."""
    base = tracking_uri.rstrip("/")
    if not base or not experiment_id or not run_id:
        return ""
    return f"{base}/#/experiments/{experiment_id}/runs/{run_id}"


def resolve_leaderboard_html_path(html_artifact_path: str | Path) -> Path | None:
    """Resolve the leaderboard HTML file from a KFP ``dsl.HTML`` artifact path.

    KFP may mount the artifact as a file path or as a directory containing the HTML.
    """
    path = Path(html_artifact_path)
    if path.is_file():
        return path
    if path.is_dir():
        for candidate in (path / "index.html", path / "leaderboard.html"):
            if candidate.is_file():
                return candidate
        html_files = sorted(path.glob("*.html"))
        if len(html_files) == 1:
            return html_files[0]
    return None


def build_mlflow_stage_map_block(
    *,
    tracking_uri: str | None = None,
    experiment_id: str | None = None,
    run_id: str | None = None,
    workspace: str | None = None,
) -> dict[str, Any]:
    """Build the ``mlflow`` object embedded in ``component_stage_map.json`` (ADR schema).

    Values default to the resolved ``KFP_MLFLOW_CONFIG`` blob; explicit arguments override.
    """
    config = resolve_mlflow_config()
    uri = (tracking_uri if tracking_uri is not None else (config.tracking_uri if config else "")).strip()
    if not uri:
        return {"tracking_enabled": False}

    exp_id = (experiment_id if experiment_id is not None else (config.experiment_id if config else "")).strip()
    parent_run_id = (run_id if run_id is not None else (config.run_id if config else "")).strip()
    ws = (workspace if workspace is not None else (config.workspace if config else "")).strip()

    block: dict[str, Any] = {
        "tracking_enabled": True,
        "tracking_uri": uri,
    }
    if exp_id:
        block["experiment_id"] = exp_id
    if parent_run_id:
        block["run_id"] = parent_run_id
    if ws:
        block["workspace"] = ws
    run_url = build_mlflow_run_url(uri, exp_id, parent_run_id)
    if run_url:
        block["run_url"] = run_url
    return block


def configure_mlflow_client(mlflow: Any, config: MlflowConfig) -> None:
    """Apply authentication, tracking URI, and workspace before MLflow API calls."""
    if config.auth_type == KUBERNETES_AUTH_TYPE:
        _apply_kubernetes_auth()
    timeout_seconds = _parse_timeout_seconds(config.timeout)
    if timeout_seconds is not None:
        # MLflow reads MLFLOW_HTTP_REQUEST_TIMEOUT as a whole-second integer.
        os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = str(timeout_seconds)
    mlflow.set_tracking_uri(config.tracking_uri)
    if config.workspace:
        _apply_workspace(mlflow, config.tracking_uri, config.workspace)


def _apply_kubernetes_auth() -> None:
    """Authenticate to MLflow with the pod's Kubernetes service-account token.

    MLflow reads ``MLFLOW_TRACKING_TOKEN`` and sends it as an ``Authorization: Bearer``
    header. Best-effort: a missing/unreadable token leaves auth unset and is surfaced by
    the eventual request failure rather than crashing here.
    """
    token_path = Path(SERVICE_ACCOUNT_TOKEN_PATH)
    try:
        token = token_path.read_text(encoding="utf-8").strip()
    except OSError:
        logger.warning(
            "authType=kubernetes but service-account token is not readable at %s; "
            "MLflow requests will be unauthenticated.",
            token_path,
        )
        return
    if not token:
        logger.warning("Service-account token at %s is empty; MLflow requests will be unauthenticated.", token_path)
        return
    os.environ["MLFLOW_TRACKING_TOKEN"] = token


def _apply_workspace(mlflow: Any, tracking_uri: str, workspace: str) -> None:
    """Route the workspace to the MLflow server on every request.

    RHOAI's multi-tenant MLflow rejects any call that does not identify a workspace
    (``INVALID_PARAMETER_VALUE: Workspace context is required``). Newer MLflow clients
    expose ``mlflow.set_workspace``; upstream/plain MLflow does not, so we attach the
    ``x-mlflow-workspace`` header on the underlying ``requests`` session instead.
    Best-effort: a failure here leaves tracking disabled but never fails the pipeline.
    """
    set_workspace = getattr(mlflow, "set_workspace", None)
    if callable(set_workspace):
        set_workspace(workspace)
        return
    try:
        _install_workspace_request_header(tracking_uri, workspace)
    except Exception:
        logger.exception("Failed to install MLflow workspace request header for %r.", workspace)


def _install_workspace_request_header(tracking_uri: str, workspace: str) -> None:
    """Monkeypatch ``requests`` so calls to the tracking host carry the workspace header.

    Scoped to the tracking host so it never affects artifact stores (e.g. S3) on other
    hosts. Idempotent: re-applying in the same process is a no-op.
    """
    from urllib.parse import urlsplit

    import requests

    host = urlsplit(tracking_uri).netloc
    if not host:
        return

    original_request = requests.Session.request
    if getattr(original_request, "_automl_workspace_patch", False):
        return

    def request_with_workspace(self, method, url, *args, **kwargs):
        try:
            if urlsplit(url).netloc == host:
                headers = dict(kwargs.get("headers") or {})
                headers.setdefault(MLFLOW_WORKSPACE_HEADER, workspace)
                kwargs["headers"] = headers
        except Exception:
            logger.debug("Could not attach MLflow workspace header", exc_info=True)
        return original_request(self, method, url, *args, **kwargs)

    request_with_workspace._automl_workspace_patch = True
    requests.Session.request = request_with_workspace


@contextmanager
def parent_mlflow_run(mlflow: Any, config: MlflowConfig, *, fallback_name: str = "") -> Iterator[Any]:
    """Open the parent run for nested logging.

    When the platform provides a ``parentRunId`` (the native RHOAI pipeline-submission
    path), that run is resumed. When it does not -- e.g. the AutoML tech-preview launcher,
    which has no MLflow experiment picker -- a parent run is created instead, under an
    experiment named ``fallback_name`` (typically the KFP run name), get-or-created when the
    platform also supplied no ``experimentId``.
    """
    configure_mlflow_client(mlflow, config)
    if config.run_id:
        with mlflow.start_run(run_id=config.run_id) as run:
            yield run
        return

    # No platform parent run: create our own. Bind an experiment first so the run does not
    # land in MLflow's Default experiment when the platform gave us no experiment id.
    if not config.experiment_id and fallback_name:
        try:
            mlflow.set_experiment(fallback_name)
        except Exception:
            logger.exception("Could not set/create MLflow experiment %r; using the default.", fallback_name)
    start_kwargs: dict[str, Any] = {}
    if config.experiment_id:
        start_kwargs["experiment_id"] = config.experiment_id
    if fallback_name:
        start_kwargs["run_name"] = fallback_name
    with mlflow.start_run(**start_kwargs) as run:
        yield run


def parse_model_name(model_name: str) -> tuple[str, int]:
    """Extract model family and stack level from an AutoGluon model name."""
    model_type = model_name.split("_")[0] if "_" in model_name else model_name
    stack_level = 1
    if "_L" in model_name:
        suffix = model_name.rsplit("_L", maxsplit=1)[-1]
        level_part = suffix.split("_", maxsplit=1)[0]
        if level_part.isdigit():
            stack_level = int(level_part)
    return model_type, stack_level


def display_model_run_name(model_name: str) -> str:
    """Return a concise MLflow run name for a refitted AutoGluon model."""
    if model_name.endswith("_FULL"):
        return model_name[: -len("_FULL")]
    return model_name


def _normalize_model_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Flatten artifact metadata that nests scores under ``test_data``."""
    test_data = metrics.get("test_data")
    if isinstance(test_data, dict) and test_data:
        return dict(test_data)
    return dict(metrics)


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    logged: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            logged[key] = float(value)
        elif hasattr(value, "item"):
            logged[key] = float(value)
    return logged


def _metrics_for_task(task_type: str, metrics: dict[str, Any]) -> dict[str, float]:
    """Select scalar metrics to log on a model child run.

    Logs every finite scalar metric AutoGluon computed for the model, minus the redundant
    variants in :data:`METRIC_EXCLUDE_KEYS`. ``task_type`` is retained for API stability and
    to allow future per-task exclusions.
    """
    normalized = _normalize_model_metrics(metrics)
    scalars = _scalar_metrics(normalized)
    return {key: value for key, value in scalars.items() if key not in METRIC_EXCLUDE_KEYS}


def _stringify_params(params: dict[str, Any]) -> dict[str, str]:
    """MLflow params must be strings."""
    return {key: str(value) for key, value in params.items()}


def _resolve_autogluon_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        for package in ("autogluon.tabular", "autogluon.core", "autogluon"):
            try:
                return version(package)
            except PackageNotFoundError:
                continue
    except Exception:
        logger.debug("Could not resolve autogluon version from package metadata", exc_info=True)
    try:
        import autogluon

        return getattr(autogluon, "__version__", "unknown")
    except Exception:
        logger.debug("Could not import autogluon for version lookup", exc_info=True)
    return "unknown"


def _resolve_kfp_version() -> str:
    """Best-effort KFP SDK version for the parent run, or ``"unknown"``."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        for package in ("kfp", "kfp-server-api"):
            try:
                return version(package)
            except PackageNotFoundError:
                continue
    except Exception:
        logger.debug("Could not resolve kfp version from package metadata", exc_info=True)
    return "unknown"


def _resolve_image() -> str:
    """Best-effort training container image reference for the parent run.

    Reads ``AUTOML_IMAGE`` (the component ``base_image``) from the installed package so the
    parent run records which image produced the models. Returns ``""`` when unavailable.
    """
    try:
        from kfp_components.utils.consts import AUTOML_IMAGE  # pyright: ignore[reportMissingImports]

        return str(AUTOML_IMAGE)
    except Exception:
        logger.debug("Could not resolve AUTOML_IMAGE for MLflow logging", exc_info=True)
        return ""


def _parse_timeout_seconds(timeout: str) -> int | None:
    """Parse a platform timeout like ``"30s"`` / ``"30"`` into whole seconds, or ``None``."""
    text = (timeout or "").strip().lower()
    if not text:
        return None
    if text.endswith("s"):
        text = text[:-1].strip()
    try:
        seconds = int(float(text))
    except ValueError:
        logger.debug("Could not parse MLflow timeout value %r; ignoring.", timeout)
        return None
    return seconds if seconds > 0 else None


def _safe_log_artifact(mlflow: Any, file_path: Path, artifact_path: str) -> bool:
    """Upload a local file to MLflow, logging and continuing on failure."""
    if not file_path.is_file():
        return False
    try:
        mlflow.log_artifact(str(file_path), artifact_path=artifact_path)
        return True
    except Exception:
        logger.exception("Failed to upload MLflow artifact %s to %s", file_path, artifact_path)
        return False


def _safe_log_artifacts_dir(mlflow: Any, dir_path: Path, artifact_path: str) -> bool:
    """Upload a local directory tree to MLflow, logging and continuing on failure."""
    if not dir_path.is_dir():
        return False
    try:
        mlflow.log_artifacts(str(dir_path), artifact_path=artifact_path)
        return True
    except Exception:
        logger.exception("Failed to upload MLflow artifacts dir %s to %s", dir_path, artifact_path)
        return False


def _write_temp_json(tmp_dir: Path, filename: str, payload: dict[str, Any]) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output = tmp_dir / filename
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def _load_plot_renderer() -> Any:
    """Return the ``mlflow_plots`` module, or ``None`` when it cannot be imported.

    Available on the AutoML runtime image (ships ``kfp_components``). In the rare
    embedded-fallback path (older images), only ``mlflow_tracking.py`` is embedded, so
    plot rendering is skipped.
    """
    try:
        from kfp_components.components.training.automl.shared import mlflow_plots

        return mlflow_plots
    except Exception:
        try:
            import mlflow_plots  # type: ignore[import-not-found]

            return mlflow_plots
        except Exception:
            logger.info("mlflow_plots module unavailable; skipping plot rendering.")
            return None


def _log_model_and_notebook_artifacts(
    mlflow: Any, model_dir: Path, *, notebook_path: Path | None = None
) -> dict[str, bool]:
    """Upload the deployment predictor (model.pkl) and a model notebook to MLflow.

    The deployment predictor is data-stripped (``clone_for_deployment``), so it is safe
    to upload. The model notebook rendered into ``model_dir`` embeds real sample rows and
    is NEVER uploaded to the tracking server; callers must pass a pre-sanitized notebook
    (typed placeholders instead of real values) via ``notebook_path`` to have a notebook
    uploaded. When ``notebook_path`` is None, no notebook is uploaded (fail safe).
    """
    results = {"model": False, "notebook": False}
    predictor_dir = model_dir / MODEL_PREDICTOR_SUBDIR
    if _safe_log_artifacts_dir(mlflow, predictor_dir, "model"):
        results["model"] = True
    if notebook_path is not None and _safe_log_artifact(mlflow, Path(notebook_path), "notebooks"):
        results["notebook"] = True
    return results


def _log_rendered_plots(
    mlflow: Any,
    *,
    task_type: str,
    model_dir: Path,
    tmp_dir: Path,
    plot_renderer: Any,
) -> int:
    """Render confusion-matrix/ROC (or back-testing) PNGs and upload them to MLflow."""
    if plot_renderer is None:
        return 0
    try:
        rendered = plot_renderer.render_model_plots(task_type, model_dir, tmp_dir)
    except Exception:
        logger.exception("Plot rendering failed for %s", model_dir)
        return 0
    uploaded = 0
    for plot_path in rendered:
        if _safe_log_artifact(mlflow, Path(plot_path), "plots"):
            uploaded += 1
    return uploaded


def _build_leaderboard_summary(
    *,
    model_names: list[str],
    valid_metrics: dict[str, dict[str, Any]],
    eval_metric: str,
) -> dict[str, Any]:
    models = []
    for model_name in model_names:
        metrics = valid_metrics.get(model_name)
        if not metrics:
            continue
        models.append(
            {
                "model_name": model_name,
                "display_name": display_model_run_name(model_name),
                "metrics": _normalize_model_metrics(metrics),
            }
        )
    return {"eval_metric": eval_metric, "models": models}


@contextmanager
def _child_mlflow_run(
    mlflow: Any,
    *,
    experiment_id: str,
    parent_run_id: str,
    run_name: str,
    tags: dict[str, str],
) -> Iterator[Any]:
    """Open a nested child run, falling back to explicit parent linkage when needed."""
    try:
        started = mlflow.start_run(run_name=run_name, nested=True)
    except Exception as exc:
        logger.warning(
            "MLflow nested child run failed for %s (%s); trying MlflowClient.create_run.",
            run_name,
            exc,
        )
    else:
        # yield outside the except so caller-body exceptions propagate unchanged
        # instead of being caught here and triggering the create_run fallback.
        with started as run:
            yield run
        return

    # MlflowClient.create_run takes tags as a plain dict (it builds RunTag objects
    # internally); passing a list of RunTag raises AttributeError.
    run_tags = {MLFLOW_PARENT_RUN_ID_TAG: parent_run_id, **tags}

    client = mlflow.MlflowClient()
    created_run = client.create_run(experiment_id=experiment_id, run_name=run_name, tags=run_tags)
    try:
        # nested=True: the parent run is still active, so reopening this child by run_id
        # would otherwise be rejected by MLflow.
        with mlflow.start_run(run_id=created_run.info.run_id, nested=True) as run:
            yield run
    except Exception as exc:
        client.set_terminated(created_run.info.run_id, status="FAILED")
        raise exc


def _aggregate_scores(model_metrics: dict[str, dict[str, Any]], eval_metric: str) -> list[float]:
    scores: list[float] = []
    for metrics in model_metrics.values():
        normalized = _normalize_model_metrics(metrics)
        if eval_metric in normalized and isinstance(normalized[eval_metric], (int, float)):
            scores.append(float(normalized[eval_metric]))
    return scores


def _log_optional_metric_artifacts(
    mlflow: Any,
    metrics_dir: Path,
    *,
    metrics: dict[str, Any],
    display_name: str,
    tmp_dir: Path,
) -> None:
    uploaded = False
    metrics_json = metrics_dir / "metrics.json"
    if _safe_log_artifact(mlflow, metrics_json, "metrics"):
        uploaded = True

    for filename in OPTIONAL_METRIC_ARTIFACTS:
        artifact_file = metrics_dir / filename
        if _safe_log_artifact(mlflow, artifact_file, "metrics"):
            uploaded = True

    if not uploaded:
        summary = _write_temp_json(
            tmp_dir,
            f"{display_name}_metrics.json",
            _normalize_model_metrics(metrics),
        )
        _safe_log_artifact(mlflow, summary, "metrics")


class MlflowExperimentLogger:
    """Incremental MLflow logger used inside the training components.

    Opens a nested child run per model **as each model finishes**, so the experiment
    updates live during the run instead of being dumped in one batch at the end. It is:

    - **Null-safe**: when MLflow is disabled every method is a no-op, so callers need no
      ``if enabled`` guards.
    - **Best-effort**: each method swallows its own exceptions and logs them, so tracking
      problems never fail the surrounding training step.

    Typical usage (inside a training component)::

        with experiment_run_logger(task_type=..., eval_metric=...) as run_logger:
            run_logger.log_header(pipeline_name=..., kfp_run_id=..., ...)
            for model_name in model_names:
                # ... compute + write this model's metrics/artifacts ...
                run_logger.log_model(model_name=..., model_dir=..., model_uri=..., metrics=...)
            run_logger.finalize(html_artifact_path=..., model_names=model_names, ...)
        logged, tracking_info = run_logger.result()
    """

    def __init__(
        self,
        mlflow: Any,
        config: MlflowConfig | None,
        *,
        task_type: str,
        eval_metric: str,
        log_model_artifacts: bool = True,
    ) -> None:
        """Store MLflow handles and tracking config; disabled when either is missing."""
        self._mlflow = mlflow
        self._config = config
        self._task_type = task_type
        self._eval_metric = eval_metric
        self._log_model_artifacts = log_model_artifacts
        self.enabled = mlflow is not None and config is not None
        # MLflow was injected by the platform (KFP_MLFLOW_CONFIG present). Distinct from
        # ``enabled``, which also requires the mlflow package and an open parent run -- so
        # callers can tell "tracking is off" apart from "tracking is on but broken".
        self.configured = config is not None
        self.parent_run_id = ""
        self.experiment_id = config.experiment_id if config else ""
        self._child_run_ids: list[str] = []
        self._child_run_errors: list[str] = []
        # Whether the parent-run writes actually reached MLflow; drives ``result()`` so a
        # run whose every write was swallowed is not reported as successfully logged.
        self._header_logged = False
        self._finalize_logged = False
        self._tracking_errors: list[str] = []
        # Populated live by the progress callback (model_name -> child run id) so the refit
        # loop enriches those runs instead of creating duplicate ones.
        self._live_child_runs: dict[str, str] = {}
        self._valid_metrics: dict[str, dict[str, Any]] = {}
        self._plot_renderer: Any = None
        self._tmp_dir: Path | None = None

    def log_header(
        self,
        *,
        pipeline_name: str,
        kfp_run_id: str,
        kfp_run_name: str = "",
        preset: str = "",
        top_n: int = 0,
        data_config: dict[str, Any] | None = None,
        dataset_uri: str = "",
    ) -> None:
        """Tag the parent run and log run-level params. Call once before ``log_model``."""
        if not self.enabled:
            return
        try:
            active = self._mlflow.active_run()
            if active is not None:
                self.parent_run_id = active.info.run_id
                self.experiment_id = active.info.experiment_id
            self._plot_renderer = _load_plot_renderer()
            self._tmp_dir = Path(tempfile.mkdtemp(prefix="automl-mlflow-"))
            autogluon_version = _resolve_autogluon_version()
            kfp_version = _resolve_kfp_version()
            image = _resolve_image()

            self._mlflow.set_tags(
                {
                    "pipeline_name": pipeline_name,
                    "kfp_run_id": kfp_run_id,
                    "kfp_run_name": kfp_run_name,
                    "autogluon_version": autogluon_version,
                    "kfp_version": kfp_version,
                    "image": image,
                    "run_type": RUN_TYPE_PIPELINE,
                }
            )
            # task_type is a run parameter (per the MLflow integration ADR), not a tag.
            parent_params: dict[str, Any] = {
                "task_type": self._task_type,
                "eval_metric": self._eval_metric,
                "autogluon_version": autogluon_version,
                "kfp_version": kfp_version,
                "image": image,
            }
            if preset:
                parent_params["preset"] = preset
            if top_n:
                parent_params["top_n"] = top_n
            if dataset_uri:
                # Non-secret dataset identity (s3://bucket/key); credentials live in the K8s secret.
                parent_params["dataset_uri"] = dataset_uri
            if data_config:
                parent_params["data_config"] = json.dumps(data_config, sort_keys=True)
            self._mlflow.log_params(_stringify_params(parent_params))
            self._header_logged = True
        except Exception as exc:
            self.record_tracking_error(f"log_header: {exc}")
            logger.exception("MLflow parent header logging failed; continuing without it.")

    def build_progress_callback(self) -> Any | None:
        """Build a Tabular AutoGluon callback that streams live candidate scores to the parent run.

        Returns ``None`` when tracking is disabled, the parent run is unknown, or the
        runtime AutoGluon lacks the callback API -- callers should then omit ``callbacks=``.
        Call after :meth:`log_header` so ``parent_run_id`` is populated.
        """
        if not self.enabled or not self.parent_run_id:
            return None
        from kfp_components.components.training.automl.shared.mlflow_callbacks import (
            build_mlflow_progress_callback,
        )

        return build_mlflow_progress_callback(
            self._mlflow,
            run_id=self.parent_run_id,
            experiment_id=self.experiment_id,
            eval_metric=self._eval_metric,
            kfp_run_id=self._config.run_id if self._config else "",
            registry=self._live_child_runs,
        )

    def prune_live_child_runs(self, keep_model_names: Iterable[str]) -> None:
        """Delete live child runs for candidates that did not make the final top-N.

        The progress callback creates a live nested run for *every* candidate AutoGluon
        trains during ``fit()`` (including all bagged base models), which is noisy once the
        leaderboard is known. Call this after selecting the top-N with the model names to
        keep -- typically the leaderboard head that will be refit and enriched by
        :meth:`log_model` -- to remove the rest. Best-effort: failures are logged, not raised.
        """
        if not self.enabled or not self._live_child_runs:
            return
        keep = set(keep_model_names)
        try:
            client = self._mlflow.MlflowClient()
        except Exception:
            logger.warning("Could not open MLflow client to prune live child runs.", exc_info=True)
            return
        for name in list(self._live_child_runs):
            if name in keep:
                continue
            run_id = self._live_child_runs.pop(name)
            try:
                client.delete_run(run_id)
            except Exception:
                logger.warning(
                    "Could not delete non-top-N MLflow child run %s (%s); leaving it in place.",
                    name,
                    run_id,
                    exc_info=True,
                )

    def log_model(
        self,
        *,
        model_name: str,
        model_dir: Path,
        model_uri: str,
        metrics: dict[str, Any],
        notebook_path: Path | None = None,
    ) -> None:
        """Create and finalize one nested child run for a single model.

        ``notebook_path`` is an optional pre-sanitized inference notebook (typed
        placeholders instead of real sample rows). Only this notebook is uploaded as an
        artifact; the data-bearing notebook rendered into ``model_dir`` is never uploaded.
        """
        if not self.enabled:
            return
        if not metrics:
            logger.warning("Skipping MLflow child run for %s: no metrics.", model_name)
            return
        try:
            self._valid_metrics[model_name] = metrics
            model_type, stack_level = parse_model_name(model_name)
            display_name = display_model_run_name(model_name)
            task_metrics = _metrics_for_task(self._task_type, metrics)

            child_params: dict[str, Any] = {
                "model_name": model_name,
                "model_type": model_type,
                "stack_level": stack_level,
                "metrics_path": f"{model_uri}/metrics",
                "predictor_path": f"{model_uri}/predictor",
                "notebook_path": f"{model_uri}/notebooks/automl_predictor_notebook.ipynb",
            }
            normalized = _normalize_model_metrics(metrics)
            if "fit_time" in normalized:
                child_params["fit_time"] = normalized["fit_time"]
            if "pred_time_val" in normalized:
                child_params["predict_time"] = normalized["pred_time_val"]

            child_tags = {
                "run_type": RUN_TYPE_MODEL,
                "model_name": model_name,
                "model_type": model_type,
                "stack_level": str(stack_level),
                "kfp_run_id": self._config.run_id if self._config else "",
            }

            tmp_dir = self._tmp_dir or Path(tempfile.mkdtemp(prefix="automl-mlflow-"))
            # Prefer enriching the live run the progress callback already created for this
            # model during fit() (matched by name); otherwise open a fresh nested run.
            live_run_id = self._live_child_runs.get(display_name) or self._live_child_runs.get(model_name)
            if live_run_id:
                # nested=True: parent run is still active, so reopening this child by run_id
                # would otherwise be rejected by MLflow.
                child_cm = self._mlflow.start_run(run_id=live_run_id, nested=True)
            else:
                child_cm = _child_mlflow_run(
                    self._mlflow,
                    experiment_id=self.experiment_id,
                    parent_run_id=self.parent_run_id,
                    run_name=display_name,
                    tags=child_tags,
                )
            with child_cm:
                self._mlflow.set_tags(child_tags)
                self._mlflow.log_params(_stringify_params(child_params))
                if task_metrics:
                    self._mlflow.log_metrics(task_metrics)
                else:
                    logger.warning("No scalar metrics to log for MLflow child run %s.", display_name)

                _log_optional_metric_artifacts(
                    self._mlflow,
                    model_dir / "metrics",
                    metrics=metrics,
                    display_name=display_name,
                    tmp_dir=tmp_dir,
                )
                _log_rendered_plots(
                    self._mlflow,
                    task_type=self._task_type,
                    model_dir=model_dir,
                    tmp_dir=tmp_dir / f"{display_name}_plots",
                    plot_renderer=self._plot_renderer,
                )
                if self._log_model_artifacts:
                    _log_model_and_notebook_artifacts(self._mlflow, model_dir, notebook_path=notebook_path)

                active_child = self._mlflow.active_run()
                if active_child is not None and active_child.info.run_id:
                    child_run_id = str(active_child.info.run_id)
                    self._child_run_ids.append(child_run_id)
        except Exception as exc:
            self._child_run_errors.append(f"{model_name}: {exc}")
            logger.exception("Failed to create MLflow child run for model %s.", model_name)

    def finalize(
        self,
        *,
        html_artifact_path: str | Path,
        model_names: list[str],
        total_fit_time_seconds: float | None = None,
    ) -> None:
        """Log parent aggregates and the leaderboard."""
        if not self.enabled:
            return
        try:
            tmp_dir = self._tmp_dir or Path(tempfile.mkdtemp(prefix="automl-mlflow-"))
            if total_fit_time_seconds is not None:
                self._mlflow.log_metric("total_fit_time_seconds", float(total_fit_time_seconds))
            scores = _aggregate_scores(self._valid_metrics, self._eval_metric)
            if scores:
                self._mlflow.log_metric("best_score", max(scores))
                self._mlflow.log_metric("worst_score", min(scores))
                self._mlflow.log_metric("mean_score", sum(scores) / len(scores))
            self._mlflow.log_metric("num_models_trained", len(self._valid_metrics))

            html_path = resolve_leaderboard_html_path(html_artifact_path)
            if html_path is not None:
                _safe_log_artifact(self._mlflow, html_path, "reports")
            else:
                logger.warning("Leaderboard HTML not found at %s.", html_artifact_path)

            summary_path = _write_temp_json(
                tmp_dir,
                "leaderboard_summary.json",
                _build_leaderboard_summary(
                    model_names=model_names,
                    valid_metrics=self._valid_metrics,
                    eval_metric=self._eval_metric,
                ),
            )
            _safe_log_artifact(self._mlflow, summary_path, "reports")

            if self._child_run_ids:
                self._mlflow.log_metric("child_run_count", len(self._child_run_ids))
            else:
                logger.warning(
                    "No MLflow child runs were created under parent run %s. Errors: %s",
                    self.parent_run_id,
                    self._child_run_errors,
                )
            if self._child_run_errors:
                self._mlflow.set_tag("child_run_errors", json.dumps(self._child_run_errors)[:500])

            if self._valid_metrics:
                best_model_name = max(
                    self._valid_metrics,
                    key=lambda name: float(
                        _normalize_model_metrics(self._valid_metrics[name]).get(self._eval_metric, float("-inf"))
                    ),
                )
                self._mlflow.log_param("best_model_name", best_model_name)
            self._finalize_logged = True
        except Exception as exc:
            self.record_tracking_error(f"finalize: {exc}")
            logger.exception("MLflow finalize step failed; continuing.")

    def record_tracking_error(self, reason: str) -> None:
        """Record why MLflow tracking could not persist results (surfaced by ``result()``)."""
        self._tracking_errors.append(reason)

    def result(self) -> tuple[bool, dict[str, str]]:
        """Return ``(logged, tracking_info)`` for recording on the component status.

        ``logged`` is True only when something actually reached MLflow -- the parent header,
        the parent finalization, or at least one child run. Because every logging method
        swallows its own exceptions, a configured-but-unreachable tracking server would
        otherwise be reported as a successful log. When ``logged`` is False but tracking was
        configured, ``tracking_info["mlflow_tracking_error"]`` carries the reason.
        """
        if self._config is None:
            return False, {}
        persisted = bool(self._header_logged or self._finalize_logged or self._child_run_ids)
        run_id_for_url = self.parent_run_id or self._config.run_id
        tracking_info: dict[str, str] = {
            "mlflow_run_id": run_id_for_url,
            "mlflow_experiment_id": self.experiment_id,
            "tracking_mode": self._config.mode,
            "mlflow_child_run_ids": ",".join(self._child_run_ids),
            "mlflow_child_run_count": str(len(self._child_run_ids)),
        }
        run_url = build_mlflow_run_url(self._config.tracking_uri, self.experiment_id, run_id_for_url)
        if run_url:
            tracking_info["mlflow_run_url"] = run_url
        if self._child_run_errors:
            tracking_info["mlflow_child_run_errors"] = json.dumps(self._child_run_errors)
        if not persisted:
            reasons = self._tracking_errors + self._child_run_errors
            tracking_info["mlflow_tracking_error"] = "; ".join(reasons) or "no MLflow write succeeded"
        return persisted, tracking_info


@contextmanager
def experiment_run_logger(
    *,
    task_type: str,
    eval_metric: str,
    log_model_artifacts: bool = True,
    run_name: str = "",
) -> Iterator[MlflowExperimentLogger]:
    """Yield an :class:`MlflowExperimentLogger` bound to the parent run.

    Resolves ``KFP_MLFLOW_CONFIG`` and imports MLflow lazily. When the platform supplies a
    ``parentRunId`` that run is resumed; otherwise a parent run (and, if needed, an
    experiment) named ``run_name`` is created. When tracking is disabled or the parent run
    cannot be opened, a disabled (no-op) logger is yielded so the caller proceeds unchanged.
    The parent run is closed when the ``with`` block exits.
    """
    config = resolve_mlflow_config()
    mlflow_mod: Any = None
    if config is not None:
        try:
            import mlflow as mlflow_mod  # type: ignore[no-redef]
        except ImportError:
            logger.warning("mlflow package is not installed in the runtime image; skipping MLflow logging.")
            mlflow_mod = None

    run_logger = MlflowExperimentLogger(
        mlflow_mod,
        config,
        task_type=task_type,
        eval_metric=eval_metric,
        log_model_artifacts=log_model_artifacts,
    )
    if not run_logger.enabled:
        if config is not None:
            # Configured but unusable -- report it as failed tracking, not as tracking off.
            run_logger.record_tracking_error("mlflow package is not installed in the runtime image")
        yield run_logger
        return

    parent_cm = parent_mlflow_run(mlflow_mod, config, fallback_name=run_name)
    try:
        parent_cm.__enter__()
    except Exception as exc:
        logger.exception("Failed to open MLflow parent run; disabling tracking for this step.")
        run_logger.enabled = False
        run_logger.record_tracking_error(f"parent run could not be opened: {exc}")
        yield run_logger
        return

    try:
        yield run_logger
    finally:
        try:
            parent_cm.__exit__(None, None, None)
        except Exception:
            logger.exception("Error while closing MLflow parent run.")
