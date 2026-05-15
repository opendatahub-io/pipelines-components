"""Pytest fixtures for AutoGluon time series training pipeline tests."""

import base64
import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure this directory is on path so test modules and integration_config can be imported
_tests_dir = Path(__file__).resolve().parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

import pytest  # noqa: E402

# .env is loaded by integration_config when get_rhoai_config/get_dspa_config run.
# ---------------------------------------------------------------------------
# Integration test configuration (RHOAI + S3 + KFP)
# Set env vars to enable integration tests; otherwise they are skipped.
# ---------------------------------------------------------------------------


def _get_rhoai_config():
    """Build integration config from environment; None if not configured."""
    from integration_config import get_rhoai_config

    return get_rhoai_config()


def _get_dspa_config():
    from integration_config import get_dspa_config

    return get_dspa_config()


@pytest.fixture(scope="session")
def rhoai_integration_config():
    """Session-scoped RHOAI integration config from env; None if not set. Use RHOAI_TOKEN (e.g. SA token for Jenkins)."""  # noqa: E501
    return _get_rhoai_config()


@pytest.fixture(scope="session")
def integration_available(rhoai_integration_config):
    """True if RHOAI integration config is present."""
    return rhoai_integration_config is not None


def _build_temp_kubeconfig(server_url, token, namespace="default"):
    """Build a minimal kubeconfig dict and write it to a temp file; return the file path."""
    server_url = (server_url or "").rstrip("/")
    config = {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [
            {
                "name": "rhoai",
                "cluster": {
                    "server": server_url,
                    "insecure-skip-tls-verify": True,
                },
            }
        ],
        "users": [{"name": "rhoai", "user": {"token": token or ""}}],
        "contexts": [
            {
                "name": "rhoai",
                "context": {"cluster": "rhoai", "user": "rhoai", "namespace": namespace or "default"},
            }
        ],
        "current-context": "rhoai",
    }
    import yaml

    fd, path = tempfile.mkstemp(suffix=".kubeconfig", prefix="automl-test-")
    os.close(fd)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return path


@pytest.fixture(scope="session")
def temp_kubeconfig_path(rhoai_integration_config):
    """Create a temporary kubeconfig from RHOAI_URL and RHOAI_TOKEN (.env) so the Kubernetes
    client does not use the default ~/.kube/config. Session-scoped; file is removed after tests.
    Yields the path to the temp file, or None when integration config is not set.
    """  # noqa: D205
    if rhoai_integration_config is None:
        yield None
        return
    path = _build_temp_kubeconfig(
        server_url=rhoai_integration_config["rhoai_url"],
        token=rhoai_integration_config["rhoai_token"],
        namespace=rhoai_integration_config["rhoai_project"],
    )
    try:
        yield path
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass


@pytest.fixture(scope="session")
def s3_client(rhoai_integration_config):
    """Session-scoped S3 client for test data upload (and optional artifact checks)."""
    if rhoai_integration_config is None:
        return None
    try:
        import boto3
    except ImportError:
        pytest.skip("boto3 not installed; install with pip install boto3")
    c = rhoai_integration_config
    return boto3.client(
        "s3",
        endpoint_url=c["s3_endpoint"],
        aws_access_key_id=c["s3_access_key"],
        aws_secret_access_key=c["s3_secret_key"],
        region_name=c["s3_region"],
        verify=False if c["s3_internal_endpoint"] else True,
    )


def _decode_jwt_sub(token: str) -> str | None:
    """Decode JWT payload (no verify) and return the 'sub' claim, or None."""
    try:
        parts = token.strip().split(".")
        if len(parts) != 3:
            return None
        payload_b64 = parts[1]
        padding = 4 - len(payload_b64) % 4
        if padding != 4:
            payload_b64 += "=" * padding
        payload = base64.urlsafe_b64decode(payload_b64)
        data = json.loads(payload)
        return data.get("sub")
    except (ValueError, json.JSONDecodeError, KeyError):
        return None


def _parse_service_account_sub(sub: str) -> tuple[str, str] | None:
    """If sub is 'system:serviceaccount:namespace:name', return (namespace, name); else None."""
    if not sub or not isinstance(sub, str):
        return None
    prefix = "system:serviceaccount:"
    if not sub.startswith(prefix):
        return None
    rest = sub[len(prefix) :].strip()
    parts = rest.split(":")
    if len(parts) != 2:
        return None
    return (parts[0].strip(), parts[1].strip())


def _ensure_admin_role_for_sa_in_namespace(
    rbac_v1,
    namespace: str,
    sa_namespace: str,
    sa_name: str,
    binding_name: str = "kfp-integration-tests-admin",
):
    """Create or replace a RoleBinding in namespace granting cluster role 'admin' to the ServiceAccount."""
    from kubernetes import client
    from kubernetes.client.rest import ApiException

    role_ref = client.V1RoleRef(
        api_group="rbac.authorization.k8s.io",
        kind="ClusterRole",
        name="admin",
    )
    subject = client.RbacV1Subject(
        kind="ServiceAccount",
        name=sa_name,
        namespace=sa_namespace,
    )
    body = client.V1RoleBinding(
        api_version="rbac.authorization.k8s.io/v1",
        kind="RoleBinding",
        metadata=client.V1ObjectMeta(name=binding_name),
        role_ref=role_ref,
        subjects=[subject],
    )
    try:
        rbac_v1.create_namespaced_role_binding(namespace, body)
    except ApiException as e:
        if e.status == 409:
            rbac_v1.replace_namespaced_role_binding(binding_name, namespace, body)
        else:
            raise


@pytest.fixture(scope="session")
def rhoai_project(rhoai_integration_config, s3_client, temp_kubeconfig_path):
    """Ensure RHOAI test project exists: create if needed via OpenShift ProjectRequest (self-provisioner), then create the S3 connection secret.

    OpenShift normally grants the ProjectRequest creator (the ServiceAccount) admin in the new
    project (same as oc new-project). If the cluster does not do that and we get 403 on the secret
    create, we create a RoleBinding granting the SA admin; that requires the SA to be allowed to
    create RoleBindings (see README_integration.md Option B).
    """  # noqa: E501
    if rhoai_integration_config is None:
        yield None
        return
    project_name = rhoai_integration_config["rhoai_project"]
    secret_name = rhoai_integration_config["s3_secret_name"]
    token = rhoai_integration_config.get("rhoai_token")
    try:
        from kubernetes import client, config
        from kubernetes.client.rest import ApiException
    except ImportError:
        pytest.skip("kubernetes client not installed; pip install kubernetes")
    try:
        if temp_kubeconfig_path:
            config.load_kube_config(config_file=temp_kubeconfig_path)
        else:
            config.load_kube_config()
    except Exception:
        try:
            config.load_incluster_config()
        except Exception:
            pytest.skip("Could not load kubeconfig or in-cluster config")
    v1 = client.CoreV1Api()
    rbac_v1 = client.RbacAuthorizationV1Api()

    # Create project if it does not exist: prefer OpenShift ProjectRequest (self-provisioner).
    project_request_group = "project.openshift.io"
    project_request_version = "v1"
    project_request_plural = "projectrequests"
    project_just_created = False
    try:
        co = client.CustomObjectsApi()
        body = {
            "apiVersion": f"{project_request_group}/{project_request_version}",
            "kind": "ProjectRequest",
            "metadata": {"name": project_name},
        }
        co.create_cluster_custom_object(
            group=project_request_group,
            version=project_request_version,
            plural=project_request_plural,
            body=body,
        )
        project_just_created = True
    except ApiException as e:
        if e.status == 409:
            pass
        elif e.status == 404 or e.status == 403:
            namespace = client.V1Namespace(metadata=client.V1ObjectMeta(name=project_name))
            try:
                v1.create_namespace(namespace)
                project_just_created = True
            except ApiException as e2:
                if e2.status != 409:
                    raise
        else:
            raise

    # OpenShift normally grants the ProjectRequest creator (user or SA) admin in the new project
    # (same as oc new-project). Try creating the secret first; only if we get 403 do we create a
    # RoleBinding to grant the SA admin (for clusters that do not auto-grant SAs).
    secret = client.V1Secret(
        metadata=client.V1ObjectMeta(name=secret_name),
        type="Opaque",
        string_data={
            "AWS_ACCESS_KEY_ID": rhoai_integration_config["s3_access_key"],
            "AWS_SECRET_ACCESS_KEY": rhoai_integration_config["s3_secret_key"],
            "AWS_S3_ENDPOINT": rhoai_integration_config["s3_internal_endpoint"]
            if rhoai_integration_config["s3_internal_endpoint"]
            else rhoai_integration_config["s3_endpoint"],
            "AWS_DEFAULT_REGION": rhoai_integration_config["s3_region"],
        },
    )

    def _create_or_replace_secret():
        try:
            v1.create_namespaced_secret(project_name, secret)
            return
        except ApiException as e:
            if e.status == 409:
                v1.replace_namespaced_secret(secret_name, project_name, secret)
                return
            if e.status == 403:
                raise
            raise

    try:
        _create_or_replace_secret()
    except ApiException as e:
        if e.status != 403:
            raise
        # Cluster did not auto-grant the SA admin in the new project. Try to add a RoleBinding.
        if not (project_just_created and token):
            pytest.fail(
                f"Cannot create secret in namespace {project_name!r}. "
                f"Grant the ServiceAccount 'edit' or 'admin' in that namespace, e.g.:\n"
                f"  oc adm policy add-role-to-user edit system:serviceaccount:<sa-namespace>:<sa-name> -n {project_name!r}"  # noqa: E501
            )
        sub = _decode_jwt_sub(token)
        sa_identity = _parse_service_account_sub(sub) if sub else None
        if not sa_identity:
            pytest.fail(
                f"Cannot create secret in namespace {project_name!r} and could not determine "
                f"ServiceAccount from token. Grant the SA 'edit' or 'admin' in that namespace."
            )
        sa_namespace, sa_name = sa_identity
        try:
            _ensure_admin_role_for_sa_in_namespace(rbac_v1, project_name, sa_namespace, sa_name)
        except ApiException as rb_e:
            if rb_e.status == 403:
                pytest.fail(
                    "ServiceAccount cannot create RoleBindings in the new project. "
                    "Either grant the SA a cluster role that allows creating rolebindings "
                    "(see README_integration.md 'Option B'), or ensure the project template "
                    "grants the requesting identity admin (default OpenShift behavior)."
                )
            raise
        try:
            _create_or_replace_secret()
        except ApiException as e2:
            if e2.status == 403:
                pytest.fail(
                    f"Cannot create secret in namespace {project_name!r} even after creating "
                    f"admin RoleBinding. Grant the ServiceAccount 'edit' or 'admin' manually, e.g.:\n"
                    f"  oc adm policy add-role-to-user edit system:serviceaccount:{sa_namespace}:{sa_name} -n {project_name!r}"  # noqa: E501
                )
            raise
    yield project_name


def _create_datascience_pipelines_application(
    namespace,
    dspa_config,
    resource_name="automl-test-dspa",
    kubeconfig_path=None,
    object_storage_url=None,
    object_storage_region=None,
    object_storage_secret_name=None,
    object_storage_bucket=None,
    object_storage_internal_url=None,
):
    """Create a DataSciencePipelinesApplication CR in the given namespace using CustomObjectsApi.

    The Data Science Pipelines Operator (DSPO) / Open Data Hub will reconcile the CR and
    deploy the pipeline server. Requires the DSPA CRD and operator to be installed.
    When kubeconfig_path is set, uses that file instead of default kubeconfig.
    When object_storage_secret_name and object_storage_bucket are set, configures
    spec.objectStorage.external; otherwise uses spec.objectStorage.internal (operator-managed MinIO).

    Returns (created_cr, error_message). On success: (created, None). On failure: (None, str).
    """
    from kubernetes import client, config
    from kubernetes.client.rest import ApiException

    try:
        if kubeconfig_path:
            config.load_kube_config(config_file=kubeconfig_path)
        else:
            config.load_kube_config()
    except Exception as e:
        try:
            config.load_incluster_config()
        except Exception as e2:
            return (
                None,
                f"Could not load kubeconfig or in-cluster config: {e!r}; in-cluster: {e2!r}",
            )

    # CRD requires spec.objectStorage: either internal (operator MinIO) or external (existing S3).
    if object_storage_secret_name and object_storage_bucket:
        from urllib.parse import urlparse

        if object_storage_internal_url:
            _parsed = urlparse(object_storage_internal_url)
            object_storage_host = _parsed.hostname or ""
            object_storage_port = str(_parsed.port) if _parsed.port else ""
            object_storage_scheme = "http"
        else:
            _parsed = urlparse(object_storage_url)
            object_storage_host = _parsed.hostname or ""
            object_storage_port = str(_parsed.port) if _parsed.port else ""
            object_storage_scheme = "https"

        object_storage = {
            "externalStorage": {
                "basePath": "",
                "bucket": object_storage_bucket,
                "host": object_storage_host,
                "port": object_storage_port,
                "region": object_storage_region,
                "s3CredentialsSecret": {
                    "accessKey": "AWS_ACCESS_KEY_ID",
                    "secretKey": "AWS_SECRET_ACCESS_KEY",
                    "secretName": object_storage_secret_name,
                },
                "scheme": object_storage_scheme,
            },
        }
    else:
        object_storage = {"internal": {}}

    body = {
        "apiVersion": f"{dspa_config['api_group']}/{dspa_config['api_version']}",
        "kind": "DataSciencePipelinesApplication",
        "metadata": {
            "name": "dspa",
            "namespace": namespace,
        },
        "spec": {
            "objectStorage": object_storage,
        },
    }
    dspa_name = body["metadata"]["name"]
    try:
        co = client.CustomObjectsApi()
        created = co.create_namespaced_custom_object(
            group=dspa_config["api_group"],
            version=dspa_config["api_version"],
            namespace=namespace,
            plural=dspa_config["plural"],
            body=body,
        )
        return (created, None)
    except ApiException as e:
        if e.status == 409:
            # DSPA already exists; fetch and reuse it.
            try:
                existing = co.get_namespaced_custom_object(
                    group=dspa_config["api_group"],
                    version=dspa_config["api_version"],
                    namespace=namespace,
                    plural=dspa_config["plural"],
                    name=dspa_name,
                )
                return (existing, None)
            except ApiException as get_e:
                return (None, f"DSPA already exists but get failed: {get_e!r}")
        detail = getattr(e, "body", None)
        if isinstance(detail, str) and detail:
            try:
                import json

                detail = json.loads(detail)
            except Exception:
                pass
        msg_parts = [
            f"DSPA creation failed: HTTP {getattr(e, 'status', '?')}",
            f"reason={getattr(e, 'reason', '')}",
        ]
        if detail and isinstance(detail, dict):
            for key in ("message", "reason", "details"):
                if key in detail and detail[key]:
                    msg_parts.append(f"{key}={detail[key]}")
        else:
            msg_parts.append(f"body={detail!r}")
        return (None, "; ".join(msg_parts))
    except Exception as e:
        return (None, f"DSPA creation failed: {type(e).__name__}: {e!r}")


def _wait_for_dspa_ready(
    namespace,
    dspa_name,
    dspa_config,
    kubeconfig_path=None,
    timeout_seconds=600,
):
    """Poll the DSPA CR until status.conditions has type=Ready and status=True, or timeout.

    Returns True when ready, False on timeout. Uses the same kubeconfig as creation.
    """
    import time

    from kubernetes import client, config
    from kubernetes.client.rest import ApiException

    try:
        if kubeconfig_path:
            config.load_kube_config(config_file=kubeconfig_path)
        else:
            config.load_kube_config()
    except Exception:
        try:
            config.load_incluster_config()
        except Exception:
            return False
    co = client.CustomObjectsApi()
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            cr = co.get_namespaced_custom_object(
                group=dspa_config["api_group"],
                version=dspa_config["api_version"],
                namespace=namespace,
                plural=dspa_config["plural"],
                name=dspa_name,
            )
        except ApiException:
            time.sleep(10)
            continue
        status = cr.get("status") or {}
        conditions = status.get("conditions") or []
        for c in conditions:
            if (c.get("type") or "") == "Ready" and (c.get("status") or "") == "True":
                return True
        time.sleep(10)
    return False


# OpenShift Route API: group, version, plural for listing routes
_ROUTE_GROUP = "route.openshift.io"
_ROUTE_VERSION = "v1"
_ROUTE_PLURAL = "routes"


def _get_dspa_route_url(namespace, route_name_prefix="ds-pipeline", timeout_seconds=300, kubeconfig_path=None):
    """Resolve the pipeline API URL from an OpenShift Route in the given namespace.

    Lists Route custom resources (route.openshift.io/v1/routes) and returns
    https://<host> for the first route whose metadata.name starts with route_name_prefix.
    If no route matches, returns the first route in the namespace. Retries until
    timeout_seconds or until a route is found.
    When kubeconfig_path is set, uses that file instead of default kubeconfig.

    Returns the URL string (with trailing slash) or None if no route is found in time.
    """
    import time

    from kubernetes import client, config
    from kubernetes.client.rest import ApiException

    try:
        if kubeconfig_path:
            config.load_kube_config(config_file=kubeconfig_path)
        else:
            config.load_kube_config()
    except Exception:
        try:
            config.load_incluster_config()
        except Exception:
            return None
    co = client.CustomObjectsApi()
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            resp = co.list_namespaced_custom_object(
                group=_ROUTE_GROUP,
                version=_ROUTE_VERSION,
                namespace=namespace,
                plural=_ROUTE_PLURAL,
            )
        except ApiException:
            time.sleep(5)
            continue
        items = resp.get("items") or []
        route = None
        for r in items:
            name = (r.get("metadata") or {}).get("name") or ""
            if name.startswith(route_name_prefix):
                route = r
                break
        if not route and items:
            route = items[0]
        if route:
            host = (route.get("spec") or {}).get("host")
            if not host and (route.get("status") or {}).get("ingress"):
                host = route["status"]["ingress"][0].get("host")
            if host:
                return f"https://{host}".rstrip("/") + "/"
        time.sleep(5)
    return None


@pytest.fixture(scope="session")
def datascience_pipelines_application(rhoai_integration_config, rhoai_project, temp_kubeconfig_path):
    """Optionally create a DataSciencePipelinesApplication CR in the test project namespace.

    Controlled by env: set RHOAI_CREATE_DSPA=true (or 1) to create the CR via Kubernetes
    CustomObjectsApi. Uses temp kubeconfig from .env (RHOAI_URL, RHOAI_TOKEN) when set.
    If creation is attempted and fails, logs the error and fails the test.
    """
    if rhoai_integration_config is None or rhoai_project is None:
        yield None
        return
    dspa_config = _get_dspa_config()
    if not dspa_config or not dspa_config.get("create"):
        yield None
        return
    try:
        from kubernetes import client, config  # noqa: F401
    except ImportError:
        pytest.skip("kubernetes client not installed; pip install kubernetes")
    created, error_message = _create_datascience_pipelines_application(
        rhoai_project,
        dspa_config,
        kubeconfig_path=temp_kubeconfig_path,
        object_storage_secret_name=rhoai_integration_config.get("s3_secret_name"),
        object_storage_url=rhoai_integration_config.get("s3_endpoint"),
        object_storage_region=rhoai_integration_config.get("s3_region"),
        object_storage_bucket=rhoai_integration_config.get("s3_bucket_artifacts")
        or rhoai_integration_config.get("s3_bucket_data"),
        object_storage_internal_url=rhoai_integration_config.get("s3_internal_endpoint"),
    )
    if created is None and error_message:
        import logging

        logging.getLogger(__name__).error("DSPA creation failed: %s", error_message)
        pytest.fail(f"DataSciencePipelinesApplication creation failed: {error_message}")

    # Wait for DSPA to become Ready, then add buffer before using the API.
    if created is not None:
        import time

        dspa_name = (created.get("metadata") or {}).get("name", "dspa")
        namespace = (created.get("metadata") or {}).get("namespace", rhoai_project)
        ready_timeout = dspa_config.get("ready_wait_timeout", 600)
        buffer_seconds = dspa_config.get("ready_buffer_seconds", 30)
        if not _wait_for_dspa_ready(
            namespace, dspa_name, dspa_config, kubeconfig_path=temp_kubeconfig_path, timeout_seconds=ready_timeout
        ):
            import logging

            logging.getLogger(__name__).warning(
                "DSPA %s/%s did not become Ready within %s s; continuing anyway",
                namespace,
                dspa_name,
                ready_timeout,
            )
        time.sleep(buffer_seconds)
    yield created


@pytest.fixture(scope="session")
def uploaded_datasets(rhoai_integration_config, s3_client):
    """Upload dataset files referenced in test_configs.TEST_CONFIGS to S3.

    Reads each unique dataset_path from the configs, uploads the file from
    tests_dir / dataset_path to S3 under key kfp-integration-test/{dataset_path},
    and returns a map: dataset_path -> {"bucket": str, "key": str}.

    Returns empty dict if integration not configured; each path is uploaded once
    even if multiple configs use the same file.
    """
    if rhoai_integration_config is None or s3_client is None:
        return {}
    from test_configs import TEST_CONFIGS

    bucket = rhoai_integration_config["s3_bucket_data"]
    prefix = "kfp-integration-test"
    result = {}
    seen_paths = set()
    for config in TEST_CONFIGS:
        rel_path = config.dataset_path
        if rel_path in seen_paths:
            continue
        seen_paths.add(rel_path)
        full_path = _tests_dir / rel_path
        if not full_path.is_file():
            pytest.skip(f"Test dataset not found: {full_path}")
        try:
            body = full_path.read_bytes()
            key = f"{prefix}/{rel_path}"
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                ContentType="text/csv",
            )
            result[rel_path] = {"bucket": bucket, "key": key}
        except Exception as e:
            pytest.skip(f"Failed to upload test data {rel_path} to S3: {e}")
    return result


@pytest.fixture(scope="session")
def kfp_client(rhoai_integration_config, datascience_pipelines_application, temp_kubeconfig_path):
    """Session-scoped KFP client pointing to RHOAI pipeline API."""
    if rhoai_integration_config is None:
        return None
    import kfp

    # If we created a DSPA, resolve the route URL from OpenShift Route; else use env.
    host = None
    if datascience_pipelines_application is not None:
        dspa_config = _get_dspa_config()
        if dspa_config:
            namespace = (datascience_pipelines_application.get("metadata") or {}).get("namespace")
            if namespace:
                host = _get_dspa_route_url(
                    namespace,
                    route_name_prefix=dspa_config.get("route_name_prefix", "ds-pipeline"),
                    timeout_seconds=dspa_config.get("route_wait_timeout", 300),
                    kubeconfig_path=temp_kubeconfig_path,
                )
    if host is None:
        host = rhoai_integration_config["rhoai_kfp_url"]
    if not host.endswith("/"):
        host = host + "/"
    if not host or not host.strip():
        pytest.skip(
            "KFP API URL not set: when RHOAI_CREATE_DSPA=true the route could not be resolved in time; "
            "otherwise set RHOAI_KFP_URL in .env"
        )

    client = kfp.Client(
        host=host,
        namespace=rhoai_integration_config["rhoai_project"],
        existing_token=rhoai_integration_config.get("rhoai_token"),
        verify_ssl=False if rhoai_integration_config.get("s3_internal_endpoint") else True,
    )
    return client


@pytest.fixture(scope="session")
def compiled_pipeline_path():
    """Return path to a compiled pipeline YAML.

    If RHOAI_COMPILED_PIPELINE_PATH is set and points to an existing file, use it directly.
    Otherwise compile the pipeline on-the-fly into a temp file.
    """
    from integration_config import _ensure_dotenv_loaded

    _ensure_dotenv_loaded()
    precompiled = os.environ.get("RHOAI_COMPILED_PIPELINE_PATH")
    if precompiled and Path(precompiled).is_file():
        yield precompiled
        return

    from kfp import compiler

    from ..pipeline import autogluon_timeseries_training_pipeline

    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    compiler.Compiler().compile(
        pipeline_func=autogluon_timeseries_training_pipeline,
        package_path=path,
    )
    yield path
    Path(path).unlink(missing_ok=True)


@pytest.fixture
def pipeline_run_timeout():
    """Timeout in seconds for waiting on a pipeline run (override via env)."""
    return int(os.environ.get("RHOAI_PIPELINE_RUN_TIMEOUT", "3600"))
