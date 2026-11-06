"""Pipeline secrets map to provider-prefixed environment variables."""

import importlib.util
from pathlib import Path

import pytest

from ..database_connection import SECRET_KEY_TO_ENV, normalize_connection


def connection_env(**fields):
    """Build the environment injected from one database secret."""
    return {SECRET_KEY_TO_ENV[key]: value for key, value in fields.items()}


@pytest.mark.parametrize(
    ("provider", "fields", "backend", "expected"),
    [
        (
            "postgres",
            {"URI": "postgresql://rag:p%40ss@[::1]:5433/docs"},
            "pgvector",
            {
                "PGVECTOR_HOST": "::1",
                "PGVECTOR_PORT": "5433",
                "PGVECTOR_DB": "docs",
                "PGVECTOR_USER": "rag",
                "PGVECTOR_PASSWORD": "p@ss",
            },
        ),
        (
            "postgres",
            {
                "host": "db.svc",
                "port": "5432",
                "user": "rag",
                "password": "secret",
                "dbname": "docs",
                "sslmode": "verify-full",
                "ca.crt": "PEM content",
            },
            "pgvector",
            {"PGVECTOR_HOST": "db.svc", "PGVECTOR_CA_CERT": "PEM content", "PGVECTOR_PASSWORD": "secret"},
        ),
        (
            "postgres",
            {"URI": "postgres://rag:secret@db/docs?sslmode=verify-full", "CA_CERT": "PEM"},
            "pgvector",
            {"PGVECTOR_CA_CERT": "PEM"},
        ),
        (
            "milvus",
            {"URI": "https://milvus:19530", "TOKEN": "token", "DATABASE": "default"},
            "milvus",
            {"MILVUS_URI": "https://milvus:19530", "MILVUS_TOKEN": "token"},
        ),
        (
            "neo4j",
            {"URI": "neo4j://graph:7687", "PASSWORD": "password", "DATABASE": "docs", "CA_CERT": "PEM"},
            "neo4j",
            {
                "NEO4J_URI": "neo4j://graph:7687",
                "NEO4J_PASSWORD": "password",
                "NEO4J_USERNAME": "neo4j",
                "NEO4J_DATABASE": "docs",
                "NEO4J_CA_CERT": "PEM",
            },
        ),
    ],
)
def test_normalizes_secret(provider, fields, backend, expected):
    """Decode DCH and shared database fields without fallback to local defaults."""
    env = connection_env(**fields)
    assert normalize_connection(provider, env) == backend
    assert expected.items() <= env.items()
    assert not any(key.startswith("AUTORAG_DB_") for key in env)
    assert normalize_connection(provider, env) == backend


@pytest.mark.parametrize(
    ("provider", "fields", "match"),
    [
        ("", {"URI": "postgres://user:password@db/docs"}, "provider is required"),
        ("pgvector", {"URI": "postgres://user:password@db/docs"}, "provider must be"),
        ("postgres", {"URI": "postgres://user:password@db:bad/docs"}, "Invalid PostgreSQL URI"),
        ("postgres", {"URI": "postgres://user:password@db:0/docs"}, "port must be"),
        ("postgres", {"URI": "postgres://user:password@db/docs?sslmode="}, "Invalid PostgreSQL URI"),
        ("postgres", {"URI": "postgres://user:password@db/docs?options=secret"}, "Invalid PostgreSQL URI"),
        ("postgres", {"URI": "postgres://user:password@db/docs", "host": "db"}, "mixes DCH"),
        ("postgres", {"host": "db"}, "Shared PostgreSQL secret requires"),
        ("neo4j", {"URI": "neo4j://graph:7687"}, "requires PASSWORD"),
        ("neo4j", {"URI": "neo4j+ssc://graph:7687", "PASSWORD": "password"}, "Invalid database URI"),
        ("neo4j", {"URI": "bolt+ssc://graph:7687", "PASSWORD": "password"}, "Invalid database URI"),
        ("milvus", {"URI": "postgres://user:password@db/docs"}, "Invalid database URI"),
        ("neo4j", {"URI": "neo4j://graph:7687", "PASSWORD": "password", "TOKEN": "secret"}, "incompatible"),
    ],
)
def test_invalid_connections_fail_without_leaking_or_mutating(provider, fields, match):
    """Do not silently drop unsupported fields or expose credential-bearing URIs."""
    env = connection_env(**fields)
    original = env.copy()
    with pytest.raises(ValueError, match=match) as error:
        normalize_connection(provider, env)
    assert env == original
    assert "user:password" not in str(error.value)


def test_explicit_provider_rejects_mixed_or_mismatched_secrets():
    """An explicit provider must not select credentials for a different backend."""
    for env in (
        {"MILVUS_URI": "http://milvus"},
        {"PGVECTOR_HOST": "db", "NEO4J_URI": "neo4j://db"},
        connection_env(URI="postgres://rag@db/docs") | {"PGVECTOR_HOST": "db"},
    ):
        with pytest.raises(ValueError, match="configuration|mixes"):
            normalize_connection("postgres", env)


def test_legacy_precedence_and_missing_configuration():
    """Omitting provider preserves existing prefixed-secret selection."""
    assert normalize_connection("", {"MILVUS_URI": "http://m", "NEO4J_URI": "neo4j://g"}) == "milvus"
    with pytest.raises(ValueError, match="No vector database configuration"):
        normalize_connection("", {})
    assert normalize_connection("", {}, allow_missing=True) == "milvus"


@pytest.mark.parametrize(
    ("env", "expected_backend"),
    [
        ({"MILVUS_URI": "https://milvus:19530", "MILVUS_TOKEN": "token"}, "milvus"),
        ({"PGVECTOR_HOST": "database", "PGVECTOR_DB": "documents", "PGVECTOR_PASSWORD": "password"}, "pgvector"),
        ({"NEO4J_URI": "neo4j://graph:7687", "NEO4J_PASSWORD": "password", "NEO4J_DATABASE": "docs"}, "neo4j"),
        ({"MILVUS_URI": "https://milvus:19530", "PGVECTOR_HOST": "database"}, "milvus"),
        ({"PGVECTOR_HOST": "database", "NEO4J_URI": "neo4j://graph:7687"}, "pgvector"),
    ],
)
def test_omitted_provider_preserves_legacy_backend_and_environment(env, expected_backend):
    """Callers omitting the new input retain their credentials and historical precedence."""
    original = env.copy()
    assert normalize_connection(environ=env) == expected_backend
    assert env == original


def test_injected_environment_names_do_not_collide():
    """DCH PASSWORD and shared PostgreSQL password must be separate optional refs."""
    assert len(set(SECRET_KEY_TO_ENV.values())) == len(SECRET_KEY_TO_ENV)


@pytest.mark.parametrize("sslmode", [None, "", "verify-full"])
def test_shared_postgres_sslmode_is_optional(sslmode):
    """Shared PostgreSQL accepts omitted or empty sslmode."""
    fields = {"host": "db", "port": "5432", "user": "rag", "password": "secret", "dbname": "docs"}
    if sslmode is not None:
        fields["sslmode"] = sslmode
    env = connection_env(**fields)
    assert normalize_connection("postgres", env) == "pgvector"
    assert env["PGVECTOR_DB"] == "docs"
    if sslmode:
        assert env["PGVECTOR_SSLMODE"] == sslmode
    else:
        assert "PGVECTOR_SSLMODE" not in env


def test_adapter_loads_without_repository_package():
    """The exact module shipped in the runtime artifact is standalone."""
    path = Path(__file__).parents[1] / "runtime_embed" / "database_connection.py"
    spec = importlib.util.spec_from_file_location("standalone_connection", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.normalize_connection("milvus", connection_env(URI="https://milvus")) == "milvus"


@pytest.mark.parametrize(
    ("provider", "fields", "expected"),
    [
        (
            "postgres",
            {
                "host": "db",
                "port": "5432",
                "user": "rag",
                "password": "password",
                "dbname": "docs",
                "schema": "tenant",
                "sslmode": "require",
            },
            {"PGVECTOR_SCHEMA": "tenant", "PGVECTOR_SSLMODE": "require"},
        ),
        (
            "postgres",
            {"URI": "postgres://rag:password@db/docs?sslmode=verify-full"},
            {"PGVECTOR_SSLMODE": "verify-full"},
        ),
        ("milvus", {"URI": "https://milvus:19530", "DATABASE": "tenant"}, {"MILVUS_DATABASE": "tenant"}),
    ],
)
def test_properties_are_forwarded_without_runtime_capability_gates(provider, fields, expected):
    """The pipeline maps properties; it does not impose additional backend capabilities."""
    env = connection_env(**fields)
    normalize_connection(provider, env)
    assert expected.items() <= env.items()
