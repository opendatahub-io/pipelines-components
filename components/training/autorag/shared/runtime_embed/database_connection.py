"""Normalize database secrets without persisting or logging credentials.

This module is embedded in KFP components. Keep it
self-contained and limited to the standard library.
"""

import logging
import os
from urllib.parse import parse_qs, unquote, urlsplit

PROVIDERS = {"postgres": "pgvector", "milvus": "milvus", "neo4j": "neo4j"}
LEGACY_KEYS = (
    "MILVUS_URI",
    "MILVUS_TOKEN",
    "MILVUS_DATABASE",
    "MILVUS_CA_CERT",
    "PGVECTOR_HOST",
    "PGVECTOR_PORT",
    "PGVECTOR_DB",
    "PGVECTOR_USER",
    "PGVECTOR_PASSWORD",
    "PGVECTOR_CA_CERT",
    "PGVECTOR_SCHEMA",
    "PGVECTOR_SSLMODE",
    "NEO4J_URI",
    "NEO4J_USERNAME",
    "NEO4J_PASSWORD",
    "NEO4J_DATABASE",
    "NEO4J_CA_CERT",
)
GENERIC_KEYS = (
    "URI",
    "TOKEN",
    "DATABASE",
    "USERNAME",
    "PASSWORD",
    "CA_CERT",
    "host",
    "port",
    "user",
    "password",
    "dbname",
    "schema",
    "sslmode",
    "ca.crt",
)
# Prefix generic keys so they cannot collide with other connections in the pod.
SECRET_KEY_TO_ENV = {key: key for key in LEGACY_KEYS} | {
    key: "AUTORAG_DB_" + key.replace(".", "_") for key in GENERIC_KEYS
}


def normalize_connection(provider="", environ=None, allow_missing=False):
    """Return the ai4rag backend and populate its legacy environment contract.

    Generic formats require an explicit DCH provider. Existing prefixed secrets
    retain their historical backend precedence when provider is omitted.
    Validation completes before mutating the environment. Errors never contain
    secret values, including invalid URIs.
    """
    env = os.environ if environ is None else environ
    if provider and provider not in PROVIDERS:
        raise ValueError("db_provider must be postgres, milvus, or neo4j.")
    generic = {key: env.get(SECRET_KEY_TO_ENV[key], "") for key in GENERIC_KEYS}
    generic = {key: value for key, value in generic.items() if value != ""}
    legacy = [
        backend
        for key, backend in (("MILVUS_URI", "milvus"), ("PGVECTOR_HOST", "pgvector"), ("NEO4J_URI", "neo4j"))
        if env.get(key)
    ]
    if not generic:
        if provider:
            backend = PROVIDERS[provider]
            if legacy != [backend]:
                raise ValueError("db_provider does not match an unambiguous database secret configuration.")
            return backend
        if legacy:
            return legacy[0]
        if allow_missing:
            logging.warning("No vector database configuration found; defaulting to milvus.")
            return "milvus"
        raise ValueError("No vector database configuration found in db_secret_name.")
    if not provider:
        raise ValueError("db_provider is required for DCH or shared PostgreSQL secrets.")
    if any(env.get(key) for key in LEGACY_KEYS):
        raise ValueError("Database secret mixes generic and provider-prefixed fields; use one format.")
    backend = PROVIDERS[provider]
    updates = {}
    if provider == "postgres":
        if generic.get("URI"):
            if any(key in generic for key in ("host", "port", "user", "dbname", "schema", "sslmode", "ca.crt")):
                raise ValueError("Database secret mixes DCH URI and shared PostgreSQL fields.")
            try:
                uri = urlsplit(generic["URI"])
                if uri.scheme not in {"postgres", "postgresql"} or not uri.hostname or not uri.username:
                    raise ValueError
                if not uri.path.strip("/") or uri.fragment:
                    raise ValueError
                query = parse_qs(uri.query, keep_blank_values=True, strict_parsing=True)
                if set(query) - {"sslmode"} or any(len(values) != 1 for values in query.values()):
                    raise ValueError
                if "sslmode" in query and not query["sslmode"][0]:
                    raise ValueError
                updates = {
                    "PGVECTOR_HOST": uri.hostname,
                    "PGVECTOR_PORT": str(5432 if uri.port is None else uri.port),
                    "PGVECTOR_DB": unquote(uri.path[1:]),
                    "PGVECTOR_USER": unquote(uri.username),
                    "PGVECTOR_PASSWORD": unquote(uri.password or ""),
                }
                sslmode = query.get("sslmode", [""])[0]
            except ValueError:
                raise ValueError("Invalid PostgreSQL URI or unsupported URI parameters in db_secret_name.") from None
            ca = generic.get("CA_CERT", "")
            allowed = {"URI", "CA_CERT"}
        else:
            required = ("host", "port", "user", "password", "dbname")
            if any(not generic.get(key) for key in required):
                raise ValueError("Shared PostgreSQL secret requires host, port, user, password, and dbname.")
            updates = {
                "PGVECTOR_" + target: generic[source]
                for source, target in (
                    ("host", "HOST"),
                    ("port", "PORT"),
                    ("user", "USER"),
                    ("password", "PASSWORD"),
                    ("dbname", "DB"),
                )
            }
            if generic.get("schema"):
                updates["PGVECTOR_SCHEMA"] = generic["schema"]
            sslmode, ca = generic.get("sslmode", ""), generic.get("ca.crt", "")
            allowed = set(required) | {"schema", "sslmode", "ca.crt"}
        if sslmode:
            updates["PGVECTOR_SSLMODE"] = sslmode
        try:
            if not 1 <= int(updates["PGVECTOR_PORT"]) <= 65535:
                raise ValueError
        except ValueError:
            raise ValueError("PostgreSQL port must be an integer from 1 to 65535.") from None
        if ca:
            updates["PGVECTOR_CA_CERT"] = ca
    else:
        if not generic.get("URI"):
            raise ValueError("DCH database secret requires URI.")
        try:
            uri = urlsplit(generic["URI"])
            schemes = {"http", "https"} if provider == "milvus" else {"neo4j", "neo4j+s", "bolt", "bolt+s"}
            if uri.scheme not in schemes or not uri.hostname or uri.fragment:
                raise ValueError
            if uri.port is not None and not 1 <= uri.port <= 65535:
                raise ValueError
            if uri.username is not None or uri.password is not None:
                raise ValueError
        except ValueError:
            raise ValueError(
                "Invalid database URI for the selected provider; supply authentication separately."
            ) from None
        prefix = provider.upper() + "_"
        updates[prefix + "URI"] = generic["URI"]
        if provider == "neo4j":
            if not generic.get("PASSWORD"):
                raise ValueError("DCH Neo4j secret requires PASSWORD.")
            allowed = {"URI", "USERNAME", "PASSWORD", "DATABASE", "CA_CERT"}
            updates[prefix + "USERNAME"] = generic.get("USERNAME", "neo4j")
            updates[prefix + "PASSWORD"] = generic["PASSWORD"]
            updates[prefix + "DATABASE"] = generic.get("DATABASE", "neo4j")
        else:
            allowed = {"URI", "TOKEN", "DATABASE", "CA_CERT"}
            if generic.get("DATABASE"):
                updates[prefix + "DATABASE"] = generic["DATABASE"]
            if generic.get("TOKEN"):
                updates[prefix + "TOKEN"] = generic["TOKEN"]
        if generic.get("CA_CERT"):
            updates[prefix + "CA_CERT"] = generic["CA_CERT"]
    if set(generic) - allowed:
        raise ValueError("Database secret contains fields incompatible with the selected provider or format.")
    env.update(updates)
    # Make subsequent resolution idempotent while keeping generic credentials out
    # of generated artifacts. The normalized environment is process-local.
    for key in set(SECRET_KEY_TO_ENV[key] for key in GENERIC_KEYS):
        env.pop(key, None)
    return backend
