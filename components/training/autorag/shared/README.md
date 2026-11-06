# AutoRAG database connections

Optimization and production indexing accept `db_provider` and `db_secret_name`.
`db_provider` uses the Data Connect Hub `provider` value: `postgres`, `milvus`, or `neo4j`.
`postgres` selects ai4rag's `pgvector` backend. It does not select a secret format.
Omitting `db_provider` preserves inference from existing provider-prefixed secrets.
Generic secrets require an explicit provider. A secret mixing generic and
provider-prefixed fields is rejected.

| Secret format | Keys |
| --- | --- |
| Legacy AutoRAG | Existing `MILVUS_*`, `PGVECTOR_*`, or `NEO4J_*` keys |
| DCH PostgreSQL/PGVector | `URI`, optional `CA_CERT` |
| DCH Milvus | `URI`, optional `TOKEN`, `DATABASE`, `CA_CERT` |
| DCH Neo4j | `URI`, `PASSWORD`, optional `USERNAME`, `DATABASE`, `CA_CERT` |
| Shared PostgreSQL | `host`, `port`, `user`, `password`, `dbname`, optional `sslmode`, `schema`, `ca.crt` |

The platform must make all required connection properties available in the
selected secret. Pipelines do not query DCH or resolve `data_connection_type_id`.
Generic keys are injected into isolated `AUTORAG_DB_*` environment variables;
the embedded adapter maps them to ai4rag's environment contract before use.
PostgreSQL URIs accept percent-encoded credentials, IPv6 hosts, a database path,
and the `sslmode` query parameter. Other URI parameters are rejected rather than
silently discarded. Certificates are PEM content, not filesystem paths.

## Property mapping

The pipeline maps the selected secret into provider-prefixed environment
variables before calling the existing ai4rag APIs. No ai4rag source or dependency
changes are part of this implementation.

| Provider | Secret property | Environment variable |
| --- | --- | --- |
| postgres | `host`, `port`, `dbname`, `user`, `password` | `PGVECTOR_HOST`, `PGVECTOR_PORT`, `PGVECTOR_DB`, `PGVECTOR_USER`, `PGVECTOR_PASSWORD` |
| postgres | `schema`, `sslmode` | `PGVECTOR_SCHEMA`, `PGVECTOR_SSLMODE` |
| postgres | `ca.crt` or DCH `CA_CERT` | `PGVECTOR_CA_CERT` |
| milvus | `URI`, `TOKEN`, `DATABASE`, `CA_CERT` | `MILVUS_URI`, `MILVUS_TOKEN`, `MILVUS_DATABASE`, `MILVUS_CA_CERT` |
| neo4j | `URI`, `USERNAME`, `PASSWORD`, `DATABASE`, `CA_CERT` | `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`, `NEO4J_CA_CERT` |

A DCH PostgreSQL `URI` is parsed into the same `PGVECTOR_*` connection fields.

## Deployment and ownership

Each pattern's indexing blueprint records `db_provider` alongside
`db_secret_name`; `settings.store_binding.provider_type` retains ai4rag's backend
name. Starter-kit generation uses the existing ai4rag generator unchanged.
Credential values are never embedded in the indexing blueprint. Starter-kit and
local development environments continue to use provider-prefixed environment
variables; the pipeline's environment mapping applies to pipeline execution.

Database/Schema claims, `vector` extension requests, secret creation, and RBAC
remain platform responsibilities. Pipeline steps consume an existing secret;
they do not create database infrastructure or modify service-account grants.
