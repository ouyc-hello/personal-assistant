# PostgreSQL migrations

`001_initial.sql` is the first production-oriented schema for the business store.

- PostgreSQL owns the transactional state and audit trail.
- The `vector` column is an indexable projection for memory retrieval.
- Milvus and Neo4j projections must keep the PostgreSQL document/entity IDs.
- Do not edit an applied migration in place; add `002_*.sql` for changes.

For local development, `assistant db-init` uses SQLAlchemy `create_all` against the configured database. It is intentionally separate from production migration execution.
