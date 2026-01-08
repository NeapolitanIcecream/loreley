# ADR 0008: Single scheduler per experiment (Postgres advisory lock)

Date: 2026-01-08

Context: Running multiple scheduler instances against the same experiment can concurrently ingest commits and write shared DB state (repo-state aggregates, file embedding cache), creating race conditions and non-deterministic behaviour.
Decision: The scheduler MUST acquire a Postgres advisory lock scoped to `experiment_id` at startup; if the lock cannot be obtained, the process exits (fail fast).
Constraints: The lock is held for the lifetime of the scheduler process using a dedicated DB connection; losing the connection releases the lock.
Consequences: Exactly one scheduler instance can operate on an experiment at a time; deployments scale workers, not schedulers, for throughput.


