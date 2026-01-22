# ADR 0027: Single-tenant DB and instance metadata marker

Date: 2026-01-22

Status: Accepted

Decision

- Each experiment uses its own Postgres database and its own scheduler/worker processes.
- `EXPERIMENT_ID` is required only for process attachment and external naming (Redis namespace/queue, job branch prefix, logs/artifacts paths).
- The database stores a single-row `InstanceMetadata` marker with the expected experiment identity and root commit.
- Scheduler/worker startup must fail fast if the marker is missing or mismatched; the upgrade path is a destructive reset (`uv run loreley reset-db --yes`).

