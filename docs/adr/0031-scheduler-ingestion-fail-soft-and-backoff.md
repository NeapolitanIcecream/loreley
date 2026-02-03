# ADR 0031: Scheduler ingestion fail-soft and backoff

Date: 2026-02-03

Status: Accepted

Decision

- Treat ingestion of succeeded jobs as a best-effort, per-job operation: failures are recorded on the job row and do not abort the scheduler tick.
- Apply an exponential retry backoff for failed ingestions based on `ingestion_attempts` and `ingestion_last_attempt_at`.
- Keep fail-fast behaviour for scheduler startup invariants (e.g. root repo-state bootstrap) and database write failures.
