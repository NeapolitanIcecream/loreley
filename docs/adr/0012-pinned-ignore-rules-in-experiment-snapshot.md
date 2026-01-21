# ADR 0012: Pin repo-state ignore rules

Date: 2026-01-09

Status: Superseded by ADR 0025.

Decision

Loreley pins repo-root ignore rules at scheduler startup (from the configured root
commit) and stores the ignore text + hash in process-local `Settings` for the
lifetime of the scheduler process. The database does not persist ignore rules as
an experiment settings snapshot.

