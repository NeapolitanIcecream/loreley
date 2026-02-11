# ADR 0038: Worker planning context batch DB loading

Date: 2026-02-11

Status: Accepted

Decision

- Load planning contexts for base and inspirations in one batch per job, instead of one DB round-trip chain per commit hash.
- Query `CommitCard` with `commit_hash IN (...)`, then query `Metric` with `commit_card_id IN (...)`, and query `MapElitesArchiveCell` with `island_id + commit_hash IN (...)`.
- Rebuild `CommitPlanningContext` objects in the original input order so planning prompt semantics remain unchanged.
- Keep existing fallback behavior for missing commit cards and missing highlights.
- Fail fast when one `(island_id, commit_hash)` maps to multiple archive rows, because that violates the single-context assumption.
