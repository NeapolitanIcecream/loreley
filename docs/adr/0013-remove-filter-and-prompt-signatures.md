# ADR 0013: Remove filter_signature and prompt_signature from experiment-scoped caches

Date: 2026-01-09

Context: Loreley locks experiment-scoped behaviour settings in `Experiment.config_snapshot` to avoid dynamic configuration complexity. Some caches still include extra signature dimensions (`filter_signature`, `prompt_signature`) that only exist to support dynamic changes.

Decision: Remove `filter_signature` from repo-state commit aggregates and key `MapElitesRepoStateAggregate` only by `(experiment_id, commit_hash)`.

Decision: Remove `prompt_signature` from trajectory chunk summaries and key `CommitChunkSummary` only by `(experiment_id, start_commit_hash, end_commit_hash, block_size)`.

Consequences: Aggregate and summary caches become simpler and aligned with the locked experiment settings model. Upgrading requires resetting the development database schema (Loreley does not ship migrations).

