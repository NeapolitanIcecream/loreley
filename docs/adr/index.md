# Architecture decision records

This index lists the ADRs that document Loreley's design and implementation decisions.
Use it when you need historical context for a subsystem, a configuration choice, or a
behavior change that is already reflected in the code.

ADR numbers are unique and chronological. `ADR 0029` resolves an earlier duplicate
`0022` numbering conflict.

## 0001-0009

- [ADR 0001: Incremental repo-state embeddings (diff-based)](0001-incremental-repo-state-embedding.md)
- [ADR 0002: Repo-state embeddings use canonical commit hashes only](0002-repo-state-commit-hash-only.md)
- [ADR 0003: Experiment config locking and dispatch (behavior params)](0003-experiment-config-locking-and-dispatch.md)
- [ADR 0004: `.loreleyignore` for embedding scope](0004-loreleyignore-for-embedding-scope.md)
- [ADR 0005: Remove legacy MAP-Elites embedding flows](0005-remove-summary-embedding-flow.md)
- [ADR 0006: Auto-start UI API from Streamlit UI entrypoint](0006-auto-start-ui-api-from-ui.md)
- [ADR 0007: Vectorized occupied-cell neighbourhood sampling for MAP-Elites](0007-vectorized-occupied-cell-neighborhood-sampling.md)
- [ADR 0008: Single scheduler per experiment (Postgres advisory lock)](0008-single-scheduler-per-experiment-lock.md)
- [ADR 0009: Immutable file embedding cache (insert-only)](0009-immutable-file-embedding-cache.md)

## 0010-0019

- [ADR 0010: Incremental-only repo-state embeddings (after bootstrap)](0010-incremental-only-repo-state-embeddings.md)
- [ADR 0011: Remove repo-state max-files cap; require interactive startup approval](0011-remove-max-files-cap-add-startup-approval.md)
- [ADR 0012: Pin repo-state ignore rules](0012-pin-repo-state-ignore-rules.md)
- [ADR 0013: Remove filter_signature and prompt_signature from experiment-scoped caches](0013-remove-filter-and-prompt-signatures.md)
- [ADR 0014: Experiment-scoped file embedding cache (remove pipeline_signature)](0014-experiment-scoped-file-embedding-cache.md)
- [ADR 0015: Experiment-attached workers and per-experiment task queues](0015-experiment-attached-workers-and-per-experiment-queues.md)
- [ADR 0016: Use Typer for the unified Loreley CLI](0016-typer-unified-cli.md)
- [ADR 0017: Use Tenacity for OpenAI retry/backoff](0017-tenacity-openai-retries.md)
- [ADR 0018: Remove legacy `script/` wrappers](0018-remove-legacy-script-wrappers.md)
- [ADR 0019: Use pathspec for pinned gitignore matching](0019-pathspec-gitignore-matching.md)

## 0020-0029

- [ADR 0020: Split built-in agent backends and extract a shared agent runner](0020-split-agent-backends-and-extract-runner.md)
- [ADR 0021: Centralize HTTP calls on httpx via an internal client module](0021-httpx-internal-http-client.md)
- [ADR 0022: Reuse UI API client connections via Streamlit resource cache](0022-ui-api-client-connection-reuse.md)
- [ADR 0023: Lock experiment behavior keyset and settings injection](0023-lock-experiment-behavior-keyset-and-settings-injection.md)
- [ADR 0024: Env-only settings; remove experiment settings snapshot](0024-env-only-settings-no-experiment-snapshot.md)
- [ADR 0025: Explicit experiment ID; remove derived config identity and legacy snapshots](0025-explicit-experiment-id-env-only-and-drop-legacy-snapshot.md)
- [ADR 0026: Experiment ID as operational naming namespace](0026-experiment-id-as-naming-namespace.md)
- [ADR 0027: Single-tenant DB and instance metadata marker](0027-single-tenant-db-and-instance-metadata.md)
- [ADR 0028: Unify git commit availability and fail-fast best-fitness branch](0028-unify-git-commit-availability-and-fail-fast-best-branch.md)
- [ADR 0029: Use Pydantic `from_attributes` for UI API response schemas](0029-pydantic-from-attributes-ui-api-schemas.md)

## 0030-0039

- [ADR 0030: DB-only repo-state embeddings and explicit bootstrap](0030-db-only-repo-state-embeddings-and-explicit-bootstrap.md)
- [ADR 0031: Scheduler ingestion fail-soft and backoff](0031-scheduler-ingestion-fail-soft-and-backoff.md)
- [ADR 0032: Simplify worker prompts; make freeform Markdown the default](0032-simplify-worker-prompts-freeform-default.md)
- [ADR 0033: Remove schema/validation modes; simplify agent outputs](0033-remove-schema-validation-and-simplify-agent-outputs.md)
- [ADR 0034: Kilocode CLI agent backend](0034-kilocode-cli-agent-backend.md)
- [ADR 0035: Default planning/coding backend is Kilocode](0035-default-agent-backend-kilocode.md)
- [ADR 0036: Single source of truth for worker commits](0036-single-source-of-truth-for-worker-commits.md)
- [ADR 0037: Sampler lightweight cell/commit sampling](0037-sampler-lightweight-cell-commit-sampling.md)
- [ADR 0038: Worker planning context batch DB loading](0038-worker-planning-context-batch-db-loading.md)
- [ADR 0039: PCA epochs and archive rebuild on refit](0039-pca-epochs-and-archive-rebuild-on-refit.md)

## 0040-0046

- [ADR 0040: Delay MAP-Elites archive until initial PCA fit](0040-delay-map-elites-archive-until-initial-pca-fit.md)
- [ADR 0041: Split MAP-Elites manager modules](0041-map-elites-manager-module-split.md)
- [ADR 0042: Sample ingest INFO logs and gate progress to TTY](0042-sampled-ingest-logs-and-tty-progress.md)
- [ADR 0043: Cache PCA history matrix for refits](0043-cache-pca-history-matrix.md)
- [ADR 0044: Repo-state embeddings must embed all cache misses](0044-repo-state-embedding-must-not-truncate-cache-misses.md)
- [ADR 0045: Config profiles for large-repo campaigns](0045-config-profiles-for-large-repo-campaigns.md)
- [ADR 0046: Agent-visible evaluation artifacts](0046-agent-visible-evaluation-artifacts.md)
