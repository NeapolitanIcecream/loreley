# ADR 0012: Pin repo-state ignore rules in the experiment snapshot

Date: 2026-01-09

Context: Repo-state embeddings run in incremental-only mode after bootstrap. Commit-scoped `.gitignore` / `.loreleyignore` changes introduce extra guards and fail-fast edge cases because eligibility can change non-locally.

Decision: Persist a **pinned root ignore rule set** (`mapelites_repo_state_ignore_text` + `mapelites_repo_state_ignore_sha256`) inside `Experiment.config_snapshot` at experiment creation time, derived from the experiment root commit. All repo-state file selection and incremental diffs use this pinned matcher for the full experiment lifecycle.

Decision: Introduce `experiment_snapshot_schema_version` and require it (and the pinned ignore keys) when loading experiment settings. Loreley does not support forward-compatible snapshot schemas; upgrades require resetting the dev database schema.

Consequences: The embedding pipeline no longer needs “ignore changed => disable incremental” logic, and experiments avoid dynamic-ignore edge cases that previously caused incremental-only ingestion failures.

