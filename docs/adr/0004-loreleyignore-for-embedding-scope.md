# ADR 0004: `.loreleyignore` for embedding scope

Date: 2026-01-06

Context: Users need a way to control which repository files participate in embeddings without changing Git tracking rules.
Decision: Support a repository-root `.loreleyignore` file and parse it with gitignore semantics via `pathspec.gitignore.GitIgnoreSpec`.
Details: In experiment runs, ignore rules are pinned at experiment creation time from the experiment root commit and persisted in `Experiment.config_snapshot` as `mapelites_repo_state_ignore_text` (with `mapelites_repo_state_ignore_sha256` as a guard). Runtime repo-state embeddings use the pinned matcher and do not consult per-commit or working-tree ignore files.
Details: Apply `.gitignore` rules first, then `.loreleyignore` rules, so `.loreleyignore` may override via negation (`!pattern`).
Constraints: Matching remains root-only (no nested `.gitignore` files and no global excludes).
Consequences: Repo-state eligibility is stable for an experiment, removing dynamic-ignore edge cases from incremental-only ingestion.

