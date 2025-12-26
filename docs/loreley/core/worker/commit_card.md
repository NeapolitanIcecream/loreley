# loreley.core.worker.commit_card

Deterministic commit-card enrichment utilities used by the evolution worker to populate bounded `CommitCard` fields for inspiration prompts and UI displays.

This module intentionally avoids LLM calls. It derives:

- a bounded list of `key_files` touched by a commit
- a bounded list of human-readable `highlights` (file-level churn summaries)

## CommitCardBuildResult

- **`CommitCardBuildResult`**: dataclass returned by the builder.
  - `key_files`: tuple of repo-relative file paths.
  - `highlights`: tuple of short strings (for example `path/to/file.py: +10/-3`).

## build_commit_card_from_git

- **`build_commit_card_from_git(worktree, base_commit, candidate_commit, max_key_files=20, max_highlights=8)`**:
  - Uses `git diff --name-only base..candidate` to enumerate touched files.
  - Uses `git diff --numstat base..candidate` to derive per-file added/deleted counts.
  - Sorts highlight candidates by total churn, then added lines.
  - Guarantees at least one highlight line so downstream prompt contracts remain valid.

## Downstream usage

The worker persists the derived data by:

- calling `build_commit_card_from_git(...)` inside `loreley.core.worker.job_store.EvolutionJobStore.persist_success()`
- clamping individual strings using `loreley.core.contracts.clamp_text(...)`
- storing the final result in `loreley.db.models.CommitCard` (`commit_cards` table)


