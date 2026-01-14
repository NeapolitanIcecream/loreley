# ADR 0019: Use pathspec for pinned gitignore matching

Date: 2026-01-14

Context: Repo-state embeddings rely on pinned repository-root ignore rules (`mapelites_repo_state_ignore_text`) and must support gitignore features such as negation and directory rules with correct semantics.
Decision: Compile the pinned ignore text with `pathspec.gitignore.GitIgnoreSpec` and use `match_file()` to filter git-root-relative paths.
Constraints: Ignore matching remains root-only (no nested `.gitignore` files and no global excludes).
Consequences: Ignore behavior matches gitignore semantics more closely; a small number of edge cases may change eligibility and should be validated by tests when upgrading.

