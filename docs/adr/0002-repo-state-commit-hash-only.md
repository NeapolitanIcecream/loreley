# ADR 0002: Repo-state embeddings use canonical commit hashes only

Date: 2026-01-04
Commits: 480319e

Context: Repo-state embeddings persist commit-level aggregates and rely on cache keys being stable and immutable across time and process restarts.
Decision: All repo-state embedding entrypoints accept a git commit identifier and immediately resolve it to a canonical full hash (`Commit.hexsha`). This canonical hash is the only value used for persistence keys, incremental aggregation, and repository snapshot reads.
Constraints: Symbolic refs (branch names, tags, `HEAD`) are not part of the MAP-Elites work loop contract; callers provide commit hashes.
Consequences: Cache correctness is guaranteed even if a caller accidentally provides a movable reference, because it is resolved before persistence. Interfaces are simplified by removing `treeish` parameters and using `commit_hash` consistently.


