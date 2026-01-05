# ADR 0001: Incremental repo-state embeddings (diff-based)

Date: 2026-01-04
Commits: a00c179

Context: Repo-state ingest enumerates the full git tree at each commit and performs per-file filtering; on large repositories this can dominate scheduler throughput.
Decision: Persist a per-commit repo-state aggregate (sum vector + eligible file count) and derive child aggregates by applying parent..child diffs (add/modify/delete/rename) using GitPython, embedding only missing blobs via the existing file cache.
Constraints: Treat `MAPELITES_REPO_STATE_MAX_FILES` as a rare safety cap; if the diff-updated count would exceed the cap or `.gitignore` changes, fall back to a full recompute for correctness.
Consequences: Most ingests become O(changed_files) and avoid full-tree traversal and full-cache reads; merges and cap-triggering commits may still require full recomputation.


