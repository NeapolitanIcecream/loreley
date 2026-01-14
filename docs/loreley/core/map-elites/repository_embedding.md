# Repository-state embeddings (file cache)

This page documents the repo-state embedding pipeline used by MAP-Elites.

## Motivation

Repo-state embeddings represent the **entire repository state** at a commit by
aggregating file-level embeddings into a single commit vector. This makes the
behaviour descriptor depend on the repository snapshot at `commit_hash`, not just a
subset of changed files.

## High-level pipeline

At a given `commit_hash`, we:

1. Try to reuse a persisted **repo-state aggregate** for the commit (fast path).
2. If missing, derive it from:
   - **Bootstrap**: compute the root commit aggregate by fully enumerating eligible files.
   - **Runtime (incremental-only)**: derive the aggregate from the single parent commit using a parent..child diff; if this is not possible, fail fast.
3. Look up each blob SHA in the **file embedding cache** and embed only cache misses.
4. Aggregate per-file embeddings into one commit vector via **uniform mean**.
5. Feed the commit vector into PCA → MAP-Elites as the behaviour descriptor.

## File enumeration and filtering

Implemented by:

- `loreley.core.map_elites.repository_files.RepositoryFileCatalog`

Eligibility is determined by a combination of:

- Root `.gitignore` + `.loreleyignore` (gitignore semantics via `pathspec`).
- `MAPELITES_PREPROCESS_ALLOWED_EXTENSIONS` / `MAPELITES_PREPROCESS_ALLOWED_FILENAMES`.
- `MAPELITES_PREPROCESS_EXCLUDED_GLOBS`.
- `MAPELITES_PREPROCESS_MAX_FILE_SIZE_KB` (oversized blobs are skipped).
- Scheduler startup approval gate: the root eligible file count is scanned at startup and must be explicitly approved by the operator (interactive y/n prompt by default, or `--yes` / `SCHEDULER_STARTUP_APPROVE=true` for non-interactive runs).

!!! note
    Ignore filtering uses `pathspec.gitignore.GitIgnoreSpec` to apply repository-root `.gitignore` + `.loreleyignore` rules with gitignore semantics. In experiment runs, the effective ignore rules are **pinned at experiment creation time** and persisted in `Experiment.config_snapshot` as `mapelites_repo_state_ignore_text` (derived from the experiment root commit). `.loreleyignore` rules are applied after `.gitignore` (so `!pattern` can re-include). Nested `.gitignore` files and global excludes are not applied.

For each eligible file we keep:

- `path` (repo-root relative)
- `blob_sha` (content fingerprint)
- `size_bytes`

## File embedding cache

Implemented by:

- `loreley.core.map_elites.file_embedding_cache.InMemoryFileEmbeddingCache`
- `loreley.core.map_elites.file_embedding_cache.DatabaseFileEmbeddingCache`
- ORM table: `loreley.db.models.MapElitesFileEmbeddingCache`

Cache key:

- `experiment_id`
- `blob_sha`

The embedding model name and output dimensionality are experiment-scoped
invariants pinned in `Experiment.config_snapshot`. The database cache stores
`embedding_model` and `dimensions` alongside vectors for validation and debugging.

The database-backed cache is **insert-only**: when multiple processes attempt to
write the same key, the first insert wins and later writes are ignored (no overwrite).

Backend selection:

- `MAPELITES_FILE_EMBEDDING_CACHE_BACKEND=db|memory` (default: `db`)

Note: The database-backed cache is experiment-scoped and requires an `experiment_id`. Use the `memory` backend for local runs without an experiment context.

## Repo-state aggregate cache (commit-level)

When `MapElitesManager` is constructed with an `experiment_id` (the scheduler does this),
repo-state embeddings persist a commit-level aggregate so future ingests can avoid
re-enumerating the full tree.

Stored in:

- ORM table: `loreley.db.models.MapElitesRepoStateAggregate`

The aggregate stores:

- `sum_vector`: sum of all per-file vectors included in the commit representation
- `file_count`: number of file paths contributing to `sum_vector`

The commit vector is derived as `sum_vector / file_count`.

### Incremental updates

When a parent aggregate exists and the commit has exactly one parent, the child aggregate is
derived from the parent by applying the parent..child diff (add/modify/delete/rename) and
embedding only the new/changed blobs. Ignore rules are pinned for the experiment lifecycle,
so ignore file changes in the evolved history do not affect eligibility. If the commit shape
is unsupported (merge commits, missing parent aggregate, or diff failures), runtime ingestion
fails fast.

## Commit aggregation

Implemented by:

- `loreley.core.map_elites.repository_state_embedding.RepositoryStateEmbedder`

Let \(v_i\) be the embedding vector for eligible file \(i\), and \(N\) be the
number of eligible files with available vectors. The repo-state commit vector is
the **uniform mean**:

\[
v_{commit} = \frac{1}{N}\sum_{i=1}^{N} v_i
\]

If multiple paths point at the same blob SHA, the corresponding \(v_i\) is the
same vector but still contributes once per file path (uniform per-file weighting).


