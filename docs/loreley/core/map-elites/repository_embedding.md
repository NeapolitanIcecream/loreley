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

- Root `.gitignore` + `.loreleyignore` (best-effort glob matching).
- `MAPELITES_PREPROCESS_ALLOWED_EXTENSIONS` / `MAPELITES_PREPROCESS_ALLOWED_FILENAMES`.
- `MAPELITES_PREPROCESS_EXCLUDED_GLOBS`.
- `MAPELITES_PREPROCESS_MAX_FILE_SIZE_KB` (oversized blobs are skipped).
- Scheduler startup approval gate: the root eligible file count is scanned at startup and must be explicitly approved by the operator (interactive y/n prompt by default, or `--yes` / `SCHEDULER_STARTUP_APPROVE=true` for non-interactive runs).

!!! note
    Ignore filtering is currently **best-effort** and only uses the repository root `.gitignore` and `.loreleyignore` at the requested `commit_hash`. `.loreleyignore` rules are applied after `.gitignore` (so `!pattern` can re-include). Nested `.gitignore` files and global excludes are not applied.

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

- `blob_sha`
- `embedding_model`
- `dimensions` (actual output vector length guard)
- `pipeline_signature`

`pipeline_signature` is a SHA-256 hash over the preprocessing/chunking/embedding
knobs that affect the produced vectors (so cache entries are invalidated when
the pipeline changes).

The database-backed cache is **insert-only**: when multiple processes attempt to
write the same key, the first insert wins and later writes are ignored (no overwrite).

Backend selection:

- `MAPELITES_FILE_EMBEDDING_CACHE_BACKEND=db|memory` (default: `db`)

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

When a parent aggregate exists, the commit has exactly one parent, and root ignore files
(`.gitignore` and `.loreleyignore`) are unchanged, the child aggregate is derived from the
parent by applying the parent..child diff (add/modify/delete/rename) and embedding only the
new/changed blobs. If these conditions are not met (merge commits, missing parent aggregate,
diff failures, or root ignore changes), runtime ingestion fails fast.

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


