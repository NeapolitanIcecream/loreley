# Repository-state embeddings (file cache)

This page documents the repo-state embedding pipeline used by MAP-Elites.

## Motivation

Repo-state embeddings represent the **entire repository state** at a commit by
aggregating file-level embeddings into a single commit vector. This makes the
behaviour descriptor depend on the repository snapshot at `treeish`, not just a
subset of changed files.

## High-level pipeline

At a given `treeish` (typically a commit hash), we:

1. Enumerate eligible files and resolve their **git blob SHA** fingerprints.
2. Look up each blob SHA in a **file embedding cache**.
3. Embed only cache misses (new/changed blobs).
4. Aggregate per-file embeddings into one commit vector via **uniform mean**.
5. Feed the commit vector into PCA → MAP-Elites as the behaviour descriptor.

## File enumeration and filtering

Implemented by:

- `loreley.core.map_elites.repository_files.RepositoryFileCatalog`

Eligibility is determined by a combination of:

- Root `.gitignore` (best-effort glob matching).
- `MAPELITES_PREPROCESS_ALLOWED_EXTENSIONS` / `MAPELITES_PREPROCESS_ALLOWED_FILENAMES`.
- `MAPELITES_PREPROCESS_EXCLUDED_GLOBS`.
- `MAPELITES_PREPROCESS_MAX_FILE_SIZE_KB` (oversized blobs are skipped).
- `MAPELITES_REPO_STATE_MAX_FILES` (optional cap; when set, the eligible list is deterministically sub-sampled).

!!! note
    `.gitignore` filtering is currently **best-effort** and only uses the repository root `.gitignore` at the requested `treeish`. Nested `.gitignore` files and global excludes are not applied.

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

Backend selection:

- `MAPELITES_FILE_EMBEDDING_CACHE_BACKEND=db|memory` (default: `db`)

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


