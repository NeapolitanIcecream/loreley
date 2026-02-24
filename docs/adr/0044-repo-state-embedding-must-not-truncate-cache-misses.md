# ADR 0044: Repo-state embeddings must embed all cache misses

Date: 2026-02-24

Context: Repo-state commit embeddings are defined as the uniform mean across all eligible repository files at a commit. Cache misses (new/modified blobs) must be embedded so aggregates remain consistent with this definition.

Decision: Remove the global chunk budget from the code embedding layer and embed cache-miss files in bounded batches. This ensures repo-state aggregates are computed from the full eligible file set while keeping request payloads and memory usage manageable.

Consequences: Bootstrap and large diffs may require multiple embedding batches; embeddings remain deterministic given pinned filters and stable batching order. Cost control should be achieved via preprocessing and chunking limits (file size, excluded globs, max chunks per file), not by silently dropping chunks.

