# ADR 0009: Immutable file embedding cache (insert-only)

Date: 2026-01-08

Context: Repo-state embeddings rely on a file-level embedding cache keyed by `(blob_sha, embedding_model, dimensions, pipeline_signature)` and on incremental aggregation being stable across time.
Decision: `MapElitesFileEmbeddingCache` entries are immutable values; writes are insert-only and conflicts are ignored (no overwrite of existing vectors).
Constraints: Correcting embeddings requires changing `pipeline_signature` (or resetting the database in development) rather than overwriting an existing cache key.
Consequences: Concurrent schedulers/workers can safely attempt to insert the same blob without causing vector drift; time-based heuristics are unnecessary for cache correctness.


