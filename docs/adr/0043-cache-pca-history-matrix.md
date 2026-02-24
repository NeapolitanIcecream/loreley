# ADR 0043: Cache PCA history matrix for refits

Date: 2026-02-24

Status: Accepted

Decision

- Maintain an internal float64 history matrix cache inside `DimensionReducer` to avoid rebuilding PCA sample matrices on every refit.
- Update the cache incrementally on history upserts and evictions; rebuild only on cache invalidation.
- Extend `reduce_commit_embeddings(...)` with an optional `reducer=...` parameter to reuse reducer state across repeated calls.
- Cache one reducer per island in `MapElitesManager` and synchronise the projection after refit alignment.
