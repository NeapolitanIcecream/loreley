# ADR 0005: Remove legacy MAP-Elites embedding flows

Date: 2026-01-06

Context: MAP-Elites uses repo-state embeddings; additional embedding flows increase complexity and schema surface without improving the active pipeline.
Decision: Keep a single repo-state embedding pipeline and remove summary embeddings, changed-files preprocessing, and `MAPELITES_EMBEDDING_MODE`.
Details: PCA history stores only `(commit_hash, vector, embedding_model)` via `PcaHistoryEntry` and `map_elites_pca_history.embedding_model`.
Details: Snapshot persistence writes incremental state with `storage_backend="cells_history_v2"` and upserts history rows idempotently per commit hash.
Constraints: Loreley does not ship migrations; schema changes are applied via `uv run loreley reset-db --yes`.
Consequences: `MAPELITES_SUMMARY_*` settings and changed-files preprocessing APIs are removed; deployments must reset DB to upgrade.


