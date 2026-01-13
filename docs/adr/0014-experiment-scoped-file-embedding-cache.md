# ADR 0014: Experiment-scoped file embedding cache (remove pipeline_signature)

Date: 2026-01-13

Context: File-level embeddings are reused across commits and must remain consistent with locked experiment behaviour settings.
Decision: Scope `MapElitesFileEmbeddingCache` by `experiment_id` and key rows by `(experiment_id, blob_sha)`; store `embedding_model` and `dimensions` for validation; remove `pipeline_signature`.
Consequences: File embeddings cannot be reused across experiments with different snapshots; cache keys are simpler and aligned with experiment config locking. Upgrading requires resetting the development database schema (Loreley does not ship migrations).

