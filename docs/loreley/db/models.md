# loreley.db.models

ORM models and enums for single-tenant experiment databases.

## Shared mixins and enums

- **`TimestampMixin`**: adds `created_at` and `updated_at` columns that default to `now()` and automatically update on modification.
- **`JobStatus`**: string-based `Enum` capturing the lifecycle of an evolution job (`PENDING`, `QUEUED`, `RUNNING`, `SUCCEEDED`, `FAILED`, `CANCELLED`).

## Core models

- **`InstanceMetadata`** (`instance_metadata` table): single-row marker for DB identity.
  - Primary key: `id=1` enforced by a check constraint.
  - Stores `schema_version`, `experiment_id_raw`, `experiment_uuid`, and the canonical `root_commit_hash`.
  - Optional `repository_slug` and `repository_canonical_origin` fields aid observability.
- **`CommitCard`** (`commit_cards` table): lightweight commit metadata used for inspiration and UI.
  - Primary key: `id` (UUID).
  - Unique constraint on `commit_hash` per database.
  - Tracks commit hash, parent hash, optional island identifier, optional `job_id`, author, subject, change summary, evaluation summary, tags, key files, and highlights.
  - Defines relationships to associated `Metric` records (via `Metric.commit_card_id`).
- **`Metric`** (`metrics` table): records individual evaluation metrics for a commit.
  - Stores metric `name`, numeric `value`, optional `unit`, whether higher values are better, and a JSONB `details` payload.
  - Links back to `CommitCard` via `commit_card_id` and maintains uniqueness per `(commit_card_id, name)`.
- **`EvolutionJob`** (`evolution_jobs` table): represents a single evolution iteration scheduled by the system.
  - Tracks current `status`, base commit, island ID, inspiration commit hashes, size-bounded job spec fields (`goal`, `constraints`, `acceptance_criteria`, `notes`, `tags`, sampling hints), human-readable `plan_summary`, priority, scheduling/processing timestamps, and last error if any.
  - Stores result/ingestion indexing fields (`result_commit_hash`, ingestion status/attempts/delta/cell index) without embedding large JSON payloads.
- **`JobArtifacts`** (`job_artifacts` table): filesystem references for cold-path artifacts produced by the worker.
  - Stores paths to planning/coding/evaluation prompts, raw outputs, and logs.
- **`CommitChunkSummary`** (`commit_chunk_summaries` table): cached trajectory summaries for commit chains.
  - Primary key: `(start_commit_hash, end_commit_hash, block_size)`.
  - Stores the summarizer model and bounded summary text for rollups.
- **`MapElitesState`** (`map_elites_states` table): persists per-island MAP-Elites snapshots.
  - Primary key: `(island_id)`.
  - Stores a JSONB `snapshot` payload containing lightweight metadata (feature bounds, PCA projection payload, last update time).
  - Archive cells and PCA history are stored incrementally in separate tables and reconstructed on load by
    `loreley.core.map_elites.snapshot.DatabaseSnapshotStore`.
  - The `snapshot` JSON must not embed `archive` or `history`. If your local database contains unsupported payloads, reset it: `uv run loreley reset-db --yes`.
- **`MapElitesArchiveCell`** (`map_elites_archive_cells` table): one row per occupied MAP-Elites archive cell.
  - Primary key: `(island_id, cell_index)`.
  - Stores the cell's `commit_hash`, `objective`, behaviour `measures`, stored `solution` vector, and `timestamp`.
  - Enables cheap per-cell upserts when a commit improves a specific cell.
- **`MapElitesPcaHistory`** (`map_elites_pca_history` table): incremental PCA history entries used to restore dimensionality reduction state.
  - Primary key: `(island_id, commit_hash)`.
  - Stores the commit embedding `vector` plus the `embedding_model` name and a `last_seen_at` marker used
    to restore ordered, bounded history windows across restarts.
- **`MapElitesFileEmbeddingCache`** (`map_elites_file_embedding_cache` table): persistent file-level embedding cache.
  - Primary key: `(blob_sha)`.
  - Stores a float array `vector` containing the file embedding, allowing repo-state embeddings to reuse unchanged file vectors across commits.
  - Stores `embedding_model` and `dimensions` alongside vectors for validation and debugging.
- **`MapElitesRepoStateAggregate`** (`map_elites_repo_state_aggregates` table): persistent commit-level aggregates for repo-state embeddings.
  - Primary key: `(commit_hash)`.
  - Stores `sum_vector` and `file_count` so the commit embedding can be derived as `sum_vector / file_count`.

