# loreley.db.models

ORM models and enums for tracking evolutionary jobs, commits, and associated metrics.

## Shared mixins and enums

- **`TimestampMixin`**: adds `created_at` and `updated_at` columns that default to `now()` and automatically update on modification.
- **`JobStatus`**: string-based `Enum` capturing the lifecycle of an evolution job (`PENDING`, `QUEUED`, `RUNNING`, `SUCCEEDED`, `FAILED`, `CANCELLED`).

## Core models

- **`Repository`** (`repositories` table): normalised view of a source code repository.
  - Stores a stable `slug` derived from either the canonical remote URL or local worktree path, the current `remote_url`, optional `root_path`, and an `extra` JSONB payload with additional metadata (canonical origin, remotes, etc.).
  - Owns a collection of `Experiment` rows and is treated as the top-level key when reasoning about experiments in a multi-repository deployment.
- **`Experiment`** (`experiments` table): captures a single experiment configuration within a repository.
  - References a `repository_id`, a stable `config_hash` computed from a subset of `Settings`, an optional human-readable `name`, a JSONB `config_snapshot` of the relevant settings, and a free-form `status`.
  - Relates to `EvolutionJob`, `CommitCard`, and `MapElitesState` so that jobs, commits, and archive state can all be grouped by experiment.
- **`CommitCard`** (`commit_cards` table): stores a lightweight, size-bounded commit representation used for inspiration and UI.
  - Tracks commit hash, parent hash, optional island identifier, optional `experiment_id`, optional `job_id`, author, subject, change summary, evaluation summary, tags, key files, and highlights.
  - Defines relationships to associated `Metric` records and jobs that use this commit as their base, and back to the owning `Experiment` when one exists.
- **`Metric`** (`metrics` table): records individual evaluation metrics for a commit.
  - Stores metric `name`, numeric `value`, optional `unit`, whether higher values are better, and a JSONB `details` payload.
  - Links back to `CommitCard` via `commit_hash` and maintains uniqueness per `(commit_hash, name)`.
- **`EvolutionJob`** (`evolution_jobs` table): represents a single evolution iteration scheduled by the system.
  - Tracks current `status`, base commit, island ID, optional `experiment_id`, inspiration commit hashes, size-bounded job spec fields (`goal`, `constraints`, `acceptance_criteria`, `notes`, `tags`, sampling hints), human-readable `plan_summary`, priority, scheduling/processing timestamps, and last error if any.
  - Stores result/ingestion indexing fields (`result_commit_hash`, ingestion status/attempts/delta/cell index) without embedding large JSON payloads.
  - Relates back to `CommitCard` via `base_commit_hash` and to `Experiment` via `experiment_id`, enabling efficient queries per base commit or experiment.
- **`JobArtifacts`** (`job_artifacts` table): filesystem references for cold-path artifacts produced by the worker.
  - Stores paths to planning/coding/evaluation prompts, raw outputs, and logs.
- **`MapElitesState`** (`map_elites_states` table): persists per-experiment, per-island snapshots of the MAP-Elites archive.
  - Uses a composite primary key `(experiment_id, island_id)` so that multiple experiments can maintain independent archives even when they share island identifiers.
  - Stores a JSONB `snapshot` payload containing **lightweight metadata** (feature bounds, PCA projection payload, schema version, and other knobs).
  - For `schema_version >= 2`, the large `archive`/`history` payloads are stored incrementally in separate tables and
    reconstructed on load by `loreley.core.map_elites.snapshot.DatabaseSnapshotBackend`.
  - For legacy rows (`schema_version < 2`) that still embed `archive`/`history` lists, the loader performs **lazy migration**
    into the incremental tables on first read and strips the large fields from `snapshot`.
- **`MapElitesArchiveCell`** (`map_elites_archive_cells` table): one row per occupied MAP-Elites archive cell.
  - Primary key: `(experiment_id, island_id, cell_index)`.
  - Stores the cell's `commit_hash`, `objective`, behaviour `measures`, stored `solution` vector, and `timestamp`.
  - Enables cheap per-cell upserts when a commit improves a specific cell.
- **`MapElitesPcaHistory`** (`map_elites_pca_history` table): incremental PCA history entries used to restore dimensionality reduction state.
  - Primary key: `(experiment_id, island_id, commit_hash)`.
  - Stores the commit's penultimate embedding `vector` plus embedding provenance (dimensions/models) and a `last_seen_at` marker used
    to restore ordered, bounded history windows across restarts.
- **`MapElitesFileEmbeddingCache`** (`map_elites_file_embedding_cache` table): persistent file-level embedding cache keyed by git blob SHA.
  - Uses a composite primary key `(blob_sha, embedding_model, dimensions, pipeline_signature)`.
  - Stores a float array `vector` containing the file embedding, allowing repo-state embeddings to reuse unchanged file vectors across commits.
