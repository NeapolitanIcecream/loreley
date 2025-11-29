# app.db.models

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
  - Relates to `EvolutionJob`, `CommitMetadata`, and `MapElitesState` so that jobs, commits, and archive state can all be grouped by experiment.
- **`CommitMetadata`** (`commits` table): stores git commit metadata and evolution context.
  - Tracks commit hash, parent hash, optional island identifier, optional `experiment_id`, author, message, evaluation summary, free-form tags, and arbitrary JSONB `extra_context`.
  - Defines relationships to associated `Metric` records and jobs that use this commit as their base, and back to the owning `Experiment` when one exists.
- **`Metric`** (`metrics` table): records individual evaluation metrics for a commit.
  - Stores metric `name`, numeric `value`, optional `unit`, whether higher values are better, and a JSONB `details` payload.
  - Links back to `CommitMetadata` via `commit_hash` and maintains uniqueness per `(commit_hash, name)`.
- **`EvolutionJob`** (`evolution_jobs` table): represents a single evolution iteration scheduled by the system.
  - Tracks current `status`, base commit, island ID, optional `experiment_id`, inspiration commit hashes, request `payload`, human-readable `plan_summary`, priority, scheduling/processing timestamps, and last error if any.
  - Relates back to `CommitMetadata` via `base_commit_hash` and to `Experiment` via `experiment_id`, enabling efficient queries per base commit or experiment.
- **`MapElitesState`** (`map_elites_states` table): persists per-experiment, per-island snapshots of the MAP-Elites archive.
  - Uses a composite primary key `(experiment_id, island_id)` so that multiple experiments can maintain independent archives even when they share island identifiers.
  - Stores a JSONB `snapshot` payload containing feature bounds, PCA history/projection metadata, and the current archive entries so that `app.core.map_elites.snapshot` and `MapElitesManager` can restore state across process restarts for a given experiment.
