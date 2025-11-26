# app.db.models

ORM models and enums for tracking evolutionary jobs, commits, and associated metrics.

## Shared mixins and enums

- **`TimestampMixin`**: adds `created_at` and `updated_at` columns that default to `now()` and automatically update on modification.
- **`JobStatus`**: string-based `Enum` capturing the lifecycle of an evolution job (`PENDING`, `QUEUED`, `RUNNING`, `SUCCEEDED`, `FAILED`, `CANCELLED`).

## Core models

- **`CommitMetadata`** (`commits` table): stores git commit metadata and evolution context.
  - Tracks commit hash, parent hash, optional island identifier, author, message, evaluation summary, free-form tags, and arbitrary JSONB `extra_context`.
  - Defines relationships to associated `Metric` records and jobs that use this commit as their base.
- **`Metric`** (`metrics` table): records individual evaluation metrics for a commit.
  - Stores metric `name`, numeric `value`, optional `unit`, whether higher values are better, and a JSONB `details` payload.
  - Links back to `CommitMetadata` via `commit_hash` and maintains uniqueness per `(commit_hash, name)`.
- **`EvolutionJob`** (`evolution_jobs` table): represents a single evolution iteration scheduled by the system.
  - Tracks current `status`, base commit, island ID, inspiration commit hashes, request `payload`, human-readable `plan_summary`, priority, scheduling/processing timestamps, and last error if any.
  - Relates back to `CommitMetadata` via `base_commit_hash`, enabling efficient queries per base commit.
- **`MapElitesState`** (`map_elites_states` table): persists per-island snapshots of the MAP-Elites archive.
  - Stores a JSONB `snapshot` payload containing feature bounds, PCA history/projection metadata, and the current archive entries so that `MapElitesManager` can restore state across process restarts.
