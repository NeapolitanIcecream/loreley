# loreley.db.models

ORM models and enums for single-tenant experiment databases.

## Shared mixins and enums

- **`TimestampMixin`**: adds `created_at` and `updated_at` columns that default to `now()` and automatically update on modification.
- **`JobStatus`**: string-based `Enum` capturing the lifecycle of an evolution
  job (`STAGED`, `PENDING`, `QUEUED`, `RUNNING`, `SUCCEEDED`, `FAILED`,
  `CANCELLED`).
- **`OperatorTaskStatus`**: string-based `Enum` for UI API background tasks (`PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`).
- **`OperatorTaskKind`**: string-based `Enum` for UI API task kinds. The first kind is `BASELINE_ENSURE`; direct repair candidate actions use `REPAIR_CANDIDATE_ACTION`.

## Core models

- **`InstanceMetadata`** (`instance_metadata` table): single-row marker for DB identity.
  - Primary key: `id=1` enforced by a check constraint.
  - Stores `schema_version`, `experiment_id_raw`, `experiment_uuid`, and the canonical `root_commit_hash`.
  - `schema_version` is the application schema marker used by native migrations and `uv run loreley db current|migrate|validate`.
  - Optional `repository_slug` and `repository_canonical_origin` fields aid observability.
- **`CommitCard`** (`commit_cards` table): lightweight commit metadata used for inspiration and UI.
  - Primary key: `id` (UUID).
  - Unique constraint on `commit_hash` per database.
  - Tracks commit hash, parent hash, optional island identifier, optional `job_id`, author, subject, change summary, evaluation summary, tags, key files, and highlights.
  - Defines relationships to associated `Metric` records (via `Metric.commit_card_id`).
- **`Metric`** (`metrics` table): records individual evaluation metrics for a commit.
  - Stores metric `name`, numeric `value`, optional `unit`, whether higher values are better, and a JSONB `details` payload.
  - Links back to `CommitCard` via `commit_card_id` and maintains uniqueness per `(commit_card_id, name)`.
- **`CampaignProgram`** (`campaign_programs` table): content-addressed campaign contract snapshot.
  - Primary key: `hash`, the raw SHA-256 of `loreley.program.md`.
  - Stores source path, optional title, raw Markdown, normalized snapshot JSON, recognized sections, and parse warnings.
- **`CampaignBaseline`** (`campaign_baselines` table): source-of-truth root evaluator baseline for a comparable campaign contract.
  - Primary key: `id` (UUID).
  - Unique constraint on `baseline_key_hash`, which covers the root commit, campaign program hash, evaluator identity, primary metric contract, runtime profile, and effective settings fingerprint.
  - Stores the root commit hash, optional `campaign_program_hash`, evaluator name/version, primary metric name and direction, runtime profile, settings fingerprint, baseline status (`valid`, `failed`, or `degraded`), metric value/unit when valid, and failure details when invalid.
  - Optionally links to the root `CommitCard` and `Metric` rows used for compatibility projections, but archive/status baseline reads should key off `campaign_baselines`.
- **`SeedPortfolio`** (`seed_portfolios` table): staged and then content-addressed campaign-level seed-planning artifact.
  - A unique request fingerprint covers the pinned model route/reasoning setting, root commit, campaign program, ordered objective contract, root evidence fingerprints, configured and effective bounded direction counts, and overlap policy.
  - Stores planner hashes/timing, status, complete selected/rejected portfolio payload, and a unique final portfolio hash. Persisted `planning` or `failed` rows block automatic replay so scheduler restarts cannot silently issue a duplicate model call.
- **`SeedDirection`** (`seed_directions` table): one implementation-ready selected brief.
  - Unique portfolio-local direction ID, ordinal, and content hash preserve the causal mechanism, first implementation, signals, neutral-result interpretation, roadmap, risks, local checks, admission intent, and selection rationale.
  - Automatic scheduling derives campaign-global attempt state from direction-linked jobs: no concurrent duplicate, at most two unsuccessful terminal attempts, and deterministic balanced reuse after a direction succeeds.
- **`CandidateCommit`** (`candidate_commits` table): durable ledger row for a worker-produced candidate commit.
  - Primary key: `id` (UUID).
  - Stores the candidate commit hash, exact source-tree hash, parent hash, nearest viable ancestor, produced job, run token, job kind, repair lineage, campaign program hash, publication state, evaluation state, archive state, lifecycle state, repair-pool counters, and optional evaluator-scoped candidate identity.
  - Failed candidates can be present here without a `CommitCard` and without entering the MAP-Elites archive.
- **`DiagnosticCapsule`** (`diagnostic_capsules` table): sanitized evaluator failure evidence safe for repair prompts.
  - Stores a bounded JSON payload, policy version, whether the policy passed, and omitted-reason codes.
  - Links to the candidate, job, and evaluation attempt when available.
- **`EvaluationAttempt`** (`evaluation_attempts` table): one evaluator outcome observed for a candidate commit.
  - Stores an immutable per-job attempt ordinal and run token, evaluator
    identity, campaign program hash, optional raw candidate identity and its
    scoped hash, protocol, measurement cache/fingerprint/id, explicit
    measurement execution and reuse kind, source attempt, evaluator-slot
    telemetry, outcome kind, failure stage/kind, repairability, safe failure
    summary, diagnostic capsule link, fixed artifact paths, artifact policy
    version, and start/finish timestamps.
- **`EvaluationMeasurement`** (`evaluation_measurements` table): one accepted,
  cacheable phased measurement. Its non-null SHA-256 cache key covers candidate
  identity, evaluator name/version, campaign program, and measurement
  fingerprint. It stores a canonical payload hash, evidence manifest, and the
  source job/attempt.
- **`EvaluationConcurrencyContract`** (`evaluation_concurrency_contracts`
  table): persisted experiment/evaluator capacity contract. Workers with a
  different `E` or limit scope for the same contract are rejected.
- **`EvaluationResourceLease`** (`evaluation_resource_leases` table):
  append-only waiter/acquisition/release telemetry for PostgreSQL advisory
  evaluator slots and per-measurement locks.
- **`OperatorTask`** (`operator_tasks` table): background task state for the local operator console.
  - Primary key: `id` (UUID).
  - Stores `kind`, `status`, JSONB request/result payloads, an optional error summary, and start/completion timestamps.
  - The first task kind is `baseline_ensure`, created by the UI API and executed with FastAPI background tasks.
  - Direct repair candidate operator actions write `repair_candidate_action` audit rows with actor, action, reason, and state transition metadata.
  - A partial unique index allows only one active `baseline_ensure` task in `pending` or `running` state.
- **`AgentAction`** (`agent_actions` table): audited Agent REST facade action record.
  - Primary key: `id` (UUID).
  - Stores actor, action type, dry-run flag, expected state, request/result payloads, error code/summary, and completion timestamp.
  - A partial unique index on `(action_type, idempotency_key)` applies when `idempotency_key` is non-empty, so retried Agent REST requests can replay a prior result instead of executing the underlying write twice.
- **`EvolutionJob`** (`evolution_jobs` table): represents a single evolution iteration scheduled by the system.
  - Tracks current `status`, base commit, island ID, inspiration commit hashes, size-bounded job spec fields (`goal`, `constraints`, `acceptance_criteria`, `notes`, `tags`, sampling hints), persistent sampling ordinal, recipe hash/reuse flag, human-readable `plan_summary`, priority, scheduling/processing timestamps, and last error if any.
  - Stores optional `campaign_program_hash` for the campaign contract used to create the job.
  - Stores candidate-publication metadata (`candidate_commit_hash`, `candidate_branch_name`, `candidate_published_at`) so a worker can durably point to a locally created or remotely published candidate even if a later step fails.
  - Stores active lease ownership fields (`heartbeat_at`, `lease_expires_at`, `run_token`, `worker_id`) so workers can renew job ownership and stale attempts can be fenced off.
  - Stores `recovery_count` so the scheduler can limit automatic stale-job recovery and eventually mark repeatedly reclaimed jobs `FAILED`.
  - Stores stable `failure_stage` and `failure_kind` fields for campaign status;
    status does not infer categories from free-form `last_error` text.
  - Stores supplied-candidate execution mode, pinned input commit, summary,
    idempotency key, sanitized provenance, and archive-ingestion policy for
    first-class manual seeds.
  - Stores result/ingestion indexing fields (`result_commit_hash`, ingestion status/attempts/delta/cell index) without embedding large JSON payloads.
  - Stores seed portfolio hash, stable direction ID and brief snapshot, plus the measured seed admission lane/reason. Descendant jobs inherit this origin provenance from their base commit.
- **`EvolutionEvent`** (`evolution_events` table): append-only history for
  scheduler, worker, ingestion, and archive facts that mutable current-state
  rows cannot retain.
  - A unique deterministic `event_key` makes duplicate delivery harmless.
  - Stores a bounded string `event_type`, optional job/run/island/commit
    identity, an aware occurrence timestamp, optional invocation ordinal and
    monotonic duration, and a small allowlisted JSON payload.
  - Indexes `(occurred_at, id)`, the per-job timeline, and event type over time.
    Payloads never contain prompts, model output, credentials, evaluator logs,
    or cold-path artifacts.
- **`JobArtifacts`** (`job_artifacts` table): filesystem references for cold-path artifacts produced by the worker.
  - Stores paths to planning/coding/evaluation prompts, raw outputs, and logs.
- **`EvaluationArtifactRecord`** (`evaluation_artifacts` table): evaluator-declared diagnostic artifact metadata materialized by a job.
  - Primary key: `id` (UUID).
  - Attempt-linked evidence is append-only and unique on
    `(evaluation_attempt_id, key)`. Legacy records without an attempt retain a
    unique `(job_id, key)` latest projection.
  - Stores job id, optional commit-card and evaluation-attempt ids, commit hash, key, kind, MIME type, label, summary, visibility, agent projection, optional storage path, payload size, SHA-256, structured diagnostics, and metadata.
  - UI API listing and downloads exclude `hidden` artifacts; Agent REST feedback includes only `agent_visible` records.
- **`CommitChunkSummary`** (`commit_chunk_summaries` table): cached trajectory summaries for commit chains.
  - Primary key: `(start_commit_hash, end_commit_hash, block_size)`.
  - Stores the summarizer model and bounded summary text for rollups.
- **`MapElitesState`** (`map_elites_states` table): persists per-island MAP-Elites snapshots.
  - Primary key: `(island_id)`.
  - Stores a JSONB `snapshot` payload containing lightweight metadata (feature bounds, PCA projection payload, last update time).
  - Archive cells and PCA history are stored incrementally in separate tables and reconstructed on load by
    `loreley.core.map_elites.snapshot.DatabaseSnapshotStore`.
  - The `snapshot` JSON must not embed `archive` or `history`. If your local database contains unsupported payloads, reset it: `uv run loreley reset-db --yes`.
- **`MapElitesArchiveCell`** (`map_elites_archive_cells` table): one row per
  retained member of a bounded per-cell Pareto front.
  - Primary key: `(island_id, cell_index, commit_hash)` with island-local commit
    uniqueness.
  - Stores ordered objective values, behaviour measures, and the archive
    timestamp. Atomic front/archive replacement also emits membership deltas to
    `evolution_events`.
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
