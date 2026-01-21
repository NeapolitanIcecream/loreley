# ADR 0025: Explicit experiment ID; remove derived config identity and legacy snapshots

Date: 2026-01-21

Status: Accepted

## Context

Loreley runs long-lived scheduler/worker processes where runtime behaviour settings are assumed stable for the lifetime of a database. Designing for dynamic per-experiment settings and forward-compatible persistence introduced unnecessary complexity (derived experiment identity, compatibility layers, runtime migrations) without user-facing feature gains.

The simplest and most legible model is:
- Settings are provided via environment variables.
- Operators keep settings consistent across processes sharing the same database.
- The system avoids derived identities and removes compatibility code paths.

## Decision

- **Experiment identity is explicit and env-only**:
  - `EXPERIMENT_ID` (UUID or slug) is required for the scheduler and worker processes.
  - Experiment identity is not derived from repository commits or other settings.

- **Remove derived experiment configuration identity**:
  - `Experiment.config_hash` is removed.
  - `Experiment.id` is the only experiment key used to scope jobs, commits, and MAP-Elites state.

- **Root commit is an operational anchor, not identity**:
  - `MAPELITES_EXPERIMENT_ROOT_COMMIT` is required for repo-state bootstrap and incremental-only ingestion.
  - The scheduler pins repo-root ignore rules (`.gitignore` + `.loreleyignore`) at startup from the root commit and keeps them process-local.

- **Remove forward compatibility for MAP-Elites snapshots**:
  - Only the incremental storage model is supported:
    - metadata in `map_elites_states.snapshot`,
    - occupied cells in `map_elites_archive_cells`,
    - PCA history in `map_elites_pca_history`.
  - Legacy payloads embedding `archive`/`history` inside `map_elites_states.snapshot` are not supported.

- **CLI simplification**:
  - There is no CLI override flag for experiment attachment.
  - Operators must set `EXPERIMENT_ID` via environment variables (or `.env`).

## Consequences

- Operators must keep environment variables consistent across scheduler/worker/UI processes that share the same database.
- Schema changes are applied via destructive resets (dev/upgrade workflow): `uv run loreley reset-db --yes`.
- Multiple experiments for the same repository are supported by choosing different `EXPERIMENT_ID` values (explicit namespaces).

