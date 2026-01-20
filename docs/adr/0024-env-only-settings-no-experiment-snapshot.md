# ADR 0024: Env-only settings; remove experiment settings snapshot

Date: 2026-01-20

Status: Accepted

## Context

Loreley can safely assume that runtime behaviour settings do not change during the lifetime of an experiment run. Designing for dynamic per-experiment configuration injection and drift prevention introduced avoidable complexity (snapshot schemas, required keysets, loader magic). The system should be easy to understand: settings come from the environment, and long-running processes do not change behaviour mid-run.

## Decision

- Loreley uses an **env-only settings model**:
  - `loreley.config.get_settings()` is the single source of configuration.
  - The database does **not** persist or apply an experiment settings snapshot.
  - UI/API reads data from the database and interprets it using the same env settings as the long-running services.

- Experiment identity is **minimal**:
  - `Experiment.config_hash` is derived from the canonical `MAPELITES_EXPERIMENT_ROOT_COMMIT` only.
  - Operational knob changes (timeouts, sampling tweaks, model choices) are not part of experiment identity.

- Repo-state ignore rules are still **pinned**, but not persisted:
  - At scheduler startup, Loreley reads repository-root `.gitignore` + `.loreleyignore` from the configured root commit.
  - The combined ignore text + SHA-256 hash are stored in `Settings.mapelites_repo_state_ignore_text` / `Settings.mapelites_repo_state_ignore_sha256` for the lifetime of the scheduler process.

- Agent backend loading is explicit and non-magical:
  - `load_agent_backend(ref, *, label)` supports an instance, a no-arg class, or a no-arg factory.
  - Loreley does not inject `Settings` into backend factories/classes.

## Consequences

- The system has fewer configuration code paths and fewer invariants to validate at runtime.
- Correctness becomes an operator responsibility: scheduler/worker/UI processes must run with consistent environment variables for a given database.
- Breaking changes are acceptable and preferred over forward compatibility:
  - `Experiment.config_snapshot` is removed from the ORM model.
  - Development upgrades require resetting the database schema (`loreley reset-db --yes`).

