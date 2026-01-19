# ADR 0023: Lock experiment behavior keyset and settings injection

Date: 2026-01-19

Context: Experiments must be reproducible across restarts and hosts. Prefix-based snapshots left behavior gaps, and agent backend factories could re-read environment variables (bypassing the persisted snapshot).
Decision: Persist an explicit experiment-scoped behavior keyset in `Experiment.config_snapshot`, derived from `Settings` fields and excluding secrets and deployment-only wiring.
Decision: Validate experiment snapshots strictly (schema version + required keys) and fail fast on missing or incompatible snapshots; development upgrades require resetting the database schema.
Decision: Inject the already-resolved effective `Settings` into agent backend factories/classes that accept a `settings` parameter, preventing implicit environment-driven drift.
Decision: Add deterministic seeds for MAP-Elites sampling and PCA to stabilise scheduling and projections.
Consequences: Runtime behavior becomes reproducible and auditable per experiment; secrets remain external; configuration drift surfaces immediately.

