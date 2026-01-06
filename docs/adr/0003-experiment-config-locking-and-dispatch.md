# ADR 0003: Experiment config locking and dispatch (behavior params)

Date: 2026-01-05

Context: Long-running experiments (and their resumed runs) must be reproducible. The scheduler and UI/API must interpret MAP-Elites state with a stable dimensionality. Workers should not rely on local environment for experiment/job behavior knobs.
Decision: Treat experiment/job behavior settings as **experiment-scoped invariants** stored in `Experiment.config_snapshot`. The scheduler is the single writer of this snapshot; workers load it from the DB and apply it as the effective behavior configuration for executing jobs.
Decision: `MAPELITES_CODE_EMBEDDING_DIMENSIONS` is required in the persisted experiment snapshot (the scheduler must set it) and is always passed to the embedding API. It is intentionally optional at process startup so UI/API/workers can boot without local env drift. `MAPELITES_DIMENSION_REDUCTION_TARGET_DIMS` is also locked per experiment.
Decision: Persisted MAP-Elites state is interpreted using the experiment snapshot dimensionality. If a stored MAP-Elites snapshot cannot be reconciled with the configured dimensionality, the system fails fast instead of silently adopting a different dimensionality.
Consequences: Persisted MAP-Elites state is interpreted consistently across restarts; worker behaviour is reproducible per experiment; local configuration drift does not affect experiment execution/interpretation, eliminating complex "auto-adapt" logic.


