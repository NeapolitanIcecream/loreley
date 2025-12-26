# loreley.core.map_elites.sampler

Sampler that turns MAP-Elites archive records into concrete `EvolutionJob` rows for further evolution.

## Protocols

- **`SupportsMapElitesRecord`**: protocol describing the record interface consumed by the sampler (commit hash, cell index, fitness, measures, solution, timestamp).
- **`SupportsMapElitesManager`**: protocol that exposes a `get_records(island_id)` method, allowing the sampler to be used against `MapElitesManager` or any compatible implementation.

## Sampling

- **`ScheduledSamplerJob`**: immutable descriptor for a newly scheduled job, exposing the `EvolutionJob` ID, island, base record, and inspiration records.
- **`MapElitesSampler`**: coordinates archive sampling and job persistence.
  - Configured via `Settings` map-elites options for dimensionality, truncation/normalization, archive grid, and sampler behaviour (`MAPELITES_DIMENSION_REDUCTION_*`, `MAPELITES_FEATURE_TRUNCATION_K`, `MAPELITES_FEATURE_NORMALIZATION_WARMUP_SAMPLES`, `MAPELITES_FEATURE_CLIP`, `MAPELITES_ARCHIVE_*`, and `MAPELITES_SAMPLER_*`).
  - `schedule_job(island_id=None, priority=None, experiment_id=None)` pulls records from the manager, chooses a base record, selects neighbours as inspirations using a configurable neighbourhood radius with optional fallback sampling, and persists a new `EvolutionJob` using bounded spec fields (goal/constraints/acceptance criteria/tags) plus sampling statistics.
  - When `experiment_id` is provided, it is coerced to a `UUID` and stored on the `EvolutionJob` row so that downstream components (scheduler ingestion, worker, MAP-Elites manager) can reliably group jobs and commits by experiment. When omitted, jobs are still scheduled and `experiment_id` remains `NULL`, which is useful for legacy or single-experiment deployments.
  - Uses `loguru` for structured logging and `rich` to print a concise confirmation when a job is enqueued.

## Neighbourhood selection

- **`_select_inspirations(...)`**: internal helper that walks outward from the base cell over the discretised behaviour grid, gathering nearby elites up to the requested inspiration count and recording selection statistics.
- **`_neighbor_indices(center_index, radius)`**: converts a flat cell index and radius into neighbouring cell indices using numpy's `unravel_index`/`ravel_multi_index`, respecting grid bounds.
