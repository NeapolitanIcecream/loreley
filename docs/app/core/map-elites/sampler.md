# app.core.map-elites.sampler

Sampler that turns MAP-Elites archive records into concrete `EvolutionJob` rows for further evolution.

## Protocols

- **`SupportsMapElitesRecord`**: protocol describing the record interface consumed by the sampler (commit hash, cell index, fitness, measures, solution, metadata, timestamp).
- **`SupportsMapElitesManager`**: protocol that exposes a `get_records(island_id)` method, allowing the sampler to be used against `MapElitesManager` or any compatible implementation.

## Sampling

- **`ScheduledSamplerJob`**: immutable descriptor for a newly scheduled job, exposing the `EvolutionJob` ID, island, base record, inspiration records, and the payload sent to the scheduler.
- **`MapElitesSampler`**: coordinates archive sampling and job persistence.
  - Configured via `Settings` map-elites options for dimensionality, feature bounds, and sampler behaviour (`MAPELITES_DIMENSION_REDUCTION_*`, `MAPELITES_FEATURE_*`, `MAPELITES_ARCHIVE_*`, and `MAPELITES_SAMPLER_*`).
  - `schedule_job(island_id=None, payload_overrides=None, priority=None)` pulls records from the manager, chooses a base record, selects neighbours as inspirations using a configurable neighbourhood radius with optional fallback sampling, builds a rich JSON payload, and persists a new `EvolutionJob` via `session_scope`.
  - Uses `loguru` for structured logging and `rich` to print a concise confirmation when a job is enqueued.

## Neighbourhood selection

- **`_select_inspirations(...)`**: internal helper that walks outward from the base cell over the discretised behaviour grid, gathering nearby elites up to the requested inspiration count and recording selection statistics.
- **`_neighbor_indices(center_index, radius)`**: converts a flat cell index and radius into neighbouring cell indices using numpy's `unravel_index`/`ravel_multi_index`, respecting grid bounds.
