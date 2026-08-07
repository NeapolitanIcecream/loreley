# loreley.core.map_elites.sampler

Sampler that turns MAP-Elites archive records into concrete `EvolutionJob` rows for further evolution.

## Protocols

- **`SupportsMapElitesManager`**: protocol that exposes `get_cell_fronts(island_id) -> Mapping[int, Sequence[str]]` (occupied cell indices → bounded Pareto members).

## Sampling

- **`ScheduledSamplerJob`**: immutable descriptor for a newly scheduled job, exposing the `EvolutionJob` ID, target island, base/inspiration commits, sampling ordinal/recipe identity, reuse flag, and optional migration provenance.
- **`MapElitesSampler`**: coordinates archive sampling and job persistence.
  - Configured via `Settings` map-elites options for dimensionality, truncation/normalization, archive grid, and sampler behaviour (`MAPELITES_DIMENSION_REDUCTION_*`, `MAPELITES_FEATURE_TRUNCATION_K`, `MAPELITES_FEATURE_NORMALIZATION_WARMUP_SAMPLES`, `MAPELITES_FEATURE_CLIP`, `MAPELITES_ARCHIVE_*`, and `MAPELITES_SAMPLER_*`).
  - `schedule_job(...)` derives a random stream from the configured seed, target island, and persistent job ordinal. Scheduler restarts therefore do not replay the beginning of a process-local random stream.
  - It chooses a behaviour cell uniformly, then a Pareto member within it, so larger fronts do not give a niche more base-selection probability. It selects neighbour inspirations using a configurable radius and optional fallback, then persists a new `EvolutionJob`.
  - The unordered `(base, inspirations)` recipe has a stable SHA-256 identity. Recent recipe hashes are supplied by the scheduler as a bounded cooldown set. The sampler resamples up to `MAPELITES_SAMPLER_MAX_RESAMPLE_ATTEMPTS`; if no alternative exists, it schedules the final recipe and records the reuse explicitly.
  - A scheduler-selected donor from another island can replace one configured inspiration. Migration is disabled when the inspiration count is zero. The base and resulting candidate remain in the target island, while donor island/commit lineage is persisted on the job.
  - Uses `loguru` for structured logging and `rich` to print a concise confirmation when a job is enqueued.

## Neighbourhood selection

- **`_select_inspirations(...)`**: internal helper that computes Chebyshev (L∞) distances from the base cell to occupied archive cells in a vectorized pass, then samples inspiration commit hashes by increasing radius (with optional fallback sampling) and records selection statistics.
- **`_neighbor_indices(center_index, radius)`**: helper that enumerates neighbouring cell indices for small grids (used by tests), respecting grid bounds.
