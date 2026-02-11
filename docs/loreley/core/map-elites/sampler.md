# loreley.core.map_elites.sampler

Sampler that turns MAP-Elites archive records into concrete `EvolutionJob` rows for further evolution.

## Protocols

- **`SupportsMapElitesManager`**: protocol that exposes a lightweight `get_cell_commits(island_id) -> Mapping[int, str]` accessor (occupied cell indices → commit hashes).

## Sampling

- **`ScheduledSamplerJob`**: immutable descriptor for a newly scheduled job, exposing the `EvolutionJob` ID, island id, `base_commit_hash`, and `inspiration_commit_hashes`.
- **`MapElitesSampler`**: coordinates archive sampling and job persistence.
  - Configured via `Settings` map-elites options for dimensionality, truncation/normalization, archive grid, and sampler behaviour (`MAPELITES_DIMENSION_REDUCTION_*`, `MAPELITES_FEATURE_TRUNCATION_K`, `MAPELITES_FEATURE_NORMALIZATION_WARMUP_SAMPLES`, `MAPELITES_FEATURE_CLIP`, `MAPELITES_ARCHIVE_*`, and `MAPELITES_SAMPLER_*`).
  - `schedule_job(island_id=None, priority=None)` samples from the cell→commit map (via `get_cell_commits`) to choose a base and select neighbour inspirations using a configurable neighbourhood radius with optional fallback sampling, then persists a new `EvolutionJob` using bounded spec fields (goal/constraints/acceptance criteria/tags) plus sampling statistics.
  - Uses `loguru` for structured logging and `rich` to print a concise confirmation when a job is enqueued.

## Neighbourhood selection

- **`_select_inspirations(...)`**: internal helper that computes Chebyshev (L∞) distances from the base cell to occupied archive cells in a vectorized pass, then samples inspiration commit hashes by increasing radius (with optional fallback sampling) and records selection statistics.
- **`_neighbor_indices(center_index, radius)`**: helper that enumerates neighbouring cell indices for small grids (used by tests), respecting grid bounds.
