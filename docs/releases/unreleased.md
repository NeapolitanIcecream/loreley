# Unreleased

These notes cover changes merged after `v0.8.4-alpha`.

## Changed

- Replaced scalar-per-cell MAP-Elites storage with bounded Pareto fronts using
  every ordered objective and its explicit `max`/`min` direction.
- Added fair seed and normal scheduling across `MAPELITES_ISLANDS`, with
  periodic, persisted cross-island migration inspirations. An empty island
  retains one bounded readiness probe after PCA warmup instead of stalling.
- Added `loreley worker --processes N`, backed by Dramatiq's spawn-based native
  process master and isolated randomized worker base clones, even when the
  caller previously initialized a `fork` context.
- Constant objectives no longer create arbitrary crowding-distance boundaries
  when a Pareto cell exceeds its configured capacity.
- Removed the `ribs` adapter/dependency and obsolete scalar archive fields.
- Removed the MAP-Elites re-export shim and the runtime discovery of
  unconfigured persisted islands. Import `MapElitesManager` from
  `loreley.core.map_elites.manager`; the configured island list is now the
  only scheduling and API contract.

## Upgrade

- Database schema version 15 discards the incomplete scalar archive and marks
  durable successful jobs for reingestion into Pareto fronts, backfilling
  missing result hashes from their persisted candidate commits.
- Convert an existing env file once with:

  ```bash
  uv run python tools/migrate_v15_config.py .env --output .env.v15
  ```

  Review the generated objective/island JSON before starting Loreley. Runtime
  code no longer accepts `MAPELITES_FITNESS_*`,
  `MAPELITES_DEFAULT_ISLAND_ID`, or other scalar-archive settings.
  Blank legacy default-island overrides migrate to the historical `main`
  fallback.
