# loreley.core.map_elites.snapshot

Helpers for applying and persisting MAP-Elites island state.

## Storage model (Postgres)

Loreley stores MAP-Elites state incrementally:

- `map_elites_states` (`MapElitesState.snapshot`): per-island metadata (feature bounds, PCA projection payload, knobs).
- `map_elites_archive_cells` (`MapElitesArchiveCell`): one row per occupied archive cell.
- `map_elites_pca_history` (`MapElitesPcaHistory`): PCA history entries used to restore dimensionality reduction state.

The read/write entry point is `DatabaseSnapshotStore`.

## Integration with `MapElitesManager`

- `MapElitesManager` requires `experiment_id` and uses `DatabaseSnapshotStore(experiment_id=...)`.
- On island initialisation it calls `store.load(island_id)` and applies the payload with `apply_snapshot(...)`.
- After ingestion and `clear_island()` it emits a `SnapshotUpdate` and calls `store.apply_update(...)`.

## Compatibility

Legacy snapshot payloads embedding `archive`/`history` inside `MapElitesState.snapshot` are not supported. Reset the database schema for upgrades (`uv run loreley reset-db --yes`).


