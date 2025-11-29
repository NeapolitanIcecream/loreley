# app.core.map-elites.snapshot

Helpers and backends for serialising and persisting MAP-Elites archive snapshots.

## Responsibilities

- **Serialisation helpers**:
  - Convert per-island PCA history (`PenultimateEmbedding`), `PCAProjection`, and `GridArchive` contents into JSON-compatible snapshot payloads.
  - Restore bounds, history, projection, archive entries, and commit-to-cell mappings from previously stored snapshots.
- **Backends**:
  - Define a small `SnapshotBackend` interface with `load(island_id)` and `save(island_id, snapshot)` methods.
  - Provide a `NullSnapshotBackend` that disables persistence and simply returns `None` on `load`.
  - Provide a `DatabaseSnapshotBackend` that stores snapshots in the `map_elites_states` table via the `MapElitesState` ORM model.

## Integration with `MapElitesManager`

- `MapElitesManager` constructs a backend through `build_snapshot_backend(experiment_id)`:
  - When `experiment_id` is `None`, a `NullSnapshotBackend` is returned and all snapshot operations become in-memory only.
  - When `experiment_id` is set, a `DatabaseSnapshotBackend` is used and snapshots are scoped by `(experiment_id, island_id)`.
- The manager decides *when* to persist:
  - On island initialisation it calls `backend.load(island_id)` and, if a payload exists, applies it with `apply_snapshot(...)`.
  - After ingestion and `clear_island()`, it calls `build_snapshot(island_id, state)` and `backend.save(island_id, snapshot)` to keep durable state up to date.


