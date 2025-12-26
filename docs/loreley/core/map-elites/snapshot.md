# loreley.core.map_elites.snapshot

Helpers and backends for serialising and persisting MAP-Elites archive snapshots.

## Responsibilities

- **Serialisation helpers**:
  - Convert per-island PCA history (`PenultimateEmbedding`), `PCAProjection`, and `GridArchive` contents into JSON-compatible snapshot payloads.
  - `PCAProjection` payloads include components, mean, explained variance, explained variance ratio, whiten flag, sample count, and fitted timestamp.
  - Restore bounds, history, projection, archive entries, and commit-to-cell mappings from previously stored snapshots.
- **Backends**:
  - Define a small `SnapshotBackend` interface with `load(island_id)` and `save(island_id, snapshot)` methods.
  - Provide an optional `apply_update(island_id, *, state, update)` method:
    - Default implementation falls back to building a full snapshot and calling `save(...)`.
    - Database backends override it to persist **incremental** updates without rewriting large JSON blobs.
  - Provide a `NullSnapshotBackend` that disables persistence and simply returns `None` on `load`.
  - Provide a `DatabaseSnapshotBackend` that stores lightweight metadata in `map_elites_states` (`MapElitesState`)
    and persists the large payload incrementally in dedicated tables:
    - `map_elites_archive_cells` (`MapElitesArchiveCell`): one row per occupied archive cell.
    - `map_elites_pca_history` (`MapElitesPcaHistory`): one row per commit hash for PCA reconstruction.

## Snapshot schema versions

`MapElitesState.snapshot` carries a small `schema_version` flag:

- **schema_version = 1 (legacy)**: `snapshot` may embed large `archive`/`history` lists directly.
- **schema_version >= 2 (incremental)**: `snapshot` is lightweight metadata only (bounds, projection, knobs),
  while `archive` and `history` are stored in the incremental tables.

When loading a legacy payload that still embeds `archive`/`history`, the database backend performs **lazy migration**
on read: it upserts rows into the incremental tables, strips the large fields from `snapshot`, and bumps
`schema_version` to 2.

## Integration with `MapElitesManager`

- `MapElitesManager` constructs a backend through `build_snapshot_backend(experiment_id)`:
  - When `experiment_id` is `None`, a `NullSnapshotBackend` is returned and all snapshot operations become in-memory only.
  - When `experiment_id` is set, a `DatabaseSnapshotBackend` is used and snapshots are scoped by `(experiment_id, island_id)`.
- The manager decides *when* to persist:
  - On island initialisation it calls `backend.load(island_id)` and, if a payload exists, applies it with `apply_snapshot(...)`.
  - After ingestion and `clear_island()`, it emits a `SnapshotUpdate` and calls `backend.apply_update(...)`:
    - PCA history/projection updates are persisted frequently (small metadata + per-commit history row).
    - Archive cell writes are persisted only when a commit actually improves a cell (single cell upsert).
    - Clearing an island deletes that island's cell/history rows and resets projection metadata.


