# loreley.core.map_elites.map_elites

High-level manager that runs the MAP-Elites pipeline on git commits and maintains per-island archives backed by the database.

## Module layout

- `loreley.core.map_elites.map_elites`: compatibility export surface for public MAP-Elites symbols.
- `loreley.core.map_elites.manager`: `MapElitesManager` orchestration and public API methods.
- `loreley.core.map_elites.types`: shared immutable/mutable data structures.
- `loreley.core.map_elites.archive_ops`: archive-centric operations and bookkeeping updates.
- `loreley.core.map_elites.db_ops`: batched database queries for vectors and fitness values.
- `loreley.core.map_elites.rebuild`: PCA-fit/refit archive seeding and rebuild routines.

## Data structures

- **`CommitEmbeddingArtifacts`**: immutable container bundling lightweight embedding artifacts for a commit. In repo-state mode this includes repo-state stats plus the final low-dimensional embedding.
- **`MapElitesRecord`**: snapshot of a single elite stored in the archive, including commit hash, island, cell index, fitness, behaviour measures, solution vector, metadata, and timestamp.
- **`MapElitesInsertionResult`**: describes the outcome of attempting to insert a commit into the archive, exposing a status flag, fitness delta, optional `MapElitesRecord`, any intermediate artifacts, and an optional human-readable message.
- **`IslandState`**: internal mutable state attached to each island, holding the `GridArchive`, behaviour bounds, PCA history/projection, and mappings between commits and archive cell indices.

## Manager

- **`MapElitesManager`**: orchestrates preprocessing, chunking, embedding, dimensionality reduction, archive updates, and snapshot persistence.
  - Configured via `Settings` map-elites options: preprocessing and repository file filtering, repo-state code embeddings (file cache), dimensionality reduction (PCA with whitening), feature normalisation/truncation (`MAPELITES_FEATURE_TRUNCATION_K`, `MAPELITES_FEATURE_NORMALIZATION_WARMUP_SAMPLES`, `MAPELITES_FEATURE_CLIP`), archive grid, fitness metric, and default island identifiers.
  - Uses the single-tenant database as the persistence boundary; all MAP-Elites state is scoped by `island_id`.
  - `ingest(commit_hash, ...)` runs the full pipeline for a commit in **repo-state** embedding mode:
    - enumerates eligible code files at `commit_hash` using pinned ignore rules from `Settings.mapelites_repo_state_ignore_text` plus basic filtering,
    - reuses a file-level embedding cache keyed by git blob SHA,
    - embeds only cache misses (new/changed blobs),
    - aggregates all file embeddings into a single commit vector via **uniform averaging**,
    - reduces it to the behaviour space (PCA), persists PCA history/projection state, and updates the island's `GridArchive` once PCA is available. During warmup (no projection yet) the archive is kept empty; when the first projection is fitted (epoch 0) the manager seeds a fresh archive from warmup commits.
  - Tracks per-island PCA history, refit counters, and the active PCA projection. When PCA is first fitted, the manager seeds the archive from warmup commits; when PCA refits, it aligns the new projection to the previous epoch (orthogonal Procrustes alignment, may include reflections when PCA axes flip, when enough anchors are available) and **rebuilds the archive cells** so all elites live in a single, consistent coordinate system.
  - Behaviour descriptors are clipped to `[-k, k]` (k from `MAPELITES_FEATURE_TRUNCATION_K`), linearly mapped into `[0, 1]^d`, and archives are constructed with fixed `[0, 1]` bounds per dimension to avoid manual per-dimension tuning and boundary crowding.
  - Delegates snapshot loading and incremental persistence to `loreley.core.map_elites.snapshot.DatabaseSnapshotStore`.

## Query helpers

- **`get_records(island_id=None)`**: returns all current `MapElitesRecord` entries for an island, rebuilding them from the underlying archive.
- **`sample_records(island_id=None, count=1)`**: randomly samples up to `count` elites from an island's archive for downstream planning or analysis.
- **`clear_island(island_id=None)`**: clears an island's archive and associated PCA history/projection state, removing all stored elites and mappings for that island.
- **`describe_island(island_id=None)`**: returns a small dict of observability stats for an island (ID, occupied cell count, total cells, QD score, and best fitness).
