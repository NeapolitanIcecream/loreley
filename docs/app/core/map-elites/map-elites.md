# app.core.map-elites.map-elites

High-level manager that runs the MAP-Elites pipeline on git commits and maintains per-island archives backed by the database.

## Data structures

- **`CommitEmbeddingArtifacts`**: immutable container bundling preprocessed files, chunked files, and the commit-level code and summary embeddings plus the final low-dimensional embedding produced for a commit.
- **`MapElitesRecord`**: snapshot of a single elite stored in the archive, including commit hash, island, cell index, fitness, behaviour measures, solution vector, metadata, and timestamp.
- **`MapElitesInsertionResult`**: describes the outcome of attempting to insert a commit into the archive, exposing a status flag, fitness delta, optional `MapElitesRecord`, any intermediate artifacts, and an optional human-readable message.
- **`IslandState`**: internal mutable state attached to each island, holding the `GridArchive`, behaviour bounds, PCA history/projection, and mappings between commits and archive cell indices.

## Manager

- **`MapElitesManager`**: orchestrates preprocessing, chunking, embedding, dimensionality reduction, archive updates, and persistence.
  - Configured via `Settings` map-elites options: preprocessing, chunking, code/summary embeddings, dimensionality reduction, feature bounds, archive grid, fitness metric, and default island identifiers.
  - `ingest(commit_hash, changed_files, metrics, island_id, repo_root, treeish, metadata, fitness_override)` runs the full pipeline for a commit: loads and preprocesses changed files, chunks them, derives code and summary embeddings, reduces them to the behaviour space, resolves a scalar fitness from metrics or overrides, and attempts to insert the result into the island's `GridArchive`.
  - Tracks per-island PCA history and projection so that new embeddings are consistent with previous ones, logging detailed progress and warnings with `loguru`.
  - Serialises archive state to and from the `MapElitesState` table using helper methods that convert numpy arrays and custom objects into JSON-compatible payloads.

## Query helpers

- **`get_records(island_id=None)`**: returns all current `MapElitesRecord` entries for an island, rebuilding them from the underlying archive.
- **`sample_records(island_id=None, count=1)`**: randomly samples up to `count` elites from an island's archive for downstream planning or analysis.
- **`clear_island(island_id=None)`**: clears an island's archive and associated PCA history/projection state, removing all stored elites and mappings for that island.
- **`describe_island(island_id=None)`**: returns a small dict of observability stats for an island (ID, occupied cell count, total cells, QD score, and best fitness).
