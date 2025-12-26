# loreley.core.map_elites.chunk

Chunking utilities for turning preprocessed code into semantically meaningful segments that can be embedded and explored by map-elites.

## Data structures

- **`FileChunk`**: represents one chunked segment of a single file, including its path, stable `chunk_id`, positional range (`start_line`/`end_line`), text `content`, `line_count`, and aggregated `change_count`.
- **`ChunkedFile`**: groups all `FileChunk` instances produced from a single file, tracking the file `path`, overall `change_count`, and `total_lines`.
- **`PreprocessedArtifact`**: lightweight protocol describing the preprocessed inputs consumed by the chunker (`path`, `change_count`, `content`).

## Chunker

- **`CodeChunker`**: splits preprocessed files into windows tuned for downstream embedding.
  - Uses `Settings` map-elites chunk configuration (`MAPELITES_CHUNK_*`) to control target and minimum lines per chunk, overlap between windows, maximum chunks per file, and keywords that hint at logical boundaries (e.g. `def`, `class`).
  - Iterates over each file, selecting break points on blank lines or boundary-looking lines where possible, and falls back to simple windowing when no better break point exists.
  - Produces `ChunkedFile` records while displaying a `rich` progress spinner and logging summary statistics via `loguru`.

## Convenience API

- **`chunk_preprocessed_files(files, settings=None)`**: helper that instantiates a `CodeChunker` and returns the list of `ChunkedFile` results for a sequence of preprocessed artifacts.
