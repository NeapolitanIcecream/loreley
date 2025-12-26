# loreley.core.map_elites.code_embedding

Commit-level code embedding utilities that consume chunked code artifacts and talk to the OpenAI embeddings API as part of the Map-Elites pipeline.

## Data structures

- **`ChunkEmbedding`**: embedding vector derived from a single `FileChunk`, storing the original chunk, its numeric embedding `vector`, and a scalar `weight` used during aggregation.
- **`FileEmbedding`**: aggregated embedding for one `ChunkedFile`, including the source file, the tuple of `ChunkEmbedding` instances, a file-level `vector`, and an overall `weight`.
- **`CommitCodeEmbedding`**: commit-level representation that bundles all `FileEmbedding` instances, the final aggregated `vector`, the embedding `model` name, and `dimensions`, plus a `chunk_count` convenience property.

## Embedder

- **`CodeEmbedder`**: orchestrates calls to the OpenAI embeddings API and aggregation logic.
  - Configured via `Settings` map-elites code embedding options (`MAPELITES_CODE_EMBEDDING_*`) controlling model name, optional output dimensions, batch size, maximum chunks per commit, retry count, and retry backoff.
  - `run(chunked_files)` filters out empty inputs, flattens chunks into a payload, embeds them in batches with a `rich` progress spinner, and turns raw vectors into `ChunkEmbedding`, `FileEmbedding`, and `CommitCodeEmbedding` objects using weighted averaging.
  - Logs detailed progress and warnings with `loguru`, including mismatched response sizes, missing owners for chunks, and empty aggregation results.

## Convenience API

- **`embed_chunked_files(chunked_files, settings=None, client=None)`**: helper that constructs a `CodeEmbedder` and returns a `CommitCodeEmbedding` for the supplied chunked files, or `None` if there is nothing worth embedding.
