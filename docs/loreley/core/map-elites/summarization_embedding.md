# loreley.core.map_elites.summarization_embedding

Optional summary-level embedding utilities that turn preprocessed code into structured natural-language summaries and a commit-level embedding vector. In repo-state mode (`MAPELITES_EMBEDDING_MODE=repo_state`), MAP-Elites uses code-only repo-state embeddings and does not currently include these summary vectors.

## Data structures

- **`FileSummary`**: immutable record describing one file-level summary (repository-relative `path`, approximate `change_count`, and the generated markdown `summary` text). The path is normalised to a `pathlib.Path`.
- **`SummaryEmbedding`**: embedding derived from a `FileSummary`, containing the original `file_summary`, its numeric `vector`, and a scalar `weight` used during aggregation.
- **`CommitSummaryEmbedding`**: commit-level representation bundling all `SummaryEmbedding` instances, the final aggregated `vector`, the `summary_model` and `embedding_model` names, `dimensions`, and a `file_count` convenience property.

## Embedder

- **`SummaryEmbedder`**: orchestrates summarisation of preprocessed files and embedding of the resulting summaries.
  - Configured via `Settings` Map-Elites summary options (`MAPELITES_SUMMARY_*`) controlling the LLM model, temperature, maximum output tokens, source excerpt character limit, and retry/backoff behaviour, and summary embedding options (`MAPELITES_SUMMARY_EMBEDDING_*`) controlling the embedding model, optional output dimensions, and batch size.
  - Uses the global OpenAI API configuration from `loreley.config.Settings` (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_API_SPEC`) when constructing the shared `OpenAI` client. `OPENAI_API_SPEC` selects whether summarisation uses:
    - the unified **Responses API** (`client.responses.create`, default), passing `_SUMMARY_INSTRUCTIONS` as `instructions` and the file prompt as `input`; or
    - the classic **Chat Completions API** (`client.chat.completions.create`), mapping `_SUMMARY_INSTRUCTIONS` to a `system` message and the file prompt to a `user` message while keeping temperature and token limits aligned.
  - `run(files)` skips empty input, calls `_summarize_files` to build `FileSummary` objects with a `rich` progress spinner and `loguru` debug/warning logs (using either Responses or Chat Completions under the hood), then calls `_embed_summaries` to batch summaries through the OpenAI embeddings API, weighting them by change count or summary length.
  - Aggregates per-file vectors into a single commit-level vector using `_weighted_average`, after sorting entries by path for stable output; returns a `CommitSummaryEmbedding` or `None` if no usable summaries or embeddings were produced.

## Convenience API

- **`summarize_preprocessed_files(files, settings=None, client=None)`**: helper that constructs a `SummaryEmbedder` (optionally injecting custom settings or OpenAI client) and returns a `CommitSummaryEmbedding` for the supplied `PreprocessedFile` sequence, or `None` when there is nothing to summarise or embed.
