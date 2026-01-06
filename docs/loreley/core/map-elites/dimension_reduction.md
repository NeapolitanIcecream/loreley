# loreley.core.map_elites.dimension_reduction

PCA-based dimensionality reduction of commit embeddings before they are fed into the MAP-Elites archive. In repo-state mode, the commit embedding is the repo-state code vector.

## Data structures

- **`PcaHistoryEntry`**: commit-level embedding recorded in PCA history. It stores the raw commit vector plus the embedding model name used to produce it.
- **`PCAProjection`**: serialisable wrapper around a fitted `sklearn.decomposition.PCA` model, capturing the mean vector, components, explained variance, explained variance ratio, whiten flag, and sample metadata, plus a `transform()` helper that projects (and when whitening is enabled, scales) new vectors.
- **`FinalEmbedding`**: low-dimensional vector that sits on the MAP-Elites grid for a commit, along with the originating `PcaHistoryEntry` and optional `PCAProjection` used.

## Reducer

- **`DimensionReducer`**: maintains a rolling history of PCA inputs and an optional PCA projection to keep the behaviour space stable.
  - Configured via `Settings` map-elites dimensionality options (`MAPELITES_DIMENSION_REDUCTION_*`) plus `MAPELITES_FEATURE_NORMALIZATION_WARMUP_SAMPLES`: target dimensions, minimum sample count (takes the max of the dimensionality minimum and the warmup), history size, refit interval, and whether to normalise input vectors.
  - `build_history_entry(...)` prepares the commit embedding from the code embedding, normalises when enabled, and returns a `PcaHistoryEntry` or `None` when no embeddings are available.
  - `reduce(entry, refit=None)` records the embedding in history, (re)fits PCA with `whiten=True` when needed, and projects into the target space, returning a `FinalEmbedding` and logging issues via `loguru` when projection cannot be computed.

## Convenience API

- **`reduce_commit_embeddings(...)`**: one-shot helper that constructs a `DimensionReducer`, builds a PCA history entry from a commit's code embedding, and returns the `FinalEmbedding` together with the updated history and projection so callers can persist state.
