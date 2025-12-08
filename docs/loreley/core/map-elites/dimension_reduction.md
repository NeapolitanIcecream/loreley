# loreley.core.map-elites.dimension_reduction

PCA-based dimensionality reduction of combined code and summary embeddings before they are fed into the MAP-Elites archive.

## Data structures

- **`PenultimateEmbedding`**: concatenated code and summary embedding for a single commit, tracking the original models and dimension counts as well as the combined vector (optionally L2-normalised).
- **`PCAProjection`**: serialisable wrapper around a fitted `sklearn.decomposition.PCA` model, capturing the mean vector, components, explained variance, explained variance ratio, whiten flag, and sample metadata, plus a `transform()` helper that projects (and when whitening is enabled, scales) new vectors.
- **`FinalEmbedding`**: low-dimensional vector that sits on the MAP-Elites grid for a commit, along with the originating `PenultimateEmbedding` and optional `PCAProjection` used.

## Reducer

- **`DimensionReducer`**: maintains rolling history of penultimate embeddings and an optional PCA projection to keep the behaviour space stable.
  - Configured via `Settings` map-elites dimensionality options (`MAPELITES_DIMENSION_REDUCTION_*`) plus `MAPELITES_FEATURE_NORMALIZATION_WARMUP_SAMPLES`: target dimensions, minimum sample count (takes the max of the dimensionality minimum and the warmup), history size, refit interval, and whether to normalise penultimate vectors.
  - `build_penultimate(...)` concatenates code and summary embeddings, normalises them when enabled, and returns a `PenultimateEmbedding` or `None` when no embeddings are available.
  - `reduce(penultimate, refit=None)` records the embedding in history, (re)fits PCA with `whiten=True` when needed, and projects into the target space, returning a `FinalEmbedding` and logging issues via `loguru` when projection cannot be computed.

## Convenience API

- **`reduce_commit_embeddings(...)`**: one-shot helper that constructs a `DimensionReducer`, builds the penultimate embedding from a commit's code and summary embeddings, and returns the `FinalEmbedding` together with the updated history and projection so callers can persist state.
