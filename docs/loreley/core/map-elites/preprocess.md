# loreley.core.map_elites.preprocess

Preprocessing utilities for turning raw repository code files into cleaned code snippets suitable for embedding and feature extraction.

## Data structures

- **`PreprocessedFile`**: lightweight record capturing the repository-relative `path`, a scalar `change_count` used for downstream weighting, and cleaned textual `content` after preprocessing.

## Preprocessor

- **`CodePreprocessor`**: filters and normalises files before embedding.
  - Uses `Settings` map-elites preprocessing options to enforce maximum file size, allowed extensions/filenames, and excluded glob patterns.
  - Loads file contents either from the working tree or a specific `commit_hash` via GitPython, applies comment stripping, tab-to-spaces conversion, blank-line collapse, and basic normalisation.
  - Exposes small helpers used by the repo-state embedding pipeline:
    - `is_code_file(path)`, `is_excluded(path)` for selection,
    - `load_text(path)` for loading content from git or disk,
    - `cleanup_text(content)` for normalisation and comment stripping.

## Convenience API

This module does not expose a changed-files preprocessing pipeline. Repo-state embeddings enumerate eligible files directly from the git tree and use `CodePreprocessor` for filtering and cleanup.
