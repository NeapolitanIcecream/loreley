# loreley.core.map_elites.preprocess

Preprocessing utilities for turning raw repository code files into cleaned code snippets suitable for embedding and feature extraction.

## Data structures

- **`ChangedFile`**: lightweight description of a file touched by a commit (path, approximate `change_count`, optional inline `content` override).
- **`PreprocessedFile`**: output record capturing the repository-relative `path`, accumulated `change_count`, and cleaned textual `content` after preprocessing.

## Preprocessor

- **`CodePreprocessor`**: filters and normalises files before embedding.
  - Uses `Settings` map-elites preprocessing options to enforce maximum file count/size, allowed extensions/filenames, and excluded glob patterns.
  - Loads file contents either from the working tree or a specific `treeish` via GitPython, applies comment stripping, tab-to-spaces conversion, blank-line collapse, and basic normalisation.
  - Exposes `run(changed_files)` which returns a list of `PreprocessedFile` objects ordered by `change_count`.
  - Reports progress via a `rich` `Progress` spinner and logs structured messages through `loguru`.

## Convenience API

- **`preprocess_changed_files(changed_files, repo_root=None, settings=None, treeish=None, repo=None)`**: functional helper that instantiates a `CodePreprocessor` and runs it over the provided list of changed files.
