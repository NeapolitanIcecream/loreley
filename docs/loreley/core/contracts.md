# loreley.core.contracts

Typed, size-bounded contracts for Loreley's hot-path data flow (planning, coding, UI, and persistence adapters).

The goal is to keep primary database rows and API payloads small and stable. Large evidence payloads (prompts, raw model outputs, logs) are treated as **cold-path artifacts** and stored on disk, referenced via `loreley.db.models.JobArtifacts`.

## Helper functions

- **`normalize_single_line(text)`**: collapses all whitespace (including newlines) into a single trimmed line.
- **`clamp_text(text, max_chars)`**: truncates text to a maximum character budget, appending an ellipsis when needed.

These helpers are used when turning free-form LLM output into stable DB/UI fields.

## Models

This module defines Pydantic models for **consumable** fields. They are not ORM models (see `loreley.db.models` for database tables).

- **`MetricBrief`**: small, stable metric snippet for prompts and UI.
  - Fields: `name`, `value`, optional `unit`, optional `higher_is_better`, optional `summary`.
- **`CommitCard`**: lightweight commit representation used for inspiration prompts and UI.
  - Size budgets (enforced by Pydantic):
    - `subject`: max 72 chars, must not contain code fences and must not start with JSON.
    - `change_summary`: max 512 chars.
    - `evaluation_summary`: optional, max 512 chars.
    - `key_files`: up to 20 paths (each max 256 chars).
    - `highlights`: 1–8 lines (each max 200 chars).
- **`ArtifactRef`**: filesystem references to cold-path artifacts (prompts, raw outputs, logs).
  - Stores optional absolute paths (max 1024 chars each).
- **`EvolutionJobSpec`**: size-bounded job specification used by the scheduler/worker boundary.
  - Includes `goal` plus bounded lists (`constraints`, `acceptance_criteria`, `notes`, `tags`) and sampling provenance (`sampling_strategy`, radius stats, etc.).

## Relationship to the database

- `loreley.db.models.CommitCard` stores bounded commit metadata used by the UI and planning context.
- `loreley.db.models.JobArtifacts` stores artifact file paths produced by a worker job.
- `loreley.db.models.EvolutionJob` stores job spec fields and ingestion tracking fields without embedding large JSON payloads.


