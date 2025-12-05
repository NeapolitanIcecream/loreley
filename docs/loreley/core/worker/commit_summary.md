# loreley.core.worker.commit_summary

Commit summarisation utilities used by the evolution worker to derive concise git commit subjects from planning and coding context.

## Domain types and errors

- **`CommitSummaryError`**: runtime error raised when the summariser cannot produce a subject line (for example, due to API errors, empty model output, or repeated failures across retries).

## CommitSummarizer

- **`CommitSummarizer`**: LLM-backed helper responsible for generating short, imperative git subjects.
  - Configured via `loreley.config.Settings` worker evolution commit options:
    - `WORKER_EVOLUTION_COMMIT_MODEL`: model identifier used with the `OpenAI` client.
    - `WORKER_EVOLUTION_COMMIT_TEMPERATURE`: sampling temperature.
    - `WORKER_EVOLUTION_COMMIT_MAX_OUTPUT_TOKENS`: upper bound on model output tokens (clamped to at least 32).
    - `WORKER_EVOLUTION_COMMIT_MAX_RETRIES`: maximum number of retry attempts on failure (minimum 1).
    - `WORKER_EVOLUTION_COMMIT_RETRY_BACKOFF_SECONDS`: linear backoff applied between retries.
    - `WORKER_EVOLUTION_COMMIT_SUBJECT_MAX_CHARS`: hard character limit for subject lines (minimum 32).
  - Lazily initialises an `OpenAI` client and applies a local `_truncate_limit` when building prompts to keep context sizes reasonable.
  - Respects the global OpenAI API surface setting `OPENAI_API_SPEC`:
    - `"responses"` (default) uses the unified Responses API (`client.responses.create`) with an `instructions` string and full prompt as `input`.
    - `"chat_completions"` uses the Chat Completions API (`client.chat.completions.create`), mapping the same instruction text to a `system` message and the prompt to a `user` message, extracting the assistant's reply from the first choice.

### Subject generation

- **`generate(job, plan, coding)`**:
  - Constructs a detailed prompt that includes:
    - The global job `goal`.
    - Plan `summary`, `rationale`, focus metrics, guardrails, constraints, acceptance criteria, and notes.
    - Coding execution summary, per-step outcomes (step IDs, statuses, summaries), and the list of tests executed.
    - The coding agent's own suggested `commit_message` as a fallback hint.
  - Calls the `OpenAI` responses API with the configured model, temperature, and token limit, plus an `instructions` string that asks for a single imperative git subject bounded by `WORKER_EVOLUTION_COMMIT_SUBJECT_MAX_CHARS` (minimum 32; enforced both in the prompt and when normalising the final subject).
  - Retries up to `_max_retries` times on `OpenAIError` or `CommitSummaryError`, waiting for `retry_backoff * attempt` seconds between attempts, regardless of whether Responses or Chat Completions is selected.
  - On success, strips and normalises whitespace, then enforces the subject character limit via `_normalise_subject()`, logging the attempt count via `loguru`.
  - On exhausting retries, raises `CommitSummaryError` with a descriptive message including the number of attempts.

### Normalisation helpers

- **`coerce_subject(text, *, default)`**:
  - Used as a safer fallback when LLM-based summarisation fails or when an existing subject must be clamped to a valid git style.
  - Collapses whitespace, falls back to a provided `default` when `text` is empty, and applies `_normalise_subject()` to respect the configured character limit.
- **`_normalise_subject(text)`**:
  - Collapses consecutive whitespace to single spaces and trims leading/trailing spaces.
  - If the cleaned subject exceeds `_subject_limit`, truncates it and appends an ellipsis to signal truncation.
- **`_build_prompt(job, plan, coding)`** / **`_truncate(text, limit=None)`**:
  - Internal helpers which format the rich multi-section prompt while bounding long summaries and step descriptions, ensuring that the most relevant context is preserved for the LLM.


