# ADR 0017: Use Tenacity for OpenAI retry/backoff

Date: 2026-01-14

Context: Loreley had multiple hand-rolled `while + sleep` retry loops around OpenAI calls (embeddings, commit summaries, trajectory chunk summaries), duplicating logic and making retry tuning inconsistent.

Decision: Use `tenacity` as the single retry/backoff mechanism for OpenAI calls and centralize configuration in `loreley/core/openai_retry.py`.

Decision: Preserve existing retry semantics by mapping `*_max_retries` to `stop_after_attempt(...)` and keeping linear backoff (`backoff_seconds * attempt_number`) via `wait_incrementing(...)`, with a loguru-compatible `before_sleep` hook.

Consequences: Retry code is shorter and consistent across call sites; future changes (retry only transient errors, add jitter, add metrics/log hooks) require edits in one place; final failure errors may include Tenacity context in tracebacks.

