# loreley.core.worker.evaluator

Evaluation utilities for Loreley's autonomous worker, responsible for running user-defined evaluation plugins in an isolated subprocess and turning their outputs into structured metrics.

## Domain types

- **`EvaluationMetric`**: single metric reported by the evaluation plugin (`name`, numeric `value`, optional `unit`, `higher_is_better` flag, and optional structured `details` mapping). Provides `as_dict()` to produce a JSON-serialisable representation.
- **`EvaluationContext`**: immutable-ish context object passed into plugins, including the git `worktree` path, optional `base_commit_hash` and `candidate_commit_hash`, optional `job_id` and high-level `goal`, an arbitrary `payload` dict (typically containing job and plan information), an optional `plan_summary`, and a free-form `metadata` dict. Paths and mappings are normalised and resolved in `__post_init__`.
- **`EvaluationResult`**: structured result returned from evaluation, containing a mandatory `summary`, a tuple of `metrics`, a tuple of `tests_executed`, a tuple of textual `logs`, and an `extra` dict for arbitrary details; its `__post_init__` enforces a non-empty summary and normalises all collections.

## Exceptions and protocols

- **`EvaluationError`**: custom runtime error raised when the evaluator cannot run the plugin successfully (import failures, bad configuration, timeouts, invalid payloads, etc.).
- **`EvaluationPlugin`**: protocol type describing callables that accept an `EvaluationContext` and return either an `EvaluationResult` or a plain mapping compatible with `EvaluationResult` fields.
- **`EvaluationCallable`**: internal alias for the concrete callable signature used by the evaluator.

## Evaluator

- **`Evaluator`**: adapter around user-defined evaluation plugins that handles import, isolation, timeouts, and coercion into `EvaluationResult`.
  - Configured via `loreley.config.Settings` worker evaluator options (`WORKER_EVALUATOR_PLUGIN`, `WORKER_EVALUATOR_PYTHON_PATHS`, `WORKER_EVALUATOR_TIMEOUT_SECONDS`, `WORKER_EVALUATOR_MAX_METRICS`).
  - **`evaluate(context)`**: validates that the `worktree` exists and is a directory, resolves or imports the plugin callable, logs the run via `loguru` and `rich`, executes the plugin in a separate process with a strict timeout, and converts the returned payload into an `EvaluationResult`, truncating the number of metrics to `max_metrics` when necessary.
  - Supports two configuration modes:
    - A dotted string reference such as `package.module:plugin` or `package.module.plugin` via `WORKER_EVALUATOR_PLUGIN`.
    - An inline callable passed at construction time (useful for tests or in-process usage), in which case no import is performed in the subprocess.
  - Extends `sys.path` using `WORKER_EVALUATOR_PYTHON_PATHS` before importing plugins, allowing evaluation logic to live outside the main application package.

## Plugin execution model

- The evaluator always runs plugins in a dedicated subprocess created via `multiprocessing.get_context("spawn")`:
  - `_plugin_subprocess_entry()` prepares the Python path, imports or reuses the plugin callable, executes it with the provided `EvaluationContext`, and sends either an `("ok", payload)` or `("error", {message, traceback})` tuple back through a `multiprocessing.Queue`.
  - The parent process waits up to `timeout` seconds for the subprocess to finish, and a small additional grace period to read from the queue.
  - If the subprocess is still alive after the timeout, the evaluator terminates it and raises `EvaluationError` with a clear timeout message.

## Payload coercion helpers

- **`_coerce_result(payload)`**: converts whatever the plugin returned into an `EvaluationResult`.
  - Accepts an existing `EvaluationResult` instance as-is.
  - When given a mapping, expects at least a non-empty `summary`, plus optional `metrics`, `tests_executed`, `logs`, and `extra` entries.
  - Raises `EvaluationError` when the payload is missing a summary or is of an unsupported type.
- **`_coerce_metrics(metrics_payload)`**: accepts a single `EvaluationMetric`, a mapping, or an iterable of these, and always returns a tuple of `EvaluationMetric` instances.
- **`_metric_from_mapping(payload)`**: turns a mapping into an `EvaluationMetric`, enforcing presence and validity of `name` and `value` fields, and validating the shape of `unit`, `higher_is_better`, and `details`.
- **`_normalise_sequence(values, label)`**: utility used to normalise `tests_executed` and `logs` into tuples of non-empty strings, accepting either a single string or an arbitrary iterable.
- **`_coerce_extra(payload)`**: normalises the `extra` field into a plain dict, rejecting non-mapping inputs with `EvaluationError`.
- **`_validate_context(context)`**: ensures that the `worktree` exists and is a directory before any plugin is run, failing fast with `EvaluationError` otherwise.


