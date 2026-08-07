# loreley.core.worker.evaluator

Evaluation utilities for Loreley's autonomous worker, responsible for running
user-defined evaluation plugins in an isolated subprocess and turning their
outputs into structured pass/fail outcomes, metrics, and diagnostic artifacts.

## Evaluator contract

Evaluator authors should use the small public contract:

- return `EvalPass(...)` when the candidate passes;
- return `EvalFail(...)` when the candidate owns the failure;
- raise an exception when the evaluator or its environment failed.

Example:

```python
from loreley.core.worker.evaluator import EvalFail, EvalPass, EvaluationContext


def plugin(context: EvaluationContext):
    ok = run_fast_checks(context.worktree)
    if not ok:
        return EvalFail(
            kind="typecheck",
            summary="mypy failed",
            details="src/app.py:42: incompatible return type",
        )
    return EvalPass(
        summary="typecheck passed",
        candidate_identity=build_artifact_sha256(context.worktree),
    )
```

Supported `EvalFail.kind` values are `compile`, `typecheck`, `lint`, `test`,
`validation`, `benchmark`, and `other`.

`EvalFail` means the candidate failed evaluation. The worker may feed a bounded,
sanitized diagnostic back to the coding agent and retry inside the same job
when the failure kind is allowlisted by worker settings. Evaluator exceptions
are treated as evaluator/infrastructure failures and do not trigger rework.

## Domain types

- **`EvaluationMetric`**: single metric reported by the evaluation plugin (`name`, numeric `value`, optional `unit`, `higher_is_better` flag, and optional structured `details` mapping). Provides `as_dict()` to produce a JSON-serialisable representation.
- **`EvaluationDiagnostic`**: bounded structured finding attached to an evaluator artifact (`kind`, `message`, `severity`, and optional `location`, `metric`, `value`, `unit`). Severity is normalized to `info`, `warning`, `error`, `regression`, or `improvement`.
- **`EvaluationArtifact`**: evaluator-declared diagnostic artifact. It can point to a file inside the evaluation worktree with `path`, provide `inline_payload`, or provide metadata-only evidence through `summary` and `diagnostics`. Each artifact has a normalized `key`, `kind`, `mime_type`, `visibility` (`agent_visible`, `human_only`, or `hidden`), and `agent_projection` (`summary`, `manifest`, or `path`).
- **`EvaluationContext`**: immutable-ish context object passed into plugins, including the git `worktree` path, optional `base_commit_hash` and `candidate_commit_hash`, optional `job_id` and high-level `goal`, an arbitrary `payload` dict (typically containing job and plan information), an optional `plan_summary`, and a free-form `metadata` dict. Paths and mappings are normalised and resolved in `__post_init__`.
- **`EvalPass`**: public passing result returned by an evaluator. It contains a
  mandatory `summary`, optional metrics, tests executed, logs, extra data,
  artifacts, and an optional `candidate_identity`.
- **`EvalFail`**: public candidate-owned failure returned by an evaluator. It
  contains `kind`, a bounded `summary`, optional bounded `details`, and optional
  artifacts.
- **`EvaluationResult`**: legacy/advanced structured passing result returned from evaluation, containing a mandatory `summary`, a tuple of `metrics`, a tuple of `tests_executed`, a tuple of textual `logs`, an `extra` dict for arbitrary details, optional evaluator artifacts, and an optional `candidate_identity`; its `__post_init__` enforces a non-empty summary and normalises all collections.
- **`EvaluationFailureResult`**: bounded evaluator-owned failure evidence for non-passing candidates. It records failure stage/kind, repairability, a safe summary, and visibility-separated evidence refs.
- **`EvaluationOutcome`**: internal/advanced compatibility envelope for
  `passed`, `candidate_failed`, `evaluator_failed`, `infrastructure_failed`, or
  `inconclusive` outcomes. Passed outcomes contain an `EvaluationResult`;
  non-passing outcomes contain an `EvaluationFailureResult`.

## Exceptions and protocols

- **`EvaluationError`**: custom runtime error raised when the evaluator cannot run the plugin successfully (import failures, bad configuration, timeouts, invalid payloads, etc.).
- **`EvaluationPlugin`**: protocol type describing callables that accept an `EvaluationContext` and return `EvalPass`, `EvalFail`, an `EvaluationResult`, an `EvaluationOutcome`, or a compatible mapping.
- **`EvaluationCallable`**: internal alias for the concrete callable signature used by the evaluator.

## Evaluator

- **`Evaluator`**: adapter around user-defined evaluation plugins that handles import, isolation, timeouts, and coercion into an internal evaluation outcome.
  - Configured via `loreley.config.Settings` worker evaluator options (`WORKER_EVALUATOR_PLUGIN`, `WORKER_EVALUATOR_PYTHON_PATHS`, `WORKER_EVALUATOR_TIMEOUT_SECONDS`, `WORKER_EVALUATOR_MAX_METRICS`).
  - **`evaluate(context)`**: validates that the `worktree` exists and is a directory, resolves or imports the plugin callable, logs the run via `loguru` and `rich`, executes the plugin in a separate process with a strict timeout, and converts the returned payload into an `EvaluationResult`, truncating the number of metrics to `max_metrics` when necessary.
  - **`evaluate_outcome(context)`**: returns the full internal `EvaluationOutcome`. `EvalPass` coerces to `outcome_kind="passed"` and `EvalFail` coerces to `outcome_kind="candidate_failed"`.
  - Supports two configuration modes:
    - A dotted string reference such as `package.module:plugin` or `package.module.plugin` via `WORKER_EVALUATOR_PLUGIN`.
    - An inline callable passed at construction time (useful for tests or in-process usage), in which case no import is performed in the subprocess.
  - Extends `sys.path` using `WORKER_EVALUATOR_PYTHON_PATHS` before importing plugins, allowing evaluation logic to live outside the main application package.

### Evaluator-equivalent candidates

Compiled or generated-artifact evaluators should set `candidate_identity` to a stable identity for the state that the metrics actually measured, such as a namespaced release-binary SHA-256. Loreley scopes that value by evaluator name, evaluator version, and campaign program, persists it on the evaluation ledger, and prevents a second equivalent candidate from occupying another archive slot. If the evaluator omits the field, archive identity remains the Git commit hash.

Identity deduplication assumes every passing measurement is already precise
enough for archive admission. Evaluators should return a non-passing
infrastructure outcome when their precision gate fails; repeated equivalent
candidates are not extra chances to select a favorable measurement.

`WORKER_EVALUATOR_VERSION` must change when measurement semantics change without a plugin source change. This keeps identities from different benchmark contracts separate.

## Evaluator-declared artifacts

Evaluation plugins may return artifacts in `EvaluationResult.artifacts`,
`EvaluationOutcome.artifacts`, or mapping payloads. Simple passing mappings use
the `artifacts` key. `artifact_records` is only read from
`EvaluationOutcome`-style mappings that include `outcome_kind`; those mappings
may also use top-level `artifacts`.

Mapping example:

```python
{
    "summary": "benchmark passed",
    "metrics": [{"name": "throughput", "value": 42.0, "higher_is_better": True}],
    "artifacts": [
        {
            "key": "benchmark-report",
            "kind": "benchmark_json",
            "mime_type": "application/json",
            "path": "reports/benchmark.json",
            "summary": "Throughput improved.",
            "visibility": "agent_visible",
            "agent_projection": "summary",
            "diagnostics": [
                {
                    "kind": "improvement",
                    "severity": "info",
                    "message": "throughput improved",
                    "metric": "throughput",
                    "value": 42.0,
                }
            ],
        }
    ],
}
```

Artifact rules:

- `path` values must resolve to regular files inside the evaluation worktree.
- `inline_payload` may be text, bytes, a mapping, or a sequence.
- A declaration with no payload must include `summary` or `diagnostics`.
- Duplicate keys, unsupported shapes, and invalid diagnostics are skipped or
  downgraded with sanitized validation warnings so evaluation can still finish.
- `hidden` artifacts are persisted for audit but are not exposed through UI API
  listing/download routes or Agent REST feedback.

## Plugin execution model

- The evaluator always runs plugins in a dedicated subprocess created via `multiprocessing.get_context("spawn")`:
  - `_plugin_subprocess_entry()` prepares the Python path, imports or reuses the plugin callable, executes it with the provided `EvaluationContext`, and sends either an `("ok", payload)` or `("error", {message, traceback})` tuple back through a `multiprocessing.Queue`.
  - The parent process waits up to `timeout` seconds for the subprocess to finish, and a small additional grace period to read from the queue.
  - If the subprocess is still alive after the timeout, the evaluator terminates it and raises `EvaluationError` with a clear timeout message.

## Payload coercion helpers

- **`_coerce_result(payload)`**: converts whatever the plugin returned into an `EvaluationResult`.
  - Accepts an existing `EvaluationResult` instance as-is.
  - When given a mapping, expects at least a non-empty `summary`, plus optional `metrics`, `tests_executed`, `logs`, `extra`, and `artifacts` entries.
  - Raises `EvaluationError` when the payload is missing a summary or is of an unsupported type.
- **`coerce_evaluation_artifacts(payload)`**: accepts an `EvaluationArtifact`, a mapping, or an iterable of either, and returns accepted artifacts plus sanitized validation warnings.
- **`_coerce_metrics(metrics_payload)`**: accepts a single `EvaluationMetric`, a mapping, or an iterable of these, and always returns a tuple of `EvaluationMetric` instances.
- **`_metric_from_mapping(payload)`**: turns a mapping into an `EvaluationMetric`, enforcing presence and validity of `name` and `value` fields, and validating the shape of `unit`, `higher_is_better`, and `details`.
- **`_normalise_sequence(values, label)`**: utility used to normalise `tests_executed` and `logs` into tuples of non-empty strings, accepting either a single string or an arbitrary iterable.
- **`_coerce_extra(payload)`**: normalises the `extra` field into a plain dict, rejecting non-mapping inputs with `EvaluationError`.
- **`_validate_context(context)`**: ensures that the `worktree` exists and is a directory before any plugin is run, failing fast with `EvaluationError` otherwise.
