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

## Phased evaluator contract

Evaluators with an expensive result that can be shared by different source
trees may opt into `phased-v1`. The protocol separates source-specific checks
from reusable measurement:

```python
from loreley.core.worker.evaluator import (
    EvalPass,
    EvaluationArtifact,
    EvaluationMeasurement,
    EvaluationPreparation,
    MeasurementEvidence,
)


class ReleaseEvaluator:
    evaluation_protocol = "phased-v1"
    evaluation_concurrency_scope = "measurement"

    def prepare(self, context):
        release = build_and_check_source(context.worktree)
        return EvaluationPreparation(
            candidate_identity=f"release-sha256:{release.sha256}",
            measurement_contract_fingerprint="corpus-v3/compiler-v1",
            state={"release_path": release.relative_path},
        )

    def measure(self, context, preparation):
        report = run_benchmark(context.worktree, preparation.state)
        return EvaluationMeasurement(
            data={"throughput": report.throughput},
            evidence=(
                MeasurementEvidence(
                    key="benchmark-report",
                    sha256=report.sha256,
                    size_bytes=report.size_bytes,
                ),
            ),
            artifacts=(
                EvaluationArtifact(
                    key="benchmark-report",
                    kind="benchmark",
                    mime_type="application/json",
                    path=report.relative_path,
                    visibility="human_only",
                ),
            ),
            cacheable=True,
        )

    def finalize(self, context, preparation, measurement, provenance):
        return EvalPass(
            summary="source checks and benchmark passed",
            candidate_identity=preparation.candidate_identity,
            metrics={"name": "throughput", "value": measurement.data["throughput"]},
        )
```

The worker always runs `prepare` before considering reuse. Two different Git
trees can share a measurement only after both pass their own source checks and
return the same candidate identity, evaluator contract, campaign program, and
measurement fingerprint. The fingerprint scopes measurement reuse, so a new
corpus or benchmark protocol cannot silently reuse an older measurement.
Archive/search identity remains the evaluator-provided candidate identity under
its evaluator and campaign contract; measuring the same binary on another
corpus does not create a second search identity.

Preparation state and measurement data must be JSON values and fit within
`WORKER_EVALUATOR_MEASUREMENT_MAX_JSON_BYTES`. Candidate identities and
fingerprints are required and limited to 512 characters. Every phase runs in a
fresh spawned subprocess under one shared timeout. `finalize` receives
`MeasurementProvenance`, including whether the measurement was reused, the
accepted measurement id, the source attempt id, and evidence hashes.

Every cacheable measurement must include at least one SHA-256
`MeasurementEvidence` entry and a materialized artifact with the same key,
size, and digest. Loreley verifies the stored bytes before acceptance and again
before reuse. Distributed workers therefore need a durable `LOGS_BASE_DIR`
visible to every worker that may reuse a measurement. Only a cacheable
measurement whose final outcome passed is accepted. The first
worker holds a PostgreSQL advisory lock until the measurement and terminal
attempt commit atomically. Concurrent workers recheck the cache after acquiring
that lock. Failed, timed-out, non-cacheable, or uncommitted measurements are
never reused.

Evaluation attempts have an immutable per-job ordinal and retain their own
artifact links. Retrying a job updates the job-level latest projection but does
not delete evidence from an earlier attempt.

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
- **`EvaluationPreparation`**: phased source-check result containing a stable
  candidate identity, measurement fingerprint, bounded JSON state, and
  source-specific artifacts.
- **`EvaluationMeasurement`**: phased measurement data, evidence manifest,
  optional artifacts, and an explicit `cacheable` decision.
- **`MeasurementEvidence`**: location-free evidence key, SHA-256, and optional
  byte size.
- **`MeasurementProvenance`**: cache key, reuse flag, accepted measurement id,
  source attempt id, and evidence supplied to `finalize`.

## Exceptions and protocols

- **`EvaluationError`**: custom runtime error raised when the evaluator cannot run the plugin successfully (import failures, bad configuration, timeouts, invalid payloads, etc.).
- **`EvaluationPlugin`**: protocol type describing callables that accept an `EvaluationContext` and return `EvalPass`, `EvalFail`, an `EvaluationResult`, an `EvaluationOutcome`, or a compatible mapping.
- **`PhasedEvaluationPlugin`**: explicit `phased-v1` protocol with callable
  `prepare`, `measure`, and `finalize` methods. Similarly named helper methods
  do not activate it.
- **`EvaluationCallable`**: internal alias for the concrete callable signature used by the evaluator.

## Evaluator

- **`Evaluator`**: adapter around user-defined evaluation plugins that handles import, isolation, timeouts, and coercion into an internal evaluation outcome.
  - Configured via `loreley.config.Settings` worker evaluator options (`WORKER_EVALUATOR_PLUGIN`, `WORKER_EVALUATOR_PYTHON_PATHS`, `WORKER_EVALUATOR_TIMEOUT_SECONDS`, `WORKER_EVALUATOR_MAX_METRICS`, `WORKER_EVALUATOR_MAX_CONCURRENCY`).
  - **`evaluate(context)`**: validates that the `worktree` exists and is a directory, resolves or imports the plugin callable, logs the run via `loguru` and `rich`, executes the plugin in a separate process with a strict timeout, and converts the returned payload into an `EvaluationResult`, truncating the number of metrics to `max_metrics` when necessary.
  - **`evaluate_outcome(context)`**: returns the full internal `EvaluationOutcome`. `EvalPass` coerces to `outcome_kind="passed"` and `EvalFail` coerces to `outcome_kind="candidate_failed"`.
  - Supports two configuration modes:
    - A dotted string reference such as `package.module:plugin` or `package.module.plugin` via `WORKER_EVALUATOR_PLUGIN`.
    - An inline callable passed at construction time (useful for tests or in-process usage), in which case no import is performed in the subprocess.
  - Extends `sys.path` using `WORKER_EVALUATOR_PYTHON_PATHS` before importing plugins, allowing evaluation logic to live outside the main application package.

### Evaluator-equivalent candidates

Compiled or generated-artifact evaluators should set `candidate_identity` to a stable identity for the state that the metrics actually measured, such as a namespaced release-binary SHA-256. Loreley scopes that value by evaluator name, evaluator version, and campaign program, persists it on the evaluation ledger, and prevents a second equivalent candidate from occupying another archive slot. If the evaluator omits the field, archive identity remains the Git commit hash.

Legacy one-shot evaluators may reuse an earlier passed result for an exact Git
tree. Phased evaluators do not take that shortcut: `prepare` must run so the
current fingerprint and source evidence are known. Archive equivalence remains
per island; status reports campaign-global unique evaluator identities
separately.

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
  - If the subprocess is still alive after the timeout, the evaluator terminates its complete process group and raises `EvaluationError` with a clear timeout message.
  - A parent-death watchdog terminates the evaluator process group when its
    worker disappears, so compiler or benchmark descendants cannot continue
    after the PostgreSQL capacity slot is released.

## Evaluator capacity

`WORKER_EVALUATOR_MAX_CONCURRENCY` defines evaluator capacity `E`, independently
of scheduler unfinished capacity `U` and configured worker processes `W`. It is
disabled by default for compatibility. When enabled, the first worker persists
the evaluator name/version, campaign program, scope, and `E` as an experiment
contract. A worker with a different `E` or scope is rejected instead of opening
another capacity pool.

PostgreSQL session advisory locks enforce slots across worker processes and
hosts. Phased evaluators can limit the whole evaluator or only `measure`;
one-shot evaluators limit the whole call. Wait, acquisition, slot, and release
timestamps are persisted and exposed in status and job provenance. There is no
SQLite production fallback.

## Framework and target boundary

Core Loreley owns phase orchestration, cache keys, first-measurement locking,
accepted-only reuse, evaluator capacity, attempt provenance, status counts, and
identity endpoints. A target evaluator owns repository-specific build flags,
source correctness gates, benchmark commands and corpora, precision policy,
metric definitions, and the decision to mark a passing measurement cacheable.

A target adapter should use only the public phased types above. It should not
query Loreley tables, implement its own measurement cache or file lock, count
campaign jobs, or stop the scheduler. Existing experiment harnesses remain
historical evidence; their target-specific evaluators can migrate to this API
without moving benchmark policy into core.

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
