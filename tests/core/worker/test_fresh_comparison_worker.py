from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

import loreley.core.fresh_comparison as comparison_service
from loreley.core.comparison_contract import validate_comparison_result
from loreley.core.worker.evaluator import EvalFail, EvalPass, EvaluationResult, Evaluator
from loreley.core.worker.evolution import EvolutionWorker, EvolutionWorkerError
from loreley.core.worker.repository import CheckoutContext
from tests.core.worker.test_evaluator import _PhasedPlugin
from tests.core.worker.test_evolution import _make_job_context


class _Lease:
    def __init__(self, events, kind, scope="measurement"):
        self.events = events
        self.kind = kind
        self.scope = scope
        self.slot_index = 0
        self.wait_seconds = 0.0
        self.lease_id = uuid4()
        self.acquired_at = datetime.now(timezone.utc)

    def release(self, reason):
        self.events.append(f"{self.kind}.release:{reason}")
        return datetime.now(timezone.utc)


class _Runtime:
    def __init__(self, events, *, scope="measurement"):
        self.events = events
        self.scope = scope
        self.cached = None

    def ensure_contract(self, **kwargs):
        self.events.append("contract")
        return "contract-key"

    def acquire_evaluator_slot(self, **kwargs):
        self.events.append("slot")
        return _Lease(self.events, "slot", self.scope)

    def acquire_measurement_lock(self, **kwargs):
        self.events.append("cache.lock")
        return _Lease(self.events, "cache")

    def lookup_measurement(self, cache_key):
        self.events.append("cache.lookup")
        return self.cached


class _ObservedPlugin(_PhasedPlugin):
    """Reuse the phased fixture's preparation and hash-linked measurement."""

    def __init__(self, events, *, scope="measurement"):
        self.events = events
        self.evaluation_concurrency_scope = scope
        self.contexts = {}
        self.failure_phase = None
        self.lower_bound = 0.01
        self.provenance = None
        self.measurement = None

    def prepare(self, context):
        self.events.append("prepare")
        self.contexts["prepare"] = context.comparison
        if self.failure_phase == "prepare":
            return EvalFail(kind="compile", summary="build failed")
        return super().prepare(context)

    def measure(self, context, preparation):
        self.events.append("measure")
        self.contexts["measure"] = context.comparison
        if self.failure_phase == "measure":
            return EvalFail(kind="benchmark", summary="load failed")
        self.measurement = super().measure(context, preparation)
        return self.measurement

    def finalize(self, context, preparation, measurement, provenance):
        self.events.append("finalize")
        self.contexts["finalize"] = context.comparison
        self.provenance = provenance
        if self.failure_phase == "finalize":
            return EvalFail(kind="validation", summary="semantic check failed")
        evidence = {}
        if context.comparison is not None:
            evidence["fresh_comparison"] = {
                "context_id": context.comparison["context_id"],
                "complete": True,
                "incumbent_value": 1.0,
                "improvement_lower_bound": self.lower_bound,
                "confidence_level": 0.95,
                "sample_count": 8,
            }
        return EvalPass(
            summary="measurement passed",
            candidate_identity=preparation.candidate_identity,
            metrics={"name": "score", "value": measurement.data["score"]},
            extra=evidence,
        )


@pytest.fixture
def worker_harness(settings, tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(comparison_service, "wait_for_bootstrap",
                        lambda **kwargs: events.append("bootstrap.ready"))
    settings.mapelites_admission_policy = "fresh_comparison"
    settings.worker_evaluator_version = "fresh-test-v1"
    plugin = _ObservedPlugin(events)
    evaluator = Evaluator(settings=settings, plugin=plugin)
    # Keep the real phase adapters and outcome coercion, replacing only process
    # transport so lifecycle ordering remains visible to this unit test.
    monkeypatch.setattr(
        evaluator, "_execute_phase_with_deadline",
        lambda phase, context, args, **kwargs: getattr(plugin, phase)(context, *args),
    )
    worker = EvolutionWorker(
        settings=settings, repository=object(), planning_agent=object(),
        coding_agent=object(), evaluator=evaluator, job_store=object(),
    )
    runtime = _Runtime(events)
    worker.evaluation_runtime = runtime
    job = _make_job_context()
    job.campaign_program_hash = "a" * 64
    checkout = CheckoutContext(
        job_id=str(job.job_id), branch_name="test", base_commit="base", worktree=tmp_path,
    )
    issued = {
        "context_id": str(uuid4()), "candidate_commit_hash": "candidate",
        "island_id": job.island_id, "cell_index": 0, "measures": [0.25],
        "projection_fingerprint": "b" * 64, "incumbent_commit_hash": "incumbent",
        "objective_name": "score", "higher_is_better": True,
    }
    prepared_calls = []
    recorded = []

    def prepare_context(**kwargs):
        events.append("comparison.prepare")
        prepared_calls.append(kwargs)
        return dict(issued)

    def record_measurement(**kwargs):
        events.append("comparison.record")
        result = kwargs["result"]
        decision = validate_comparison_result(
            result.extra.get("fresh_comparison"), issued, result.metrics,
        )
        recorded.append((kwargs, decision))

    monkeypatch.setattr(comparison_service, "prepare_context", prepare_context)
    monkeypatch.setattr(comparison_service, "record_measurement", record_measurement)
    return SimpleNamespace(
        worker=worker, settings=settings, plugin=plugin, runtime=runtime, job=job,
        checkout=checkout, events=events, issued=issued, prepared_calls=prepared_calls,
        recorded=recorded,
    )


def _run(harness):
    return harness.worker._evaluate_or_reuse(
        job_ctx=harness.job, checkout=harness.checkout, plan=None,
        candidate_commit="candidate", source_tree_hash="source-tree",
    )


@pytest.mark.parametrize("scope", ["measurement", "whole"])
def test_fresh_context_issued_after_preparation_and_capacity_slot(worker_harness, scope):
    h = worker_harness
    h.plugin.evaluation_concurrency_scope = scope
    h.runtime.scope = scope
    outcome = _run(h)
    assert outcome.outcome_kind == "passed"
    assert h.events.index("comparison.prepare") > h.events.index("prepare")
    assert h.events.index("comparison.prepare") > h.events.index("slot")
    assert h.events.index("measure") > h.events.index("comparison.prepare")
    assert h.plugin.contexts["prepare"] is None
    assert h.plugin.contexts["measure"] == h.issued
    assert h.plugin.contexts["finalize"] == h.issued
    request = h.prepared_calls[0]["request"]
    assert request.candidate_identity == "binary:abc"
    assert request.measurement_contract_fingerprint == "benchmark-v1"
    assert request.job_id == h.job.job_id
    assert request.run_token == h.job.run_token
    assert len(h.recorded) == 1


def test_fresh_comparison_bypasses_both_measurement_cache_and_source_reuse(worker_harness):
    h = worker_harness
    h.runtime.cached = object()  # Any lookup/reuse would fail immediately.

    def forbidden_lookup(**kwargs):
        raise AssertionError("fresh comparison reused a source-tree outcome")

    h.worker.job_store = SimpleNamespace(find_reusable_evaluation=forbidden_lookup)
    outcome = _run(h)
    assert "cache.lock" not in h.events
    assert "cache.lookup" not in h.events
    assert h.plugin.measurement.cacheable is False
    assert outcome.measurement_payload["cacheable"] is False
    assert outcome.measurement_executed is True
    assert outcome.measurement_reused is False
    assert outcome.reuse_kind == "none"
    assert outcome._runtime_leases == []


@pytest.mark.parametrize("phase", ["prepare", "measure", "finalize"])
def test_only_passing_final_outcome_records_comparison(worker_harness, phase):
    h = worker_harness
    h.plugin.failure_phase = phase
    outcome = _run(h)
    assert outcome.outcome_kind == "candidate_failed"
    assert h.recorded == []
    assert "comparison.record" not in h.events
    assert bool(h.prepared_calls) is (phase != "prepare")
    if phase == "prepare":
        assert "slot" not in h.events
    else:
        assert any(event.startswith("slot.release:") for event in h.events)


def test_invalid_comparison_evidence_is_infrastructure_failure(worker_harness):
    h = worker_harness
    h.plugin.lower_bound = 1000.0  # Greater than measured gain, invalid interval.
    outcome = _run(h)
    assert outcome.outcome_kind == "infrastructure_failed"
    assert outcome.failure.failure_kind == "phased_evaluator_failed"
    assert "exceeds" in outcome.failure.safe_failure_summary
    assert h.recorded == []
    assert outcome.measurement_executed is True
    assert outcome.measurement_payload["cacheable"] is False
    assert any(event.startswith("slot.release:") for event in h.events)


def test_context_preparation_failure_does_not_claim_measurement_executed(
    worker_harness, monkeypatch,
):
    h = worker_harness

    def stale_context(**kwargs):
        raise ValueError("projection is not frozen")

    monkeypatch.setattr(comparison_service, "prepare_context", stale_context)
    outcome = _run(h)
    assert outcome.outcome_kind == "infrastructure_failed"
    assert "projection is not frozen" in outcome.failure.safe_failure_summary
    assert outcome.measurement_executed is False
    assert "measure" not in h.events
    assert "finalize" not in h.events
    assert not h.recorded
    assert any(event.startswith("slot.release:") for event in h.events)


def test_valid_non_improvement_remains_a_passing_measured_candidate(worker_harness):
    h = worker_harness
    h.plugin.lower_bound = -0.02
    outcome = _run(h)
    assert outcome.outcome_kind == "passed"
    assert len(h.recorded) == 1
    assert h.recorded[0][1].allowed is False


@pytest.mark.parametrize("mode", ["default", "seed"])
def test_default_and_seed_retain_original_measurement_reuse(worker_harness, mode):
    h = worker_harness
    if mode == "default":
        h.settings.mapelites_admission_policy = "pareto"
    else:
        h.job.is_seed_job = True
    preparation = _PhasedPlugin().prepare(None)
    cached_measurement = _PhasedPlugin().measure(None, preparation)
    h.runtime.cached = SimpleNamespace(
        id=uuid4(), measurement=cached_measurement, source_evaluation_attempt_id=None,
    )
    outcome = _run(h)
    assert outcome.outcome_kind == "passed"
    assert "cache.lock" in h.events
    assert "cache.lookup" in h.events
    assert "measure" not in h.events
    assert h.prepared_calls == []
    assert h.recorded == []
    assert h.plugin.contexts["finalize"] is None
    assert cached_measurement.cacheable is True
    assert outcome.measurement_executed is False
    assert outcome.measurement_reused is True
    assert outcome.reuse_kind == "measurement"


def test_nonphased_evaluator_fails_before_invocation_or_cached_reuse(worker_harness):
    h = worker_harness

    def forbidden_call(*args, **kwargs):
        raise AssertionError("one-shot evaluator must not run in fresh mode")

    h.worker.evaluator = SimpleNamespace(
        evaluator_name="legacy", evaluator_version="v1", evaluate_outcome=forbidden_call,
    )
    h.worker.job_store = SimpleNamespace(find_reusable_evaluation=forbidden_call)
    with pytest.raises(EvolutionWorkerError, match="requires a phased-v1"):
        _run(h)
    assert not h.prepared_calls
    assert not h.recorded


@pytest.mark.parametrize("mode", ["default", "seed"])
def test_legacy_evaluator_still_runs_for_default_or_seed(worker_harness, mode):
    h = worker_harness
    if mode == "default":
        h.settings.mapelites_admission_policy = "pareto"
    else:
        h.job.is_seed_job = True
    calls = []

    def evaluate(context):
        calls.append(context)
        return EvaluationResult(summary="legacy passed")

    h.worker.evaluator = SimpleNamespace(evaluate=evaluate)
    assert _run(h).outcome_kind == "passed"
    assert len(calls) == 1
    assert calls[0].comparison is None
    assert not h.prepared_calls
    assert not h.recorded
