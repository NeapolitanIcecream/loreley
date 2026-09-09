from __future__ import annotations

import hashlib
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest

import loreley.core.fresh_comparison as comparison
from loreley.core.evolution_events import COMPARISON_PREPARED, EvolutionEventValidationError
from loreley.core.worker.evaluation_runtime import EvaluationRuntimeError


def _binding():
    commit = "c" * 40
    job = SimpleNamespace(island_id="main", campaign_program_hash="a" * 64)
    attempt = SimpleNamespace(
        evaluator_name="paired",
        evaluator_version="1",
        campaign_program_hash=job.campaign_program_hash,
        candidate_identity="sha256:binary",
        measurement_contract_fingerprint="petclinic-v1",
    )
    context = {
        "candidate_commit_hash": commit,
        "island_id": job.island_id,
        "campaign_program_hash": job.campaign_program_hash,
        "evaluator_name": attempt.evaluator_name,
        "evaluator_version": attempt.evaluator_version,
        "candidate_identity_sha256": hashlib.sha256(
            attempt.candidate_identity.encode()
        ).hexdigest(),
        "contract_sha256": hashlib.sha256(
            attempt.measurement_contract_fingerprint.encode()
        ).hexdigest(),
    }
    return context, attempt, job, commit


@pytest.mark.parametrize(
    "policy,is_seed,expected",
    [
        ("pareto", False, False),
        ("fresh_comparison", False, True),
        ("fresh_comparison", True, False),
    ],
)
def test_policy_is_opt_in_and_seeds_use_existing_admission(policy, is_seed, expected):
    settings = SimpleNamespace(mapelites_admission_policy=policy)
    assert comparison.enabled(settings, is_seed_job=is_seed) is expected


def test_attempt_binding_accepts_exact_durable_identity() -> None:
    context, attempt, job, commit = _binding()
    comparison._verify_attempt(context, attempt, job, commit)


@pytest.mark.parametrize(
    "field",
    [
        "candidate_commit_hash",
        "island_id",
        "campaign_program_hash",
        "evaluator_name",
        "evaluator_version",
        "candidate_identity_sha256",
        "contract_sha256",
    ],
)
def test_attempt_binding_rejects_altered_or_missing_context_field(field) -> None:
    context, attempt, job, commit = _binding()
    for remove in (False, True):
        changed = dict(context)
        if remove:
            del changed[field]
        else:
            changed[field] = "forged"
        with pytest.raises(EvaluationRuntimeError, match="does not match"):
            comparison._verify_attempt(changed, attempt, job, commit)


@pytest.mark.parametrize(
    "field",
    [
        "evaluator_name",
        "evaluator_version",
        "candidate_identity",
        "measurement_contract_fingerprint",
        "campaign_program_hash",
    ],
)
def test_attempt_binding_rejects_changed_successful_attempt(field) -> None:
    context, attempt, job, commit = _binding()
    setattr(attempt, field, "changed")
    with pytest.raises(EvaluationRuntimeError, match="does not match|different campaign"):
        comparison._verify_attempt(context, attempt, job, commit)


def test_postgres_requirement_fails_before_locking_other_databases() -> None:
    session = Mock()
    session.get_bind.return_value.dialect.name = "sqlite"
    with pytest.raises(EvaluationRuntimeError, match="PostgreSQL"):
        comparison.lock_comparison(session, SimpleNamespace(experiment_id="test"))
    session.execute.assert_not_called()


def test_lock_key_is_stable_per_experiment_and_does_not_depend_on_python_hash() -> None:
    session = Mock()
    session.get_bind.return_value.dialect.name = "postgresql"
    settings = SimpleNamespace(experiment_id="one")
    comparison.lock_comparison(session, settings)
    first = session.execute.call_args.args[1]["key"]
    comparison.lock_comparison(session, settings)
    assert session.execute.call_args.args[1]["key"] == first
    comparison.lock_comparison(session, SimpleNamespace(experiment_id="two"))
    assert session.execute.call_args.args[1]["key"] != first
    assert -(2**63) <= first < 2**63


@pytest.mark.parametrize(
    "payload", [{"context_id": "x" * 257}, {"context_id": "x", "unexpected": 1}]
)
def test_event_identity_cannot_be_silently_truncated_or_dropped(monkeypatch, payload):
    writer = Mock()
    monkeypatch.setattr(comparison, "record_evolution_event", writer)
    with pytest.raises(
        (EvaluationRuntimeError, EvolutionEventValidationError),
        match="identity contract|rejected payload fields",
    ):
        comparison._write_event(
            Mock(),
            event_type=COMPARISON_PREPARED,
            job_id=uuid4(),
            run_token=uuid4(),
            context={"island_id": "main", "candidate_commit_hash": "c" * 40},
            payload=payload,
        )
    writer.assert_not_called()
