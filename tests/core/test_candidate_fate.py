from __future__ import annotations

from types import SimpleNamespace

from loreley.core.candidate_fate import derive_candidate_fate
from loreley.db.models import JobStatus


def test_succeeded_new_archive_insert_derives_elite_inserted() -> None:
    job = SimpleNamespace(
        status=JobStatus.SUCCEEDED,
        result_commit_hash="abc123",
        ingestion_status="succeeded",
        ingestion_status_code=2,
        ingestion_cell_index=7,
    )
    candidate = SimpleNamespace(evaluation_status="passed", repair_state="audit_only")

    fate = derive_candidate_fate(
        job=job,
        candidate=candidate,
        current_archive_cell_index=7,
        current_archive_member=True,
    )

    assert fate.label == "elite_inserted"
    assert fate.reason == "Candidate entered an empty archive niche. archive status_code=2. cell=7."


def test_succeeded_archive_improvement_derives_elite_replaced() -> None:
    job = SimpleNamespace(
        status="succeeded",
        result_commit_hash="abc123",
        ingestion_status="succeeded",
        ingestion_status_code=1,
        ingestion_cell_index=3,
    )
    candidate = SimpleNamespace(evaluation_status="passed", repair_state="audit_only")

    fate = derive_candidate_fate(
        job=job,
        candidate=candidate,
        current_archive_cell_index=3,
        current_archive_member=True,
    )

    assert fate.label == "elite_replaced"
    assert fate.reason == "Candidate improved an occupied archive niche. archive status_code=1. cell=3."


def test_current_archive_membership_derives_elite_retained() -> None:
    candidate = SimpleNamespace(
        commit_hash="abc123",
        evaluation_status="passed",
        archive_status="member",
        repair_state="audit_only",
    )

    fate = derive_candidate_fate(
        candidate=candidate,
        current_archive_cell_index=4,
        current_archive_member=True,
    )

    assert fate.label == "elite_retained"
    assert fate.reason == "Candidate is a current archive elite. cell=4."


def test_skipped_passed_candidate_derives_valid_not_elite() -> None:
    job = SimpleNamespace(
        status="succeeded",
        result_commit_hash="abc123",
        ingestion_status="skipped",
        ingestion_message="Commit not inserted; objective below cell threshold.",
    )
    candidate = SimpleNamespace(evaluation_status="passed", archive_status="rejected")

    fate = derive_candidate_fate(job=job, candidate=candidate)

    assert fate.label == "valid_not_elite"
    assert (
        fate.reason
        == "Candidate passed evaluation but did not enter the archive. Commit not inserted; objective below cell threshold."
    )


def test_passed_without_ingestion_derives_valid_not_considered() -> None:
    job = SimpleNamespace(status="succeeded", result_commit_hash="abc123")
    candidate = SimpleNamespace(evaluation_status="passed", archive_status="not_considered")

    fate = derive_candidate_fate(job=job, candidate=candidate)

    assert fate.label == "valid_not_considered"
    assert fate.reason == "Candidate passed evaluation but archive insertion has not been recorded."


def test_repair_eligible_failure_derives_repair_pending() -> None:
    candidate = SimpleNamespace(
        evaluation_status="candidate_failed",
        repair_state="eligible",
        failure_stage="evaluation",
        failure_kind="regression",
    )

    fate = derive_candidate_fate(candidate=candidate)

    assert fate.label == "repair_pending"
    assert fate.reason == "Candidate repair_state=eligible. Failure stage=evaluation kind=regression."


def test_policy_failure_derives_policy_failed() -> None:
    candidate = SimpleNamespace(
        evaluation_status="candidate_failed",
        repair_state="ineligible",
        failure_stage="policy",
        failure_kind="scope_violation",
        failure_summary="Protected path changed.",
    )

    fate = derive_candidate_fate(candidate=candidate)

    assert fate.label == "policy_failed"
    assert (
        fate.reason
        == "Campaign or evaluator policy rejected the candidate. Failure stage=policy kind=scope_violation. Protected path changed."
    )


def test_discarded_lifecycle_derives_discarded_for_sampling() -> None:
    candidate = SimpleNamespace(
        evaluation_status="passed",
        lifecycle_status="discarded",
        archive_status="rejected",
    )

    fate = derive_candidate_fate(candidate=candidate)

    assert fate.label == "discarded_for_sampling"
    assert fate.reason == "Candidate lifecycle_status=discarded; excluded from default future sampling."
