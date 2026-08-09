from __future__ import annotations

from loreley.core.worker.candidate_identity import (
    evaluation_identity_key,
    normalize_candidate_identity,
)


def test_candidate_identity_is_normalized_and_bounded() -> None:
    assert normalize_candidate_identity(" binary\nsha ") == "binary sha"
    assert len(normalize_candidate_identity("x" * 600) or "") == 512


def test_evaluation_identity_key_scopes_the_evaluator_contract() -> None:
    base = evaluation_identity_key(
        candidate_identity="binary-sha256:abc",
        evaluator_name="example:evaluate",
        evaluator_version="source-sha256:v1",
        campaign_program_hash="campaign-a",
    )
    same = evaluation_identity_key(
        candidate_identity="binary-sha256:abc",
        evaluator_name="example:evaluate",
        evaluator_version="source-sha256:v1",
        campaign_program_hash="campaign-a",
    )
    changed_protocol = evaluation_identity_key(
        candidate_identity="binary-sha256:abc",
        evaluator_name="example:evaluate",
        evaluator_version="source-sha256:v2",
        campaign_program_hash="campaign-a",
    )
    changed_measurement = evaluation_identity_key(
        candidate_identity="binary-sha256:abc",
        evaluator_name="example:evaluate",
        evaluator_version="source-sha256:v1",
        campaign_program_hash="campaign-a",
        measurement_contract_fingerprint="different-validation-corpus",
    )

    assert base == same
    assert base != changed_protocol
    assert base == changed_measurement
    assert len(base or "") == 64


def test_missing_candidate_identity_disables_equivalence_deduplication() -> None:
    assert (
        evaluation_identity_key(
            candidate_identity=None,
            evaluator_name="example:evaluate",
            evaluator_version="v1",
            campaign_program_hash="campaign-a",
        )
        is None
    )
