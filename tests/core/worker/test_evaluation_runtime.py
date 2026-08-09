from __future__ import annotations

import math

import pytest

from loreley.core.worker.evaluation_runtime import (
    EvaluationRuntimeError,
    evaluation_contract_key,
    measurement_cache_key,
    measurement_payload_sha256,
)
from loreley.core.worker.evaluator import EvaluationPreparation


def _preparation(*, identity: str = "binary:abc", fingerprint: str = "bench-v1"):
    return EvaluationPreparation(
        candidate_identity=identity,
        measurement_contract_fingerprint=fingerprint,
        state={"build": "release"},
    )


def test_measurement_cache_key_covers_the_complete_contract() -> None:
    base = measurement_cache_key(
        preparation=_preparation(),
        evaluator_name="demo",
        evaluator_version="1",
        campaign_program_hash="a" * 64,
    )

    variants = {
        measurement_cache_key(
            preparation=_preparation(identity="binary:def"),
            evaluator_name="demo",
            evaluator_version="1",
            campaign_program_hash="a" * 64,
        ),
        measurement_cache_key(
            preparation=_preparation(fingerprint="bench-v2"),
            evaluator_name="demo",
            evaluator_version="1",
            campaign_program_hash="a" * 64,
        ),
        measurement_cache_key(
            preparation=_preparation(),
            evaluator_name="demo",
            evaluator_version="2",
            campaign_program_hash="a" * 64,
        ),
        measurement_cache_key(
            preparation=_preparation(),
            evaluator_name="demo",
            evaluator_version="1",
            campaign_program_hash="b" * 64,
        ),
    }

    assert base not in variants
    assert len(variants) == 4


def test_evaluator_contract_key_does_not_fork_when_e_or_scope_changes() -> None:
    first = evaluation_contract_key(
        experiment_id="exp",
        evaluator_name="demo",
        evaluator_version="1",
        campaign_program_hash="a" * 64,
    )
    second = evaluation_contract_key(
        experiment_id="exp",
        evaluator_name="demo",
        evaluator_version="1",
        campaign_program_hash="a" * 64,
    )

    assert first == second


def test_measurement_payload_hash_is_canonical_and_rejects_nan() -> None:
    assert measurement_payload_sha256({"b": 2, "a": 1}) == measurement_payload_sha256(
        {"a": 1, "b": 2}
    )
    with pytest.raises(EvaluationRuntimeError, match="canonical JSON"):
        measurement_payload_sha256({"value": math.nan})
