from __future__ import annotations

import json

from loreley.core.evaluation import (
    AdaptiveSamplingConfig,
    HoeffdingConfidenceSequence,
    MeasurementContract,
)


def _contract(**changes):
    values = {
        "workload_fingerprint": "corpus-sha256:abc",
        "metric_name": "throughput_ratio",
        "metric_unit": "ratio",
        "estimand": "equal-weight mean across declared corpus strata",
        "higher_is_better": True,
        "interval_method": HoeffdingConfidenceSequence(0.0, 2.0),
        "sampling": AdaptiveSamplingConfig(
            min_samples=8,
            max_samples=32,
            batch_size=4,
            target_ci_half_width=0.02,
            stratum_weights={"large": 0.5, "small": 0.5},
        ),
        "stratum_weights": {"large": 0.5, "small": 0.5},
        "metadata": {"pairing": "root_candidate", "compiler": "clang"},
    }
    values.update(changes)
    return MeasurementContract(**values)


def test_measurement_contract_has_stable_canonical_fingerprint() -> None:
    first = _contract(stratum_weights={"small": 0.5, "large": 0.5})
    second = _contract(stratum_weights={"large": 0.5, "small": 0.5})

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64
    payload = first.as_dict()
    assert payload["interval"]["capability"] == "anytime_valid"
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_measurement_contract_changes_when_evidence_semantics_change() -> None:
    base = _contract()
    changed_workload = _contract(workload_fingerprint="corpus-sha256:def")
    changed_bounds = _contract(
        interval_method=HoeffdingConfidenceSequence(0.0, 3.0)
    )
    changed_budget = _contract(
        sampling=AdaptiveSamplingConfig(
            min_samples=8,
            max_samples=64,
            batch_size=4,
            target_ci_half_width=0.02,
            stratum_weights={"large": 0.5, "small": 0.5},
        )
    )

    assert len(
        {base.fingerprint, changed_workload.fingerprint, changed_bounds.fingerprint,
         changed_budget.fingerprint}
    ) == 4
