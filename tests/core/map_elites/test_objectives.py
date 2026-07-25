from __future__ import annotations

import math

import pytest

from loreley.core.map_elites.objectives import (
    ObjectiveContract,
    ObjectiveContractError,
    ObjectiveSpec,
)


def _metric(name: str, value: object, *, higher_is_better: object) -> dict[str, object]:
    return {
        "name": name,
        "value": value,
        "higher_is_better": higher_is_better,
    }


def test_objective_contract_resolves_raw_and_all_max_scores() -> None:
    contract = ObjectiveContract(
        (
            ObjectiveSpec(name="throughput", direction="max"),
            ObjectiveSpec(name="p99_latency_ms", direction="min"),
        )
    )

    resolved = contract.resolve(
        (
            _metric("p99_latency_ms", 12.5, higher_is_better=False),
            _metric("throughput", "80", higher_is_better=True),
            _metric("diagnostic_only", 3, higher_is_better=True),
        )
    )

    assert resolved.values == (80.0, 12.5)
    assert resolved.scores == (80.0, -12.5)
    assert contract.primary.name == "throughput"
    assert contract.names == ("throughput", "p99_latency_ms")


def test_objective_contract_fingerprint_is_canonical_and_order_sensitive() -> None:
    first = ObjectiveContract(
        (
            ObjectiveSpec(name="quality", direction="max"),
            ObjectiveSpec(name="latency", direction="min"),
        )
    )
    same = ObjectiveContract(
        (
            ObjectiveSpec(name="quality", direction="max"),
            ObjectiveSpec(name="latency", direction="min"),
        )
    )
    reordered = ObjectiveContract(tuple(reversed(first.specs)))

    assert first.fingerprint == same.fingerprint
    assert first.as_payload() == [
        {"name": "quality", "direction": "max"},
        {"name": "latency", "direction": "min"},
    ]
    assert first.fingerprint != reordered.fingerprint


@pytest.mark.parametrize(
    ("metrics", "message"),
    [
        (
            (_metric("quality", 1.0, higher_is_better=True),),
            "Missing configured objective",
        ),
        (
            (
                _metric("quality", math.nan, higher_is_better=True),
                _metric("latency", 2.0, higher_is_better=False),
            ),
            "finite",
        ),
        (
            (
                _metric("quality", math.inf, higher_is_better=True),
                _metric("latency", 2.0, higher_is_better=False),
            ),
            "finite",
        ),
        (
            (
                _metric("quality", 1.0, higher_is_better=False),
                _metric("latency", 2.0, higher_is_better=False),
            ),
            "direction",
        ),
        (
            (
                _metric("quality", True, higher_is_better=True),
                _metric("latency", 2.0, higher_is_better=False),
            ),
            "boolean",
        ),
    ],
)
def test_objective_contract_rejects_incomplete_or_invalid_metrics(
    metrics: tuple[dict[str, object], ...],
    message: str,
) -> None:
    contract = ObjectiveContract(
        (
            ObjectiveSpec(name="quality", direction="max"),
            ObjectiveSpec(name="latency", direction="min"),
        )
    )

    with pytest.raises(ObjectiveContractError, match=message):
        contract.resolve(metrics)


def test_objective_contract_rejects_duplicate_names_and_duplicate_metrics() -> None:
    with pytest.raises(ObjectiveContractError, match="unique"):
        ObjectiveContract(
            (
                ObjectiveSpec(name="score", direction="max"),
                ObjectiveSpec(name="score", direction="min"),
            )
        )

    contract = ObjectiveContract((ObjectiveSpec(name="score", direction="max"),))
    with pytest.raises(ObjectiveContractError, match="Duplicate metric"):
        contract.resolve(
            (
                _metric("score", 1.0, higher_is_better=True),
                _metric("score", 2.0, higher_is_better=True),
            )
        )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (False, False),
        ("false", False),
        ("0", False),
        (True, True),
        ("true", True),
        ("1", True),
    ],
)
def test_objective_contract_parses_direction_without_truthiness(
    raw: object,
    expected: bool,
) -> None:
    direction = "max" if expected else "min"
    contract = ObjectiveContract((ObjectiveSpec(name="score", direction=direction),))

    resolved = contract.resolve((_metric("score", 1.0, higher_is_better=raw),))

    assert resolved.values == (1.0,)
