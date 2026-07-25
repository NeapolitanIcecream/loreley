from __future__ import annotations

import random
from itertools import permutations

import numpy as np

from loreley.core.map_elites.pareto_archive import ParetoCandidate, ParetoGridArchive


def _archive(
    *,
    capacity: int = 8,
    objective_count: int = 2,
) -> ParetoGridArchive:
    return ParetoGridArchive(
        dims=(4, 4),
        ranges=((0.0, 1.0), (0.0, 1.0)),
        objective_count=objective_count,
        max_front_size=capacity,
        epsilon=1.0e-9,
    )


def _candidate(
    commit_hash: str,
    objectives: tuple[float, ...],
    *,
    measures: tuple[float, float] = (0.2, 0.2),
    timestamp: float = 1.0,
) -> ParetoCandidate:
    return ParetoCandidate(
        commit_hash=commit_hash,
        objective_values=objectives,
        objective_scores=objectives,
        measures=measures,
        timestamp=timestamp,
    )


def test_same_cell_keeps_nondominated_tradeoffs_and_rejects_dominated_candidate() -> None:
    archive = _archive()

    outcomes = archive.add_many(
        (
            _candidate("a", (10.0, 1.0)),
            _candidate("b", (1.0, 10.0)),
            _candidate("c", (6.0, 6.0)),
            _candidate("d", (5.0, 5.0)),
        )
    )

    cell_index = int(archive.index_of(np.asarray([[0.2, 0.2]])).item())
    assert {entry.commit_hash for entry in archive.front(cell_index)} == {"a", "b", "c"}
    assert [outcome.retained for outcome in outcomes] == [True, True, True, False]
    assert archive.stats.num_elites == 3
    assert archive.stats.num_occupied == 1


def test_equivalent_objectives_keep_deterministic_commit_representative() -> None:
    archive = _archive()

    archive.add(_candidate("z-later", (4.0, 4.0), timestamp=1.0))
    outcome = archive.add(_candidate("a-stable", (4.0, 4.0), timestamp=2.0))

    assert outcome.retained is True
    assert outcome.removed_commit_hashes == ("z-later",)
    assert [entry.commit_hash for entry in archive.records()] == ["a-stable"]


def test_crowding_capacity_preserves_two_objective_boundaries() -> None:
    archive = _archive(capacity=2)

    archive.add_many(
        (
            _candidate("quality-boundary", (10.0, 0.0)),
            _candidate("latency-boundary", (0.0, 10.0)),
            _candidate("middle", (6.0, 6.0)),
        )
    )

    assert {entry.commit_hash for entry in archive.records()} == {
        "quality-boundary",
        "latency-boundary",
    }


def test_crowding_ignores_constant_objectives_when_selecting_boundaries() -> None:
    archive = _archive(capacity=2, objective_count=3)

    archive.add_many(
        (
            _candidate("z-left", (1.0, 0.0, 10.0)),
            _candidate("a-middle", (1.0, 6.0, 6.0)),
            _candidate("y-right", (1.0, 10.0, 0.0)),
        )
    )

    assert {entry.commit_hash for entry in archive.records()} == {
        "z-left",
        "y-right",
    }


def test_batch_pruning_is_independent_of_candidate_order() -> None:
    candidates = (
        _candidate("a", (10.0, 0.0)),
        _candidate("b", (0.0, 10.0)),
        _candidate("c", (8.0, 4.0)),
        _candidate("d", (4.0, 8.0)),
        _candidate("e", (6.0, 6.0)),
    )
    retained_sets: set[tuple[str, ...]] = set()

    for ordering in permutations(candidates):
        archive = _archive(capacity=3)
        archive.add_many(ordering)
        retained_sets.add(tuple(entry.commit_hash for entry in archive.records()))

    assert len(retained_sets) == 1


def test_sampling_chooses_a_cell_before_a_front_member() -> None:
    archive = _archive()
    archive.add_many(
        (
            _candidate("front-a", (10.0, 1.0), measures=(0.1, 0.1)),
            _candidate("front-b", (1.0, 10.0), measures=(0.1, 0.1)),
            _candidate("solo", (5.0, 5.0), measures=(0.9, 0.9)),
        )
    )

    counts = {"front": 0, "solo": 0}
    rng = random.Random(42)
    for _ in range(4000):
        sampled = archive.sample(1, rng=rng)[0]
        if sampled.commit_hash == "solo":
            counts["solo"] += 1
        else:
            counts["front"] += 1

    assert 1700 <= counts["front"] <= 2300
    assert 1700 <= counts["solo"] <= 2300


def test_same_commit_can_be_retained_by_independent_islands() -> None:
    first = _archive()
    second = _archive()
    candidate = _candidate("shared", (3.0, 7.0))

    assert first.add(candidate).retained is True
    assert second.add(candidate).retained is True
    assert [entry.commit_hash for entry in first.records()] == ["shared"]
    assert [entry.commit_hash for entry in second.records()] == ["shared"]
