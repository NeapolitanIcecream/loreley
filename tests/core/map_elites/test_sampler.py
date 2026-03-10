from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
from types import SimpleNamespace

from loreley.config import Settings
from loreley.core.map_elites.sampler import MapElitesSampler


@dataclass(slots=True)
class FakeRecord:
    commit_hash: str
    cell_index: int
    objective: float = 0.0


class FakeManager:
    def __init__(self, records: Sequence[FakeRecord]) -> None:
        self._records = tuple(records)

    def get_cell_commits(self, island_id: str | None = None) -> Mapping[int, str]:  # noqa: ARG002
        return {record.cell_index: record.commit_hash for record in self._records}

    def get_cell_objectives(self, island_id: str | None = None) -> Mapping[int, float]:  # noqa: ARG002
        return {record.cell_index: record.objective for record in self._records}


def make_sampler(settings: Settings, records: Sequence[FakeRecord]) -> MapElitesSampler:
    settings.mapelites_dimensionality_target_dims = 2
    settings.mapelites_archive_cells_per_dim = 3
    settings.mapelites_sampler_inspiration_count = 3
    settings.mapelites_sampler_neighbor_radius = 1
    settings.mapelites_sampler_neighbor_max_radius = 1
    settings.mapelites_sampler_fallback_sample_size = 4

    manager = FakeManager(records)
    import random

    python_rng = random.Random(1234)
    return MapElitesSampler(manager=manager, settings=settings, rng=python_rng)


def test_neighbor_indices_within_grid_and_exclude_center(settings: Settings) -> None:
    center_index = int(np.ravel_multi_index((1, 1), (3, 3)))
    sampler = make_sampler(settings, records=[])

    neighbors = sampler._neighbor_indices(center_index, radius=1)  # type: ignore[attr-defined]
    assert neighbors

    coords = {
        tuple(int(v) for v in np.unravel_index(idx, sampler._grid_shape))  # type: ignore[attr-defined]
        for idx in neighbors
    }

    assert (1, 1) not in coords
    for r, c in coords:
        assert 0 <= r < 3
        assert 0 <= c < 3
        assert max(abs(r - 1), abs(c - 1)) <= 1


def test_select_inspirations_respects_inspiration_count(settings: Settings) -> None:
    records = [FakeRecord(commit_hash=f"c{i}", cell_index=i) for i in range(9)]
    sampler = make_sampler(settings, records=records)
    base = records[4]
    cell_commits = {record.cell_index: record.commit_hash for record in records}

    inspirations, stats = sampler._select_inspirations(  # type: ignore[attr-defined]
        base_cell_index=base.cell_index,
        base_commit_hash=base.commit_hash,
        cell_commits=cell_commits,
    )

    assert len(inspirations) <= settings.mapelites_sampler_inspiration_count
    assert base.commit_hash not in set(inspirations)
    assert stats["radius_used"] <= settings.mapelites_sampler_neighbor_max_radius


def test_select_inspirations_does_not_call_neighbor_indices(monkeypatch, settings: Settings) -> None:
    import random

    settings.mapelites_dimensionality_target_dims = 12
    settings.mapelites_archive_cells_per_dim = 4
    settings.mapelites_sampler_inspiration_count = 3
    settings.mapelites_sampler_neighbor_radius = 2
    settings.mapelites_sampler_neighbor_max_radius = 3
    settings.mapelites_sampler_fallback_sample_size = 0

    shape = tuple(
        settings.mapelites_archive_cells_per_dim for _ in range(settings.mapelites_dimensionality_target_dims)
    )
    base_coord = tuple(1 for _ in range(settings.mapelites_dimensionality_target_dims))
    base_index = int(np.ravel_multi_index(base_coord, shape))

    neighbor1 = list(base_coord)
    neighbor1[0] = 2
    neighbor1_index = int(np.ravel_multi_index(tuple(neighbor1), shape))

    neighbor2 = list(base_coord)
    neighbor2[1] = 3
    neighbor2_index = int(np.ravel_multi_index(tuple(neighbor2), shape))

    records = [
        FakeRecord(commit_hash="base", cell_index=base_index),
        FakeRecord(commit_hash="n1", cell_index=neighbor1_index),
        FakeRecord(commit_hash="n2", cell_index=neighbor2_index),
    ]
    sampler = MapElitesSampler(
        manager=FakeManager(records),
        settings=settings,
        rng=random.Random(1234),
    )
    cell_commits = {record.cell_index: record.commit_hash for record in records}

    def explode(self, center_index: int, radius: int) -> list[int]:  # noqa: ARG001
        raise RuntimeError("_neighbor_indices should not be used by _select_inspirations")

    monkeypatch.setattr(MapElitesSampler, "_neighbor_indices", explode)

    inspirations, stats = sampler._select_inspirations(  # type: ignore[attr-defined]
        base_cell_index=base_index,
        base_commit_hash="base",
        cell_commits=cell_commits,
    )
    assert len(inspirations) == 2
    assert set(inspirations) == {"n1", "n2"}
    assert stats["radius_used"] == settings.mapelites_sampler_neighbor_radius
    assert stats["radius_used"] <= settings.mapelites_sampler_neighbor_max_radius


def test_schedule_job_with_and_without_records(monkeypatch, settings: Settings) -> None:
    empty_manager = FakeManager(records=[])
    sampler_empty = MapElitesSampler(manager=empty_manager, settings=settings)
    assert sampler_empty.schedule_job() is None

    records = [FakeRecord(commit_hash=f"c{i}", cell_index=i) for i in range(4)]
    sampler = MapElitesSampler(manager=FakeManager(records), settings=settings)

    captured_calls: list[dict[str, Any]] = []

    def fake_persist_job(  # type: ignore[unused-argument]
        self,
        *,
        island_id,
        base_commit_hash,
        inspiration_commit_hashes,
        selection_stats,
        iteration_hint,
        priority,
    ):
        captured_calls.append(
            {
                "island_id": island_id,
                "base_commit_hash": base_commit_hash,
                "inspiration_commit_hashes": inspiration_commit_hashes,
                "selection_stats": selection_stats,
                "iteration_hint": iteration_hint,
                "priority": priority,
            }
        )
        return SimpleNamespace(id=uuid4())

    monkeypatch.setattr(MapElitesSampler, "_persist_job", fake_persist_job)

    # Jobs are still scheduled for non-empty archives.
    job = sampler.schedule_job()
    assert job is not None
    assert job.job_id is not None
    assert job.base_commit_hash in {record.commit_hash for record in records}
    assert captured_calls


def test_schedule_job_prefers_higher_objective_cells_when_available(
    monkeypatch,
    settings: Settings,
) -> None:
    records = [
        FakeRecord(commit_hash="low", cell_index=0, objective=0.5),
        FakeRecord(commit_hash="high", cell_index=1, objective=5.0),
    ]

    class RecordingRng:
        def __init__(self) -> None:
            self.random_values = [0.99]
            self.choice_calls = 0
            self.shuffle_calls = 0

        def choice(self, items: Sequence[tuple[int, str]]) -> tuple[int, str]:
            self.choice_calls += 1
            return items[0]

        def random(self) -> float:
            return self.random_values.pop(0)

        def shuffle(self, items: list[int]) -> None:
            self.shuffle_calls += 1

    sampler = MapElitesSampler(
        manager=FakeManager(records),
        settings=settings,
        rng=RecordingRng(),
    )

    monkeypatch.setattr(
        MapElitesSampler,
        "_persist_job",
        lambda *_args, **_kwargs: SimpleNamespace(id=uuid4()),
    )

    job = sampler.schedule_job()

    assert job is not None
    assert job.base_commit_hash == "high"


def test_schedule_job_excludes_base_commits_selected_earlier_in_batch(
    monkeypatch,
    settings: Settings,
) -> None:
    records = [
        FakeRecord(commit_hash="c0", cell_index=0, objective=0.5),
        FakeRecord(commit_hash="c1", cell_index=1, objective=0.6),
        FakeRecord(commit_hash="c2", cell_index=2, objective=0.7),
    ]
    sampler = make_sampler(settings, records=records)

    monkeypatch.setattr(
        MapElitesSampler,
        "_persist_job",
        lambda *_args, **_kwargs: SimpleNamespace(id=uuid4()),
    )

    snapshot = sampler.get_sampling_snapshot()
    assert snapshot is not None

    first = sampler.schedule_job(
        sampling_snapshot=snapshot,
        excluded_base_commits=set(),
    )
    assert first is not None

    second = sampler.schedule_job(
        sampling_snapshot=snapshot,
        excluded_base_commits={first.base_commit_hash},
    )
    assert second is not None
    assert second.base_commit_hash != first.base_commit_hash
