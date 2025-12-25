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
    fitness: float = 1.0
    measures: Sequence[float] = ()
    solution: Sequence[float] = ()
    timestamp: float = 0.0


class FakeManager:
    def __init__(self, records: Sequence[FakeRecord]) -> None:
        self._records = tuple(records)

    def get_records(self, island_id: str | None = None) -> tuple[FakeRecord, ...]:
        return self._records


def make_sampler(settings: Settings, records: Sequence[FakeRecord]) -> MapElitesSampler:
    settings.mapelites_dimensionality_target_dims = 2
    settings.mapelites_archive_cells_per_dim = 3
    settings.mapelites_sampler_inspiration_count = 3
    settings.mapelites_sampler_neighbor_radius = 1
    settings.mapelites_sampler_neighbor_max_radius = 1
    settings.mapelites_sampler_fallback_sample_size = 4

    manager = FakeManager(records)
    rng = np.random.default_rng(1234)
    # MapElitesSampler accepts random.Random; we use a simple wrapper here
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
    records_by_cell = {record.cell_index: record for record in records}

    inspirations, stats = sampler._select_inspirations(base, records_by_cell)  # type: ignore[attr-defined]

    assert len(inspirations) <= settings.mapelites_sampler_inspiration_count
    assert base.commit_hash not in {rec.commit_hash for rec in inspirations}
    assert stats["radius_used"] <= settings.mapelites_sampler_neighbor_max_radius


def test_schedule_job_with_and_without_records(monkeypatch, settings: Settings) -> None:
    empty_manager = FakeManager(records=[])
    sampler_empty = MapElitesSampler(manager=empty_manager, settings=settings)
    assert sampler_empty.schedule_job() is None

    records = [FakeRecord(commit_hash=f"c{i}", cell_index=i) for i in range(4)]
    sampler = MapElitesSampler(manager=FakeManager(records), settings=settings)

    captured_experiment_ids: list[Any] = []

    def fake_persist_job(  # type: ignore[unused-argument]
        self,
        *,
        island_id,
        base,
        inspirations,
        selection_stats,
        iteration_hint,
        priority,
        experiment_id,
    ):
        captured_experiment_ids.append(experiment_id)
        return SimpleNamespace(id=uuid4())

    monkeypatch.setattr(MapElitesSampler, "_persist_job", fake_persist_job)

    # When no experiment_id is provided, jobs are still scheduled and the
    # persisted row should see a null experiment identifier.
    job = sampler.schedule_job()
    assert job is not None
    assert job.job_id is not None
    assert job.base_commit_hash in {record.commit_hash for record in records}
    assert captured_experiment_ids[-1] is None

    # When an experiment_id is provided, it should be threaded through
    # to the persistence layer unchanged.
    exp_id = uuid4()
    job_with_exp = sampler.schedule_job(experiment_id=exp_id)
    assert job_with_exp is not None
    assert captured_experiment_ids[-1] == exp_id


