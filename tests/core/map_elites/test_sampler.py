from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
from types import SimpleNamespace

from loreley.config import Settings
from loreley.core.map_elites import sampler as sampler_module
from loreley.core.map_elites.sampler import (
    MapElitesSampler,
    ScheduleJobRequest,
    sampling_recipe_hash,
)


@dataclass(slots=True)
class FakeRecord:
    commit_hash: str
    cell_index: int


class FakeManager:
    def __init__(self, records: Sequence[FakeRecord]) -> None:
        self._records = tuple(records)

    def get_cell_fronts(
        self,
        island_id: str | None = None,  # noqa: ARG002
    ) -> Mapping[int, tuple[str, ...]]:
        fronts: dict[int, list[str]] = {}
        for record in self._records:
            fronts.setdefault(record.cell_index, []).append(record.commit_hash)
        return {
            cell_index: tuple(commits)
            for cell_index, commits in fronts.items()
        }


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
    cell_fronts = {
        record.cell_index: (record.commit_hash,)
        for record in records
    }

    inspirations, stats = sampler._select_inspirations(  # type: ignore[attr-defined]
        base_cell_index=base.cell_index,
        base_commit_hash=base.commit_hash,
        cell_fronts=cell_fronts,
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
    cell_fronts = {
        record.cell_index: (record.commit_hash,)
        for record in records
    }

    def explode(self, center_index: int, radius: int) -> list[int]:  # noqa: ARG001
        raise RuntimeError("_neighbor_indices should not be used by _select_inspirations")

    monkeypatch.setattr(MapElitesSampler, "_neighbor_indices", explode)

    inspirations, stats = sampler._select_inspirations(  # type: ignore[attr-defined]
        base_cell_index=base_index,
        base_commit_hash="base",
        cell_fronts=cell_fronts,
    )
    assert len(inspirations) == 2
    assert set(inspirations) == {"n1", "n2"}
    assert stats["radius_used"] == settings.mapelites_sampler_neighbor_radius
    assert stats["radius_used"] <= settings.mapelites_sampler_neighbor_max_radius


def test_neighbor_candidate_positions_keep_initial_radius_shell_semantics() -> None:
    distances = np.asarray([0, 1, 2, 3], dtype=np.int64)

    initial = sampler_module._neighbor_candidate_positions(  # noqa: SLF001
        distances=distances,
        radius=2,
        first_radius=2,
    )
    later = sampler_module._neighbor_candidate_positions(  # noqa: SLF001
        distances=distances,
        radius=3,
        first_radius=2,
    )

    assert initial == [1, 2]
    assert later == [3]


def test_fallback_inspiration_candidates_exclude_base_and_selected_commits() -> None:
    candidates = sampler_module._fallback_inspiration_candidates(  # noqa: SLF001
        base_cell_index=1,
        cell_fronts={
            1: ("base",),
            2: ("already-selected",),
            3: ("candidate",),
            4: (),
        },
        selected_commits={"base", "already-selected"},
    )

    assert candidates == ("candidate",)


def test_schedule_job_with_and_without_records(monkeypatch, settings: Settings) -> None:
    empty_manager = FakeManager(records=[])
    sampler_empty = MapElitesSampler(manager=empty_manager, settings=settings)
    assert sampler_empty.schedule_job() is None

    records = [FakeRecord(commit_hash=f"c{i}", cell_index=i) for i in range(4)]
    sampler = MapElitesSampler(manager=FakeManager(records), settings=settings)

    captured_calls: list[Any] = []

    def fake_persist_job(self: MapElitesSampler, request: Any) -> SimpleNamespace:  # noqa: ARG001
        captured_calls.append(request)
        return SimpleNamespace(id=uuid4())

    monkeypatch.setattr(MapElitesSampler, "_persist_job", fake_persist_job)

    # Jobs are still scheduled for non-empty archives.
    job = sampler.schedule_job(
        ScheduleJobRequest(
            migration_source_island_id="source",
            migration_commit_hash="migration",
        )
    )
    assert job is not None
    assert job.job_id is not None
    assert job.base_commit_hash in {record.commit_hash for record in records}
    assert job.migration_source_island_id == "source"
    assert job.migration_commit_hash == "migration"
    assert "migration" in job.inspiration_commit_hashes
    assert len(captured_calls) == 1
    assert captured_calls[0].migration_source_island_id == "source"
    assert captured_calls[0].migration_commit_hash == "migration"


def test_zero_inspiration_capacity_disables_migration(
    monkeypatch,
    settings: Settings,
) -> None:
    settings.mapelites_sampler_inspiration_count = 0
    sampler = MapElitesSampler(
        manager=FakeManager([FakeRecord(commit_hash="base", cell_index=0)]),
        settings=settings,
    )
    captured_calls: list[Any] = []

    def fake_persist_job(self: MapElitesSampler, request: Any) -> SimpleNamespace:  # noqa: ARG001
        captured_calls.append(request)
        return SimpleNamespace(id=uuid4())

    monkeypatch.setattr(MapElitesSampler, "_persist_job", fake_persist_job)

    job = sampler.schedule_job(
        ScheduleJobRequest(
            migration_source_island_id="source",
            migration_commit_hash="migration",
        )
    )

    assert job is not None
    assert job.inspiration_commit_hashes == ()
    assert job.migration_source_island_id is None
    assert job.migration_commit_hash is None
    assert captured_calls[0].migration_source_island_id is None
    assert captured_calls[0].migration_commit_hash is None


def test_schedule_job_chooses_cells_without_front_size_bias(
    monkeypatch,
    settings: Settings,
) -> None:
    records = [
        FakeRecord(commit_hash="a", cell_index=0),
        FakeRecord(commit_hash="b", cell_index=0),
        FakeRecord(commit_hash="c", cell_index=0),
        FakeRecord(commit_hash="solo", cell_index=1),
    ]
    import random
    sampler = MapElitesSampler(
        manager=FakeManager(records),
        settings=settings,
        rng=random.Random(77),
    )

    monkeypatch.setattr(
        MapElitesSampler,
        "_persist_job",
        lambda *_args, **_kwargs: SimpleNamespace(id=uuid4()),
    )

    counts = {"front": 0, "solo": 0}
    snapshot = sampler.get_sampling_snapshot()
    assert snapshot is not None
    for _ in range(1000):
        job = sampler.schedule_job(ScheduleJobRequest(sampling_snapshot=snapshot))
        assert job is not None
        if job.base_commit_hash == "solo":
            counts["solo"] += 1
        else:
            counts["front"] += 1
    assert 400 <= counts["front"] <= 600
    assert 400 <= counts["solo"] <= 600


def test_schedule_job_excludes_base_commits_selected_earlier_in_batch(
    monkeypatch,
    settings: Settings,
) -> None:
    records = [
        FakeRecord(commit_hash="c0", cell_index=0),
        FakeRecord(commit_hash="c1", cell_index=1),
        FakeRecord(commit_hash="c2", cell_index=2),
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
        ScheduleJobRequest(
            sampling_snapshot=snapshot,
            excluded_base_commits=set(),
        )
    )
    assert first is not None

    second = sampler.schedule_job(
        ScheduleJobRequest(
            sampling_snapshot=snapshot,
            excluded_base_commits={first.base_commit_hash},
        )
    )
    assert second is not None
    assert second.base_commit_hash != first.base_commit_hash


def test_sampling_ordinal_is_restart_stable(
    monkeypatch,
    settings: Settings,
) -> None:
    records = [FakeRecord(commit_hash=f"c{i}", cell_index=i) for i in range(9)]
    settings.mapelites_sampler_seed = 20260804
    monkeypatch.setattr(
        MapElitesSampler,
        "_persist_job",
        lambda *_args, **_kwargs: SimpleNamespace(id=uuid4()),
    )

    first_sampler = make_sampler(settings, records=records)
    second_sampler = make_sampler(settings, records=list(reversed(records)))
    first = first_sampler.schedule_job(ScheduleJobRequest(sampling_ordinal=71))
    second = second_sampler.schedule_job(ScheduleJobRequest(sampling_ordinal=71))

    assert first is not None and second is not None
    assert (
        first.base_commit_hash,
        first.inspiration_commit_hashes,
        first.sampling_recipe_hash,
    ) == (
        second.base_commit_hash,
        second.inspiration_commit_hashes,
        second.sampling_recipe_hash,
    )


def test_recipe_cooldown_resamples_instead_of_replaying(
    monkeypatch,
    settings: Settings,
) -> None:
    records = [FakeRecord(commit_hash=f"c{i}", cell_index=i) for i in range(9)]
    settings.mapelites_sampler_seed = 20260804
    settings.mapelites_sampler_max_resample_attempts = 32
    monkeypatch.setattr(
        MapElitesSampler,
        "_persist_job",
        lambda *_args, **_kwargs: SimpleNamespace(id=uuid4()),
    )
    sampler = make_sampler(settings, records=records)

    first = sampler.schedule_job(ScheduleJobRequest(sampling_ordinal=76))
    assert first is not None and first.sampling_recipe_hash
    replacement = sampler.schedule_job(
        ScheduleJobRequest(
            sampling_ordinal=76,
            excluded_recipe_hashes={first.sampling_recipe_hash},
        )
    )

    assert replacement is not None
    assert replacement.sampling_recipe_hash != first.sampling_recipe_hash
    assert replacement.sampling_recipe_reused is False


def test_recipe_cooldown_marks_unavoidable_reuse(
    monkeypatch,
    settings: Settings,
) -> None:
    settings.mapelites_sampler_inspiration_count = 0
    settings.mapelites_sampler_max_resample_attempts = 3
    expected_hash = sampling_recipe_hash("only", ())
    captured: list[Any] = []

    def fake_persist(_self: MapElitesSampler, request: Any) -> SimpleNamespace:
        captured.append(request)
        return SimpleNamespace(id=uuid4())

    monkeypatch.setattr(MapElitesSampler, "_persist_job", fake_persist)
    sampler = MapElitesSampler(
        manager=FakeManager([FakeRecord(commit_hash="only", cell_index=0)]),
        settings=settings,
    )

    job = sampler.schedule_job(
        ScheduleJobRequest(
            sampling_ordinal=3,
            excluded_recipe_hashes={expected_hash},
        )
    )

    assert job is not None
    assert job.sampling_recipe_hash == expected_hash
    assert job.sampling_recipe_reused is True
    assert captured[0].selection_stats["recipe_resample_attempts"] == 3


def test_sampling_recipe_hash_ignores_inspiration_order() -> None:
    assert sampling_recipe_hash("base", ("a", "b")) == sampling_recipe_hash(
        "base",
        ("b", "a"),
    )
