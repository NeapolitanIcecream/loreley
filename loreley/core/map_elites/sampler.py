"""Schedule evolution jobs based on MAP-Elites archives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from math import prod
import random
from typing import Any, Mapping, Protocol, Sequence
from uuid import UUID

import numpy as np
from loguru import logger
from rich.console import Console
from sqlalchemy.exc import SQLAlchemyError

from loreley.config import Settings, get_settings
from loreley.db.base import session_scope
from loreley.db.models import EvolutionJob, JobStatus

console = Console()
log = logger.bind(module="map_elites.sampler")

__all__ = ["ScheduledSamplerJob", "MapElitesSampler"]


class SupportsMapElitesRecord(Protocol):
    """Protocol describing the record surface the sampler consumes.

    All attributes are treated as read-only from the sampler's perspective.
    """

    @property
    def commit_hash(self) -> str: ...

    @property
    def cell_index(self) -> int: ...

    @property
    def fitness(self) -> float: ...

    @property
    def measures(self) -> Sequence[float]: ...

    @property
    def solution(self) -> Sequence[float]: ...

    @property
    def timestamp(self) -> float: ...


class SupportsMapElitesManager(Protocol):
    """Protocol describing the manager interface required by the sampler."""

    def get_records(
        self,
        island_id: str | None = None,
    ) -> tuple[SupportsMapElitesRecord, ...]:
        """Return archive records for an island."""
        ...


@dataclass(slots=True, frozen=True)
class ScheduledSamplerJob:
    """Result descriptor for a job scheduled via the sampler."""

    job_id: UUID
    island_id: str
    base: SupportsMapElitesRecord
    inspirations: tuple[SupportsMapElitesRecord, ...]

    @property
    def base_commit_hash(self) -> str:
        return self.base.commit_hash

    @property
    def inspiration_commit_hashes(self) -> tuple[str, ...]:
        return tuple(record.commit_hash for record in self.inspirations)


class MapElitesSampler:
    """Translate MAP-Elites archives into EvolutionJob rows."""

    def __init__(
        self,
        manager: SupportsMapElitesManager,
        *,
        settings: Settings | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.manager = manager
        self.settings = settings or get_settings()
        self._rng = rng or random.Random()
        self._target_dims = max(1, self.settings.mapelites_dimensionality_target_dims)
        self._cells_per_dim = max(2, self.settings.mapelites_archive_cells_per_dim)
        self._grid_shape = tuple(self._cells_per_dim for _ in range(self._target_dims))
        self._inspiration_count = max(0, self.settings.mapelites_sampler_inspiration_count)
        self._neighbor_radius = max(0, self.settings.mapelites_sampler_neighbor_radius)
        self._max_neighbor_radius = max(
            self._neighbor_radius,
            self.settings.mapelites_sampler_neighbor_max_radius,
        )
        self._fallback_sample_size = max(
            0,
            self.settings.mapelites_sampler_fallback_sample_size,
        )
        self._default_priority = self.settings.mapelites_sampler_default_priority
        self._default_island = self.settings.mapelites_default_island_id or "default"

    def schedule_job(
        self,
        *,
        island_id: str | None = None,
        priority: int | None = None,
        experiment_id: UUID | str | None = None,
    ) -> ScheduledSamplerJob | None:
        """Select base/inspiration commits and persist an EvolutionJob."""
        effective_island = island_id or self._default_island
        records = list(self.manager.get_records(effective_island))
        if not records:
            log.warning("Cannot schedule job; island {} archive is empty", effective_island)
            return None

        base_record = self._rng.choice(records)
        records_by_cell = {record.cell_index: record for record in records}
        inspirations, selection_stats = self._select_inspirations(base_record, records_by_cell)
        iteration_hint = None
        radius_used = selection_stats.get("radius_used")
        initial_radius = selection_stats.get("initial_radius")
        if radius_used is not None:
            iteration_hint = f"MAP-Elites radius {radius_used} (initial {initial_radius})"

        exp_id: UUID | None = None
        if experiment_id is not None:
            if isinstance(experiment_id, UUID):
                exp_id = experiment_id
            else:
                exp_id = UUID(str(experiment_id))

        job = self._persist_job(
            island_id=effective_island,
            base=base_record,
            inspirations=inspirations,
            selection_stats=selection_stats,
            iteration_hint=iteration_hint,
            priority=priority,
            experiment_id=exp_id,
        )
        if not job:
            return None

        console.log(
            f"[bold green]Queued evolution job[/] island={effective_island} "
            f"base={base_record.commit_hash} inspirations={len(inspirations)}",
        )

        return ScheduledSamplerJob(
            job_id=job.id,
            island_id=effective_island,
            base=base_record,
            inspirations=inspirations,
        )

    def _select_inspirations(
        self,
        base: SupportsMapElitesRecord,
        records_by_cell: Mapping[int, SupportsMapElitesRecord],
    ) -> tuple[tuple[SupportsMapElitesRecord, ...], dict[str, Any]]:
        if self._inspiration_count <= 0:
            return tuple(), {
                "initial_radius": self._neighbor_radius,
                "radius_used": 0,
                "fallback_inspirations": 0,
            }

        selected: list[SupportsMapElitesRecord] = []
        selected_commits = {base.commit_hash}
        radius_used = 0
        min_radius = max(0, self._neighbor_radius)
        max_radius = max(min_radius, self._max_neighbor_radius)
        radius = max(1, min_radius) if max_radius > 0 else 1

        if records_by_cell and max_radius > 0:
            try:
                base_coords = np.asarray(
                    np.unravel_index(base.cell_index, self._grid_shape),
                    dtype=np.int64,
                )
            except ValueError:
                base_coords = None

            if base_coords is not None:
                # We only care about occupied archive cells. Enumerating the full
                # Chebyshev ball in d dimensions is (2r+1)^d and quickly becomes
                # intractable; instead we compute Chebyshev distances to occupied
                # cells in a single vectorized pass (O(N * d)).
                items = tuple(records_by_cell.items())
                cell_indices = np.asarray([idx for idx, _ in items], dtype=np.int64)
                records = [record for _, record in items]
                max_index = prod(self._grid_shape) - 1
                if max_index >= 0:
                    valid_mask = cell_indices >= 0
                    try:
                        valid_mask &= cell_indices <= max_index
                    except OverflowError:
                        # If max_index exceeds int64 range, any int64 cell index is
                        # necessarily <= max_index. Keep only non-negative values.
                        pass
                    if not np.all(valid_mask):
                        mask_list = valid_mask.tolist()
                        cell_indices = cell_indices[valid_mask]
                        records = [rec for rec, keep in zip(records, mask_list) if keep]
                if cell_indices.size > 0:
                    try:
                        coords = np.asarray(
                            np.unravel_index(cell_indices, self._grid_shape),
                            dtype=np.int64,
                        ).T
                    except ValueError:
                        coords = None

                    if coords is not None:
                        dist = np.max(np.abs(coords - base_coords), axis=1)

                        while radius <= max_radius and len(selected) < self._inspiration_count:
                            # Preserve the existing semantics:
                            # - The first iteration considers all occupied cells within
                            #   the configured initial radius (<= radius).
                            # - Subsequent iterations only consider the new shell
                            #   (distance == radius) to avoid redundant rescans.
                            if radius == max(1, min_radius):
                                candidate_positions = np.flatnonzero((dist > 0) & (dist <= radius))
                            else:
                                candidate_positions = np.flatnonzero(dist == radius)

                            positions = candidate_positions.tolist()
                            self._rng.shuffle(positions)
                            added_this_radius = False
                            for pos in positions:
                                record = records[pos]
                                if record.commit_hash in selected_commits:
                                    continue
                                selected.append(record)
                                selected_commits.add(record.commit_hash)
                                added_this_radius = True
                                if len(selected) >= self._inspiration_count:
                                    break
                            if added_this_radius:
                                radius_used = radius
                            radius += 1

        fallback_inspirations = 0
        if len(selected) < self._inspiration_count and self._fallback_sample_size > 0:
            needed = self._inspiration_count - len(selected)
            fallback_candidates = [
                record
                for record in records_by_cell.values()
                if record.commit_hash not in selected_commits
            ]
            if fallback_candidates:
                self._rng.shuffle(fallback_candidates)
                fallback_slice = fallback_candidates[: min(needed, self._fallback_sample_size)]
                fallback_inspirations = len(fallback_slice)
                selected.extend(fallback_slice)

        if len(selected) > self._inspiration_count:
            selected = selected[: self._inspiration_count]

        stats = {
            "initial_radius": self._neighbor_radius,
            "max_radius": self._max_neighbor_radius,
            "radius_used": radius_used,
            "fallback_inspirations": fallback_inspirations,
        }
        return tuple(selected), stats

    def _neighbor_indices(self, center_index: int, radius: int) -> list[int]:
        if radius <= 0:
            return []
        try:
            coordinates = tuple(
                int(value) for value in np.unravel_index(center_index, self._grid_shape)
            )
        except ValueError:
            return []

        ranges = [
            range(max(0, coord - radius), min(dim, coord + radius + 1))
            for coord, dim in zip(coordinates, self._grid_shape)
        ]

        neighbors: list[int] = []
        for candidate in product(*ranges):
            if candidate == coordinates:
                continue
            if max(abs(c - base) for c, base in zip(candidate, coordinates)) > radius:
                continue
            neighbor_index = int(np.ravel_multi_index(candidate, self._grid_shape))
            neighbors.append(neighbor_index)

        self._rng.shuffle(neighbors)
        return neighbors

    def _persist_job(
        self,
        *,
        island_id: str,
        base: SupportsMapElitesRecord,
        inspirations: Sequence[SupportsMapElitesRecord],
        selection_stats: Mapping[str, Any],
        iteration_hint: str | None,
        priority: int | None,
        experiment_id: UUID | None,
    ) -> EvolutionJob | None:
        job_priority = self._default_priority if priority is None else priority
        goal = (self.settings.worker_evolution_global_goal or "").strip() or None
        if not goal:
            log.error("Cannot schedule job; WORKER_EVOLUTION_GLOBAL_GOAL is empty.")
            return None
        job = EvolutionJob(
            status=JobStatus.PENDING,
            base_commit_hash=base.commit_hash,
            island_id=island_id,
            experiment_id=experiment_id,
            inspiration_commit_hashes=[record.commit_hash for record in inspirations],
            goal=goal,
            constraints=[],
            acceptance_criteria=[],
            notes=[],
            tags=[],
            iteration_hint=iteration_hint,
            sampling_strategy="grid_neighbors",
            sampling_initial_radius=int(selection_stats.get("initial_radius", 0)),
            sampling_radius_used=int(selection_stats.get("radius_used", 0)),
            sampling_fallback_inspirations=int(selection_stats.get("fallback_inspirations", 0)),
            is_seed_job=False,
            priority=job_priority,
            scheduled_at=datetime.now(timezone.utc),
        )
        try:
            with session_scope() as session:
                session.add(job)
                session.flush()
        except SQLAlchemyError as exc:
            log.error("Failed to persist evolution job for island {}: {}", island_id, exc)
            return None
        return job

