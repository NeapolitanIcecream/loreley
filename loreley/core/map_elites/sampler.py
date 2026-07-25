"""Schedule evolution jobs based on MAP-Elites archives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from math import prod
import random
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Protocol
from uuid import UUID

import numpy as np
from loguru import logger
from rich.console import Console
from sqlalchemy.exc import SQLAlchemyError

from loreley.config import Settings, get_settings, resolve_default_island_id
from loreley.core.campaign_program import (
    CampaignProjectionInput,
    CampaignProgramSnapshot,
    apply_campaign_program_projection,
)
from loreley.db.base import session_scope
from loreley.db.models import EvolutionJob, JobStatus

console = Console()
log = logger.bind(module="map_elites.sampler")

__all__ = [
    "MapElitesSampler",
    "SamplingSnapshot",
    "ScheduleJobRequest",
    "ScheduledSamplerJob",
]


class SupportsMapElitesManager(Protocol):
    """Protocol describing the manager interface required by the sampler."""

    def get_cell_fronts(
        self,
        island_id: str | None = None,
    ) -> Mapping[int, Sequence[str]]:
        """Return occupied behavior cells mapped to Pareto members."""
        ...


@dataclass(slots=True, frozen=True)
class ScheduledSamplerJob:
    """Result descriptor for a job scheduled via the sampler."""

    job_id: UUID
    island_id: str
    base_commit_hash: str
    inspiration_commit_hashes: tuple[str, ...]
    migration_source_island_id: str | None = None
    migration_commit_hash: str | None = None


@dataclass(slots=True, frozen=True)
class SamplingSnapshot:
    """Precomputed archive view reused across a scheduling tick."""

    island_id: str
    cell_fronts: Mapping[int, tuple[str, ...]]
    items: tuple[tuple[int, str], ...]
    neighbor_cell_indices: np.ndarray | None
    neighbor_commits: tuple[str, ...]
    neighbor_coords: np.ndarray | None


@dataclass(slots=True, frozen=True)
class ScheduleJobRequest:
    """Inputs that may vary for one sampler scheduling attempt."""

    island_id: str | None = None
    priority: int | None = None
    sampling_snapshot: SamplingSnapshot | None = None
    cell_fronts: Mapping[int, Sequence[str]] | None = None
    excluded_base_commits: Collection[str] | None = None
    campaign_program: CampaignProgramSnapshot | None = None
    migration_source_island_id: str | None = None
    migration_commit_hash: str | None = None


@dataclass(slots=True, frozen=True)
class _PreparedSamplerJob:
    """Fully selected job fields ready for persistence."""

    island_id: str
    base_commit_hash: str
    inspiration_commit_hashes: tuple[str, ...]
    selection_stats: Mapping[str, Any]
    iteration_hint: str | None
    priority: int | None
    campaign_program: CampaignProgramSnapshot | None = None
    migration_source_island_id: str | None = None
    migration_commit_hash: str | None = None


@dataclass(slots=True)
class _InspirationSelectionState:
    selected: list[str]
    selected_commits: set[str]


@dataclass(slots=True, frozen=True)
class _NeighborRadiusConfig:
    radius: int
    min_radius: int
    max_radius: int


def _occupied_neighbor_arrays(
    items: Sequence[tuple[int, str]],
    *,
    grid_shape: tuple[int, ...],
) -> tuple[np.ndarray | None, tuple[str, ...], np.ndarray | None]:
    if not items:
        return None, tuple(), None

    cell_indices = np.asarray([idx for idx, _ in items], dtype=np.int64)
    commits = tuple(commit for _, commit in items)
    max_index = prod(grid_shape) - 1
    valid_mask = cell_indices >= 0
    if max_index >= 0:
        try:
            valid_mask &= cell_indices <= max_index
        except OverflowError:
            # If max_index exceeds int64 range, any int64 cell index is
            # necessarily <= max_index. Keep only non-negative values.
            pass
    if not np.all(valid_mask):
        mask_list = valid_mask.tolist()
        cell_indices = cell_indices[valid_mask]
        commits = tuple(commit for commit, keep in zip(commits, mask_list) if keep)
    if cell_indices.size <= 0:
        return None, tuple(), None

    try:
        coords = np.asarray(
            np.unravel_index(cell_indices, grid_shape),
            dtype=np.int64,
        ).T
    except ValueError:
        return None, tuple(), None
    return cell_indices, commits, coords


def _neighbor_candidate_positions(
    *,
    distances: np.ndarray,
    radius: int,
    first_radius: int,
) -> list[int]:
    # Preserve the existing semantics:
    # - The first iteration considers all occupied cells within the configured
    #   initial radius (<= radius).
    # - Subsequent iterations only consider the new shell (distance == radius)
    #   to avoid redundant rescans.
    if int(radius) == int(first_radius):
        candidates = np.flatnonzero((distances > 0) & (distances <= radius))
    else:
        candidates = np.flatnonzero(distances == radius)
    return candidates.tolist()


def _fallback_inspiration_candidates(
    *,
    base_cell_index: int,
    cell_fronts: Mapping[int, Sequence[str]],
    selected_commits: Collection[str],
) -> tuple[str, ...]:
    selected = set(selected_commits)
    return tuple(
        commit_hash
        for cell_index, commits in cell_fronts.items()
        if cell_index != base_cell_index
        for commit_hash in commits
        if commit_hash and commit_hash not in selected
    )


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
        if rng is None:
            seed = int(getattr(self.settings, "mapelites_sampler_seed", 0) or 0)
            if seed < 0:
                seed = 0
            rng = random.Random(seed)
        self._rng = rng
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
        self._default_island = resolve_default_island_id(self.settings)

    def schedule_job(
        self,
        request: ScheduleJobRequest | None = None,
    ) -> ScheduledSamplerJob | None:
        """Select base/inspiration commits and persist an EvolutionJob."""
        request = request or ScheduleJobRequest()
        effective_island = request.island_id or self._default_island
        snapshot = self._resolve_sampling_snapshot(request)
        if snapshot is None or not snapshot.cell_fronts:
            log.warning("Cannot schedule job; island {} archive is empty", effective_island)
            return None

        prepared = self._prepare_job(request=request, snapshot=snapshot)
        if prepared is None:
            return None
        job = self._persist_job(prepared)
        if not job:
            return None

        console.log(
            f"[bold green]Queued evolution job[/] island={prepared.island_id} "
            f"base={prepared.base_commit_hash} "
            f"inspirations={len(prepared.inspiration_commit_hashes)}",
        )

        return ScheduledSamplerJob(
            job_id=job.id,
            island_id=prepared.island_id,
            base_commit_hash=prepared.base_commit_hash,
            inspiration_commit_hashes=prepared.inspiration_commit_hashes,
            migration_source_island_id=prepared.migration_source_island_id,
            migration_commit_hash=prepared.migration_commit_hash,
        )

    def _resolve_sampling_snapshot(
        self,
        request: ScheduleJobRequest,
    ) -> SamplingSnapshot | None:
        if request.sampling_snapshot is not None:
            return request.sampling_snapshot
        effective_island = request.island_id or self._default_island
        if request.cell_fronts is not None:
            return self._build_sampling_snapshot(
                island_id=effective_island,
                cell_fronts=request.cell_fronts,
            )
        return self.get_sampling_snapshot(effective_island)

    def _prepare_job(
        self,
        *,
        request: ScheduleJobRequest,
        snapshot: SamplingSnapshot,
    ) -> _PreparedSamplerJob | None:
        base_selection = self._select_base_candidate(
            snapshot=snapshot,
            excluded_base_commits=request.excluded_base_commits,
        )
        if base_selection is None:
            log.info(
                "Cannot schedule job; island {} has no remaining unique base commits for this batch",
                snapshot.island_id,
            )
            return None

        base_cell_index, base_commit_hash = base_selection
        migration_source, migration_commit = self._migration_for_base(
            request=request,
            base_commit_hash=base_commit_hash,
        )
        inspirations, selection_stats = self._select_inspirations(
            base_cell_index=base_cell_index,
            base_commit_hash=base_commit_hash,
            cell_fronts=snapshot.cell_fronts,
            sampling_snapshot=snapshot,
        )
        inspirations = self._include_migration_inspiration(
            inspirations,
            base_commit_hash=base_commit_hash,
            migration_commit_hash=migration_commit,
        )
        return _PreparedSamplerJob(
            island_id=snapshot.island_id,
            base_commit_hash=base_commit_hash,
            inspiration_commit_hashes=inspirations,
            selection_stats=selection_stats,
            iteration_hint=self._iteration_hint(selection_stats),
            priority=request.priority,
            campaign_program=request.campaign_program,
            migration_source_island_id=migration_source,
            migration_commit_hash=migration_commit,
        )

    def _migration_for_base(
        self,
        *,
        request: ScheduleJobRequest,
        base_commit_hash: str,
    ) -> tuple[str | None, str | None]:
        if (
            self._inspiration_count <= 0
            or request.migration_commit_hash == base_commit_hash
        ):
            return None, None
        return request.migration_source_island_id, request.migration_commit_hash

    def _include_migration_inspiration(
        self,
        inspirations: tuple[str, ...],
        *,
        base_commit_hash: str,
        migration_commit_hash: str | None,
    ) -> tuple[str, ...]:
        if (
            not migration_commit_hash
            or migration_commit_hash == base_commit_hash
            or migration_commit_hash in inspirations
        ):
            return inspirations
        return (
            *inspirations[: max(0, self._inspiration_count - 1)],
            migration_commit_hash,
        )

    @staticmethod
    def _iteration_hint(selection_stats: Mapping[str, Any]) -> str | None:
        radius_used = selection_stats.get("radius_used")
        if radius_used is None:
            return None
        initial_radius = selection_stats.get("initial_radius")
        return f"MAP-Elites radius {radius_used} (initial {initial_radius})"

    def get_cell_fronts_snapshot(
        self,
        island_id: str | None = None,
    ) -> tuple[str, Mapping[int, tuple[str, ...]]] | None:
        """Return a stable occupied-cell snapshot for a scheduling tick."""

        snapshot = self.get_sampling_snapshot(island_id)
        if snapshot is None:
            return None
        return snapshot.island_id, dict(snapshot.cell_fronts)

    def get_sampling_snapshot(
        self,
        island_id: str | None = None,
    ) -> SamplingSnapshot | None:
        """Return a precomputed archive snapshot for batch scheduling."""

        effective_island = island_id or self._default_island
        cell_fronts = self.manager.get_cell_fronts(effective_island)
        if not cell_fronts:
            return None
        return self._build_sampling_snapshot(
            island_id=effective_island,
            cell_fronts=cell_fronts,
        )

    def _build_sampling_snapshot(
        self,
        *,
        island_id: str,
        cell_fronts: Mapping[int, Sequence[str]],
    ) -> SamplingSnapshot:
        items: list[tuple[int, str]] = []
        cleaned_fronts: dict[int, tuple[str, ...]] = {}

        for raw_index, raw_commits in cell_fronts.items():
            try:
                cell_index = int(raw_index)
            except (TypeError, ValueError):
                continue
            commits = tuple(
                dict.fromkeys(
                    str(commit or "").strip()
                    for commit in raw_commits
                    if str(commit or "").strip()
                )
            )
            if not commits:
                continue
            cleaned_fronts[cell_index] = commits
            items.extend((cell_index, commit) for commit in commits)

        items.sort()
        neighbor_cell_indices: np.ndarray | None = None
        neighbor_commits: tuple[str, ...] = tuple()
        neighbor_coords: np.ndarray | None = None

        if items:
            neighbor_cell_indices, neighbor_commits, neighbor_coords = _occupied_neighbor_arrays(
                items,
                grid_shape=self._grid_shape,
            )

        return SamplingSnapshot(
            island_id=island_id,
            cell_fronts=cleaned_fronts,
            items=tuple(items),
            neighbor_cell_indices=neighbor_cell_indices,
            neighbor_commits=neighbor_commits,
            neighbor_coords=neighbor_coords,
        )

    def _select_base_candidate(
        self,
        *,
        snapshot: SamplingSnapshot,
        excluded_base_commits: Collection[str] | None,
    ) -> tuple[int, str] | None:
        excluded = {str(commit).strip() for commit in (excluded_base_commits or ()) if str(commit).strip()}
        available_cells = [
            (
                cell_index,
                tuple(
                    commit
                    for commit in commits
                    if commit and commit not in excluded
                ),
            )
            for cell_index, commits in snapshot.cell_fronts.items()
        ]
        available_cells = [
            (cell_index, commits)
            for cell_index, commits in available_cells
            if commits
        ]
        if not available_cells:
            return None
        cell_index, commits = self._rng.choice(available_cells)
        return cell_index, self._rng.choice(commits)

    def _select_inspirations(
        self,
        *,
        base_cell_index: int,
        base_commit_hash: str,
        cell_fronts: Mapping[int, Sequence[str]],
        sampling_snapshot: SamplingSnapshot | None = None,
    ) -> tuple[tuple[str, ...], dict[str, Any]]:
        if self._inspiration_count <= 0:
            return tuple(), {
                "initial_radius": self._neighbor_radius,
                "radius_used": 0,
                "fallback_inspirations": 0,
            }

        selected: list[str] = []
        selected_commits = {base_commit_hash}
        radius_used = 0
        min_radius = max(0, self._neighbor_radius)
        max_radius = max(min_radius, self._max_neighbor_radius)
        radius = max(1, min_radius) if max_radius > 0 else 1

        if cell_fronts and max_radius > 0:
            radius_used = self._select_neighbor_inspirations(
                base_cell_index=base_cell_index,
                cell_fronts=cell_fronts,
                sampling_snapshot=sampling_snapshot,
                state=_InspirationSelectionState(
                    selected=selected,
                    selected_commits=selected_commits,
                ),
                radius_config=_NeighborRadiusConfig(
                    radius=radius,
                    min_radius=min_radius,
                    max_radius=max_radius,
                ),
            )

        fallback_inspirations = 0
        if len(selected) < self._inspiration_count and self._fallback_sample_size > 0:
            fallback = self._select_fallback_inspirations(
                base_cell_index=base_cell_index,
                cell_fronts=cell_fronts,
                selected_commits=selected_commits,
                needed=self._inspiration_count - len(selected),
            )
            fallback_inspirations = len(fallback)
            selected.extend(fallback)

        if len(selected) > self._inspiration_count:
            selected = selected[: self._inspiration_count]

        stats = {
            "initial_radius": self._neighbor_radius,
            "max_radius": self._max_neighbor_radius,
            "radius_used": radius_used,
            "fallback_inspirations": fallback_inspirations,
        }
        return tuple(selected), stats

    def _select_neighbor_inspirations(
        self,
        *,
        base_cell_index: int,
        cell_fronts: Mapping[int, Sequence[str]],
        sampling_snapshot: SamplingSnapshot | None,
        state: _InspirationSelectionState,
        radius_config: _NeighborRadiusConfig,
    ) -> int:
        try:
            base_coords = np.asarray(
                np.unravel_index(base_cell_index, self._grid_shape),
                dtype=np.int64,
            )
        except ValueError:
            return 0

        commits, coords = self._neighbor_candidates_for_selection(
            cell_fronts=cell_fronts,
            sampling_snapshot=sampling_snapshot,
        )
        if coords is None:
            return 0

        dist = np.max(np.abs(coords - base_coords), axis=1)
        radius_used = 0
        radius = radius_config.radius
        first_radius = max(1, radius_config.min_radius)
        while radius <= radius_config.max_radius and len(state.selected) < self._inspiration_count:
            positions = _neighbor_candidate_positions(
                distances=dist,
                radius=radius,
                first_radius=first_radius,
            )
            self._rng.shuffle(positions)
            added_this_radius = self._append_neighbor_inspirations(
                positions=positions,
                commits=commits,
                selected=state.selected,
                selected_commits=state.selected_commits,
            )
            if added_this_radius:
                radius_used = radius
            radius += 1
        return radius_used

    def _neighbor_candidates_for_selection(
        self,
        *,
        cell_fronts: Mapping[int, Sequence[str]],
        sampling_snapshot: SamplingSnapshot | None,
    ) -> tuple[Sequence[str], np.ndarray | None]:
        coords = sampling_snapshot.neighbor_coords if sampling_snapshot is not None else None
        if coords is not None:
            return sampling_snapshot.neighbor_commits, coords

        # We only care about occupied archive cells. Enumerating the full
        # Chebyshev ball in d dimensions is (2r+1)^d and quickly becomes
        # intractable; instead we compute Chebyshev distances to occupied
        # cells in a single vectorized pass (O(N * d)).
        _cell_indices, commits, coords = _occupied_neighbor_arrays(
            tuple(
                (cell_index, commit_hash)
                for cell_index, front in cell_fronts.items()
                for commit_hash in front
            ),
            grid_shape=self._grid_shape,
        )
        return commits, coords

    def _append_neighbor_inspirations(
        self,
        *,
        positions: Sequence[int],
        commits: Sequence[str],
        selected: list[str],
        selected_commits: set[str],
    ) -> bool:
        added = False
        for pos in positions:
            commit_hash = commits[pos]
            if not commit_hash or commit_hash in selected_commits:
                continue
            selected.append(commit_hash)
            selected_commits.add(commit_hash)
            added = True
            if len(selected) >= self._inspiration_count:
                break
        return added

    def _select_fallback_inspirations(
        self,
        *,
        base_cell_index: int,
        cell_fronts: Mapping[int, Sequence[str]],
        selected_commits: Collection[str],
        needed: int,
    ) -> tuple[str, ...]:
        candidates = list(
            _fallback_inspiration_candidates(
                base_cell_index=base_cell_index,
                cell_fronts=cell_fronts,
                selected_commits=selected_commits,
            )
        )
        if not candidates:
            return ()
        self._rng.shuffle(candidates)
        return tuple(candidates[: min(max(0, int(needed)), self._fallback_sample_size)])

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
        request: _PreparedSamplerJob | None = None,
        **options: Any,
    ) -> EvolutionJob | None:
        request = self._resolve_persist_request(request, options)
        job = self._build_evolution_job(request)
        if job is None:
            return None
        try:
            with session_scope() as session:
                session.add(job)
                session.flush()
        except SQLAlchemyError as exc:
            log.error(
                "Failed to persist evolution job for island {}: {}",
                request.island_id,
                exc,
            )
            return None
        return job

    @staticmethod
    def _resolve_persist_request(
        request: _PreparedSamplerJob | None,
        options: Mapping[str, Any],
    ) -> _PreparedSamplerJob:
        if request is not None:
            if options:
                raise TypeError("Pass either a prepared job or keyword options, not both.")
            return request
        return _PreparedSamplerJob(**dict(options))

    def _build_evolution_job(
        self,
        request: _PreparedSamplerJob,
    ) -> EvolutionJob | None:
        default_goal = (self.settings.worker_evolution_global_goal or "").strip() or None
        projection = apply_campaign_program_projection(
            CampaignProjectionInput(
                snapshot=request.campaign_program,
                goal=default_goal,
                constraints=(),
                acceptance_criteria=(),
                notes=(),
                default_goal=default_goal,
            )
        )
        if not projection.goal:
            log.error("Cannot schedule job; WORKER_EVOLUTION_GLOBAL_GOAL is empty.")
            return None
        selection_stats = request.selection_stats
        return EvolutionJob(
            status=JobStatus.PENDING,
            base_commit_hash=request.base_commit_hash,
            island_id=request.island_id,
            inspiration_commit_hashes=list(request.inspiration_commit_hashes),
            migration_source_island_id=request.migration_source_island_id,
            migration_commit_hash=request.migration_commit_hash,
            goal=projection.goal,
            constraints=projection.constraints,
            acceptance_criteria=projection.acceptance_criteria,
            notes=projection.notes,
            tags=[],
            iteration_hint=request.iteration_hint,
            sampling_strategy="grid_neighbors",
            sampling_initial_radius=int(selection_stats.get("initial_radius", 0)),
            sampling_radius_used=int(selection_stats.get("radius_used", 0)),
            sampling_fallback_inspirations=int(
                selection_stats.get("fallback_inspirations", 0)
            ),
            is_seed_job=False,
            job_kind="evolution",
            campaign_program_hash=(
                request.campaign_program.raw_sha256
                if request.campaign_program
                else None
            ),
            priority=(
                self._default_priority
                if request.priority is None
                else request.priority
            ),
            scheduled_at=datetime.now(timezone.utc),
        )
