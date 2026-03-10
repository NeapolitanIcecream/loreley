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
from loreley.db.base import session_scope
from loreley.db.models import EvolutionJob, JobStatus

console = Console()
log = logger.bind(module="map_elites.sampler")

__all__ = ["SamplingSnapshot", "ScheduledSamplerJob", "MapElitesSampler"]


class SupportsMapElitesManager(Protocol):
    """Protocol describing the manager interface required by the sampler."""

    def get_cell_commits(self, island_id: str | None = None) -> Mapping[int, str]:
        """Return occupied cell indices mapped to commit hashes."""
        ...


@dataclass(slots=True, frozen=True)
class ScheduledSamplerJob:
    """Result descriptor for a job scheduled via the sampler."""

    job_id: UUID
    island_id: str
    base_commit_hash: str
    inspiration_commit_hashes: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class SamplingSnapshot:
    """Precomputed archive view reused across a scheduling tick."""

    island_id: str
    cell_commits: Mapping[int, str]
    cell_objectives: Mapping[int, float]
    items: tuple[tuple[int, str], ...]
    neighbor_cell_indices: np.ndarray | None
    neighbor_commits: tuple[str, ...]
    neighbor_coords: np.ndarray | None


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
        *,
        island_id: str | None = None,
        priority: int | None = None,
        sampling_snapshot: SamplingSnapshot | None = None,
        cell_commits: Mapping[int, str] | None = None,
        cell_objectives: Mapping[int, float] | None = None,
        excluded_base_commits: Collection[str] | None = None,
    ) -> ScheduledSamplerJob | None:
        """Select base/inspiration commits and persist an EvolutionJob."""
        effective_island = island_id or self._default_island
        snapshot = sampling_snapshot
        if snapshot is None:
            if cell_commits is not None:
                snapshot = self._build_sampling_snapshot(
                    island_id=effective_island,
                    cell_commits=cell_commits,
                    cell_objectives=cell_objectives,
                )
            else:
                snapshot = self.get_sampling_snapshot(effective_island)
        if snapshot is None or not snapshot.cell_commits:
            log.warning("Cannot schedule job; island {} archive is empty", effective_island)
            return None
        effective_island = snapshot.island_id

        base_selection = self._select_base_candidate(
            snapshot=snapshot,
            excluded_base_commits=excluded_base_commits,
        )
        if base_selection is None:
            log.info(
                "Cannot schedule job; island {} has no remaining unique base commits for this batch",
                effective_island,
            )
            return None
        base_cell_index, base_commit_hash = base_selection

        inspirations, selection_stats = self._select_inspirations(
            base_cell_index=base_cell_index,
            base_commit_hash=base_commit_hash,
            cell_commits=snapshot.cell_commits,
            sampling_snapshot=snapshot,
        )
        iteration_hint = None
        radius_used = selection_stats.get("radius_used")
        initial_radius = selection_stats.get("initial_radius")
        if radius_used is not None:
            iteration_hint = f"MAP-Elites radius {radius_used} (initial {initial_radius})"

        job = self._persist_job(
            island_id=effective_island,
            base_commit_hash=base_commit_hash,
            inspiration_commit_hashes=inspirations,
            selection_stats=selection_stats,
            iteration_hint=iteration_hint,
            priority=priority,
        )
        if not job:
            return None

        console.log(
            f"[bold green]Queued evolution job[/] island={effective_island} "
            f"base={base_commit_hash} inspirations={len(inspirations)}",
        )

        return ScheduledSamplerJob(
            job_id=job.id,
            island_id=effective_island,
            base_commit_hash=base_commit_hash,
            inspiration_commit_hashes=tuple(inspirations),
        )

    def get_cell_commits_snapshot(
        self,
        island_id: str | None = None,
    ) -> tuple[str, Mapping[int, str]] | None:
        """Return a stable occupied-cell snapshot for a scheduling tick."""

        snapshot = self.get_sampling_snapshot(island_id)
        if snapshot is None:
            return None
        return snapshot.island_id, dict(snapshot.cell_commits)

    def get_sampling_snapshot(
        self,
        island_id: str | None = None,
    ) -> SamplingSnapshot | None:
        """Return a precomputed archive snapshot for batch scheduling."""

        effective_island = island_id or self._default_island
        cell_commits = self.manager.get_cell_commits(effective_island)
        if not cell_commits:
            return None
        cell_objectives = self._load_cell_objectives(effective_island)
        return self._build_sampling_snapshot(
            island_id=effective_island,
            cell_commits=cell_commits,
            cell_objectives=cell_objectives,
        )

    def _load_cell_objectives(self, island_id: str) -> Mapping[int, float]:
        getter = getattr(self.manager, "get_cell_objectives", None)
        if not callable(getter):
            return {}
        try:
            raw = getter(island_id)
        except TypeError:
            raw = getter()
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("Failed to load cell objectives for island {}: {}", island_id, exc)
            return {}

        objectives: dict[int, float] = {}
        if not isinstance(raw, Mapping):
            return objectives
        for cell_index, value in raw.items():
            try:
                idx = int(cell_index)
                objective = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(objective):
                continue
            objectives[idx] = objective
        return objectives

    def _build_sampling_snapshot(
        self,
        *,
        island_id: str,
        cell_commits: Mapping[int, str],
        cell_objectives: Mapping[int, float] | None = None,
    ) -> SamplingSnapshot:
        items: list[tuple[int, str]] = []
        cleaned_commits: dict[int, str] = {}
        objectives = cell_objectives or {}
        cleaned_objectives: dict[int, float] = {}

        for raw_index, raw_commit in cell_commits.items():
            commit_hash = str(raw_commit or "").strip()
            if not commit_hash:
                continue
            try:
                cell_index = int(raw_index)
            except (TypeError, ValueError):
                continue
            items.append((cell_index, commit_hash))
            cleaned_commits[cell_index] = commit_hash
            value = objectives.get(cell_index)
            if value is None:
                continue
            try:
                objective = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(objective):
                cleaned_objectives[cell_index] = objective

        items.sort(key=lambda item: item[0])
        neighbor_cell_indices: np.ndarray | None = None
        neighbor_commits: tuple[str, ...] = tuple()
        neighbor_coords: np.ndarray | None = None

        if items:
            cell_indices = np.asarray([idx for idx, _ in items], dtype=np.int64)
            commits = tuple(commit for _, commit in items)
            max_index = prod(self._grid_shape) - 1
            valid_mask = cell_indices >= 0
            if max_index >= 0:
                try:
                    valid_mask &= cell_indices <= max_index
                except OverflowError:
                    pass
            if not np.all(valid_mask):
                mask_list = valid_mask.tolist()
                cell_indices = cell_indices[valid_mask]
                commits = tuple(commit for commit, keep in zip(commits, mask_list) if keep)
            if cell_indices.size > 0:
                try:
                    neighbor_coords = np.asarray(
                        np.unravel_index(cell_indices, self._grid_shape),
                        dtype=np.int64,
                    ).T
                    neighbor_cell_indices = cell_indices
                    neighbor_commits = commits
                except ValueError:
                    neighbor_coords = None
                    neighbor_cell_indices = None

        return SamplingSnapshot(
            island_id=island_id,
            cell_commits=cleaned_commits,
            cell_objectives=cleaned_objectives,
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
        candidates = [item for item in snapshot.items if item[1] and item[1] not in excluded]
        if not candidates:
            return None

        weights = self._build_base_sampling_weights(
            items=candidates,
            cell_objectives=snapshot.cell_objectives,
        )
        if weights is None:
            return self._rng.choice(candidates)
        return self._weighted_choice(candidates, weights)

    def _build_base_sampling_weights(
        self,
        *,
        items: Sequence[tuple[int, str]],
        cell_objectives: Mapping[int, float],
    ) -> tuple[float, ...] | None:
        if not items or not cell_objectives:
            return None

        values = np.asarray(
            [float(cell_objectives.get(cell_index, float("nan"))) for cell_index, _ in items],
            dtype=np.float64,
        )
        finite_mask = np.isfinite(values)
        if not np.any(finite_mask):
            return None

        finite_values = values[finite_mask]
        min_value = float(np.min(finite_values))
        weights = np.ones(len(items), dtype=np.float64)
        weights[finite_mask] = (finite_values - min_value) + 1.0
        return tuple(float(weight) for weight in weights.tolist())

    def _weighted_choice(
        self,
        items: Sequence[tuple[int, str]],
        weights: Sequence[float],
    ) -> tuple[int, str]:
        total_weight = float(sum(weights))
        if total_weight <= 0.0:
            return self._rng.choice(list(items))

        random_fn = getattr(self._rng, "random", None)
        if not callable(random_fn):
            return self._rng.choice(list(items))

        threshold = float(random_fn()) * total_weight
        cumulative = 0.0
        last_item = items[-1]
        for item, raw_weight in zip(items, weights):
            weight = max(0.0, float(raw_weight))
            cumulative += weight
            if cumulative >= threshold:
                return item
            last_item = item
        return last_item

    def _select_inspirations(
        self,
        *,
        base_cell_index: int,
        base_commit_hash: str,
        cell_commits: Mapping[int, str],
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

        if cell_commits and max_radius > 0:
            try:
                base_coords = np.asarray(
                    np.unravel_index(base_cell_index, self._grid_shape),
                    dtype=np.int64,
                )
            except ValueError:
                base_coords = None

            if base_coords is not None:
                coords = sampling_snapshot.neighbor_coords if sampling_snapshot is not None else None
                commits: Sequence[str]
                if coords is not None:
                    commits = sampling_snapshot.neighbor_commits
                else:
                    # We only care about occupied archive cells. Enumerating the full
                    # Chebyshev ball in d dimensions is (2r+1)^d and quickly becomes
                    # intractable; instead we compute Chebyshev distances to occupied
                    # cells in a single vectorized pass (O(N * d)).
                    items = tuple(cell_commits.items())
                    cell_indices = np.asarray([idx for idx, _ in items], dtype=np.int64)
                    commits = [commit for _, commit in items]
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
                            commits = [commit for commit, keep in zip(commits, mask_list) if keep]
                    if cell_indices.size > 0:
                        try:
                            coords = np.asarray(
                                np.unravel_index(cell_indices, self._grid_shape),
                                dtype=np.int64,
                            ).T
                        except ValueError:
                            coords = None
                    else:
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
                            commit_hash = commits[pos]
                            if not commit_hash or commit_hash in selected_commits:
                                continue
                            selected.append(commit_hash)
                            selected_commits.add(commit_hash)
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
                commit_hash
                for cell_index, commit_hash in cell_commits.items()
                if cell_index != base_cell_index
                and commit_hash
                and commit_hash not in selected_commits
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
        base_commit_hash: str,
        inspiration_commit_hashes: Sequence[str],
        selection_stats: Mapping[str, Any],
        iteration_hint: str | None,
        priority: int | None,
    ) -> EvolutionJob | None:
        job_priority = self._default_priority if priority is None else priority
        goal = (self.settings.worker_evolution_global_goal or "").strip() or None
        if not goal:
            log.error("Cannot schedule job; WORKER_EVOLUTION_GLOBAL_GOAL is empty.")
            return None
        job = EvolutionJob(
            status=JobStatus.PENDING,
            base_commit_hash=base_commit_hash,
            island_id=island_id,
            inspiration_commit_hashes=list(inspiration_commit_hashes),
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
