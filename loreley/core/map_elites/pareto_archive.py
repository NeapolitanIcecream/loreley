"""A bounded Pareto front in every fixed behavior-grid cell."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "ParetoAddOutcome",
    "ParetoArchiveStats",
    "ParetoCandidate",
    "ParetoGridArchive",
]

Vector = tuple[float, ...]


def _parse_dimensions(values: Sequence[int]) -> tuple[int, ...]:
    dimensions = tuple(int(value) for value in values)
    if not dimensions or any(value < 1 for value in dimensions):
        raise ValueError("Pareto archive dimensions must all be positive.")
    return dimensions


def _parse_ranges(
    values: Sequence[tuple[float, float]],
    *,
    dimension_count: int,
) -> tuple[tuple[float, float], ...]:
    ranges = tuple((float(low), float(high)) for low, high in values)
    if len(ranges) != dimension_count:
        raise ValueError("Pareto archive dimensions and ranges must have equal length.")
    if any(
        not math.isfinite(low) or not math.isfinite(high) or low >= high
        for low, high in ranges
    ):
        raise ValueError("Pareto archive ranges must be finite and increasing.")
    return ranges


def _parse_positive_int(value: int, *, message: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise ValueError(message)
    return parsed


def _parse_epsilon(value: float) -> float:
    epsilon = float(value)
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("Pareto epsilon must be finite and non-negative.")
    return epsilon


@dataclass(slots=True, frozen=True)
class ParetoCandidate:
    """One evaluated candidate ready for behavior-cell admission."""

    commit_hash: str
    objective_values: Vector
    objective_scores: Vector
    measures: Vector
    timestamp: float

    def __post_init__(self) -> None:
        commit_hash = str(self.commit_hash or "").strip()
        if not commit_hash:
            raise ValueError("Pareto candidate commit hash cannot be empty.")
        object.__setattr__(self, "commit_hash", commit_hash)
        object.__setattr__(
            self,
            "objective_values",
            tuple(float(value) for value in self.objective_values),
        )
        object.__setattr__(
            self,
            "objective_scores",
            tuple(float(value) for value in self.objective_scores),
        )
        object.__setattr__(
            self,
            "measures",
            tuple(float(value) for value in self.measures),
        )
        object.__setattr__(self, "timestamp", float(self.timestamp))
        numeric_values = (
            *self.objective_values,
            *self.objective_scores,
            *self.measures,
            self.timestamp,
        )
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("Pareto candidate values must all be finite.")


@dataclass(slots=True, frozen=True)
class ParetoAddOutcome:
    """Admission outcome for one candidate."""

    cell_index: int
    retained: bool
    removed_commit_hashes: tuple[str, ...] = ()

    @property
    def status(self) -> int:
        return 1 if self.retained else 0


@dataclass(slots=True, frozen=True)
class ParetoArchiveStats:
    """Small stats surface with unambiguous multi-objective meanings."""

    num_elites: int
    num_occupied: int
    coverage: float


class ParetoGridArchive:
    """Fixed grid whose occupied cells contain bounded Pareto fronts."""

    def __init__(
        self,
        *,
        dims: Sequence[int],
        ranges: Sequence[tuple[float, float]],
        objective_count: int,
        max_front_size: int,
        epsilon: float,
    ) -> None:
        parsed_dims = _parse_dimensions(dims)
        self.dims = parsed_dims
        self.ranges = _parse_ranges(ranges, dimension_count=len(parsed_dims))
        self.objective_count = _parse_positive_int(
            objective_count,
            message="Pareto archive requires at least one objective.",
        )
        self.max_front_size = _parse_positive_int(
            max_front_size,
            message="Pareto front capacity must be positive.",
        )
        self.epsilon = _parse_epsilon(epsilon)
        self._fronts: dict[int, tuple[ParetoCandidate, ...]] = {}
        self._commit_to_cell: dict[str, int] = {}

    @property
    def empty(self) -> bool:
        return not self._commit_to_cell

    @property
    def stats(self) -> ParetoArchiveStats:
        total_cells = math.prod(self.dims)
        occupied = len(self._fronts)
        return ParetoArchiveStats(
            num_elites=len(self._commit_to_cell),
            num_occupied=occupied,
            coverage=(occupied / total_cells) if total_cells else 0.0,
        )

    def clear(self) -> None:
        self._fronts.clear()
        self._commit_to_cell.clear()

    def index_of(self, measures: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        matrix = np.asarray(measures, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.dims):
            raise ValueError(
                "Behavior measures shape mismatch "
                f"(expected=(*, {len(self.dims)}) got={matrix.shape})."
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Behavior measures must be finite.")

        coordinates: list[np.ndarray] = []
        for dimension, (low, high), values in zip(self.dims, self.ranges, matrix.T):
            if np.any(values < low) or np.any(values > high):
                raise ValueError(
                    f"Behavior measures must remain inside [{low}, {high}]."
                )
            scaled = (values - low) / (high - low)
            coordinate = np.floor(scaled * dimension).astype(np.int64)
            coordinates.append(np.clip(coordinate, 0, dimension - 1))
        return np.ravel_multi_index(tuple(coordinates), self.dims).astype(np.int64)

    def front(self, cell_index: int) -> tuple[ParetoCandidate, ...]:
        return self._fronts.get(int(cell_index), ())

    def records(self) -> tuple[ParetoCandidate, ...]:
        return tuple(
            candidate
            for cell_index in sorted(self._fronts)
            for candidate in self._fronts[cell_index]
        )

    def add(self, candidate: ParetoCandidate) -> ParetoAddOutcome:
        return self.add_many((candidate,))[0]

    def add_many(
        self,
        candidates: Sequence[ParetoCandidate],
    ) -> tuple[ParetoAddOutcome, ...]:
        pending = tuple(candidates)
        if not pending:
            return ()
        commits = self._validate_batch(pending)
        cell_indices = self._candidate_cell_indices(pending)
        before_commits = set(self._commit_to_cell)
        affected_cells = set(cell_indices)
        self._detach_commits(commits, affected_cells=affected_cells)
        additions_by_cell = self._group_by_cell(cell_indices, pending)
        self._merge_affected_cells(affected_cells, additions_by_cell=additions_by_cell)
        self._rebuild_commit_index()
        return self._build_add_outcomes(
            pending,
            cell_indices=cell_indices,
            before_commits=before_commits,
        )

    def sample(
        self,
        count: int,
        *,
        rng: random.Random | Any | None = None,
    ) -> tuple[ParetoCandidate, ...]:
        if self.empty or count <= 0:
            return ()
        chooser = rng or random
        occupied_cells = tuple(sorted(self._fronts))
        return tuple(
            chooser.choice(self._fronts[chooser.choice(occupied_cells)])
            for _ in range(int(count))
        )

    def data(self) -> Mapping[str, np.ndarray]:
        """Return column arrays for bulk serialization and API adapters."""

        records = self.records()
        if not records:
            return {}
        indices = [self._commit_to_cell[record.commit_hash] for record in records]
        return {
            "index": np.asarray(indices, dtype=np.int64),
            "objective_values": np.asarray(
                [record.objective_values for record in records],
                dtype=np.float64,
            ),
            "objective_scores": np.asarray(
                [record.objective_scores for record in records],
                dtype=np.float64,
            ),
            "measures": np.asarray(
                [record.measures for record in records], dtype=np.float64
            ),
            "commit_hash": np.asarray(
                [record.commit_hash for record in records],
                dtype=object,
            ),
            "timestamp": np.asarray(
                [record.timestamp for record in records], dtype=np.float64
            ),
        }

    def _validate_candidate(self, candidate: ParetoCandidate) -> None:
        if len(candidate.objective_values) != self.objective_count:
            raise ValueError(
                "Raw objective vector length mismatch "
                f"(expected={self.objective_count} got={len(candidate.objective_values)})."
            )
        if len(candidate.objective_scores) != self.objective_count:
            raise ValueError(
                "Dominance score vector length mismatch "
                f"(expected={self.objective_count} got={len(candidate.objective_scores)})."
            )
        if len(candidate.measures) != len(self.dims):
            raise ValueError(
                "Behavior measures length mismatch "
                f"(expected={len(self.dims)} got={len(candidate.measures)})."
            )

    def _validate_batch(
        self,
        candidates: Sequence[ParetoCandidate],
    ) -> tuple[str, ...]:
        commits = tuple(candidate.commit_hash for candidate in candidates)
        if len(set(commits)) != len(commits):
            raise ValueError("A Pareto archive batch cannot contain duplicate commits.")
        for candidate in candidates:
            self._validate_candidate(candidate)
        return commits

    def _candidate_cell_indices(
        self,
        candidates: Sequence[ParetoCandidate],
    ) -> tuple[int, ...]:
        measures = np.asarray(
            [candidate.measures for candidate in candidates],
            dtype=np.float64,
        )
        return tuple(int(value) for value in self.index_of(measures))

    def _detach_commits(
        self,
        commit_hashes: Sequence[str],
        *,
        affected_cells: set[int],
    ) -> None:
        for commit_hash in commit_hashes:
            old_cell = self._commit_to_cell.pop(commit_hash, None)
            if old_cell is None:
                continue
            affected_cells.add(old_cell)
            remaining = tuple(
                candidate
                for candidate in self._fronts.get(old_cell, ())
                if candidate.commit_hash != commit_hash
            )
            self._store_front(old_cell, remaining)

    @staticmethod
    def _group_by_cell(
        cell_indices: Sequence[int],
        candidates: Sequence[ParetoCandidate],
    ) -> dict[int, list[ParetoCandidate]]:
        grouped: dict[int, list[ParetoCandidate]] = {}
        for cell_index, candidate in zip(cell_indices, candidates):
            grouped.setdefault(cell_index, []).append(candidate)
        return grouped

    def _merge_affected_cells(
        self,
        cell_indices: set[int],
        *,
        additions_by_cell: Mapping[int, Sequence[ParetoCandidate]],
    ) -> None:
        for cell_index in sorted(cell_indices):
            combined = (
                *self._fronts.get(cell_index, ()),
                *additions_by_cell.get(cell_index, ()),
            )
            self._store_front(cell_index, self._select_front(combined))

    def _store_front(
        self,
        cell_index: int,
        front: Sequence[ParetoCandidate],
    ) -> None:
        if front:
            self._fronts[cell_index] = tuple(front)
        else:
            self._fronts.pop(cell_index, None)

    def _build_add_outcomes(
        self,
        candidates: Sequence[ParetoCandidate],
        *,
        cell_indices: Sequence[int],
        before_commits: set[str],
    ) -> tuple[ParetoAddOutcome, ...]:
        after_commits = set(self._commit_to_cell)
        removed = tuple(sorted(before_commits.difference(after_commits)))
        removed_for_candidate = removed if len(candidates) == 1 else ()
        return tuple(
            ParetoAddOutcome(
                cell_index=cell_index,
                retained=candidate.commit_hash in after_commits,
                removed_commit_hashes=removed_for_candidate,
            )
            for cell_index, candidate in zip(cell_indices, candidates)
        )

    def _select_front(
        self,
        candidates: Sequence[ParetoCandidate],
    ) -> tuple[ParetoCandidate, ...]:
        if not candidates:
            return ()

        representatives: list[ParetoCandidate] = []
        for candidate in sorted(candidates, key=lambda item: item.commit_hash):
            if any(
                self._equivalent(candidate.objective_scores, retained.objective_scores)
                for retained in representatives
            ):
                continue
            representatives.append(candidate)

        nondominated = [
            candidate
            for candidate in representatives
            if not any(
                other.commit_hash != candidate.commit_hash
                and self._dominates(other.objective_scores, candidate.objective_scores)
                for other in representatives
            )
        ]
        if len(nondominated) > self.max_front_size:
            crowding = self._crowding_distances(nondominated)
            nondominated = sorted(
                nondominated,
                key=lambda item: (
                    -crowding[item.commit_hash],
                    item.commit_hash,
                ),
            )[: self.max_front_size]
        return tuple(sorted(nondominated, key=lambda item: item.commit_hash))

    def _equivalent(self, first: Vector, second: Vector) -> bool:
        return all(
            abs(first_value - second_value) <= self.epsilon
            for first_value, second_value in zip(first, second)
        )

    def _dominates(self, first: Vector, second: Vector) -> bool:
        no_worse = all(
            first_value >= second_value - self.epsilon
            for first_value, second_value in zip(first, second)
        )
        strictly_better = any(
            first_value > second_value + self.epsilon
            for first_value, second_value in zip(first, second)
        )
        return no_worse and strictly_better

    def _crowding_distances(
        self,
        front: Sequence[ParetoCandidate],
    ) -> dict[str, float]:
        distances = {candidate.commit_hash: 0.0 for candidate in front}
        if len(front) <= 2:
            return {commit_hash: math.inf for commit_hash in distances}

        for objective_index in range(self.objective_count):
            ordered = sorted(
                front,
                key=lambda candidate: (
                    candidate.objective_scores[objective_index],
                    candidate.commit_hash,
                ),
            )
            low = ordered[0].objective_scores[objective_index]
            high = ordered[-1].objective_scores[objective_index]
            distances[ordered[0].commit_hash] = math.inf
            distances[ordered[-1].commit_hash] = math.inf
            span = high - low
            if span <= self.epsilon:
                continue
            for index in range(1, len(ordered) - 1):
                commit_hash = ordered[index].commit_hash
                if math.isinf(distances[commit_hash]):
                    continue
                previous_value = ordered[index - 1].objective_scores[objective_index]
                next_value = ordered[index + 1].objective_scores[objective_index]
                distances[commit_hash] += (next_value - previous_value) / span
        return distances

    def _rebuild_commit_index(self) -> None:
        self._commit_to_cell = {
            candidate.commit_hash: cell_index
            for cell_index, front in self._fronts.items()
            for candidate in front
        }
