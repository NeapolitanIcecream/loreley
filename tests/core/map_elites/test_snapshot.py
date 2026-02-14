from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from loreley.core.map_elites.dimension_reduction import PCAProjection, PcaHistoryEntry
from loreley.core.map_elites.snapshot import (
    apply_snapshot,
    serialize_projection,
)


class DummyArchive:
    """Minimal archive stub used to test snapshot serialisation logic."""

    def __init__(
        self,
        data: Mapping[str, Any] | None = None,
        *,
        add_status: np.ndarray | None = None,
        add_value: np.ndarray | None = None,
    ) -> None:
        self._data = data or {}
        self.empty = not bool(self._data)
        self.solution_dim = 2
        self.dims = (4, 4)
        self._add_status = add_status
        self._add_value = add_value
        self.add_calls: list[dict[str, Any]] = []

    def data(self) -> Mapping[str, Any]:
        return self._data

    def add(
        self,
        solution: np.ndarray,
        objective: np.ndarray,
        measures: np.ndarray,
        *,
        commit_hash: np.ndarray,
        timestamp: np.ndarray,
    ) -> Mapping[str, np.ndarray]:
        # Capture arguments so tests can assert that entries were restored.
        self.add_calls.append(
            {
                "solution": solution,
                "objective": objective,
                "measures": measures,
                "commit_hash": commit_hash,
                "timestamp": timestamp,
            }
        )
        batch_size = int(np.asarray(objective).reshape(-1).shape[0])
        status = (
            np.asarray(self._add_status, dtype=np.int64).reshape(-1)
            if self._add_status is not None
            else np.ones(batch_size, dtype=np.int64)
        )
        value = (
            np.asarray(self._add_value, dtype=np.float64).reshape(-1)
            if self._add_value is not None
            else np.asarray(objective, dtype=np.float64).reshape(-1)
        )
        if status.shape[0] != batch_size or value.shape[0] != batch_size:
            raise AssertionError("DummyArchive add_info shape mismatch")
        return {
            "status": status,
            "value": value,
        }

    def index_of(self, measures: np.ndarray) -> np.ndarray:
        batch = np.asarray(measures, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        return np.zeros(batch.shape[0], dtype=np.int64)


@dataclass
class DummyState:
    """Lightweight stand-in for `IslandState` used in snapshot tests."""

    archive: DummyArchive
    lower_bounds: np.ndarray = field(default_factory=lambda: np.array([-1.0, -1.0]))
    upper_bounds: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0]))
    history: tuple[PcaHistoryEntry, ...] = field(default_factory=tuple)
    projection: PCAProjection | None = None
    commit_to_index: dict[str, int] = field(default_factory=dict)
    index_to_commit: dict[int, str] = field(default_factory=dict)


def _make_history_entry() -> PcaHistoryEntry:
    return PcaHistoryEntry(
        commit_hash="c1",
        vector=(1.0, 2.0),
        embedding_model="code",
    )


def _make_projection() -> PCAProjection:
    return PCAProjection(
        feature_count=2,
        components=((1.0, 0.0), (0.0, 1.0)),
        mean=(0.0, 0.0),
        explained_variance=(1.0, 1.0),
        explained_variance_ratio=(1.0, 0.0),
        sample_count=10,
        epoch=0,
        fitted_at=123.0,
        whiten=True,
        rotation=None,
    )


def test_build_and_apply_snapshot_round_trip_basic() -> None:
    # Prepare an in-memory state with a single archive entry.
    archive_data = {
        "index": [0],
        "objective": [1.23],
        "measures": [[0.1, 0.2]],
        "solution": [[0.1, 0.2]],
        "commit_hash": ["c1"],
        "timestamp": [42.0],
    }
    entry = _make_history_entry()
    original_state = DummyState(
        archive=DummyArchive(archive_data),
        lower_bounds=np.array([-2.0, -2.0]),
        upper_bounds=np.array([2.0, 2.0]),
        history=(entry,),
        projection=_make_projection(),
    )

    snapshot = {
        "island_id": "main",
        "lower_bounds": original_state.lower_bounds.tolist(),
        "upper_bounds": original_state.upper_bounds.tolist(),
        "history": [
            {
                "commit_hash": entry.commit_hash,
                "vector": [float(v) for v in entry.vector],
                "embedding_model": str(entry.embedding_model),
            }
        ],
        "projection": serialize_projection(original_state.projection),
        "archive": [
            {
                "index": 0,
                "objective": 1.23,
                "measures": [0.1, 0.2],
                "solution": [0.1, 0.2],
                "commit_hash": "c1",
                "timestamp": 42.0,
            }
        ],
    }

    # Apply snapshot onto a fresh, empty state.
    restored_state = DummyState(archive=DummyArchive())
    commit_to_island: dict[str, str] = {}

    apply_snapshot(
        state=restored_state,
        snapshot=snapshot,
        island_id="main",
        commit_to_island=commit_to_island,
    )

    # Bounds, history and projection are restored.
    assert np.allclose(restored_state.lower_bounds, original_state.lower_bounds)
    assert np.allclose(restored_state.upper_bounds, original_state.upper_bounds)
    assert len(restored_state.history) == 1
    assert restored_state.history[0].commit_hash == entry.commit_hash
    assert restored_state.projection is not None
    assert restored_state.projection.feature_count == original_state.projection.feature_count  # type: ignore[union-attr]

    # Archive entries and commit mappings are restored.
    assert restored_state.index_to_commit == {0: "c1"}
    assert restored_state.commit_to_index == {"c1": 0}
    assert commit_to_island == {"c1": "main"}
    assert len(restored_state.archive.add_calls) == 1


def test_apply_snapshot_only_maps_entries_with_positive_add_status() -> None:
    snapshot = {
        "island_id": "main",
        "lower_bounds": [-1.0, -1.0],
        "upper_bounds": [1.0, 1.0],
        "history": [],
        "projection": None,
        "archive": [
            {
                "index": 0,
                "objective": 1.0,
                "measures": [0.1, 0.2],
                "solution": [0.1, 0.2],
                "commit_hash": "c1",
                "timestamp": 10.0,
            },
            {
                "index": 1,
                "objective": 0.5,
                "measures": [0.3, 0.4],
                "solution": [0.3, 0.4],
                "commit_hash": "c2",
                "timestamp": 11.0,
            },
        ],
    }

    restored_state = DummyState(
        archive=DummyArchive(
            add_status=np.asarray([2, 0], dtype=np.int64),
            add_value=np.asarray([1.0, -0.5], dtype=np.float64),
        )
    )
    commit_to_island: dict[str, str] = {}

    apply_snapshot(
        state=restored_state,
        snapshot=snapshot,
        island_id="main",
        commit_to_island=commit_to_island,
    )

    assert restored_state.index_to_commit == {0: "c1"}
    assert restored_state.commit_to_index == {"c1": 0}
    assert commit_to_island == {"c1": "main"}
    assert len(restored_state.archive.add_calls) == 1


