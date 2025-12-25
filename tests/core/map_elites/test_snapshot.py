from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import uuid

import numpy as np

from loreley.core.map_elites.dimension_reduction import PCAProjection, PenultimateEmbedding
from loreley.core.map_elites.snapshot import (
    DatabaseSnapshotBackend,
    NullSnapshotBackend,
    apply_snapshot,
    build_snapshot,
    build_snapshot_backend,
)


class DummyArchive:
    """Minimal archive stub used to test snapshot serialisation logic."""

    def __init__(self, data: Mapping[str, Any] | None = None) -> None:
        self._data = data or {}
        self.empty = not bool(self._data)
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
    ) -> None:
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


@dataclass
class DummyState:
    """Lightweight stand-in for `IslandState` used in snapshot tests."""

    archive: DummyArchive
    lower_bounds: np.ndarray = field(default_factory=lambda: np.array([-1.0, -1.0]))
    upper_bounds: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0]))
    history: tuple[PenultimateEmbedding, ...] = field(default_factory=tuple)
    projection: PCAProjection | None = None
    commit_to_index: dict[str, int] = field(default_factory=dict)
    index_to_commit: dict[int, str] = field(default_factory=dict)


def _make_penultimate() -> PenultimateEmbedding:
    return PenultimateEmbedding(
        commit_hash="c1",
        vector=(1.0, 2.0),
        code_dimensions=2,
        summary_dimensions=0,
        code_model="code",
        summary_model=None,
        summary_embedding_model=None,
    )


def _make_projection() -> PCAProjection:
    return PCAProjection(
        feature_count=2,
        components=((1.0, 0.0), (0.0, 1.0)),
        mean=(0.0, 0.0),
        explained_variance=(1.0, 1.0),
        explained_variance_ratio=(1.0, 0.0),
        sample_count=10,
        fitted_at=123.0,
        whiten=True,
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
    penultimate = _make_penultimate()
    original_state = DummyState(
        archive=DummyArchive(archive_data),
        lower_bounds=np.array([-2.0, -2.0]),
        upper_bounds=np.array([2.0, 2.0]),
        history=(penultimate,),
        projection=_make_projection(),
    )

    snapshot = build_snapshot("main", original_state)

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
    assert restored_state.history[0].commit_hash == penultimate.commit_hash
    assert restored_state.projection is not None
    assert restored_state.projection.feature_count == original_state.projection.feature_count  # type: ignore[union-attr]

    # Archive entries and commit mappings are restored.
    assert restored_state.index_to_commit == {0: "c1"}
    assert restored_state.commit_to_index == {"c1": 0}
    assert commit_to_island == {"c1": "main"}
    assert len(restored_state.archive.add_calls) == 1


def test_build_snapshot_backend_picks_null_or_db_backend() -> None:
    null_backend = build_snapshot_backend(None)
    assert isinstance(null_backend, NullSnapshotBackend)

    some_id = uuid.uuid4()
    db_backend = build_snapshot_backend(some_id)
    assert isinstance(db_backend, DatabaseSnapshotBackend)
    assert db_backend.experiment_id == some_id


