from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import SQLAlchemyError

import loreley.core.map_elites.snapshot as snapshot_module
from loreley.core.map_elites.dimension_reduction import PCAProjection
from loreley.core.map_elites.objectives import ObjectiveContract, ObjectiveSpec
from loreley.core.map_elites.pareto_archive import ParetoGridArchive
from loreley.core.map_elites.snapshot import (
    DatabaseSnapshotStore,
    SnapshotElite,
    apply_snapshot,
    serialize_projection,
)
from loreley.core.map_elites.types import IslandState


def _contract() -> ObjectiveContract:
    return ObjectiveContract(
        (
            ObjectiveSpec(name="quality", direction="max"),
            ObjectiveSpec(name="latency", direction="min"),
        )
    )


def _state() -> IslandState:
    return IslandState(
        archive=ParetoGridArchive(
            dims=(4, 4),
            ranges=((0.0, 1.0), (0.0, 1.0)),
            objective_count=2,
            max_front_size=4,
            epsilon=1.0e-9,
        ),
        lower_bounds=np.asarray([0.0, 0.0]),
        upper_bounds=np.asarray([1.0, 1.0]),
    )


def _projection() -> PCAProjection:
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


def test_apply_snapshot_restores_complete_pareto_front_and_projection() -> None:
    contract = _contract()
    snapshot = {
        "island_id": "main",
        "objective_contract": contract.as_payload(),
        "objective_contract_fingerprint": contract.fingerprint,
        "lower_bounds": [0.0, 0.0],
        "upper_bounds": [1.0, 1.0],
        "samples_since_fit": 3,
        "history": [
            {
                "commit_hash": "a",
                "vector": [1.0, 2.0],
                "embedding_model": "code",
            }
        ],
        "projection": serialize_projection(_projection()),
        "archive": [
            {
                "index": 0,
                "commit_hash": "a",
                "objective_values": [10.0, 5.0],
                "measures": [0.1, 0.1],
                "timestamp": 10.0,
            },
            {
                "index": 0,
                "commit_hash": "b",
                "objective_values": [8.0, 3.0],
                "measures": [0.1, 0.1],
                "timestamp": 11.0,
            },
        ],
    }
    state = _state()

    apply_snapshot(
        state=state,
        snapshot=snapshot,
        island_id="main",
        objective_contract=contract,
    )

    assert state.samples_since_fit == 3
    assert state.projection is not None
    assert state.history[0].commit_hash == "a"
    assert state.index_to_commits == {0: ("a", "b")}
    assert state.commit_to_index == {"a": 0, "b": 0}
    assert {
        candidate.objective_scores
        for candidate in state.archive.records()
    } == {(10.0, -5.0), (8.0, -3.0)}


def test_apply_snapshot_fails_on_objective_contract_mismatch() -> None:
    stored = _contract()
    configured = ObjectiveContract((ObjectiveSpec(name="quality", direction="max"),))
    snapshot = {
        "objective_contract": stored.as_payload(),
        "objective_contract_fingerprint": stored.fingerprint,
        "archive": [],
    }

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        apply_snapshot(
            state=_state(),
            snapshot=snapshot,
            island_id="main",
            objective_contract=configured,
        )


def test_apply_snapshot_rejects_persisted_dominated_member() -> None:
    contract = _contract()
    snapshot = {
        "objective_contract": contract.as_payload(),
        "objective_contract_fingerprint": contract.fingerprint,
        "archive": [
            {
                "index": 0,
                "commit_hash": "winner",
                "objective_values": [10.0, 1.0],
                "measures": [0.1, 0.1],
                "timestamp": 1.0,
            },
            {
                "index": 0,
                "commit_hash": "dominated",
                "objective_values": [5.0, 2.0],
                "measures": [0.1, 0.1],
                "timestamp": 2.0,
            },
        ],
    }

    with pytest.raises(ValueError, match="dominated"):
        apply_snapshot(
            state=_state(),
            snapshot=snapshot,
            island_id="main",
            objective_contract=contract,
        )


def test_replace_front_deletes_only_target_cell_then_inserts_all_members() -> None:
    store = DatabaseSnapshotStore()
    elites = (
        SnapshotElite(
            cell_index=2,
            commit_hash="a",
            objective_values=(10.0, 5.0),
            measures=(0.6, 0.1),
            timestamp=1.0,
        ),
        SnapshotElite(
            cell_index=2,
            commit_hash="b",
            objective_values=(8.0, 3.0),
            measures=(0.6, 0.1),
            timestamp=2.0,
        ),
    )

    class DummySession:
        def __init__(self) -> None:
            self.statements: list[object] = []

        def execute(self, statement: object) -> None:
            self.statements.append(statement)

    session = DummySession()
    store._replace_front(  # type: ignore[arg-type, attr-defined]
        session,
        island_id="main",
        elites=elites,
    )

    assert len(session.statements) == 2
    delete_sql = str(
        session.statements[0].compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )
    insert_sql = str(session.statements[1])
    assert "cell_index = 2" in delete_sql
    assert "objective_values" in insert_sql
    assert "solution" not in insert_sql


def test_prune_history_entries_deletes_rows_beyond_limit() -> None:
    store = DatabaseSnapshotStore()

    class DummyScalarResult:
        def __init__(self, values: list[str]) -> None:
            self._values = values

        def scalars(self) -> "DummyScalarResult":
            return self

        def all(self) -> list[str]:
            return list(self._values)

    class DummySession:
        def __init__(self) -> None:
            self.statements: list[object] = []

        def execute(self, statement: object) -> DummyScalarResult:
            self.statements.append(statement)
            return DummyScalarResult(["old-1", "old-2"] if len(self.statements) == 1 else [])

    session = DummySession()
    store._prune_history_entries(  # type: ignore[arg-type, attr-defined]
        session,
        island_id="main",
        history_limit=3,
    )

    compiled = str(
        session.statements[1].compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    ).upper()
    assert "DELETE FROM MAP_ELITES_PCA_HISTORY" in compiled
    assert "OLD-1" in compiled
    assert "OLD-2" in compiled


def test_snapshot_load_raises_on_database_error() -> None:
    store = DatabaseSnapshotStore()

    @contextmanager
    def broken_scope():  # type: ignore[no-untyped-def]
        raise SQLAlchemyError("boom")
        yield

    original_scope = snapshot_module.session_scope
    snapshot_module.session_scope = broken_scope
    try:
        with pytest.raises(
            snapshot_module.SnapshotStoreError,
            match="Failed to load MAP-Elites snapshot",
        ):
            store.load("main")
    finally:
        snapshot_module.session_scope = original_scope


def test_snapshot_apply_update_raises_on_database_error() -> None:
    store = DatabaseSnapshotStore()

    @contextmanager
    def broken_scope():  # type: ignore[no-untyped-def]
        raise SQLAlchemyError("boom")
        yield

    original_scope = snapshot_module.session_scope
    snapshot_module.session_scope = broken_scope
    try:
        with pytest.raises(
            snapshot_module.SnapshotStoreError,
            match="Failed to persist MAP-Elites snapshot",
        ):
            store.apply_update(
                "main",
                update=snapshot_module.SnapshotUpdate(
                    objective_contract=_contract(),
                ),
            )
    finally:
        snapshot_module.session_scope = original_scope
