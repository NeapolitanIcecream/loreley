from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from loreley.config import resolve_objective_contract
from loreley.core.map_elites.code_embedding import CommitCodeEmbedding
from loreley.core.map_elites.comparison import (
    ComparisonAdmission, ComparisonContextError, StaleComparisonError,
)
from loreley.core.map_elites.dimension_reduction import PCAProjection
from loreley.core.map_elites.manager import MapElitesManager
from loreley.core.map_elites.objectives import ObjectiveSpec
from loreley.core.map_elites.pareto_archive import ParetoCandidate, ParetoGridArchive
from loreley.core.map_elites.snapshot import serialize_projection


def _candidate(commit, score=1.0, measures=(0.25,)):
    return ParetoCandidate(commit, (score,), (score,), measures, 1.0)


def _archive(objective_count=1):
    return ParetoGridArchive(
        dims=(2,), ranges=((0.0, 1.0),), objective_count=objective_count,
        max_front_size=4, epsilon=1e-9,
    )


def test_conditional_replacement_bypasses_old_lucky_score():
    archive = _archive()
    incumbent = _candidate("incumbent", 1000.0)
    archive.add(incumbent)
    outcome = archive.replace_if_current(
        _candidate("candidate", 1.0), expected_cell_index=0,
        incumbent_commit_hash="incumbent",
    )
    assert outcome.retained
    assert outcome.removed_commit_hashes == ("incumbent",)
    assert archive.front(0)[0].commit_hash == "candidate"
    assert incumbent.objective_values == (1000.0,)


@pytest.mark.parametrize(
    ("candidate", "cell", "incumbent", "message"),
    [
        (_candidate("candidate"), 0, "outdated", "no longer current"),
        (_candidate("candidate"), 0, None, "no longer current"),
        (_candidate("candidate", measures=(0.75,)), 0, "incumbent", "compared cell"),
        (_candidate("incumbent"), 0, "incumbent", "already retained"),
        (_candidate("candidate"), False, "incumbent", "must be an integer"),
    ],
)
def test_conditional_replacement_invalid_context_does_not_mutate(candidate, cell, incumbent, message):
    archive = _archive()
    archive.add(_candidate("incumbent", 1000.0))
    before = archive.records()
    with pytest.raises(ValueError, match=message):
        archive.replace_if_current(
            candidate, expected_cell_index=cell, incumbent_commit_hash=incumbent,
        )
    assert archive.records() == before
    assert archive.stats.num_elites == 1


def test_conditional_empty_cell_requires_it_still_to_be_empty():
    archive = _archive()
    candidate = _candidate("candidate")
    outcome = archive.replace_if_current(
        candidate, expected_cell_index=0, incumbent_commit_hash=None,
    )
    assert outcome.removed_commit_hashes == ()
    with pytest.raises(ValueError, match="no longer current"):
        archive.replace_if_current(
            _candidate("other"), expected_cell_index=0, incumbent_commit_hash=None,
        )
    assert archive.records() == (candidate,)


def test_conditional_replacement_rejects_multiple_objectives():
    archive = _archive(objective_count=2)
    incumbent = ParetoCandidate("incumbent", (1.0, 1.0), (1.0, 1.0), (0.25,), 1.0)
    archive.add(incumbent)
    with pytest.raises(ValueError, match="exactly one objective"):
        archive.replace_if_current(
            ParetoCandidate("candidate", (2.0, 2.0), (2.0, 2.0), (0.25,), 2.0),
            expected_cell_index=0, incumbent_commit_hash="incumbent",
        )
    assert archive.records() == (incumbent,)


def test_conditional_replacement_rejects_multiple_incumbents():
    archive = _archive()
    incumbents = (_candidate("one"), _candidate("two"))
    # A corrupt or future multi-member single-objective front must not be erased.
    archive._fronts[0] = incumbents
    archive._commit_to_cell = {"one": 0, "two": 0}
    with pytest.raises(ValueError, match="multi-member"):
        archive.replace_if_current(
            _candidate("candidate"), expected_cell_index=0, incumbent_commit_hash="one",
        )
    assert archive.records() == incumbents


class _SnapshotStore:
    def __init__(self, payload):
        self.payload = payload
        self.updates = []
        self.loads = []
        self.fail = False

    def load(self, island_id, *, history_limit=None, session=None):
        self.loads.append((island_id, session))
        return deepcopy(self.payload)

    def apply_update(self, island_id, *, update, session=None):
        if self.fail:
            raise RuntimeError("snapshot failed")
        self.updates.append((island_id, update, session))
        if update.front_replace:
            cells = {entry.cell_index for entry in update.front_replace}
            self.payload["archive"] = [
                entry for entry in self.payload["archive"] if entry["index"] not in cells
            ] + [
                {
                    "index": entry.cell_index,
                    "commit_hash": entry.commit_hash,
                    "objective_values": list(entry.objective_values),
                    "measures": list(entry.measures),
                    "timestamp": entry.timestamp,
                }
                for entry in update.front_replace
            ]


@pytest.fixture
def comparison_manager(settings, monkeypatch):
    settings.mapelites_islands = ("main",)
    settings.mapelites_objectives = (ObjectiveSpec(name="score", direction="max"),)
    settings.mapelites_dimensionality_target_dims = 1
    settings.mapelites_archive_cells_per_dim = 2
    settings.mapelites_dimensionality_refit_interval = 0
    settings.mapelites_dimensionality_penultimate_normalize = False
    settings.mapelites_feature_truncation_k = 1.0
    projection = PCAProjection(
        feature_count=1, components=((1.0,),), mean=(0.0,),
        explained_variance=(1.0,), explained_variance_ratio=(1.0,),
        sample_count=8, epoch=1, fitted_at=0.0, whiten=False,
    )
    contract = resolve_objective_contract(settings)
    store = _SnapshotStore({
        "island_id": "main",
        "objective_contract": contract.as_payload(),
        "objective_contract_fingerprint": contract.fingerprint,
        "lower_bounds": [0.0], "upper_bounds": [1.0],
        "projection": serialize_projection(projection),
        "history": [], "samples_since_fit": 8,
        "archive": [{
            "index": 0, "commit_hash": "incumbent", "objective_values": [1000.0],
            "measures": [0.25], "timestamp": 1.0,
        }],
    })
    manager = MapElitesManager(settings=settings)
    manager._snapshot_store = store
    monkeypatch.setattr(
        manager, "_embed_repo_state_for_ingest",
        lambda **kwargs: SimpleNamespace(
            code_embedding=CommitCodeEmbedding((), (-0.5,), "local-test", 1), stats=None,
        ),
    )
    return manager, store


def _ingest(manager, context, *, allow=True, **kwargs):
    return manager.ingest_comparison(
        ComparisonAdmission(
            commit_hash="candidate",
            metrics=[{"name": "score", "value": 1.0, "higher_is_better": True}],
            comparison_context=context,
            replacement_allowed=allow,
        ),
        **kwargs,
    )


def test_prepare_uses_durable_snapshot_and_does_not_advance_pca(comparison_manager):
    manager, store = comparison_manager
    manager.get_records("main")
    store.payload["archive"][0]["commit_hash"] = "new-incumbent"
    session = object()
    context = manager.prepare_comparison(commit_hash="candidate", snapshot_session=session)
    assert context["incumbent_commit_hash"] == "new-incumbent"
    assert context["cell_index"] == 0
    assert context["measures"] == [0.25]
    assert context["objective_name"] == "score"
    assert context["higher_is_better"] is True
    assert json.loads(json.dumps(context)) == context
    assert store.loads[-1] == ("main", session)
    assert not store.updates
    assert manager._archives["main"].history == ()
    assert manager._archives["main"].samples_since_fit == 8


def test_manager_persists_conditional_admission_without_incumbent_refresh(comparison_manager):
    manager, store = comparison_manager
    context = manager.prepare_comparison(commit_hash="candidate")
    result = _ingest(manager, context)
    assert result.inserted
    assert result.record.commit_hash == "candidate"
    assert result.record.objective_values == (1.0,)
    assert result.delta == 1.0
    update = store.updates[-1][1]
    assert update.archive_change_reason == "fresh_comparison"
    assert update.archive_replace is None
    assert update.history_upsert is None
    assert [entry.commit_hash for entry in update.front_replace] == ["candidate"]
    manager.reload_island("main")
    assert manager.get_cell_fronts("main") == {0: ("candidate",)}


def test_manager_denied_comparison_preserves_archive_and_history(comparison_manager):
    manager, store = comparison_manager
    context = manager.prepare_comparison(commit_hash="candidate")
    before = manager.get_records("main")
    result = _ingest(manager, context, allow=False)
    assert not result.inserted
    assert result.artifacts.final_embedding is not None
    assert manager.get_records("main") == before
    assert not store.updates
    assert store.payload["history"] == []


@pytest.mark.parametrize("allow", [True, False])
def test_manager_rejects_stale_incumbent_even_when_denied(comparison_manager, allow):
    manager, store = comparison_manager
    context = manager.prepare_comparison(commit_hash="candidate")
    store.payload["archive"][0]["commit_hash"] = "new-incumbent"
    with pytest.raises(StaleComparisonError, match="no longer current"):
        _ingest(manager, context, allow=allow)
    assert not store.updates
    assert manager.get_cell_fronts("main") == {0: ("new-incumbent",)}


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("candidate_commit_hash", "wrong"), ("island_id", "wrong"),
        ("cell_index", 1), ("measures", [0.26]), ("projection_fingerprint", "wrong"),
        ("objective_name", "other"), ("higher_is_better", "true"),
        ("incumbent_commit_hash", None),
    ],
)
def test_manager_context_rejection_leaves_archive_unchanged(comparison_manager, key, value):
    manager, store = comparison_manager
    context = manager.prepare_comparison(commit_hash="candidate")
    context[key] = value
    with pytest.raises(ComparisonContextError):
        _ingest(manager, context)
    assert not store.updates
    assert manager.get_cell_fronts("main") == {0: ("incumbent",)}


def test_manager_persistence_failure_does_not_poison_memory(comparison_manager):
    manager, store = comparison_manager
    context = manager.prepare_comparison(commit_hash="candidate")
    store.fail = True
    with pytest.raises(RuntimeError, match="snapshot failed"):
        _ingest(manager, context)
    assert manager.get_cell_fronts("main") == {0: ("incumbent",)}
    assert not store.updates
    store.fail = False
    assert _ingest(manager, context).inserted


def test_manager_handles_empty_cell_and_rejects_duplicate_ingestion(comparison_manager):
    manager, store = comparison_manager
    store.payload["archive"] = []
    context = manager.prepare_comparison(commit_hash="candidate")
    assert context["incumbent_commit_hash"] is None
    assert _ingest(manager, context).inserted
    with pytest.raises(StaleComparisonError, match="already retained"):
        _ingest(manager, context)
    assert len(store.updates) == 1


@pytest.mark.parametrize("change", ["refit", "unfitted", "objectives"])
def test_manager_requires_single_objective_frozen_projection(comparison_manager, change):
    manager, store = comparison_manager
    if change == "refit":
        manager.settings.mapelites_dimensionality_refit_interval = 10
    elif change == "unfitted":
        store.payload["projection"] = None
    else:
        manager.settings.mapelites_objectives = (
            ObjectiveSpec(name="score", direction="max"),
            ObjectiveSpec(name="latency", direction="min"),
        )
        manager._objective_contract = resolve_objective_contract(manager.settings)
    with pytest.raises(ComparisonContextError):
        manager.prepare_comparison(commit_hash="candidate")
    assert not store.updates
