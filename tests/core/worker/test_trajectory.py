from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast

from sqlalchemy.orm import Session

from loreley.config import Settings
from loreley.core.worker.trajectory import (
    build_inspiration_trajectory_rollup,
    find_lca,
    get_full_chunk_pairs_from_tip,
    walk_unique_path,
)
from loreley.db.models import CommitCard


class _Result:
    def __init__(self, value: Any) -> None:
        self._value = value

    def scalar_one_or_none(self) -> Any:
        return self._value


class _FakeSession:
    def __init__(self, cards: dict[str, Any]) -> None:
        self._cards = dict(cards)

    def get(self, model: Any, key: Any) -> Any:
        if model is CommitCard:
            return self._cards.get(str(key))
        return None

    def execute(self, _stmt: Any) -> _Result:
        return _Result(None)

    @contextmanager
    def begin_nested(self):
        yield self

    def add(self, _obj: Any) -> None:  # pragma: no cover - not used in these tests
        return

    def flush(self) -> None:  # pragma: no cover - not used in these tests
        return


def _card(commit_hash: str, parent: str | None, summary: str) -> Any:
    return SimpleNamespace(
        commit_hash=commit_hash,
        parent_commit_hash=parent,
        change_summary=summary,
    )


def test_find_lca_and_walk_unique_path() -> None:
    # R -> B -> C1 -> C2
    #      └-> D1 -> D2
    cards = {
        "r": _card("r", None, "root"),
        "b": _card("b", "r", "base"),
        "c1": _card("c1", "b", "c1"),
        "c2": _card("c2", "c1", "c2"),
        "d1": _card("d1", "b", "d1"),
        "d2": _card("d2", "d1", "d2"),
    }
    session = cast(Session, _FakeSession(cards))

    lca = find_lca(base_commit_hash="d2", inspiration_commit_hash="c2", session=session)
    assert lca == "b"

    path = walk_unique_path(lca_commit_hash="b", inspiration_commit_hash="c2", session=session)
    assert [card.commit_hash for card in path] == ["c1", "c2"]


def test_get_full_chunk_pairs_from_tip_produces_root_aligned_pairs() -> None:
    # Build a linear chain: r -> c1 -> ... -> c20
    cards: dict[str, Any] = {"r": _card("r", None, "root")}
    parent = "r"
    for i in range(1, 21):
        commit = f"c{i}"
        cards[commit] = _card(commit, parent, f"step {i}")
        parent = commit

    session = cast(Session, _FakeSession(cards))
    pairs = get_full_chunk_pairs_from_tip(
        inspiration_commit_hash="c20",
        block_size=8,
        session=session,
    )
    assert pairs == [("c8", "c16"), ("r", "c8")]


def test_build_rollup_renders_bounded_raw_sections_and_omission_count(settings: Settings) -> None:
    # r -> c1 -> ... -> c40 with base=c3 (unique path length 37: c4..c40)
    cards: dict[str, Any] = {"r": _card("r", None, "root")}
    parent = "r"
    for i in range(1, 41):
        commit = f"c{i}"
        cards[commit] = _card(commit, parent, f"step {i}")
        parent = commit

    session = cast(Session, _FakeSession(cards))
    settings.worker_planning_trajectory_block_size = 8
    settings.worker_planning_trajectory_max_raw_steps = 6
    settings.worker_planning_trajectory_max_chunks = 0

    rollup = build_inspiration_trajectory_rollup(
        base_commit_hash="c3",
        inspiration_commit_hash="c40",
        session=session,
        settings=settings,
    )

    assert rollup.meta["unique_steps_count"] == 37
    assert rollup.meta["omitted_steps"] == 24
    text = "\n".join(rollup.lines)
    assert "unique_steps_count: 37" in text
    assert "Earliest unique steps" in text
    assert "Recent unique steps" in text
    assert "c4:" in text
    assert "c33:" in text
    assert "Omitted 24 older unique step" in text


def test_build_rollup_does_not_duplicate_when_path_is_short(settings: Settings) -> None:
    # r -> a1 -> a2 -> a3 (unique path length 3)
    cards = {
        "r": _card("r", None, "root"),
        "a1": _card("a1", "r", "step 1"),
        "a2": _card("a2", "a1", "step 2"),
        "a3": _card("a3", "a2", "step 3"),
    }
    session = cast(Session, _FakeSession(cards))
    settings.worker_planning_trajectory_block_size = 8
    settings.worker_planning_trajectory_max_raw_steps = 6
    settings.worker_planning_trajectory_max_chunks = 0

    rollup = build_inspiration_trajectory_rollup(
        base_commit_hash="r",
        inspiration_commit_hash="a3",
        session=session,
        settings=settings,
    )
    text = "\n".join(rollup.lines)
    assert "Recent unique steps" in text
    assert "Earliest unique steps" not in text
    assert rollup.meta["omitted_steps"] == 0


def test_build_rollup_forces_earliest_steps_when_no_partial_block(settings: Settings) -> None:
    # r -> c1 -> ... -> c40 with base=r (unique path starts on a chunk boundary)
    cards: dict[str, Any] = {"r": _card("r", None, "root")}
    parent = "r"
    for i in range(1, 41):
        commit = f"c{i}"
        cards[commit] = _card(commit, parent, f"step {i}")
        parent = commit

    session = cast(Session, _FakeSession(cards))
    settings.worker_planning_trajectory_block_size = 8
    settings.worker_planning_trajectory_max_raw_steps = 6
    settings.worker_planning_trajectory_max_chunks = 0

    rollup = build_inspiration_trajectory_rollup(
        base_commit_hash="r",
        inspiration_commit_hash="c40",
        session=session,
        settings=settings,
    )

    text = "\n".join(rollup.lines)
    assert "Earliest unique steps" in text
    assert "c1:" in text
    assert "c2:" in text


def test_build_rollup_marks_zero_unique_steps_for_ancestor_inspiration(settings: Settings) -> None:
    # r -> b -> c (inspiration=b is an ancestor of base=c)
    cards = {
        "r": _card("r", None, "root"),
        "b": _card("b", "r", "b"),
        "c": _card("c", "b", "c"),
    }
    session = cast(Session, _FakeSession(cards))
    settings.worker_planning_trajectory_max_chunks = 0

    rollup = build_inspiration_trajectory_rollup(
        base_commit_hash="c",
        inspiration_commit_hash="b",
        session=session,
        settings=settings,
    )
    assert rollup.meta["unique_steps_count"] == 0
    assert any("unique_steps_count: 0" in line for line in rollup.lines)


