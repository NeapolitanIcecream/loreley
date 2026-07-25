from __future__ import annotations

from pathlib import Path

from loreley.config import resolve_default_island_id
from loreley.core.map_elites.manager import MapElitesManager
from loreley.core.map_elites.sampler import MapElitesSampler


class _RecordingManager:
    def __init__(self) -> None:
        self.calls: list[str | None] = []

    def get_cell_fronts(
        self,
        island_id: str | None = None,
    ) -> dict[int, tuple[str, ...]]:
        self.calls.append(island_id)
        return {0: ("c1",)}


def test_resolve_default_island_uses_first_configured_island(settings) -> None:
    settings.mapelites_islands = ("explore", "main")

    assert resolve_default_island_id(settings) == "explore"


def test_sampler_uses_first_configured_island(settings) -> None:
    settings.mapelites_islands = ("explore", "main")
    manager = _RecordingManager()

    sampler = MapElitesSampler(manager, settings=settings)
    snapshot = sampler.get_cell_fronts_snapshot()

    assert snapshot == ("explore", {0: ("c1",)})
    assert manager.calls == ["explore"]


def test_manager_uses_first_configured_island(settings) -> None:
    settings.mapelites_islands = ("explore", "main")

    manager = MapElitesManager(settings=settings, repo_root=Path("."))
    manager._snapshot_store = type(  # type: ignore[attr-defined]
        "_StubSnapshotStore",
        (),
        {"load": staticmethod(lambda *_args, **_kwargs: None)},
    )()

    description = manager.describe_island()

    assert description["island_id"] == "explore"
