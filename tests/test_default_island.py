from __future__ import annotations

from pathlib import Path

from loreley.config import resolve_default_island_id
from loreley.core.map_elites.manager import MapElitesManager
from loreley.core.map_elites.sampler import MapElitesSampler


class _RecordingManager:
    def __init__(self) -> None:
        self.calls: list[str | None] = []

    def get_cell_commits(self, island_id: str | None = None) -> dict[int, str]:
        self.calls.append(island_id)
        return {0: "c1"}


def test_resolve_default_island_falls_back_to_main_for_blank_setting(settings) -> None:
    """Blank island config should not fork behavior away from the configured default."""

    settings.mapelites_default_island_id = ""

    assert resolve_default_island_id(settings) == "main"


def test_sampler_uses_main_when_default_island_setting_is_blank(settings) -> None:
    """Regression: sampler should not fall back to the legacy `default` island name."""

    settings.mapelites_default_island_id = ""
    manager = _RecordingManager()

    sampler = MapElitesSampler(manager, settings=settings)
    snapshot = sampler.get_cell_commits_snapshot()

    assert snapshot == ("main", {0: "c1"})
    assert manager.calls == ["main"]


def test_manager_uses_main_when_default_island_setting_is_blank(settings) -> None:
    """Regression: manager should resolve blank default-island settings to `main`."""

    settings.mapelites_default_island_id = ""
    manager = MapElitesManager(settings=settings, repo_root=Path("."))
    manager._snapshot_store = type(  # type: ignore[attr-defined]
        "_StubSnapshotStore",
        (),
        {"load": staticmethod(lambda *_args, **_kwargs: None)},
    )()

    description = manager.describe_island()

    assert description["island_id"] == "main"
