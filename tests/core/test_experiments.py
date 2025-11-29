from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.core.experiments import (
    _build_slug_from_source,
    build_experiment_config_snapshot,
    hash_experiment_config,
)


def test_build_slug_from_source_basic() -> None:
    slug = _build_slug_from_source("https://github.com/Owner/Repo.git")
    assert slug == "github.com/owner/repo"


def test_experiment_config_hash_stable(settings: Settings) -> None:
    snapshot_1 = build_experiment_config_snapshot(settings)
    hash_1 = hash_experiment_config(snapshot_1)

    # Mutate an unrelated setting; hash should remain unchanged.
    settings.app_name = "SomethingElse"
    snapshot_2 = build_experiment_config_snapshot(settings)
    hash_2 = hash_experiment_config(snapshot_2)

    assert snapshot_1 == snapshot_2
    assert hash_1 == hash_2


