from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from loreley.config import resolve_objective_contract
from tests.support import TestSettings


def test_campaign_program_approve_policy_is_rejected_until_implemented() -> None:
    """Regression: approve policy was accepted while no approval flow existed."""

    with pytest.raises(ValidationError) as exc_info:
        TestSettings(CAMPAIGN_PROGRAM_CHANGE_POLICY="approve")

    errors = exc_info.value.errors()
    assert errors[0]["loc"] == ("CAMPAIGN_PROGRAM_CHANGE_POLICY",)


def test_multi_island_and_objective_contract_load_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAPELITES_ISLANDS", '["main","explore"]')
    monkeypatch.setenv(
        "MAPELITES_OBJECTIVES",
        '[{"name":"quality","direction":"max"},{"name":"latency_ms","direction":"min"}]',
    )

    settings = TestSettings()
    contract = resolve_objective_contract(settings)

    assert settings.mapelites_islands == ("main", "explore")
    assert contract.names == ("quality", "latency_ms")
    assert contract.primary.direction == "max"


@pytest.mark.parametrize(
    "islands",
    [
        [],
        [""],
        ["main", " main "],
    ],
)
def test_island_contract_requires_nonempty_unique_names(islands: list[str]) -> None:
    with pytest.raises(ValidationError, match="island"):
        TestSettings(MAPELITES_ISLANDS=islands)


def test_objective_contract_requires_unique_names() -> None:
    with pytest.raises(ValidationError, match="unique"):
        TestSettings(
            MAPELITES_OBJECTIVES=[
                {"name": "score", "direction": "max"},
                {"name": "score", "direction": "min"},
            ]
        )


def test_migration_interval_accepts_zero_as_disabled() -> None:
    settings = TestSettings(MAPELITES_MIGRATION_INTERVAL_JOBS=0)

    assert settings.mapelites_migration_interval_jobs == 0
    with pytest.raises(ValidationError):
        TestSettings(MAPELITES_MIGRATION_INTERVAL_JOBS=-1)


def test_randomized_worker_repositories_include_the_process_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr("loreley.config.uuid.uuid4", lambda: SimpleNamespace(hex="a" * 32))
    monkeypatch.setattr("loreley.config.os.getpid", lambda: 101)
    first = TestSettings(
        WORKER_REPO_WORKTREE=str(tmp_path / "worker"),
        WORKER_REPO_WORKTREE_RANDOMIZE=True,
        WORKER_REPO_WORKTREE_RANDOM_SUFFIX_LEN=1,
    )
    monkeypatch.setattr("loreley.config.os.getpid", lambda: 202)
    second = TestSettings(
        WORKER_REPO_WORKTREE=str(tmp_path / "worker"),
        WORKER_REPO_WORKTREE_RANDOMIZE=True,
        WORKER_REPO_WORKTREE_RANDOM_SUFFIX_LEN=1,
    )

    assert first.worker_repo_worktree.endswith("worker-pid-101-a")
    assert second.worker_repo_worktree.endswith("worker-pid-202-a")
    assert first.worker_repo_worktree != second.worker_repo_worktree
