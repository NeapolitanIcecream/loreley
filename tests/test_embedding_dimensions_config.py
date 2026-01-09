from __future__ import annotations

from pathlib import Path

import pytest
from git import Repo

from loreley.config import Settings
from loreley.core.experiments import ExperimentError, build_experiment_config_snapshot


def _init_repo(tmp_path: Path) -> Repo:
    repo = Repo.init(tmp_path)
    with repo.config_writer() as cfg:
        cfg.set_value("user", "name", "Test User")
        cfg.set_value("user", "email", "test@example.com")
    return repo


def _commit_all(repo: Repo, message: str) -> str:
    repo.git.add(A=True)
    commit = repo.index.commit(message)
    return commit.hexsha


def test_settings_does_not_require_embedding_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAPELITES_CODE_EMBEDDING_DIMENSIONS", raising=False)

    settings = Settings(_env_file=None)
    assert settings.mapelites_code_embedding_dimensions is None


def test_experiment_snapshot_requires_embedding_dimensions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("MAPELITES_CODE_EMBEDDING_DIMENSIONS", raising=False)

    settings = Settings(mapelites_code_embedding_dimensions=None, _env_file=None)
    repo = _init_repo(tmp_path)
    (tmp_path / "a.py").write_text("print('a')\n", encoding="utf-8")
    root_commit = _commit_all(repo, "init")
    settings.mapelites_experiment_root_commit = root_commit
    with pytest.raises(ExperimentError):
        build_experiment_config_snapshot(settings, repo=repo)


