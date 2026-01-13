from __future__ import annotations

from pathlib import Path

import pytest
from git import Repo

from loreley.config import Settings
from loreley.core.map_elites.file_embedding_cache import DatabaseFileEmbeddingCache
from loreley.core.map_elites.repository_state_embedding import RepoStateEmbeddingError, RepositoryStateEmbedder


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


def test_repo_state_incremental_only_raises_when_no_cache_hit_and_no_incremental_path(
    tmp_path: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_repo(tmp_path)
    settings.mapelites_preprocess_allowed_extensions = [".py"]
    settings.mapelites_preprocess_allowed_filenames = []
    settings.mapelites_preprocess_excluded_globs = []
    settings.mapelites_preprocess_max_file_size_kb = 64

    (tmp_path / "a.py").write_text("print('a')\n", encoding="utf-8")
    c1 = _commit_all(repo, "c1")

    cache = DatabaseFileEmbeddingCache(
        experiment_id="00000000-0000-0000-0000-000000000000",
        embedding_model="stub",
        requested_dimensions=2,
    )
    embedder = RepositoryStateEmbedder(
        settings=settings,
        cache=cache,
        repo=repo,
        experiment_id="00000000-0000-0000-0000-000000000000",
    )

    monkeypatch.setattr(embedder, "_load_aggregate", lambda **_kwargs: None)
    monkeypatch.setattr(embedder, "_try_incremental_aggregate", lambda **_kwargs: None)

    with pytest.raises(RepoStateEmbeddingError, match="incremental-only"):
        _ = embedder.run(commit_hash=c1, repo_root=tmp_path, mode="incremental_only")


