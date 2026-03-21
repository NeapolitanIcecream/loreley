from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from git import Repo
import pytest
from sqlalchemy import delete

from loreley.config import get_settings
import loreley.db.base as db_base
from loreley.db.models import MapElitesRepoStateAggregate


def init_repo(tmp_path: Path) -> Repo:
    repo = Repo.init(tmp_path)
    with repo.config_writer() as cfg:
        cfg.set_value("user", "name", "Benchmark User")
        cfg.set_value("user", "email", "benchmark@example.com")
    return repo


def commit_all(repo: Repo, message: str) -> str:
    repo.git.add(A=True)
    commit = repo.index.commit(message)
    return commit.hexsha


def commit_empty(repo: Repo, message: str) -> str:
    repo.git.commit("--allow-empty", "-m", message)
    return repo.head.commit.hexsha


def blob_sha(repo: Repo, commit_hash: str, path: str) -> str:
    return repo.git.rev_parse(f"{commit_hash}:{path}").strip()


def vector_for_sha(sha: str, *, dims: int) -> tuple[float, ...]:
    digest = bytes.fromhex(sha[:40])
    width = max(1, int(dims))
    return tuple(digest[idx % len(digest)] / 255.0 for idx in range(width))


def sum_vectors(blob_shas: Sequence[str], *, dims: int) -> tuple[float, ...]:
    totals = [0.0] * max(1, int(dims))
    for sha in blob_shas:
        vector = vector_for_sha(sha, dims=dims)
        for idx, value in enumerate(vector):
            totals[idx] += float(value)
    return tuple(totals)


def prepare_postgres_benchmark(
    monkeypatch: pytest.MonkeyPatch,
    *,
    root_commit: str,
) -> None:
    if not os.getenv("DATABASE_URL"):
        pytest.skip("Set DATABASE_URL to run Postgres-backed benchmarks.")

    monkeypatch.setenv("MAPELITES_EXPERIMENT_ROOT_COMMIT", root_commit)
    get_settings.cache_clear()
    db_base._session_factory.cache_clear()
    db_base.get_engine.cache_clear()
    db_base.reset_database_schema(include_console_log=False)


def clear_commit_aggregate(commit_hash: str) -> None:
    with db_base.session_scope() as session:
        stmt = delete(MapElitesRepoStateAggregate).where(
            MapElitesRepoStateAggregate.commit_hash == str(commit_hash),
        )
        session.execute(stmt)


def clear_commit_aggregates(commit_hashes: Sequence[str]) -> None:
    cleaned = [str(commit_hash).strip() for commit_hash in commit_hashes if str(commit_hash).strip()]
    if not cleaned:
        return
    with db_base.session_scope() as session:
        stmt = delete(MapElitesRepoStateAggregate).where(
            MapElitesRepoStateAggregate.commit_hash.in_(cleaned),
        )
        session.execute(stmt)
