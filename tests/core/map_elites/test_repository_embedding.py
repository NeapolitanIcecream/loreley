from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from git import Repo

from loreley.config import Settings
from loreley.core.map_elites.code_embedding import CommitCodeEmbedding, FileEmbedding
from loreley.core.map_elites.repository_files import list_repository_files
from loreley.core.map_elites.repository_state_embedding import RepositoryStateEmbedder


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


def _stub_embed_chunked_files(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Monkeypatch repo-state embedding to avoid OpenAI calls.

    Returns a dict capturing call metadata.
    """

    import loreley.core.map_elites.repository_state_embedding as rse

    calls: dict[str, Any] = {"count": 0, "file_counts": []}

    def _vector_for_content(text: str) -> tuple[float, float]:
        digest = hashlib.sha1(text.encode("utf-8")).digest()
        return (digest[0] / 255.0, digest[1] / 255.0)

    def _fake_embed(chunked_files, *, settings=None, client=None):  # type: ignore[no-untyped-def]
        calls["count"] += 1
        calls["file_counts"].append(len(chunked_files))
        file_embeddings: list[FileEmbedding] = []
        for file in chunked_files:
            text = "\n".join(chunk.content for chunk in file.chunks)
            vec = _vector_for_content(text)
            file_embeddings.append(
                FileEmbedding(
                    file=file,
                    chunk_embeddings=(),
                    vector=vec,
                    weight=1.0,
                )
            )
        commit_vec = _mean([fe.vector for fe in file_embeddings])
        return CommitCodeEmbedding(
            files=tuple(file_embeddings),
            vector=commit_vec,
            model="stub",
            dimensions=len(commit_vec),
        )

    monkeypatch.setattr(rse, "embed_chunked_files", _fake_embed)
    return calls


def _mean(vectors: list[tuple[float, ...]]) -> tuple[float, ...]:
    if not vectors:
        return ()
    dims = len(vectors[0])
    if dims == 0:
        return ()
    totals = [0.0] * dims
    for vec in vectors:
        assert len(vec) == dims
        for i in range(dims):
            totals[i] += float(vec[i])
    return tuple(value / len(vectors) for value in totals)


def test_repository_file_catalog_respects_gitignore_and_extension_filter(
    tmp_path: Path,
    settings: Settings,
) -> None:
    repo = _init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("print('a')\n", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("print('ignored')\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("not code\n", encoding="utf-8")

    commit = _commit_all(repo, "init")

    settings.mapelites_preprocess_allowed_extensions = [".py"]
    settings.mapelites_preprocess_allowed_filenames = []
    settings.mapelites_preprocess_excluded_globs = []
    settings.mapelites_preprocess_max_file_size_kb = 64

    files = list_repository_files(
        repo_root=tmp_path,
        treeish=commit,
        settings=settings,
        repo=repo,
    )
    paths = [f.path.as_posix() for f in files]
    assert paths == ["a.py"]
    assert files[0].blob_sha
    assert files[0].size_bytes > 0


def test_repository_file_catalog_respects_repo_state_max_files_cap(
    tmp_path: Path,
    settings: Settings,
) -> None:
    repo = _init_repo(tmp_path)
    (tmp_path / "a.py").write_text("print('a')\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("print('b')\n", encoding="utf-8")
    (tmp_path / "c.py").write_text("print('c')\n", encoding="utf-8")
    commit = _commit_all(repo, "init")

    settings.mapelites_preprocess_allowed_extensions = [".py"]
    settings.mapelites_preprocess_allowed_filenames = []
    settings.mapelites_preprocess_excluded_globs = []
    settings.mapelites_preprocess_max_file_size_kb = 64
    settings.mapelites_repo_state_max_files = 2

    files = list_repository_files(
        repo_root=tmp_path,
        treeish=commit,
        settings=settings,
        repo=repo,
    )
    paths = [f.path.as_posix() for f in files]
    assert len(paths) == 2
    assert set(paths).issubset({"a.py", "b.py", "c.py"})
    # Deterministic across calls.
    files_again = list_repository_files(
        repo_root=tmp_path,
        treeish=commit,
        settings=settings,
        repo=repo,
    )
    assert [f.path.as_posix() for f in files_again] == paths


def test_repository_state_embedder_uses_cache_hits_and_misses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings: Settings,
) -> None:
    repo = _init_repo(tmp_path)
    settings.mapelites_preprocess_allowed_extensions = [".py"]
    settings.mapelites_preprocess_allowed_filenames = []
    settings.mapelites_preprocess_excluded_globs = []
    settings.mapelites_preprocess_max_file_size_kb = 64

    (tmp_path / "a.py").write_text("print('a1')\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("print('b1')\n", encoding="utf-8")
    c1 = _commit_all(repo, "c1")

    # Modify a.py and add c.py, leave b.py unchanged.
    (tmp_path / "a.py").write_text("print('a2')\n", encoding="utf-8")
    (tmp_path / "c.py").write_text("print('c1')\n", encoding="utf-8")
    c2 = _commit_all(repo, "c2")

    calls = _stub_embed_chunked_files(monkeypatch)

    embedder = RepositoryStateEmbedder(settings=settings, cache_backend="memory", repo=repo)

    e1, s1 = embedder.run(commit_hash=c1, repo_root=tmp_path, treeish=c1)
    assert e1 is not None
    assert s1.eligible_files == 2
    assert s1.unique_blobs == 2
    assert s1.cache_hits == 0
    assert s1.cache_misses == 2
    assert calls["count"] == 1
    assert calls["file_counts"][-1] == 2

    e2, s2 = embedder.run(commit_hash=c2, repo_root=tmp_path, treeish=c2)
    assert e2 is not None
    assert s2.eligible_files == 3
    assert s2.unique_blobs == 3
    assert s2.cache_hits == 1  # b.py unchanged
    assert s2.cache_misses == 2  # a.py changed + c.py new
    assert calls["count"] == 2
    assert calls["file_counts"][-1] == 2

    # Commit vector is uniform mean across all eligible file paths.
    repo_files = list_repository_files(repo_root=tmp_path, treeish=c2, settings=settings, repo=repo)
    cached = embedder.cache.get_many(sorted({f.blob_sha for f in repo_files}))
    expected = _mean([cached[f.blob_sha] for f in repo_files])
    assert e2.vector == pytest.approx(expected)


def test_repository_state_embedder_deduplicates_duplicate_blobs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    settings: Settings,
) -> None:
    repo = _init_repo(tmp_path)
    settings.mapelites_preprocess_allowed_extensions = [".py"]
    settings.mapelites_preprocess_allowed_filenames = []
    settings.mapelites_preprocess_excluded_globs = []
    settings.mapelites_preprocess_max_file_size_kb = 64

    # Two files with identical content => same blob SHA.
    (tmp_path / "a.py").write_text("print('same')\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("print('same')\n", encoding="utf-8")
    c1 = _commit_all(repo, "c1")

    calls = _stub_embed_chunked_files(monkeypatch)
    embedder = RepositoryStateEmbedder(settings=settings, cache_backend="memory", repo=repo)

    embedding, stats = embedder.run(commit_hash=c1, repo_root=tmp_path, treeish=c1)
    assert embedding is not None
    assert stats.eligible_files == 2
    assert stats.unique_blobs == 1
    assert stats.cache_misses == 1
    assert stats.files_embedded == 1
    assert stats.files_aggregated == 2
    assert calls["count"] == 1
    assert calls["file_counts"][-1] == 1  # embedded once for the shared blob


