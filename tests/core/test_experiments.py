from __future__ import annotations

from pathlib import Path

from git import Repo

from loreley.config import Settings
from loreley.core.experiments import (
    _build_slug_from_source,
    build_experiment_config_snapshot,
    hash_experiment_config,
    _normalise_remote_url,
)


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


def test_build_slug_from_source_basic() -> None:
    slug = _build_slug_from_source("https://github.com/Owner/Repo.git")
    assert slug == "github.com/owner/repo"


def test_experiment_config_hash_stable(settings: Settings, tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("print('a')\n", encoding="utf-8")
    root_commit = _commit_all(repo, "init")
    settings.mapelites_experiment_root_commit = root_commit

    snapshot_1 = build_experiment_config_snapshot(settings, repo=repo)
    hash_1 = hash_experiment_config(snapshot_1)

    # Mutate an unrelated setting; hash should remain unchanged.
    settings.app_name = "SomethingElse"
    snapshot_2 = build_experiment_config_snapshot(settings, repo=repo)
    hash_2 = hash_experiment_config(snapshot_2)

    assert snapshot_1 == snapshot_2
    assert hash_1 == hash_2


def test_experiment_config_snapshot_includes_mapelites_and_excludes_unrelated(
    settings: Settings,
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    (tmp_path / "a.py").write_text("print('a')\n", encoding="utf-8")
    root_commit = _commit_all(repo, "init")
    settings.mapelites_experiment_root_commit = root_commit

    snapshot = build_experiment_config_snapshot(settings, repo=repo)

    # Map-Elites and evaluator knobs should be present.
    assert "mapelites_preprocess_max_file_size_kb" in snapshot
    assert "worker_evaluator_timeout_seconds" in snapshot
    assert "worker_evolution_global_goal" in snapshot

    # Unrelated environment fields should not be part of the experiment key.
    assert "app_name" not in snapshot
    assert "environment" not in snapshot


def test_experiment_config_hash_changes_when_experiment_knob_changes(
    settings: Settings,
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    (tmp_path / "a.py").write_text("print('a')\n", encoding="utf-8")
    root_commit = _commit_all(repo, "init")
    settings.mapelites_experiment_root_commit = root_commit

    snapshot_1 = build_experiment_config_snapshot(settings, repo=repo)
    hash_1 = hash_experiment_config(snapshot_1)

    # Tweak a MAP-Elites setting that should affect experiment identity.
    settings.mapelites_sampler_inspiration_count += 1
    snapshot_2 = build_experiment_config_snapshot(settings, repo=repo)
    hash_2 = hash_experiment_config(snapshot_2)

    assert snapshot_1 != snapshot_2
    assert hash_1 != hash_2


def test_experiment_config_snapshot_encodes_default_negative_infinity(
    settings: Settings,
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    (tmp_path / "a.py").write_text("print('a')\n", encoding="utf-8")
    root_commit = _commit_all(repo, "init")
    settings.mapelites_experiment_root_commit = root_commit

    snapshot = build_experiment_config_snapshot(settings, repo=repo)
    assert snapshot["mapelites_archive_threshold_min"] == {"__float__": "-inf"}


def test_experiment_config_hash_distinguishes_non_finite_floats(
    settings: Settings,
    tmp_path: Path,
) -> None:
    repo = _init_repo(tmp_path)
    (tmp_path / "a.py").write_text("print('a')\n", encoding="utf-8")
    root_commit = _commit_all(repo, "init")
    settings.mapelites_experiment_root_commit = root_commit

    settings.mapelites_archive_threshold_min = float("-inf")
    snapshot_minf = build_experiment_config_snapshot(settings, repo=repo)
    hash_minf = hash_experiment_config(snapshot_minf)

    settings.mapelites_archive_threshold_min = float("inf")
    snapshot_inf = build_experiment_config_snapshot(settings, repo=repo)
    hash_inf = hash_experiment_config(snapshot_inf)

    settings.mapelites_archive_threshold_min = float("nan")
    snapshot_nan = build_experiment_config_snapshot(settings, repo=repo)
    hash_nan = hash_experiment_config(snapshot_nan)

    assert snapshot_minf["mapelites_archive_threshold_min"] == {"__float__": "-inf"}
    assert snapshot_inf["mapelites_archive_threshold_min"] == {"__float__": "inf"}
    assert snapshot_nan["mapelites_archive_threshold_min"] == {"__float__": "nan"}

    assert snapshot_minf != snapshot_inf
    assert snapshot_minf != snapshot_nan
    assert snapshot_inf != snapshot_nan

    assert hash_minf != hash_inf
    assert hash_minf != hash_nan
    assert hash_inf != hash_nan


def test_normalise_remote_url_canonicalises_and_strips_credentials() -> None:
    https = "https://user:pass@example.com:8443/Owner/Repo.git"
    ssh = "git@github.com:Owner/Repo.git"

    https_norm = _normalise_remote_url(https)
    ssh_norm = _normalise_remote_url(ssh)

    # Credentials and query/fragment should be stripped.
    assert "user" not in https_norm
    assert "pass" not in https_norm
    assert https_norm.startswith("https://example.com:8443/")

    # SCP-style URLs are normalised into a proper ssh:// form.
    assert ssh_norm.startswith("ssh://git@github.com/")
    assert ssh_norm.endswith("/Owner/Repo.git")


