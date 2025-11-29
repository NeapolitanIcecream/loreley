from __future__ import annotations

from loreley.config import Settings
from loreley.core.experiments import (
    _build_slug_from_source,
    build_experiment_config_snapshot,
    hash_experiment_config,
    _normalise_remote_url,
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


def test_experiment_config_snapshot_includes_mapelites_and_excludes_unrelated(
    settings: Settings,
) -> None:
    snapshot = build_experiment_config_snapshot(settings)

    # Map-Elites and evaluator knobs should be present.
    assert "mapelites_preprocess_max_files" in snapshot
    assert "worker_evaluator_timeout_seconds" in snapshot

    # Unrelated environment fields should not be part of the experiment key.
    assert "app_name" not in snapshot
    assert "environment" not in snapshot


def test_experiment_config_hash_changes_when_experiment_knob_changes(
    settings: Settings,
) -> None:
    snapshot_1 = build_experiment_config_snapshot(settings)
    hash_1 = hash_experiment_config(snapshot_1)

    # Tweak a MAP-Elites setting that should affect experiment identity.
    settings.mapelites_sampler_inspiration_count += 1
    snapshot_2 = build_experiment_config_snapshot(settings)
    hash_2 = hash_experiment_config(snapshot_2)

    assert snapshot_1 != snapshot_2
    assert hash_1 != hash_2


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


