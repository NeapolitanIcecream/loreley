from __future__ import annotations

from loreley.core.experiments import (
    _hash_experiment_config,
    _build_slug_from_source,
    _normalise_remote_url,
)


def test_build_slug_from_source_basic() -> None:
    slug = _build_slug_from_source("https://github.com/Owner/Repo.git")
    assert slug == "github.com/owner/repo"


def test_experiment_config_hash_depends_only_on_root_commit() -> None:
    hash_1 = _hash_experiment_config(root_commit="a" * 40)
    hash_2 = _hash_experiment_config(root_commit="b" * 40)
    assert hash_1 != hash_2

    # Same input => stable hash.
    hash_1_again = _hash_experiment_config(root_commit="a" * 40)
    assert hash_1_again == hash_1


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


