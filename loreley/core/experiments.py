from __future__ import annotations

"""Helpers for deriving repository and experiment dimensions.

This module is responsible for:
  - Normalising a git repository into a stable Repository row.
  - Deriving an Experiment from the current Settings configuration.
"""

import hashlib
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

from git import Repo
from git.exc import BadName, GitCommandError, InvalidGitRepositoryError, NoSuchPathError
from loguru import logger
from rich.console import Console
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from loreley.config import Settings, get_settings
from loreley.core.map_elites.repository_files import ROOT_IGNORE_FILES
from loreley.db.base import session_scope
from loreley.db.models import Experiment, Repository

console = Console()
log = logger.bind(module="core.experiments")

__all__ = [
    "ExperimentError",
    "canonicalise_repository",
    "derive_experiment",
    "get_or_create_experiment",
]


class ExperimentError(RuntimeError):
    """Raised when the repository/experiment context cannot be resolved."""

def _hash_experiment_config(*, root_commit: str) -> str:
    """Return a stable experiment hash for the current env-only settings model."""

    seed = f"root_commit:{(root_commit or '').strip()}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _normalise_remote_url(raw: str) -> str:
    """Return a canonical remote URL without credentials.

    Handles both standard URLs (https://host/owner/repo.git) and scp-style
    SSH URLs (git@github.com:owner/repo.git).
    """

    url = raw.strip()
    if not url:
        return ""

    # Convert scp-style SSH URL (git@github.com:owner/repo.git) into a proper
    # URL so that urlparse can handle it. We preserve the username for SSH
    # remotes (git@) but strip credentials for HTTPS.
    if "://" not in url and "@" in url and ":" in url.split("@", 1)[1]:
        user_host, path = url.split(":", 1)
        url = f"ssh://{user_host}/{path}"

    parsed = urlparse(url)
    scheme = parsed.scheme or "ssh"
    host = parsed.hostname or ""
    path = parsed.path or ""

    # Decide whether to keep the username in the canonical form.
    # - For HTTPS/HTTP we drop any username to avoid leaking credentials.
    # - For SSH-style remotes we keep the username (e.g. git@github.com).
    username = parsed.username or ""
    if scheme in ("http", "https"):
        userinfo = ""
    else:
        userinfo = f"{username}@" if username else ""

    netloc = host
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    if userinfo:
        netloc = f"{userinfo}{netloc}"

    # Drop password, query and fragment for storage and hashing.
    return urlunparse((scheme, netloc, path, "", "", ""))


def _build_slug_from_source(source: str) -> str:
    """Normalise an arbitrary source string into a repository slug."""

    text = source.strip()
    if not text:
        return "default"

    # Try to interpret the source as a URL (including scp-style SSH).
    candidate = text
    if "://" not in candidate and "@" in candidate and ":" in candidate.split("@", 1)[1]:
        user_host, path = candidate.split(":", 1)
        candidate = f"ssh://{user_host}/{path}"

    parsed = urlparse(candidate)
    host = parsed.hostname
    path = parsed.path or ""

    if host:
        # URL-like input: build slug from host + path.
        if path.endswith(".git"):
            path = path[: -len(".git")]
        base = f"{host}{path}"
    else:
        # Fallback: treat as a plain path or arbitrary string.
        base = text
        if base.endswith(".git"):
            base = base[: -len(".git")]

    base = base.lower()
    slug = re.sub(r"[^a-z0-9._/-]+", "-", base).strip("-")
    return slug or "default"


def canonicalise_repository(
    *,
    settings: Settings | None = None,
    repo_root: Path | str | None = None,
    repo: Repo | None = None,
) -> Repository:
    """Resolve or create a Repository row for the given git worktree.

    The caller is responsible for ensuring that the DB is reachable.
    """

    settings = settings or get_settings()
    root = Path(repo_root or settings.worker_repo_worktree).expanduser().resolve()

    try:
        repo_obj = repo or Repo(root)
    except (InvalidGitRepositoryError, NoSuchPathError) as exc:
        message = f"Path {root} is not a valid git repository."
        log.error("{}: {}", message, exc)
        raise ExperimentError(message) from exc

    origin_url: str | None = None
    try:
        origin = getattr(repo_obj.remotes, "origin", None)
        if origin is not None:
            origin_url = str(origin.url)
    except Exception:  # pragma: no cover - defensive
        origin_url = None

    canonical_origin = _normalise_remote_url(origin_url) if origin_url else ""
    if canonical_origin:
        parsed = urlparse(canonical_origin)
        host = parsed.hostname or "local"
        source = f"{host}{parsed.path}"
    else:
        source = str(root)

    slug = _build_slug_from_source(source)

    extra: dict[str, Any] = {
        "canonical_origin": canonical_origin or None,
        "root_path": str(root),
        "remotes": [
            {"name": remote.name, "url": _normalise_remote_url(str(remote.url))}
            for remote in getattr(repo_obj, "remotes", [])
        ],
    }

    try:
        with session_scope() as session:
            stmt = select(Repository).where(Repository.slug == slug)
            existing = session.execute(stmt).scalar_one_or_none()
            if existing:
                # Best-effort refresh of metadata; do not fail the call.
                updated = False
                if canonical_origin and existing.remote_url != canonical_origin:
                    existing.remote_url = canonical_origin
                    updated = True
                if not existing.root_path:
                    existing.root_path = str(root)
                    updated = True
                if extra and existing.extra != extra:
                    # Merge rather than overwrite to avoid losing prior context.
                    merged = dict(existing.extra or {})
                    merged.update(extra)
                    existing.extra = merged
                    updated = True
                if updated:
                    console.log(
                        "[cyan]Updated repository metadata[/] slug={} path={}".format(
                            existing.slug,
                            existing.root_path,
                        ),
                    )
                return existing

            repo_row = Repository(
                slug=slug,
                remote_url=canonical_origin or None,
                root_path=str(root),
                extra=extra,
            )
            session.add(repo_row)
            session.flush()
            console.log(
                "[bold green]Registered repository[/] slug={} path={}".format(
                    repo_row.slug,
                    repo_row.root_path,
                ),
            )
            log.info(
                "Registered repository slug={} remote_url={} root_path={}",
                repo_row.slug,
                repo_row.remote_url,
                repo_row.root_path,
            )
            return repo_row
    except SQLAlchemyError as exc:  # pragma: no cover - DB failure handling
        log.error("Failed to resolve repository {}: {}", slug, exc)
        raise ExperimentError(f"Failed to resolve repository {slug}: {exc}") from exc

def _ensure_commit_available(*, repo: Repo, commit_hash: str) -> str:
    """Return canonical hash for commit, fetching from remotes when needed."""

    commit = (commit_hash or "").strip()
    if not commit:
        raise ExperimentError("MAPELITES_EXPERIMENT_ROOT_COMMIT is required.")
    try:
        return str(getattr(repo.commit(commit), "hexsha", "") or "").strip()
    except BadName:
        pass

    console.log(f"[yellow]Fetching missing commit[/] {commit}")
    try:
        repo.git.fetch("--all", "--tags")
        return str(getattr(repo.commit(commit), "hexsha", "") or "").strip()
    except GitCommandError as exc:
        raise ExperimentError(f"Cannot fetch commit {commit}: {exc}") from exc
    except BadName as exc:
        raise ExperimentError(f"Commit {commit} not found after fetch.") from exc


def _load_root_ignore_text_from_commit(*, repo: Repo, commit_hash: str) -> str:
    """Return pinned root ignore rules by reading ignore files from a commit."""

    commit = (commit_hash or "").strip()
    if not commit:
        return ""
    chunks: list[str] = []
    for filename in ROOT_IGNORE_FILES:
        try:
            chunks.append(repo.git.show(f"{commit}:{filename}"))
        except (GitCommandError, BadName):
            chunks.append("")
    return "\n".join(chunks).strip()


def derive_experiment(settings: Settings, repository: Repository, *, repo: Repo) -> Experiment:
    """Return or create an Experiment row for the given repository/settings."""

    root_ref = (settings.mapelites_experiment_root_commit or "").strip()
    if not root_ref:
        raise ExperimentError("MAPELITES_EXPERIMENT_ROOT_COMMIT is required to derive an experiment.")
    canonical_root = _ensure_commit_available(repo=repo, commit_hash=root_ref)
    if not canonical_root:
        raise ExperimentError(f"Cannot resolve root commit {root_ref!r}.")

    config_hash = _hash_experiment_config(root_commit=canonical_root)

    try:
        with session_scope() as session:
            stmt = select(Experiment).where(
                Experiment.repository_id == repository.id,
                Experiment.config_hash == config_hash,
            )
            existing = session.execute(stmt).scalar_one_or_none()
            if existing:
                return existing

            name = f"{repository.slug}-{config_hash[:8]}"
            experiment = Experiment(
                repository_id=repository.id,
                config_hash=config_hash,
                name=name,
                status="active",
            )
            session.add(experiment)
            session.flush()
            console.log(
                "[bold green]Created experiment[/] id={} repo={} hash={}".format(
                    experiment.id,
                    repository.slug,
                    config_hash[:8],
                ),
            )
            log.info(
                "Created experiment id={} repository_id={} hash={}",
                experiment.id,
                experiment.repository_id,
                config_hash,
            )
            return experiment
    except SQLAlchemyError as exc:  # pragma: no cover - DB failure handling
        log.error(
            "Failed to derive experiment for repository {}: {}",
            repository.slug,
            exc,
        )
        raise ExperimentError(
            f"Failed to derive experiment for repository {repository.slug}: {exc}",
        ) from exc


def get_or_create_experiment(
    *,
    settings: Settings | None = None,
    repo_root: Path | str | None = None,
) -> tuple[Repository, Experiment, Settings]:
    """Resolve the Repository and Experiment for the current process.

    This helper is intended to be called once during scheduler startup so that
    all jobs and MAP-Elites state produced by that scheduler share the same
    experiment identifier.
    """

    settings = settings or get_settings()
    root_candidate = repo_root or settings.scheduler_repo_root or settings.worker_repo_worktree
    root = Path(root_candidate).expanduser().resolve()

    try:
        repo_obj = Repo(root)
    except (InvalidGitRepositoryError, NoSuchPathError) as exc:
        message = f"Scheduler repo {root} is not a git repository."
        log.error("{}: {}", message, exc)
        raise ExperimentError(message) from exc

    repository = canonicalise_repository(settings=settings, repo_root=root, repo=repo_obj)
    root_ref = (settings.mapelites_experiment_root_commit or "").strip()
    if not root_ref:
        raise ExperimentError(
            "MAPELITES_EXPERIMENT_ROOT_COMMIT is required for scheduler startup.",
        )
    canonical_root = _ensure_commit_available(repo=repo_obj, commit_hash=root_ref)
    ignore_text = _load_root_ignore_text_from_commit(repo=repo_obj, commit_hash=canonical_root)
    ignore_sha = hashlib.sha256(ignore_text.encode("utf-8")).hexdigest()

    # Pin root ignore rules in memory for the full scheduler process lifetime.
    # This deliberately does not persist settings inside the database.
    settings = settings.model_copy(
        update={
            "mapelites_experiment_root_commit": canonical_root,
            "mapelites_repo_state_ignore_text": ignore_text,
            "mapelites_repo_state_ignore_sha256": ignore_sha,
        }
    )

    experiment = derive_experiment(settings, repository, repo=repo_obj)

    console.log(
        "[bold cyan]Using experiment[/] id={} repo={} hash={}".format(
            experiment.id,
            repository.slug,
            experiment.config_hash[:8],
        ),
    )
    log.info(
        "Using experiment id={} repository_slug={} hash={}",
        experiment.id,
        repository.slug,
        experiment.config_hash,
    )
    return repository, experiment, settings


