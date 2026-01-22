from __future__ import annotations

"""Helpers for single-tenant repository and instance bootstrap."""

from dataclasses import dataclass
import hashlib
import re
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from git import Repo
from git.exc import BadName, GitCommandError, InvalidGitRepositoryError, NoSuchPathError
from loguru import logger
from rich.console import Console

from loreley.config import Settings, get_settings
from loreley.core.map_elites.repository_files import ROOT_IGNORE_FILES
from loreley.db.base import INSTANCE_SCHEMA_VERSION, session_scope
from loreley.db.models import InstanceMetadata

console = Console()
log = logger.bind(module="core.experiments")

__all__ = [
    "ExperimentError",
    "RepositoryIdentity",
    "bootstrap_instance",
]


class ExperimentError(RuntimeError):
    """Raised when the repository/instance context cannot be resolved."""


@dataclass(frozen=True, slots=True)
class RepositoryIdentity:
    """Resolved identity for the repository backing this instance."""

    slug: str
    canonical_origin: str | None
    root_path: str


def _normalise_remote_url(raw: str) -> str:
    """Return a canonical remote URL without credentials."""

    url = raw.strip()
    if not url:
        return ""

    if "://" not in url and "@" in url and ":" in url.split("@", 1)[1]:
        user_host, path = url.split(":", 1)
        url = f"ssh://{user_host}/{path}"

    parsed = urlparse(url)
    scheme = parsed.scheme or "ssh"
    host = parsed.hostname or ""
    path = parsed.path or ""

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

    return urlunparse((scheme, netloc, path, "", "", ""))


def _build_slug_from_source(source: str) -> str:
    """Normalise an arbitrary source string into a repository slug."""

    text = source.strip()
    if not text:
        return "default"

    candidate = text
    if "://" not in candidate and "@" in candidate and ":" in candidate.split("@", 1)[1]:
        user_host, path = candidate.split(":", 1)
        candidate = f"ssh://{user_host}/{path}"

    parsed = urlparse(candidate)
    host = parsed.hostname
    path = parsed.path or ""

    if host:
        if path.endswith(".git"):
            path = path[: -len(".git")]
        base = f"{host}{path}"
    else:
        base = text
        if base.endswith(".git"):
            base = base[: -len(".git")]

    base = base.lower()
    slug = re.sub(r"[^a-z0-9._/-]+", "-", base).strip("-")
    return slug or "default"


def _resolve_repository_identity(*, repo: Repo, root: Path) -> RepositoryIdentity:
    origin_url: str | None = None
    try:
        origin = getattr(repo.remotes, "origin", None)
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
    return RepositoryIdentity(
        slug=slug,
        canonical_origin=canonical_origin or None,
        root_path=str(root),
    )


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


def _root_commit_matches(stored: str, configured: str) -> bool:
    stored = (stored or "").strip()
    configured = (configured or "").strip()
    if not stored or not configured:
        return False
    return stored.startswith(configured) or configured.startswith(stored)


def _validate_instance_metadata(
    *,
    settings: Settings,
    repository: RepositoryIdentity,
    canonical_root: str,
) -> None:
    from loreley.naming import resolve_experiment_identity

    identity = resolve_experiment_identity(settings.experiment_id)
    with session_scope() as session:
        meta = session.get(InstanceMetadata, 1)
        if meta is None:
            raise ExperimentError(
                "Instance metadata is missing. "
                "Reset the database schema with `uv run loreley reset-db --yes`.",
            )
        if int(meta.schema_version or 0) != INSTANCE_SCHEMA_VERSION:
            raise ExperimentError(
                "Instance metadata schema_version mismatch. "
                "Reset the database schema with `uv run loreley reset-db --yes`.",
            )
        if str(meta.experiment_id_raw or "").strip() != identity.raw:
            raise ExperimentError(
                "EXPERIMENT_ID does not match the database marker. "
                "Reset the database schema with `uv run loreley reset-db --yes`.",
            )
        if str(meta.experiment_uuid or "") != str(identity.uuid):
            raise ExperimentError(
                "EXPERIMENT_ID UUID mapping does not match the database marker. "
                "Reset the database schema with `uv run loreley reset-db --yes`.",
            )
        if not _root_commit_matches(str(meta.root_commit_hash or ""), canonical_root):
            raise ExperimentError(
                "MAPELITES_EXPERIMENT_ROOT_COMMIT does not match the database marker. "
                "Reset the database schema with `uv run loreley reset-db --yes`.",
            )

        updated = False
        if str(meta.root_commit_hash or "") != canonical_root:
            meta.root_commit_hash = canonical_root
            updated = True
        if repository.slug and meta.repository_slug != repository.slug:
            meta.repository_slug = repository.slug
            updated = True
        if repository.canonical_origin and meta.repository_canonical_origin != repository.canonical_origin:
            meta.repository_canonical_origin = repository.canonical_origin
            updated = True
        if updated:
            console.log(
                "[cyan]Updated instance metadata[/] experiment_id={} repo={}".format(
                    identity.raw,
                    repository.slug,
                ),
            )
            log.info(
                "Updated instance metadata experiment_id={} repository_slug={}",
                identity.raw,
                repository.slug,
            )


def bootstrap_instance(
    *,
    settings: Settings | None = None,
    repo_root: Path | str | None = None,
) -> tuple[RepositoryIdentity, Settings]:
    """Resolve repository identity and pin root ignore rules for the scheduler."""

    settings = settings or get_settings()
    root_candidate = repo_root or settings.scheduler_repo_root or settings.worker_repo_worktree
    root = Path(root_candidate).expanduser().resolve()

    try:
        repo_obj = Repo(root)
    except (InvalidGitRepositoryError, NoSuchPathError) as exc:
        message = f"Scheduler repo {root} is not a git repository."
        log.error("{}: {}", message, exc)
        raise ExperimentError(message) from exc

    repository = _resolve_repository_identity(repo=repo_obj, root=root)
    root_ref = (settings.mapelites_experiment_root_commit or "").strip()
    if not root_ref:
        raise ExperimentError(
            "MAPELITES_EXPERIMENT_ROOT_COMMIT is required for scheduler startup.",
        )
    canonical_root = _ensure_commit_available(repo=repo_obj, commit_hash=root_ref)
    ignore_text = _load_root_ignore_text_from_commit(repo=repo_obj, commit_hash=canonical_root)
    ignore_sha = hashlib.sha256(ignore_text.encode("utf-8")).hexdigest()

    settings = settings.model_copy(
        update={
            "mapelites_experiment_root_commit": canonical_root,
            "mapelites_repo_state_ignore_text": ignore_text,
            "mapelites_repo_state_ignore_sha256": ignore_sha,
        }
    )

    _validate_instance_metadata(
        settings=settings,
        repository=repository,
        canonical_root=canonical_root,
    )

    console.log(
        "[bold cyan]Using instance[/] experiment={} repo={}".format(
            settings.experiment_id,
            repository.slug,
        ),
    )
    log.info(
        "Using instance experiment_id={} repository_slug={}",
        settings.experiment_id,
        repository.slug,
    )
    return repository, settings


