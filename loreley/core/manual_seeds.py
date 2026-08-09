"""First-class, idempotent manual seed submission.

Manual seeds are ordinary evolution jobs whose candidate commit already exists.
Workers skip the planning and coding stages, then use the same evaluator,
measurement cache, provenance store, and archive ingestion path as generated
candidates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from git import Repo
from git.exc import BadName, GitCommandError, InvalidGitRepositoryError, NoSuchPathError
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
import yaml

from loreley.config import Settings, resolve_default_island_id
from loreley.core.campaign_program import (
    CampaignProjectionInput,
    apply_campaign_program_projection,
    load_campaign_program_from_repo,
    persist_campaign_program,
)
from loreley.core.contracts import normalize_single_line
from loreley.core.experiments import bootstrap_instance
from loreley.db.base import ensure_database_schema, get_engine, session_scope
from loreley.db.locks import (
    AdvisoryLock,
    release_pg_advisory_lock,
    try_acquire_pg_advisory_lock,
    uuid_to_pg_bigint_lock_key,
)
from loreley.db.models import CandidateCommit, EvolutionJob, JobStatus
from loreley.naming import resolve_experiment_uuid

MANUAL_SEED_MANIFEST_SCHEMA_VERSION = 1


class ManualSeedError(ValueError):
    """Raised when a manual seed manifest cannot be imported safely."""


@dataclass(frozen=True, slots=True)
class ManualSeedSpec:
    """One user-supplied candidate commit and its stable logical key."""

    key: str
    commit: str
    remote_ref: str
    summary: str
    goal: str | None = None
    island_id: str | None = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ManualSeedManifest:
    """Validated manual seed manifest plus content provenance."""

    schema_version: int
    seeds: tuple[ManualSeedSpec, ...]
    sha256: str
    source_name: str


@dataclass(frozen=True, slots=True)
class ManualSeedImportResult:
    """Atomic import result; existing matching submissions are skipped."""

    manifest_sha256: str
    created_job_ids: tuple[str, ...]
    existing_job_ids: tuple[str, ...]

    @property
    def created(self) -> int:
        return len(self.created_job_ids)

    @property
    def existing(self) -> int:
        return len(self.existing_job_ids)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created"] = self.created
        payload["existing"] = self.existing
        return payload


@dataclass(frozen=True, slots=True)
class _ResolvedManualSeed:
    spec: ManualSeedSpec
    ordinal: int
    commit_hash: str
    remote_ref: str
    island_id: str
    submission_key: str
    definition_sha256: str
    goal: str
    constraints: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]
    notes: tuple[str, ...]
    tags: tuple[str, ...]


def load_manual_seed_manifest(path: str | Path) -> ManualSeedManifest:
    """Load a strict YAML or JSON manual seed manifest."""

    source = Path(path).expanduser().resolve()
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise ManualSeedError(f"Could not read manual seed manifest: {exc}") from exc
    try:
        payload = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        raise ManualSeedError(f"Manual seed manifest is not valid YAML/JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ManualSeedError("Manual seed manifest must be a mapping.")

    version = _coerce_schema_version(payload.get("schema_version", payload.get("version", 1)))
    raw_seeds = payload.get("seeds")
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ManualSeedError("Manual seed manifest must contain a non-empty 'seeds' list.")

    seeds = tuple(_parse_seed(item, index=index) for index, item in enumerate(raw_seeds))
    keys = [seed.key for seed in seeds]
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    if duplicate_keys:
        raise ManualSeedError(
            "Manual seed keys must be unique within a manifest: " + ", ".join(duplicate_keys)
        )
    return ManualSeedManifest(
        schema_version=version,
        seeds=seeds,
        sha256=hashlib.sha256(raw).hexdigest(),
        source_name=source.name,
    )


def import_manual_seed_manifest(
    *,
    settings: Settings,
    manifest_path: str | Path,
) -> ManualSeedImportResult:
    """Validate and atomically import archive-eligible manual-seed jobs."""

    ensure_database_schema(settings=settings)
    lock = _acquire_import_lock(settings)
    try:
        repo_root = str(
            settings.scheduler_repo_root or settings.worker_repo_worktree or ""
        ).strip()
        if not repo_root:
            raise ManualSeedError(
                "SCHEDULER_REPO_ROOT or WORKER_REPO_WORKTREE is required."
            )
        _, effective_settings = bootstrap_instance(
            settings=settings,
            repo_root=repo_root,
        )
        return _import_manual_seed_manifest(
            settings=effective_settings,
            manifest_path=manifest_path,
        )
    finally:
        release_pg_advisory_lock(lock)


def _import_manual_seed_manifest(
    *,
    settings: Settings,
    manifest_path: str | Path,
) -> ManualSeedImportResult:
    """Import while holding the experiment's scheduler advisory lock."""

    manifest = load_manual_seed_manifest(manifest_path)
    repo_root, repo = _load_target_repo(settings)
    root_hash = _resolve_root_commit(settings=settings, repo=repo)
    campaign = load_campaign_program_from_repo(repo_root)
    campaign_hash = campaign.snapshot.raw_sha256 if campaign.snapshot is not None else None
    resolved = _resolve_seeds(
        settings=settings,
        repo=repo,
        root_hash=root_hash,
        campaign_snapshot=campaign.snapshot,
        campaign_program_hash=campaign_hash,
        manifest=manifest,
    )
    _reject_duplicate_commits(resolved)

    created: list[str] = []
    existing: list[str] = []
    imported_at = datetime.now(timezone.utc)
    with session_scope() as session:
        total_jobs = int(
            session.execute(select(func.count(EvolutionJob.id))).scalar_one()
        )
        raw_max_total = settings.scheduler_max_total_jobs
        if raw_max_total is None or int(raw_max_total) <= 0:
            raise ManualSeedError(
                "SCHEDULER_MAX_TOTAL_JOBS must be a positive integer before importing seeds."
            )
        max_total_jobs = int(raw_max_total)
        if campaign.snapshot is not None and campaign.raw_markdown is not None:
            persist_campaign_program(
                session=session,
                snapshot=campaign.snapshot,
                raw_markdown=campaign.raw_markdown,
            )
        for seed in resolved:
            row = session.execute(
                select(EvolutionJob).where(
                    EvolutionJob.external_submission_key == seed.submission_key
                )
            ).scalar_one_or_none()
            if row is not None:
                _require_matching_existing_seed(row=row, seed=seed)
                existing.append(str(row.id))
                continue
            prior_seed = session.execute(
                select(EvolutionJob).where(
                    EvolutionJob.job_kind == "manual_seed",
                    EvolutionJob.input_candidate_commit_hash == seed.commit_hash,
                )
            ).scalar_one_or_none()
            if prior_seed is not None:
                raise ManualSeedError(
                    f"Commit {seed.commit_hash[:12]} is already registered as a manual seed "
                    "under a different key or island."
                )
            prior_candidate = session.execute(
                select(CandidateCommit.id).where(CandidateCommit.commit_hash == seed.commit_hash)
            ).scalar_one_or_none()
            if prior_candidate is not None:
                raise ManualSeedError(
                    f"Commit {seed.commit_hash[:12]} is already a campaign candidate; "
                    "manual seeds must introduce a new candidate commit."
                )
            if total_jobs >= max_total_jobs:
                raise ManualSeedError(
                    "Manual seed import would exceed SCHEDULER_MAX_TOTAL_JOBS="
                    f"{max_total_jobs}. Increase the campaign endpoint before importing."
                )
            job = EvolutionJob(
                status=JobStatus.STAGED,
                base_commit_hash=root_hash,
                island_id=seed.island_id,
                inspiration_commit_hashes=[],
                goal=seed.goal,
                constraints=list(seed.constraints),
                acceptance_criteria=list(seed.acceptance_criteria),
                notes=list(seed.notes),
                tags=list(seed.tags),
                iteration_hint="Manual seed: evaluate the supplied commit without agent generation.",
                sampling_strategy="manual_seed",
                sampling_initial_radius=None,
                sampling_radius_used=None,
                sampling_fallback_inspirations=None,
                is_seed_job=True,
                job_kind="manual_seed",
                execution_mode="evaluate_existing",
                input_candidate_commit_hash=seed.commit_hash,
                input_candidate_summary=seed.spec.summary,
                external_submission_key=seed.submission_key,
                input_provenance=_seed_provenance(
                    manifest=manifest,
                    seed=seed,
                    campaign_program_hash=campaign_hash,
                ),
                archive_ingestion_enabled=True,
                campaign_program_hash=campaign_hash,
                priority=int(settings.mapelites_sampler_default_priority) + 1,
                scheduled_at=None,
                created_at=imported_at + timedelta(microseconds=seed.ordinal),
            )
            try:
                with session.begin_nested():
                    session.add(job)
                    session.flush()
            except IntegrityError:
                row = session.execute(
                    select(EvolutionJob).where(
                        EvolutionJob.external_submission_key == seed.submission_key
                    )
                ).scalar_one_or_none()
                if row is None:
                    conflicting_commit = session.execute(
                        select(EvolutionJob).where(
                            EvolutionJob.job_kind == "manual_seed",
                            EvolutionJob.input_candidate_commit_hash == seed.commit_hash,
                        )
                    ).scalar_one_or_none()
                    if conflicting_commit is not None:
                        raise ManualSeedError(
                            f"Commit {seed.commit_hash[:12]} was concurrently registered "
                            "as a different manual seed."
                        )
                    raise
                _require_matching_existing_seed(row=row, seed=seed)
                existing.append(str(row.id))
                continue
            created.append(str(job.id))
            total_jobs += 1

    return ManualSeedImportResult(
        manifest_sha256=manifest.sha256,
        created_job_ids=tuple(created),
        existing_job_ids=tuple(existing),
    )


def _parse_seed(value: object, *, index: int) -> ManualSeedSpec:
    if not isinstance(value, Mapping):
        raise ManualSeedError(f"seeds[{index}] must be a mapping.")
    key = _required_line(value.get("key"), field=f"seeds[{index}].key", limit=128)
    commit = _required_line(
        value.get("commit", value.get("commit_hash")),
        field=f"seeds[{index}].commit",
        limit=256,
    )
    remote_ref = _required_line(
        value.get("remote_ref"),
        field=f"seeds[{index}].remote_ref",
        limit=512,
    )
    _validate_remote_ref(remote_ref, field=f"seeds[{index}].remote_ref")
    summary = _required_text(value.get("summary"), field=f"seeds[{index}].summary", limit=4096)
    goal = _optional_text(value.get("goal"), limit=512)
    island_id = _optional_line(value.get("island_id"), limit=64)
    tags = _string_tuple(value.get("tags", ()), field=f"seeds[{index}].tags", limit=64)
    metadata = value.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ManualSeedError(f"seeds[{index}].metadata must be a mapping.")
    try:
        serialized_metadata = json.loads(json.dumps(dict(metadata), ensure_ascii=True))
    except (TypeError, ValueError) as exc:
        raise ManualSeedError(f"seeds[{index}].metadata must be JSON-serializable.") from exc
    if len(json.dumps(serialized_metadata, separators=(",", ":"))) > 65536:
        raise ManualSeedError(f"seeds[{index}].metadata exceeds 64 KiB.")
    return ManualSeedSpec(
        key=key,
        commit=commit,
        remote_ref=remote_ref,
        summary=summary,
        goal=goal,
        island_id=island_id,
        tags=tags,
        metadata=serialized_metadata,
    )


def _coerce_schema_version(value: object) -> int:
    try:
        version = int(value)
    except (TypeError, ValueError) as exc:
        raise ManualSeedError("Manual seed manifest schema_version must be an integer.") from exc
    if version != MANUAL_SEED_MANIFEST_SCHEMA_VERSION:
        raise ManualSeedError(
            "Unsupported manual seed manifest schema_version="
            f"{version}; expected {MANUAL_SEED_MANIFEST_SCHEMA_VERSION}."
        )
    return version


def _load_target_repo(settings: Settings) -> tuple[Path, Repo]:
    raw = str(settings.scheduler_repo_root or settings.worker_repo_worktree or "").strip()
    if not raw:
        raise ManualSeedError("SCHEDULER_REPO_ROOT or WORKER_REPO_WORKTREE is required.")
    root = Path(raw).expanduser().resolve()
    try:
        repo = Repo(root, search_parent_directories=True)
    except (InvalidGitRepositoryError, NoSuchPathError) as exc:
        raise ManualSeedError(f"Manual seed repository is not a Git repository: {root}") from exc
    worktree = Path(repo.working_tree_dir or root).resolve()
    return worktree, repo


def _resolve_root_commit(*, settings: Settings, repo: Repo) -> str:
    raw = normalize_single_line(str(settings.mapelites_experiment_root_commit or ""))
    if not raw:
        raise ManualSeedError("MAPELITES_EXPERIMENT_ROOT_COMMIT is required.")
    return _resolve_commit(repo=repo, value=raw, field="MAPELITES_EXPERIMENT_ROOT_COMMIT")


def _resolve_seeds(
    *,
    settings: Settings,
    repo: Repo,
    root_hash: str,
    campaign_snapshot: Any | None,
    campaign_program_hash: str | None,
    manifest: ManualSeedManifest,
) -> tuple[_ResolvedManualSeed, ...]:
    configured_islands = tuple(str(item).strip() for item in settings.mapelites_islands if str(item).strip())
    default_island = resolve_default_island_id(settings)
    resolved: list[_ResolvedManualSeed] = []
    for ordinal, spec in enumerate(manifest.seeds, start=1):
        commit_hash = _resolve_commit(repo=repo, value=spec.commit, field=f"seed {spec.key!r}")
        _require_worker_remote_ref(
            repo=repo,
            remote_url=str(settings.worker_repo_remote_url or ""),
            remote_ref=spec.remote_ref,
            commit_hash=commit_hash,
            seed_key=spec.key,
        )
        if commit_hash == root_hash:
            raise ManualSeedError(f"Manual seed {spec.key!r} resolves to the experiment root commit.")
        try:
            repo.git.merge_base("--is-ancestor", root_hash, commit_hash)
        except GitCommandError as exc:
            raise ManualSeedError(
                f"Manual seed {spec.key!r} commit is not a descendant of the experiment root."
            ) from exc
        commit_object = repo.commit(commit_hash)
        parent_hashes = tuple(str(parent.hexsha) for parent in commit_object.parents)
        if parent_hashes != (root_hash,):
            raise ManualSeedError(
                f"Manual seed {spec.key!r} must be a single commit directly on the "
                "experiment root; observed parents="
                f"{[parent[:12] for parent in parent_hashes]}."
            )
        island_id = spec.island_id or default_island
        if configured_islands and island_id not in configured_islands:
            raise ManualSeedError(
                f"Manual seed {spec.key!r} targets unconfigured island {island_id!r}."
            )
        projection = apply_campaign_program_projection(
            CampaignProjectionInput(
                snapshot=campaign_snapshot,
                goal=spec.goal or "",
                constraints=(),
                acceptance_criteria=(),
                notes=(),
                default_goal=(settings.worker_evolution_global_goal or "").strip(),
                preserve_existing_goal=True,
            )
        )
        goal = (
            projection.goal
            or "Evaluate a user-supplied seed candidate against the campaign contract."
        )
        tags = tuple(_manual_seed_tags(spec.tags))
        definition = {
            "acceptance_criteria": list(projection.acceptance_criteria),
            "archive_ingestion_enabled": True,
            "campaign_program_hash": campaign_program_hash or "",
            "commit_hash": commit_hash,
            "constraints": list(projection.constraints),
            "execution_mode": "evaluate_existing",
            "goal": goal,
            "island_id": island_id,
            "job_kind": "manual_seed",
            "key": spec.key,
            "metadata": dict(spec.metadata or {}),
            "notes": list(projection.notes),
            "remote_ref": spec.remote_ref,
            "summary": spec.summary,
            "tags": list(tags),
        }
        definition_sha256 = _canonical_sha256(definition)
        submission_key = _canonical_sha256(
            {
                "experiment_id": str(settings.experiment_id or ""),
                "kind": "manual_seed",
                "seed_key": spec.key,
            }
        )
        resolved.append(
            _ResolvedManualSeed(
                spec=spec,
                ordinal=ordinal,
                commit_hash=commit_hash,
                remote_ref=spec.remote_ref,
                island_id=island_id,
                submission_key=submission_key,
                definition_sha256=definition_sha256,
                goal=goal,
                constraints=tuple(projection.constraints),
                acceptance_criteria=tuple(projection.acceptance_criteria),
                notes=tuple(projection.notes),
                tags=tags,
            )
        )
    return tuple(resolved)


def _resolve_commit(*, repo: Repo, value: str, field: str) -> str:
    try:
        commit = repo.commit(value)
    except (BadName, ValueError) as exc:
        raise ManualSeedError(f"{field} does not resolve to a Git commit: {value!r}.") from exc
    return str(commit.hexsha)


def _reject_duplicate_commits(seeds: Sequence[_ResolvedManualSeed]) -> None:
    by_commit: dict[str, list[str]] = {}
    for seed in seeds:
        by_commit.setdefault(seed.commit_hash, []).append(seed.spec.key)
    duplicates = {commit: keys for commit, keys in by_commit.items() if len(keys) > 1}
    if duplicates:
        detail = "; ".join(f"{commit[:12]}: {', '.join(keys)}" for commit, keys in duplicates.items())
        raise ManualSeedError(f"Manual seed commits must be unique within a manifest: {detail}")


def _require_matching_existing_seed(*, row: EvolutionJob, seed: _ResolvedManualSeed) -> None:
    provenance = dict(getattr(row, "input_provenance", {}) or {})
    matches = (
        str(getattr(row, "job_kind", "") or "") == "manual_seed"
        and bool(getattr(row, "is_seed_job", False))
        and str(getattr(row, "execution_mode", "") or "") == "evaluate_existing"
        and str(getattr(row, "input_candidate_commit_hash", "") or "") == seed.commit_hash
        and str(getattr(row, "island_id", "") or "") == seed.island_id
        and bool(getattr(row, "archive_ingestion_enabled", False))
        and str(getattr(row, "input_candidate_summary", "") or "") == seed.spec.summary
        and str(getattr(row, "goal", "") or "") == seed.goal
        and tuple(getattr(row, "constraints", ()) or ()) == seed.constraints
        and tuple(getattr(row, "acceptance_criteria", ()) or ())
        == seed.acceptance_criteria
        and tuple(getattr(row, "notes", ()) or ()) == seed.notes
        and tuple(getattr(row, "tags", ()) or ()) == seed.tags
        and str(provenance.get("definition_sha256") or "") == seed.definition_sha256
    )
    if not matches:
        raise ManualSeedError(
            f"Manual seed key {seed.spec.key!r} was already imported with a different definition. "
            "Use a new key for a materially different seed."
        )


def _seed_provenance(
    *,
    manifest: ManualSeedManifest,
    seed: _ResolvedManualSeed,
    campaign_program_hash: str | None,
) -> dict[str, Any]:
    return {
        "source_type": "manual_seed_manifest",
        "schema_version": manifest.schema_version,
        "manifest_sha256": manifest.sha256,
        "manifest_name": manifest.source_name,
        "seed_key": seed.spec.key,
        "seed_ordinal": seed.ordinal,
        "remote_name": "origin",
        "remote_ref": seed.remote_ref,
        "definition_sha256": seed.definition_sha256,
        "campaign_program_hash": campaign_program_hash,
        "metadata": dict(seed.spec.metadata or {}),
    }


def _manual_seed_tags(tags: Sequence[str]) -> list[str]:
    values = ["manual_seed", *tags]
    return list(dict.fromkeys(values))[:32]


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _acquire_import_lock(settings: Settings) -> AdvisoryLock:
    experiment_id = resolve_experiment_uuid(settings.experiment_id)
    lock = try_acquire_pg_advisory_lock(
        engine=get_engine(),
        key=uuid_to_pg_bigint_lock_key(experiment_id),
    )
    if lock is None:
        raise ManualSeedError(
            "Manual seeds are a pre-start operation; stop the experiment scheduler "
            "before importing them."
        )
    return lock


def _validate_remote_ref(remote_ref: str, *, field: str) -> None:
    valid_shape = re.fullmatch(
        r"refs/(?:heads|tags)/[A-Za-z0-9][A-Za-z0-9._/-]*",
        remote_ref,
    )
    if not valid_shape or any(
        token in remote_ref
        for token in ("..", "@{", "\\", " ", "//", "~", "^", ":", "?", "[")
    ) or remote_ref.endswith(("/", ".", ".lock")):
        raise ManualSeedError(
            f"{field} must be a full, fetchable Git ref such as refs/heads/loreley-seeds/foo."
        )


def _require_worker_remote_ref(
    *,
    repo: Repo,
    remote_url: str,
    remote_ref: str,
    commit_hash: str,
    seed_key: str,
) -> None:
    if not remote_url.strip():
        raise ManualSeedError("WORKER_REPO_REMOTE_URL is required for manual seeds.")
    try:
        output = repo.git.ls_remote("--refs", remote_url, remote_ref)
    except GitCommandError as exc:
        raise ManualSeedError(
            f"Could not verify worker remote ref for manual seed {seed_key!r}."
        ) from exc
    rows = [line.split() for line in output.splitlines() if line.strip()]
    matches = [parts for parts in rows if len(parts) == 2 and parts[1] == remote_ref]
    if len(matches) != 1:
        raise ManualSeedError(
            f"Manual seed {seed_key!r} remote_ref {remote_ref!r} was not found exactly once."
        )
    observed = matches[0][0].lower()
    if observed != commit_hash.lower():
        raise ManualSeedError(
            f"Manual seed {seed_key!r} remote_ref does not resolve to the declared commit."
        )


def _required_line(value: object, *, field: str, limit: int) -> str:
    normalized = _optional_line(value, limit=limit)
    if normalized is None:
        raise ManualSeedError(f"{field} is required.")
    return normalized


def _required_text(value: object, *, field: str, limit: int) -> str:
    normalized = _optional_text(value, limit=limit)
    if normalized is None:
        raise ManualSeedError(f"{field} is required.")
    return normalized


def _optional_line(value: object, *, limit: int) -> str | None:
    if value is None:
        return None
    normalized = normalize_single_line(str(value))
    if len(normalized) > limit:
        raise ManualSeedError(f"Manifest value exceeds {limit} characters.")
    return normalized or None


def _optional_text(value: object, *, limit: int) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if len(normalized) > limit:
        raise ManualSeedError(f"Manifest value exceeds {limit} characters.")
    return normalized or None


def _string_tuple(value: object, *, field: str, limit: int) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ManualSeedError(f"{field} must be a list of strings.")
    result: list[str] = []
    for index, item in enumerate(value):
        normalized = _optional_line(item, limit=limit)
        if normalized is None:
            raise ManualSeedError(f"{field}[{index}] must be a non-empty string.")
        result.append(normalized)
    return tuple(dict.fromkeys(result))


__all__ = [
    "MANUAL_SEED_MANIFEST_SCHEMA_VERSION",
    "ManualSeedError",
    "ManualSeedImportResult",
    "ManualSeedManifest",
    "ManualSeedSpec",
    "import_manual_seed_manifest",
    "load_manual_seed_manifest",
]
