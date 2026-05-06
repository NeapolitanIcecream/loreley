"""Cold-path artifact store for the evolution worker.

Artifacts are large, audit/debug oriented payloads (prompts, raw outputs, logs).
They must not be embedded in primary DB rows. Instead, store them on disk and
persist only their paths in the database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import shutil
from typing import Any, Literal, Mapping, Sequence
from uuid import UUID

from loguru import logger

from loreley.config import Settings, get_settings
from loreley.core.contracts import clamp_text, normalize_single_line
from loreley.core.worker.coding import CodingAgentResponse
from loreley.core.worker.evaluator import (
    ArtifactAgentProjection,
    ArtifactValidationWarning,
    ArtifactVisibility,
    EvaluationArtifact,
    EvaluationDiagnostic,
    EvaluationOutcome,
    EvaluationResult,
)
from loreley.core.worker.planning import PlanningAgentResponse

log = logger.bind(module="worker.artifacts")

__all__ = [
    "FixedJobArtifactPaths",
    "FailureJobArtifactWriteRequest",
    "JobArtifactWriteResult",
    "JobArtifactWriteRequest",
    "MaterializedEvaluationArtifact",
    "resolve_worker_instance_id",
    "worker_runtime_metadata",
    "write_failure_job_artifacts",
    "write_job_artifacts",
]


@dataclass(frozen=True, slots=True)
class FixedJobArtifactPaths:
    planning_prompt_path: str | None = None
    planning_raw_output_path: str | None = None
    planning_plan_json_path: str | None = None
    coding_prompt_path: str | None = None
    coding_raw_output_path: str | None = None
    coding_execution_json_path: str | None = None
    evaluation_json_path: str | None = None
    evaluation_logs_path: str | None = None

    def as_dict(self) -> dict[str, str]:
        return {
            key: value
            for key, value in {
                "planning_prompt_path": self.planning_prompt_path,
                "planning_raw_output_path": self.planning_raw_output_path,
                "planning_plan_json_path": self.planning_plan_json_path,
                "coding_prompt_path": self.coding_prompt_path,
                "coding_raw_output_path": self.coding_raw_output_path,
                "coding_execution_json_path": self.coding_execution_json_path,
                "evaluation_json_path": self.evaluation_json_path,
                "evaluation_logs_path": self.evaluation_logs_path,
            }.items()
            if value
        }


@dataclass(frozen=True, slots=True)
class MaterializedEvaluationArtifact:
    key: str
    kind: str
    mime_type: str
    label: str | None
    summary: str | None
    visibility: ArtifactVisibility
    agent_projection: ArtifactAgentProjection
    storage_path: str | None
    size_bytes: int | None
    sha256: str | None
    diagnostics: tuple[EvaluationDiagnostic, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def manifest(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "kind": self.kind,
            "mime_type": self.mime_type,
            "label": self.label,
            "visibility": self.visibility,
            "agent_projection": self.agent_projection,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "diagnostic_count": len(self.diagnostics),
        }


@dataclass(frozen=True, slots=True)
class JobArtifactWriteResult:
    fixed: FixedJobArtifactPaths
    evaluation_artifacts: tuple[MaterializedEvaluationArtifact, ...] = ()
    validation_warnings: tuple[ArtifactValidationWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class JobArtifactWriteRequest:
    job_id: UUID
    plan: PlanningAgentResponse
    coding: CodingAgentResponse
    evaluation: EvaluationResult
    base_commit_hash: str
    candidate_commit_hash: str
    commit_message: str
    run_token: UUID | None = None
    worktree: Path | None = None
    settings: Settings | None = None


@dataclass(frozen=True, slots=True)
class FailureJobArtifactWriteRequest:
    job_id: UUID
    base_commit_hash: str
    candidate_commit_hash: str | None
    message: str
    outcome: EvaluationOutcome
    run_token: UUID | None = None
    plan: PlanningAgentResponse | None = None
    coding: CodingAgentResponse | None = None
    worktree: Path | None = None
    settings: Settings | None = None


_MIME_EXTENSIONS: dict[str, str] = {
    "text/plain": ".txt",
    "application/json": ".json",
    "image/svg+xml": ".svg",
    "image/png": ".png",
    "text/html": ".html",
    "application/octet-stream": ".bin",
}


def _resolve_artifacts_dir(
    settings: Settings,
    job_id: UUID,
    *,
    run_token: UUID | None = None,
) -> Path:
    if settings.logs_base_dir:
        base_dir = Path(settings.logs_base_dir).expanduser()
    else:
        base_dir = Path.cwd()
    from loreley.naming import safe_namespace_or_none

    exp_ns = safe_namespace_or_none(getattr(settings, "experiment_id", None))
    logs_root = (base_dir / "logs" / exp_ns) if exp_ns else (base_dir / "logs")
    root = logs_root / "worker" / "artifacts" / str(job_id)
    if run_token is not None:
        root = root / str(run_token)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def resolve_worker_instance_id() -> str:
    """Return a stable worker instance identifier for logs, leases, and artifacts."""

    return (os.getenv("LORELEY_WORKER_INSTANCE_ID") or "").strip() or f"pid-{os.getpid()}"


def worker_runtime_metadata() -> dict[str, Any]:
    """Return non-sensitive runtime metadata for diagnostics."""

    instance_id = resolve_worker_instance_id()
    return {
        "instance_id": instance_id,
        "pid": os.getpid(),
    }


def _materialize_evaluation_artifacts(
    *,
    root: Path,
    evaluation: EvaluationResult,
    worktree: Path | None,
    settings: Settings,
) -> tuple[tuple[MaterializedEvaluationArtifact, ...], tuple[ArtifactValidationWarning, ...]]:
    warnings: list[ArtifactValidationWarning] = list(evaluation.artifact_validation_warnings or ())
    if not settings.worker_evaluation_artifacts_enabled:
        return (), tuple(warnings)
    if not evaluation.artifacts:
        return (), tuple(warnings)

    artifacts: list[MaterializedEvaluationArtifact] = []
    allowed_mime_types = _normalized_mime_set(settings.worker_evaluation_artifact_allowed_mime_types)
    max_bytes = max(0, int(settings.worker_evaluation_artifact_max_bytes))
    resolved_worktree = Path(worktree).expanduser().resolve() if worktree is not None else Path.cwd().resolve()

    for index, artifact in enumerate(evaluation.artifacts):
        materialized, artifact_warnings = _materialize_one_artifact(
            root=root,
            artifact=artifact,
            artifact_index=index,
            worktree=resolved_worktree,
            allowed_mime_types=allowed_mime_types,
            max_bytes=max_bytes,
        )
        warnings.extend(artifact_warnings)
        if materialized is not None:
            artifacts.append(materialized)
    return tuple(artifacts), tuple(warnings)


def _materialize_one_artifact(
    *,
    root: Path,
    artifact: EvaluationArtifact,
    artifact_index: int,
    worktree: Path,
    allowed_mime_types: set[str],
    max_bytes: int,
) -> tuple[MaterializedEvaluationArtifact | None, tuple[ArtifactValidationWarning, ...]]:
    has_path = artifact.path is not None
    has_inline = artifact.inline_payload is not None
    if has_path and has_inline:
        return _invalid_artifact_result(
            artifact=artifact,
            artifact_index=artifact_index,
            code="multiple_payload_sources",
            message="Artifact declared both path and inline_payload; raw bytes were not stored.",
            input_ref=f"artifacts[{artifact_index}].inline_payload",
        )

    if artifact.mime_type not in allowed_mime_types:
        return _invalid_artifact_result(
            artifact=artifact,
            artifact_index=artifact_index,
            code="unsupported_mime_type",
            message="Artifact MIME type is not allowed for storage.",
            input_ref=f"artifacts[{artifact_index}].mime_type",
        )

    if not has_path and not has_inline:
        if _has_metadata_payload(artifact):
            return _metadata_only_artifact(artifact), ()
        return _invalid_artifact_result(
            artifact=artifact,
            artifact_index=artifact_index,
            code="missing_payload",
            message="Artifact has no path, inline payload, summary, or diagnostics.",
            input_ref=f"artifacts[{artifact_index}]",
        )

    artifact_dir = root / "evaluation_artifacts" / artifact.key
    artifact_dir.mkdir(parents=True, exist_ok=True)
    target = artifact_dir / _artifact_filename(artifact)

    if has_path:
        written_path, warning = _copy_path_payload(
            artifact=artifact,
            artifact_index=artifact_index,
            worktree=worktree,
            target=target,
            max_bytes=max_bytes,
        )
    else:
        written_path, warning = _write_inline_payload(
            artifact=artifact,
            artifact_index=artifact_index,
            target=target,
            max_bytes=max_bytes,
        )
    if warning is not None:
        return _artifact_warning_result(artifact, warning)
    assert written_path is not None
    return _stored_materialized_artifact(artifact, written_path), ()


def _invalid_artifact_result(
    *,
    artifact: EvaluationArtifact,
    artifact_index: int,
    code: str,
    message: str,
    input_ref: str,
) -> tuple[MaterializedEvaluationArtifact | None, tuple[ArtifactValidationWarning, ...]]:
    warning = _warning(
        artifact=artifact,
        artifact_index=artifact_index,
        code=code,
        action="metadata_only" if _has_metadata_payload(artifact) else "skipped",
        message=message,
        input_ref=input_ref,
    )
    return _artifact_warning_result(artifact, warning)


def _artifact_warning_result(
    artifact: EvaluationArtifact,
    warning: ArtifactValidationWarning,
) -> tuple[MaterializedEvaluationArtifact | None, tuple[ArtifactValidationWarning, ...]]:
    materialized = _metadata_only_artifact(artifact) if _has_metadata_payload(artifact) else None
    return materialized, (warning,)


def _copy_path_payload(
    *,
    artifact: EvaluationArtifact,
    artifact_index: int,
    worktree: Path,
    target: Path,
    max_bytes: int,
) -> tuple[Path | None, ArtifactValidationWarning | None]:
    source, path_warning = _resolve_safe_source_path(
        raw_path=artifact.path,
        worktree=worktree,
        artifact=artifact,
        artifact_index=artifact_index,
    )
    if path_warning is not None:
        return None, path_warning
    assert source is not None
    if source.stat().st_size > max_bytes:
        return None, _warning(
            artifact=artifact,
            artifact_index=artifact_index,
            code="artifact_too_large",
            action="metadata_only" if _has_metadata_payload(artifact) else "skipped",
            message="Artifact raw payload exceeds the configured size limit.",
            input_ref=f"artifacts[{artifact_index}].path",
        )
    shutil.copyfile(source, target)
    return target, None


def _write_inline_payload(
    *,
    artifact: EvaluationArtifact,
    artifact_index: int,
    target: Path,
    max_bytes: int,
) -> tuple[Path | None, ArtifactValidationWarning | None]:
    payload = _inline_payload_bytes(artifact.inline_payload, artifact.mime_type)
    if len(payload) > max_bytes:
        return None, _warning(
            artifact=artifact,
            artifact_index=artifact_index,
            code="artifact_too_large",
            action="metadata_only" if _has_metadata_payload(artifact) else "skipped",
            message="Artifact inline payload exceeds the configured size limit.",
            input_ref=f"artifacts[{artifact_index}].inline_payload",
        )
    target.write_bytes(payload)
    return target, None


def _stored_materialized_artifact(
    artifact: EvaluationArtifact,
    target: Path,
) -> MaterializedEvaluationArtifact:
    return MaterializedEvaluationArtifact(
        key=artifact.key,
        kind=artifact.kind,
        mime_type=artifact.mime_type,
        label=artifact.label,
        summary=artifact.summary,
        visibility=artifact.visibility,
        agent_projection=artifact.agent_projection,
        storage_path=str(target),
        size_bytes=target.stat().st_size,
        sha256=_sha256_file(target),
        diagnostics=tuple(artifact.diagnostics),
        metadata=dict(artifact.metadata or {}),
    )


def _metadata_only_artifact(artifact: EvaluationArtifact) -> MaterializedEvaluationArtifact:
    return MaterializedEvaluationArtifact(
        key=artifact.key,
        kind=artifact.kind,
        mime_type=artifact.mime_type,
        label=artifact.label,
        summary=artifact.summary,
        visibility=artifact.visibility,
        agent_projection="manifest" if artifact.agent_projection == "path" else artifact.agent_projection,
        storage_path=None,
        size_bytes=None,
        sha256=None,
        diagnostics=tuple(artifact.diagnostics),
        metadata=dict(artifact.metadata or {}),
    )


def _resolve_safe_source_path(
    *,
    raw_path: Path | str | None,
    worktree: Path,
    artifact: EvaluationArtifact,
    artifact_index: int,
) -> tuple[Path | None, ArtifactValidationWarning | None]:
    if raw_path is None:
        return None, _warning(
            artifact=artifact,
            artifact_index=artifact_index,
            code="missing_path",
            action="metadata_only" if _has_metadata_payload(artifact) else "skipped",
            message="Artifact path is missing.",
            input_ref=f"artifacts[{artifact_index}].path",
        )
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = worktree / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        return None, _warning(
            artifact=artifact,
            artifact_index=artifact_index,
            code="missing_path",
            action="metadata_only" if _has_metadata_payload(artifact) else "skipped",
            message="Artifact path does not resolve to a readable file.",
            input_ref=f"artifacts[{artifact_index}].path",
        )
    if resolved.is_dir():
        return None, _warning(
            artifact=artifact,
            artifact_index=artifact_index,
            code="directory_path",
            action="metadata_only" if _has_metadata_payload(artifact) else "skipped",
            message="Artifact path points to a directory.",
            input_ref=f"artifacts[{artifact_index}].path",
        )
    if not _is_relative_to(resolved, worktree):
        return None, _warning(
            artifact=artifact,
            artifact_index=artifact_index,
            code="path_escape",
            action="metadata_only" if _has_metadata_payload(artifact) else "skipped",
            message="Artifact path escapes the evaluation worktree.",
            input_ref=f"artifacts[{artifact_index}].path",
        )
    if not resolved.is_file():
        return None, _warning(
            artifact=artifact,
            artifact_index=artifact_index,
            code="not_file",
            action="metadata_only" if _has_metadata_payload(artifact) else "skipped",
            message="Artifact path does not point to a regular file.",
            input_ref=f"artifacts[{artifact_index}].path",
        )
    return resolved, None


def _inline_payload_bytes(payload: Any, mime_type: str) -> bytes:
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    if isinstance(payload, str):
        return payload.encode("utf-8")
    if isinstance(payload, Mapping) or (
        isinstance(payload, Sequence) and not isinstance(payload, str | bytes | bytearray)
    ):
        return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    if mime_type.startswith("text/"):
        return str(payload or "").encode("utf-8")
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _artifact_filename(artifact: EvaluationArtifact) -> str:
    extension = _MIME_EXTENSIONS.get(artifact.mime_type)
    if extension is None:
        guessed = mimetypes.guess_extension(artifact.mime_type)
        extension = guessed or ".bin"
    return f"{artifact.key}{extension}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_mime_set(values: Sequence[str]) -> set[str]:
    return {
        normalize_single_line(str(value)).lower()
        for value in values
        if normalize_single_line(str(value)).lower()
    }


def _has_metadata_payload(artifact: EvaluationArtifact) -> bool:
    return bool(artifact.summary or artifact.diagnostics)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _warning(
    *,
    artifact: EvaluationArtifact,
    artifact_index: int,
    code: str,
    action: Literal["skipped", "downgraded", "metadata_only"],
    message: str,
    input_ref: str,
) -> ArtifactValidationWarning:
    return ArtifactValidationWarning(
        artifact_index=artifact_index,
        artifact_key=artifact.key,
        code=clamp_text(normalize_single_line(code).lower(), 64),
        action=action,
        message=clamp_text(normalize_single_line(message), 240),
        input_ref=clamp_text(normalize_single_line(input_ref), 128),
    )


def write_job_artifacts(request: JobArtifactWriteRequest) -> JobArtifactWriteResult:
    """Write fixed and evaluator-declared artifacts to worker-managed storage."""

    settings = request.settings or get_settings()
    root = _resolve_artifacts_dir(settings, request.job_id, run_token=request.run_token)
    worker = worker_runtime_metadata()
    fixed = _write_fixed_job_artifacts(root=root, request=request, worker=worker)
    evaluation_artifacts, validation_warnings = _write_evaluation_result_artifacts(
        root=root,
        request=request,
        settings=settings,
        worker=worker,
    )
    log.info(
        "Wrote fixed_artifacts={} evaluation_artifacts={} validation_warnings={} for job {}",
        len(fixed.as_dict()),
        len(evaluation_artifacts),
        len(validation_warnings),
        request.job_id,
    )
    return JobArtifactWriteResult(
        fixed=fixed,
        evaluation_artifacts=evaluation_artifacts,
        validation_warnings=validation_warnings,
    )


def write_failure_job_artifacts(request: FailureJobArtifactWriteRequest) -> JobArtifactWriteResult:
    """Write available worker artifacts for a failed candidate attempt."""

    settings = request.settings or get_settings()
    root = _resolve_artifacts_dir(settings, request.job_id, run_token=request.run_token)
    worker = worker_runtime_metadata()
    fixed = _write_failure_fixed_job_artifacts(root=root, request=request, worker=worker)
    evaluation_artifacts, validation_warnings = _write_failure_outcome_artifacts(
        root=root,
        request=request,
        settings=settings,
        worker=worker,
    )
    log.info(
        "Wrote failure fixed_artifacts={} evaluation_artifacts={} validation_warnings={} for job {}",
        len(fixed.as_dict()),
        len(evaluation_artifacts),
        len(validation_warnings),
        request.job_id,
    )
    return JobArtifactWriteResult(
        fixed=fixed,
        evaluation_artifacts=evaluation_artifacts,
        validation_warnings=validation_warnings,
    )


def _write_failure_fixed_job_artifacts(
    *,
    root: Path,
    request: FailureJobArtifactWriteRequest,
    worker: Mapping[str, Any],
) -> FixedJobArtifactPaths:
    planning_prompt_path: str | None = None
    planning_raw_path: str | None = None
    planning_plan_path: str | None = None
    coding_prompt_path: str | None = None
    coding_raw_path: str | None = None
    coding_execution_path: str | None = None

    if request.plan is not None:
        planning_prompt = root / "planning_prompt.txt"
        planning_raw = root / "planning_raw_output.txt"
        planning_plan = root / "planning_plan.json"
        _write_text(planning_prompt, request.plan.prompt)
        _write_text(planning_raw, request.plan.raw_output)
        _write_json(
            planning_plan,
            {
                "job_id": str(request.job_id),
                "base_commit_hash": request.base_commit_hash,
                "candidate_commit_hash": request.candidate_commit_hash,
                "worker": worker,
                "plan": request.plan.plan.as_dict(),
                "backend": {
                    "command": list(request.plan.command),
                    "stderr": request.plan.stderr,
                    "attempts": request.plan.attempts,
                    "duration_seconds": request.plan.duration_seconds,
                },
            },
        )
        planning_prompt_path = str(planning_prompt)
        planning_raw_path = str(planning_raw)
        planning_plan_path = str(planning_plan)

    if request.coding is not None:
        coding_prompt = root / "coding_prompt.txt"
        coding_raw = root / "coding_raw_output.txt"
        coding_exec = root / "coding_execution.json"
        _write_text(coding_prompt, request.coding.prompt)
        _write_text(coding_raw, request.coding.raw_output)
        _write_json(
            coding_exec,
            {
                "job_id": str(request.job_id),
                "base_commit_hash": request.base_commit_hash,
                "candidate_commit_hash": request.candidate_commit_hash,
                "worker": worker,
                "report": request.coding.report.as_dict(),
                "backend": {
                    "command": list(request.coding.command),
                    "stderr": request.coding.stderr,
                    "attempts": request.coding.attempts,
                    "duration_seconds": request.coding.duration_seconds,
                },
            },
        )
        coding_prompt_path = str(coding_prompt)
        coding_raw_path = str(coding_raw)
        coding_execution_path = str(coding_exec)

    return FixedJobArtifactPaths(
        planning_prompt_path=planning_prompt_path,
        planning_raw_output_path=planning_raw_path,
        planning_plan_json_path=planning_plan_path,
        coding_prompt_path=coding_prompt_path,
        coding_raw_output_path=coding_raw_path,
        coding_execution_json_path=coding_execution_path,
        evaluation_json_path=str(root / "evaluation.json"),
        evaluation_logs_path=str(root / "evaluation_logs.txt"),
    )


def _write_failure_outcome_artifacts(
    *,
    root: Path,
    request: FailureJobArtifactWriteRequest,
    settings: Settings,
    worker: Mapping[str, Any],
) -> tuple[tuple[MaterializedEvaluationArtifact, ...], tuple[ArtifactValidationWarning, ...]]:
    evaluation_json = root / "evaluation.json"
    evaluation_logs = root / "evaluation_logs.txt"
    synthetic = EvaluationResult(
        summary=request.message or request.outcome.outcome_kind,
        artifacts=request.outcome.artifacts,
        artifact_validation_warnings=request.outcome.artifact_validation_warnings,
    )
    evaluation_artifacts, validation_warnings = _materialize_evaluation_artifacts(
        root=root,
        evaluation=synthetic,
        worktree=request.worktree,
        settings=settings,
    )
    _write_json(
        evaluation_json,
        {
            "job_id": str(request.job_id),
            "base_commit_hash": request.base_commit_hash,
            "candidate_commit_hash": request.candidate_commit_hash,
            "worker": worker,
            "message": request.message,
            "outcome": request.outcome.as_dict(),
            "evaluation_artifacts": [artifact.manifest() for artifact in evaluation_artifacts],
            "artifact_validation_warnings": [warning.as_dict() for warning in validation_warnings],
        },
    )
    failure_summary = ""
    if request.outcome.failure is not None:
        failure_summary = request.outcome.failure.safe_failure_summary
    _write_text(evaluation_logs, failure_summary or request.message)
    return evaluation_artifacts, validation_warnings


def _write_fixed_job_artifacts(
    *,
    root: Path,
    request: JobArtifactWriteRequest,
    worker: Mapping[str, Any],
) -> FixedJobArtifactPaths:
    planning_prompt = root / "planning_prompt.txt"
    planning_raw = root / "planning_raw_output.txt"
    planning_plan = root / "planning_plan.json"
    _write_text(planning_prompt, request.plan.prompt)
    _write_text(planning_raw, request.plan.raw_output)
    _write_json(
        planning_plan,
        {
            "job_id": str(request.job_id),
            "base_commit_hash": request.base_commit_hash,
            "candidate_commit_hash": request.candidate_commit_hash,
            "commit_message": request.commit_message,
            "worker": worker,
            "plan": request.plan.plan.as_dict(),
            "backend": {
                "command": list(request.plan.command),
                "stderr": request.plan.stderr,
                "attempts": request.plan.attempts,
                "duration_seconds": request.plan.duration_seconds,
            },
        },
    )

    coding_prompt = root / "coding_prompt.txt"
    coding_raw = root / "coding_raw_output.txt"
    coding_exec = root / "coding_execution.json"
    _write_text(coding_prompt, request.coding.prompt)
    _write_text(coding_raw, request.coding.raw_output)
    _write_json(
        coding_exec,
        {
            "job_id": str(request.job_id),
            "base_commit_hash": request.base_commit_hash,
            "candidate_commit_hash": request.candidate_commit_hash,
            "commit_message": request.commit_message,
            "worker": worker,
            "report": request.coding.report.as_dict(),
            "backend": {
                "command": list(request.coding.command),
                "stderr": request.coding.stderr,
                "attempts": request.coding.attempts,
                "duration_seconds": request.coding.duration_seconds,
            },
        },
    )
    return FixedJobArtifactPaths(
        planning_prompt_path=str(planning_prompt),
        planning_raw_output_path=str(planning_raw),
        planning_plan_json_path=str(planning_plan),
        coding_prompt_path=str(coding_prompt),
        coding_raw_output_path=str(coding_raw),
        coding_execution_json_path=str(coding_exec),
        evaluation_json_path=str(root / "evaluation.json"),
        evaluation_logs_path=str(root / "evaluation_logs.txt"),
    )


def _write_evaluation_result_artifacts(
    *,
    root: Path,
    request: JobArtifactWriteRequest,
    settings: Settings,
    worker: Mapping[str, Any],
) -> tuple[tuple[MaterializedEvaluationArtifact, ...], tuple[ArtifactValidationWarning, ...]]:
    evaluation_json = root / "evaluation.json"
    evaluation_logs = root / "evaluation_logs.txt"
    evaluation_artifacts, validation_warnings = _materialize_evaluation_artifacts(
        root=root,
        evaluation=request.evaluation,
        worktree=request.worktree,
        settings=settings,
    )
    _write_json(
        evaluation_json,
        {
            "job_id": str(request.job_id),
            "base_commit_hash": request.base_commit_hash,
            "candidate_commit_hash": request.candidate_commit_hash,
            "commit_message": request.commit_message,
            "worker": worker,
            "summary": request.evaluation.summary,
            "metrics": [metric.as_dict() for metric in request.evaluation.metrics],
            "tests_executed": list(request.evaluation.tests_executed),
            "logs": list(request.evaluation.logs),
            "extra": dict(request.evaluation.extra or {}),
            "evaluation_artifacts": [artifact.manifest() for artifact in evaluation_artifacts],
            "artifact_validation_warnings": [warning.as_dict() for warning in validation_warnings],
        },
    )
    _write_text(evaluation_logs, "\n".join(str(line) for line in request.evaluation.logs))
    return evaluation_artifacts, validation_warnings
