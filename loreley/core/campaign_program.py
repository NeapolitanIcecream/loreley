from __future__ import annotations

"""Campaign program parsing, projection, and persistence helpers."""

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

from loguru import logger
from sqlalchemy import select

from loreley.core.contracts import clamp_text, normalize_single_line

log = logger.bind(module="campaign_program")

DEFAULT_CAMPAIGN_PROGRAM_PATH = "loreley.program.md"
CAMPAIGN_PROGRAM_SCHEMA_VERSION = 1

_HEADING_RE = re.compile(r"^(?P<marks>#{1,6})\s+(?P<title>.+?)\s*#*\s*$")
_SECTION_ALIASES: dict[str, str] = {
    "goal": "Goal",
    "primary metric": "Primary metric",
    "correctness gates": "Correctness gates",
    "editable scope": "Editable scope",
    "protected scope": "Protected scope",
    "evaluation budget": "Evaluation budget",
    "complexity policy": "Complexity policy",
    "failure policy": "Failure policy",
    "logging policy": "Logging policy",
}
_RECOGNIZED_SECTION_ORDER: tuple[str, ...] = (
    "Goal",
    "Primary metric",
    "Correctness gates",
    "Editable scope",
    "Protected scope",
    "Evaluation budget",
    "Complexity policy",
    "Failure policy",
    "Logging policy",
)
_DEFAULT_PROTECTED_SCOPE = DEFAULT_CAMPAIGN_PROGRAM_PATH
_DIRECTION_ALIASES: dict[str, str] = {
    "higher": "higher_is_better",
    "higher is better": "higher_is_better",
    "higher_is_better": "higher_is_better",
    "maximize": "higher_is_better",
    "maximise": "higher_is_better",
    "max": "higher_is_better",
    "lower": "lower_is_better",
    "lower is better": "lower_is_better",
    "lower_is_better": "lower_is_better",
    "minimize": "lower_is_better",
    "minimise": "lower_is_better",
    "min": "lower_is_better",
}
_KV_RE = re.compile(r"^\s*[-*]?\s*([A-Za-z][A-Za-z0-9 _-]{0,40})\s*:\s*(.+?)\s*$")
_BULLET_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)(?P<text>.+?)\s*$")


@dataclass(frozen=True, slots=True)
class PrimaryMetric:
    """Normalized primary metric declared by a campaign program."""

    name: str | None = None
    direction: str | None = None
    unit: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "name": self.name,
            "direction": self.direction,
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class CampaignProgramSection:
    """Raw Markdown section retained for provenance and metadata."""

    title: str
    canonical_title: str | None
    body: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "canonical_title": self.canonical_title,
            "body": self.body,
        }

    def as_metadata_dict(self) -> dict[str, Any]:
        return {
            "title": clamp_text(normalize_single_line(self.title), 128),
            "canonical_title": self.canonical_title,
        }


@dataclass(frozen=True, slots=True)
class CampaignProgramSnapshot:
    """Normalized, bounded campaign program snapshot."""

    schema_version: int
    source_path: str
    raw_sha256: str
    normalized_sha256: str | None
    title: str | None
    goal: str | None
    primary_metric: PrimaryMetric | None
    correctness_gates: tuple[str, ...] = ()
    editable_scope: tuple[str, ...] = ()
    protected_scope: tuple[str, ...] = ()
    evaluation_budget: tuple[str, ...] = ()
    complexity_policy: tuple[str, ...] = ()
    failure_policy: tuple[str, ...] = ()
    logging_policy: tuple[str, ...] = ()
    recognized_sections: tuple[str, ...] = ()
    unknown_sections: tuple[CampaignProgramSection, ...] = ()
    parse_warnings: tuple[dict[str, Any], ...] = ()

    @property
    def hash(self) -> str:
        return self.raw_sha256

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_path": self.source_path,
            "raw_sha256": self.raw_sha256,
            "normalized_sha256": self.normalized_sha256,
            "title": self.title,
            "goal": self.goal,
            "primary_metric": self.primary_metric.as_dict() if self.primary_metric else None,
            "correctness_gates": list(self.correctness_gates),
            "editable_scope": list(self.editable_scope),
            "protected_scope": list(self.protected_scope),
            "evaluation_budget": list(self.evaluation_budget),
            "complexity_policy": list(self.complexity_policy),
            "failure_policy": list(self.failure_policy),
            "logging_policy": list(self.logging_policy),
            "recognized_sections": list(self.recognized_sections),
            "unknown_sections": [section.as_dict() for section in self.unknown_sections],
            "parse_warnings": [dict(item) for item in self.parse_warnings],
        }

    def as_payload_dict(self) -> dict[str, Any]:
        """Return the bounded evaluator/artifact projection of the snapshot."""

        payload = self.as_dict()
        payload["unknown_sections"] = [
            section.as_metadata_dict() for section in self.unknown_sections
        ]
        return payload


@dataclass(frozen=True, slots=True)
class CampaignProgramLoadResult:
    """Result of reading a campaign program from a repository."""

    snapshot: CampaignProgramSnapshot | None
    raw_markdown: str | None = None
    source_path: Path | None = None

    @property
    def found(self) -> bool:
        return self.snapshot is not None


@dataclass(frozen=True, slots=True)
class CampaignJobProjection:
    """Bounded job fields derived from a campaign program."""

    goal: str | None = None
    constraints: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProjectedJobFields:
    """Final job fields after caller values and campaign projection are merged."""

    goal: str | None
    constraints: list[str]
    acceptance_criteria: list[str]
    notes: list[str]


def parse_campaign_program(
    raw_bytes: bytes,
    *,
    source_path: str = DEFAULT_CAMPAIGN_PROGRAM_PATH,
) -> CampaignProgramSnapshot:
    """Parse recognized sections from a campaign program Markdown document."""

    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    raw_markdown = raw_bytes.decode("utf-8", errors="replace")
    title, sections = _parse_markdown_sections(raw_markdown)

    recognized: dict[str, list[str]] = {name: [] for name in _RECOGNIZED_SECTION_ORDER}
    unknown_sections: list[CampaignProgramSection] = []
    recognized_titles: list[str] = []
    warnings: list[dict[str, Any]] = []

    for section in sections:
        canonical = _canonical_section_title(section.title)
        if canonical is None:
            unknown_sections.append(section)
            continue
        if canonical not in recognized_titles:
            recognized_titles.append(canonical)
        recognized[canonical].append(section.body)

    if not recognized_titles:
        warnings.append(
            _warning(
                "no_recognized_sections",
                "No recognized campaign program sections were found.",
            )
        )

    primary_metric, metric_warnings = _parse_primary_metric(
        "\n\n".join(recognized["Primary metric"])
    )
    warnings.extend(metric_warnings)

    protected_scope = _parse_item_section(recognized["Protected scope"], max_items=19)
    if _DEFAULT_PROTECTED_SCOPE not in protected_scope:
        protected_scope = (*protected_scope, _DEFAULT_PROTECTED_SCOPE)

    snapshot_without_hash = {
        "schema_version": CAMPAIGN_PROGRAM_SCHEMA_VERSION,
        "source_path": source_path,
        "title": clamp_text(normalize_single_line(title or ""), 256) or None,
        "goal": _parse_goal(recognized["Goal"]),
        "primary_metric": primary_metric.as_dict() if primary_metric else None,
        "correctness_gates": list(_parse_item_section(recognized["Correctness gates"], max_items=20)),
        "editable_scope": list(_parse_scope_section(recognized["Editable scope"], warnings=warnings)),
        "protected_scope": list(_parse_scope_patterns(protected_scope, "Protected scope", warnings)),
        "evaluation_budget": list(_parse_item_section(recognized["Evaluation budget"], max_items=20)),
        "complexity_policy": list(_parse_item_section(recognized["Complexity policy"], max_items=20)),
        "failure_policy": list(_parse_item_section(recognized["Failure policy"], max_items=20)),
        "logging_policy": list(_parse_item_section(recognized["Logging policy"], max_items=20)),
        "recognized_sections": recognized_titles,
        "unknown_sections": [section.as_dict() for section in unknown_sections],
        "parse_warnings": warnings,
    }
    normalized_sha256 = hashlib.sha256(
        json.dumps(snapshot_without_hash, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    return CampaignProgramSnapshot(
        schema_version=CAMPAIGN_PROGRAM_SCHEMA_VERSION,
        source_path=source_path,
        raw_sha256=raw_sha256,
        normalized_sha256=normalized_sha256,
        title=snapshot_without_hash["title"],
        goal=snapshot_without_hash["goal"],
        primary_metric=primary_metric,
        correctness_gates=tuple(snapshot_without_hash["correctness_gates"]),
        editable_scope=tuple(snapshot_without_hash["editable_scope"]),
        protected_scope=tuple(snapshot_without_hash["protected_scope"]),
        evaluation_budget=tuple(snapshot_without_hash["evaluation_budget"]),
        complexity_policy=tuple(snapshot_without_hash["complexity_policy"]),
        failure_policy=tuple(snapshot_without_hash["failure_policy"]),
        logging_policy=tuple(snapshot_without_hash["logging_policy"]),
        recognized_sections=tuple(recognized_titles),
        unknown_sections=tuple(unknown_sections),
        parse_warnings=tuple(warnings),
    )


def load_campaign_program_from_repo(
    repo_root: Path,
    *,
    relative_path: str = DEFAULT_CAMPAIGN_PROGRAM_PATH,
) -> CampaignProgramLoadResult:
    """Load the default campaign program from a repository root if present."""

    repo = Path(repo_root).expanduser().resolve()
    source_path = repo / relative_path
    if not source_path.exists():
        log.debug(
            "Campaign program file not found repo_root={} source_path={}",
            repo,
            relative_path,
        )
        return CampaignProgramLoadResult(snapshot=None, raw_markdown=None, source_path=source_path)
    try:
        raw_bytes = source_path.read_bytes()
    except OSError as exc:
        log.warning(
            "Campaign program file could not be read repo_root={} source_path={} error={}",
            repo,
            relative_path,
            exc,
        )
        return CampaignProgramLoadResult(snapshot=None, raw_markdown=None, source_path=source_path)
    snapshot = parse_campaign_program(raw_bytes, source_path=relative_path)
    log.debug(
        "Campaign program parsed hash={} source_path={} recognized_sections={} warnings={}",
        snapshot.raw_sha256,
        relative_path,
        list(snapshot.recognized_sections),
        len(snapshot.parse_warnings),
    )
    return CampaignProgramLoadResult(
        snapshot=snapshot,
        raw_markdown=raw_bytes.decode("utf-8", errors="replace"),
        source_path=source_path,
    )


def persist_campaign_program(
    *,
    session: Any,
    snapshot: CampaignProgramSnapshot,
    raw_markdown: str,
) -> None:
    """Upsert a campaign program snapshot into the content-addressed table."""

    from loreley.db.models import CampaignProgram

    row = session.get(CampaignProgram, snapshot.raw_sha256)
    if row is None:
        row = CampaignProgram(
            hash=snapshot.raw_sha256,
            schema_version=snapshot.schema_version,
            source_path=snapshot.source_path,
            title=snapshot.title,
            raw_markdown=raw_markdown,
            normalized_snapshot=snapshot.as_dict(),
            recognized_sections=list(snapshot.recognized_sections),
            parse_warnings=[dict(item) for item in snapshot.parse_warnings],
        )
        session.add(row)
        return
    row.schema_version = snapshot.schema_version
    row.source_path = snapshot.source_path
    row.title = snapshot.title
    row.raw_markdown = raw_markdown
    row.normalized_snapshot = snapshot.as_dict()
    row.recognized_sections = list(snapshot.recognized_sections)
    row.parse_warnings = [dict(item) for item in snapshot.parse_warnings]


def load_campaign_program_snapshot_by_hash(
    *,
    session: Any,
    program_hash: str | None,
) -> CampaignProgramSnapshot | None:
    """Load a normalized campaign program snapshot from the DB."""

    value = normalize_single_line(str(program_hash or ""))
    if not value:
        return None
    from loreley.db.models import CampaignProgram

    row = session.execute(
        select(CampaignProgram).where(CampaignProgram.hash == value)
    ).scalar_one_or_none()
    if row is None:
        return None
    return campaign_program_snapshot_from_mapping(row.normalized_snapshot)


def campaign_program_snapshot_from_mapping(
    payload: Mapping[str, Any] | None,
) -> CampaignProgramSnapshot | None:
    """Rehydrate a snapshot from a DB JSON payload."""

    if not isinstance(payload, Mapping):
        return None
    primary_payload = payload.get("primary_metric")
    primary_metric = None
    if isinstance(primary_payload, Mapping):
        primary_metric = PrimaryMetric(
            name=_optional_line(primary_payload.get("name"), 128),
            direction=_optional_line(primary_payload.get("direction"), 32),
            unit=_optional_line(primary_payload.get("unit"), 32),
        )
    unknown_sections = tuple(
        CampaignProgramSection(
            title=str(item.get("title") or ""),
            canonical_title=(
                str(item.get("canonical_title"))
                if item.get("canonical_title") is not None
                else None
            ),
            body=str(item.get("body") or ""),
        )
        for item in _mapping_sequence(payload.get("unknown_sections"))
    )
    return CampaignProgramSnapshot(
        schema_version=int(payload.get("schema_version") or CAMPAIGN_PROGRAM_SCHEMA_VERSION),
        source_path=str(payload.get("source_path") or DEFAULT_CAMPAIGN_PROGRAM_PATH),
        raw_sha256=str(payload.get("raw_sha256") or ""),
        normalized_sha256=_optional_line(payload.get("normalized_sha256"), 64),
        title=_optional_line(payload.get("title"), 256),
        goal=_optional_line(payload.get("goal"), 512),
        primary_metric=primary_metric,
        correctness_gates=_bounded_string_tuple(payload.get("correctness_gates"), limit=200),
        editable_scope=_bounded_string_tuple(payload.get("editable_scope"), limit=200),
        protected_scope=_bounded_string_tuple(payload.get("protected_scope"), limit=200),
        evaluation_budget=_bounded_string_tuple(payload.get("evaluation_budget"), limit=200),
        complexity_policy=_bounded_string_tuple(payload.get("complexity_policy"), limit=200),
        failure_policy=_bounded_string_tuple(payload.get("failure_policy"), limit=200),
        logging_policy=_bounded_string_tuple(payload.get("logging_policy"), limit=200),
        recognized_sections=_bounded_string_tuple(payload.get("recognized_sections"), limit=64),
        unknown_sections=unknown_sections,
        parse_warnings=tuple(dict(item) for item in _mapping_sequence(payload.get("parse_warnings"))),
    )


def project_campaign_program(
    snapshot: CampaignProgramSnapshot | None,
) -> CampaignJobProjection:
    """Project a normalized program into bounded EvolutionJob fields."""

    if snapshot is None:
        return CampaignJobProjection()

    constraints: list[str] = []
    constraints.extend(_prefixed_items("Correctness gate", snapshot.correctness_gates))
    constraints.extend(_prefixed_items("Editable scope", snapshot.editable_scope))
    constraints.extend(_prefixed_items("Protected scope", snapshot.protected_scope))
    constraints.extend(_prefixed_items("Evaluation budget", snapshot.evaluation_budget))
    constraints.extend(_prefixed_items("Failure policy", snapshot.failure_policy))

    acceptance: list[str] = []
    metric = _primary_metric_line(snapshot.primary_metric)
    if metric:
        acceptance.append(metric)
    acceptance.extend(_prefixed_items("Correctness gate", snapshot.correctness_gates))

    notes: list[str] = []
    notes.extend(_prefixed_items("Complexity policy", snapshot.complexity_policy))
    notes.extend(_prefixed_items("Logging policy", snapshot.logging_policy))
    if snapshot.title:
        notes.append(clamp_text(f"Campaign program: {snapshot.title}", 200))
    notes.append(clamp_text(f"Campaign program hash: {snapshot.raw_sha256[:12]}", 200))

    return CampaignJobProjection(
        goal=snapshot.goal,
        constraints=tuple(_bounded_dedupe(constraints, max_items=20, max_chars=200)),
        acceptance_criteria=tuple(_bounded_dedupe(acceptance, max_items=20, max_chars=200)),
        notes=tuple(_bounded_dedupe(notes, max_items=20, max_chars=200)),
    )


def apply_campaign_program_projection(
    *,
    snapshot: CampaignProgramSnapshot | None,
    goal: str | None,
    constraints: Sequence[str] | None = None,
    acceptance_criteria: Sequence[str] | None = None,
    notes: Sequence[str] | None = None,
    default_goal: str | None = None,
    preserve_existing_goal: bool = False,
) -> ProjectedJobFields:
    """Merge caller-supplied job fields with program-derived defaults."""

    projection = project_campaign_program(snapshot)
    existing_goal = normalize_single_line(goal or "") or None
    default_goal_norm = normalize_single_line(default_goal or "") or None
    if projection.goal and (
        not existing_goal
        or (not preserve_existing_goal and existing_goal == default_goal_norm)
    ):
        existing_goal = projection.goal

    existing_constraints = _bounded_dedupe(constraints or (), max_items=20, max_chars=200)
    if not existing_constraints and projection.constraints:
        existing_constraints = list(projection.constraints)

    existing_acceptance = _bounded_dedupe(
        acceptance_criteria or (),
        max_items=20,
        max_chars=200,
    )
    if not existing_acceptance and projection.acceptance_criteria:
        existing_acceptance = list(projection.acceptance_criteria)

    existing_notes = _bounded_dedupe(notes or (), max_items=20, max_chars=200)
    if not existing_notes and projection.notes:
        existing_notes = list(projection.notes)

    return ProjectedJobFields(
        goal=clamp_text(existing_goal or "", 512) or None,
        constraints=existing_constraints,
        acceptance_criteria=existing_acceptance,
        notes=existing_notes,
    )


def campaign_program_artifact_payload(
    snapshot: CampaignProgramSnapshot | None,
) -> dict[str, Any] | None:
    """Return prompt-safe campaign program metadata for worker artifacts."""

    if snapshot is None:
        return None
    return {
        "hash": snapshot.raw_sha256,
        "source_path": snapshot.source_path,
        "title": snapshot.title,
        "normalized_snapshot": snapshot.as_payload_dict(),
    }


def campaign_program_evaluator_payload(
    snapshot: CampaignProgramSnapshot | None,
) -> dict[str, Any] | None:
    """Return evaluator-facing campaign program metadata without raw Markdown."""

    if snapshot is None:
        return None
    return {
        "hash": snapshot.raw_sha256,
        "source_path": snapshot.source_path,
        "title": snapshot.title,
        "snapshot": snapshot.as_payload_dict(),
    }


def _parse_markdown_sections(raw_markdown: str) -> tuple[str | None, list[CampaignProgramSection]]:
    title: str | None = None
    current_title: str | None = None
    current_body: list[str] = []
    sections: list[CampaignProgramSection] = []

    def _flush() -> None:
        nonlocal current_title, current_body
        if current_title is None:
            current_body = []
            return
        canonical = _canonical_section_title(current_title)
        sections.append(
            CampaignProgramSection(
                title=current_title,
                canonical_title=canonical,
                body="\n".join(current_body).strip(),
            )
        )
        current_title = None
        current_body = []

    for line in raw_markdown.splitlines():
        match = _HEADING_RE.match(line)
        if match:
            heading = normalize_single_line(match.group("title").strip())
            if len(match.group("marks")) == 1 and title is None:
                title = heading
            _flush()
            current_title = heading
            continue
        if current_title is not None:
            current_body.append(line)
    _flush()
    return title, sections


def _canonical_section_title(title: str) -> str | None:
    normalized = normalize_single_line(title).strip().lower()
    normalized = re.sub(r"[:：]+$", "", normalized).strip()
    return _SECTION_ALIASES.get(normalized)


def _parse_goal(bodies: Sequence[str]) -> str | None:
    text = "\n\n".join(body for body in bodies if body.strip()).strip()
    return clamp_text(normalize_single_line(text), 512) or None


def _parse_primary_metric(body: str) -> tuple[PrimaryMetric | None, list[dict[str, Any]]]:
    if not body.strip():
        return None, []
    warnings: list[dict[str, Any]] = []
    values: dict[str, str] = {}
    prose: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _KV_RE.match(stripped)
        if not match:
            prose.append(stripped)
            continue
        key = normalize_single_line(match.group(1)).lower().replace(" ", "_").replace("-", "_")
        value = normalize_single_line(match.group(2))
        if key in {"name", "direction", "unit"} and value:
            values[key] = value
        else:
            prose.append(stripped)

    if not values:
        warnings.append(
            _warning(
                "primary_metric_prose",
                "Primary metric should use key/value lines: name, direction, unit.",
                section="Primary metric",
            )
        )
        return None, warnings

    direction = values.get("direction")
    normalized_direction = None
    if direction:
        normalized_direction = _DIRECTION_ALIASES.get(direction.strip().lower())
        if normalized_direction is None:
            warnings.append(
                _warning(
                    "primary_metric_direction_unknown",
                    "Primary metric direction was not recognized.",
                    section="Primary metric",
                )
            )
    if prose:
        warnings.append(
            _warning(
                "primary_metric_extra_prose",
                "Primary metric contained prose outside supported key/value fields.",
                section="Primary metric",
            )
        )
    if not values.get("name"):
        warnings.append(
            _warning(
                "primary_metric_missing_name",
                "Primary metric key/value form did not include name.",
                section="Primary metric",
            )
        )
    return (
        PrimaryMetric(
            name=_optional_line(values.get("name"), 128),
            direction=normalized_direction,
            unit=_optional_line(values.get("unit"), 32),
        ),
        warnings,
    )


def _parse_item_section(bodies: Sequence[str], *, max_items: int) -> tuple[str, ...]:
    values: list[str] = []
    for body in bodies:
        for line in body.splitlines():
            item = _line_item_text(line)
            if item:
                values.append(item)
    if not values:
        text = normalize_single_line("\n\n".join(bodies))
        if text:
            values.append(text)
    return tuple(_bounded_dedupe(values, max_items=max_items, max_chars=200))


def _parse_scope_section(
    bodies: Sequence[str],
    *,
    warnings: list[dict[str, Any]],
) -> tuple[str, ...]:
    return _parse_scope_patterns(_parse_item_section(bodies, max_items=20), "Editable scope", warnings)


def _parse_scope_patterns(
    patterns: Sequence[str],
    section: str,
    warnings: list[dict[str, Any]],
) -> tuple[str, ...]:
    safe: list[str] = []
    for raw in patterns:
        pattern = _strip_inline_code(normalize_single_line(raw))
        reason = unsafe_scope_pattern_reason(pattern)
        if reason is not None:
            warnings.append(
                _warning(
                    "unsafe_scope_pattern",
                    f"Unsafe {section.lower()} pattern will fail the worker scope gate.",
                    section=section,
                    pattern=clamp_text(pattern, 120),
                    reason=reason,
                )
            )
        safe.append(pattern)
    return tuple(_bounded_dedupe(safe, max_items=20, max_chars=200))


def unsafe_scope_pattern_reason(pattern: str) -> str | None:
    """Return a machine-readable reason when a repo-relative scope pattern is unsafe."""

    value = normalize_single_line(pattern)
    if not value:
        return "empty"
    if "\x00" in value:
        return "nul_byte"
    if "\\" in value:
        return "not_posix"
    if value.startswith("/") or re.match(r"^[A-Za-z]:", value):
        return "absolute_path"
    parts = PurePosixPath(value).parts
    if any(part == ".." for part in parts):
        return "path_traversal"
    if any(part == ".git" for part in parts):
        return "git_internal"
    if value.startswith("!"):
        return "negation_not_supported"
    return None


def _line_item_text(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    match = _BULLET_RE.match(stripped)
    if match:
        stripped = match.group("text").strip()
    return clamp_text(normalize_single_line(_strip_inline_code(stripped)), 200) or None


def _strip_inline_code(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped.startswith("`") and stripped.endswith("`"):
        return stripped.strip("`").strip()
    return stripped


def _primary_metric_line(metric: PrimaryMetric | None) -> str | None:
    if metric is None or not metric.name:
        return None
    direction = ""
    if metric.direction == "higher_is_better":
        direction = "higher is better"
    elif metric.direction == "lower_is_better":
        direction = "lower is better"
    unit = f", unit={metric.unit}" if metric.unit else ""
    suffix = f" ({direction}{unit})" if direction or unit else ""
    return clamp_text(f"Primary metric: {metric.name}{suffix}", 200)


def _prefixed_items(prefix: str, values: Sequence[str]) -> list[str]:
    return [
        clamp_text(f"{prefix}: {normalize_single_line(value)}", 200)
        for value in values
        if normalize_single_line(value)
    ]


def _bounded_dedupe(
    values: Sequence[str],
    *,
    max_items: int,
    max_chars: int,
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = clamp_text(normalize_single_line(str(value or "")), max_chars)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
        if len(result) >= max_items:
            break
    return result


def _bounded_string_tuple(payload: Any, *, limit: int, max_items: int = 20) -> tuple[str, ...]:
    if payload is None:
        return ()
    if isinstance(payload, str):
        candidates: tuple[Any, ...] = (payload,)
    else:
        try:
            candidates = tuple(payload)
        except TypeError:
            candidates = (payload,)
    return tuple(_bounded_dedupe([str(item) for item in candidates], max_items=max_items, max_chars=limit))


def _mapping_sequence(payload: Any) -> tuple[Mapping[str, Any], ...]:
    if payload is None:
        return ()
    if isinstance(payload, Mapping):
        return (payload,)
    try:
        values = tuple(payload)
    except TypeError:
        return ()
    return tuple(item for item in values if isinstance(item, Mapping))


def _optional_line(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    text = clamp_text(normalize_single_line(str(value)), limit)
    return text or None


def _warning(
    code: str,
    message: str,
    *,
    section: str | None = None,
    pattern: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "code": clamp_text(normalize_single_line(code).lower(), 64),
        "message": clamp_text(normalize_single_line(message), 240),
    }
    if section:
        payload["section"] = clamp_text(normalize_single_line(section), 64)
    if pattern:
        payload["pattern"] = clamp_text(normalize_single_line(pattern), 120)
    if reason:
        payload["reason"] = clamp_text(normalize_single_line(reason), 64)
    return payload
