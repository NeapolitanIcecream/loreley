"""Campaign-level seed portfolio contracts, planning, and admission policy."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from loreley.config import Settings, get_settings
from loreley.core.map_elites.objectives import ObjectiveContract
from loreley.core.usage import LLMUsageEventPayload
from loreley.core.worker.agent import (
    AgentBackend,
    AgentInvocation,
    AgentTask,
    load_agent_backend,
    run_agent_task,
)

SEED_PORTFOLIO_SCHEMA_VERSION = 1
MAX_SEED_PORTFOLIO_DIRECTIONS = 16
MAX_SEED_PORTFOLIO_PAIRWISE_OVERLAPS = (
    MAX_SEED_PORTFOLIO_DIRECTIONS * (MAX_SEED_PORTFOLIO_DIRECTIONS - 1) // 2
)

IMMEDIATE_EVIDENCE_LANE = "immediate_evidence"
EXPLORATORY_STEPPING_STONE_LANE = "exploratory_stepping_stone"

SeedAdmissionIntent = Literal[
    "immediate_evidence",
    "exploratory_stepping_stone",
]

_DIRECTION_ID_RE = re.compile(r"^[a-z][a-z0-9-]{2,63}$")
_DEFAULT_SEED_PORTFOLIO_BACKEND = (
    "loreley.core.worker.agent.backends.kilocode_cli:kilocode_seed_portfolio_backend"
)


class SeedPortfolioError(RuntimeError):
    """Base error for seed portfolio construction and persistence."""


class SeedPortfolioValidationError(SeedPortfolioError):
    """Raised when a planner output violates the portfolio policy."""


class SeedPortfolioPlanningError(SeedPortfolioError):
    """Raised when the portfolio agent cannot produce a valid contract."""


def _normalized_line(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def canonical_json_hash(payload: object) -> str:
    """Return a stable SHA-256 for a JSON-compatible payload."""

    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class SeedDirection(BaseModel):
    """One implementation-ready causal direction selected for the seed slate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    direction_id: Annotated[str, Field(min_length=3, max_length=64)]
    title: Annotated[str, Field(min_length=3, max_length=120)]
    bottleneck: Annotated[str, Field(min_length=10, max_length=800)]
    causal_mechanism: Annotated[str, Field(min_length=10, max_length=1200)]
    likely_files: Annotated[
        tuple[Annotated[str, Field(min_length=1, max_length=256)], ...],
        Field(min_length=1, max_length=16),
    ]
    first_implementation: Annotated[str, Field(min_length=10, max_length=1600)]
    expected_immediate_signals: Annotated[
        tuple[Annotated[str, Field(min_length=3, max_length=300)], ...],
        Field(min_length=1, max_length=8),
    ]
    acceptable_neutral_results: Annotated[
        tuple[Annotated[str, Field(min_length=3, max_length=300)], ...],
        Field(min_length=1, max_length=8),
    ]
    roadmap: Annotated[
        tuple[Annotated[str, Field(min_length=3, max_length=400)], ...],
        Field(min_length=2, max_length=3),
    ]
    risks: Annotated[
        tuple[Annotated[str, Field(min_length=3, max_length=300)], ...],
        Field(min_length=1, max_length=8),
    ]
    local_checks: Annotated[
        tuple[Annotated[str, Field(min_length=3, max_length=300)], ...],
        Field(min_length=1, max_length=8),
    ]
    admission_intent: SeedAdmissionIntent
    selection_reason: Annotated[str, Field(min_length=10, max_length=800)]

    @field_validator("direction_id")
    @classmethod
    def _validate_direction_id(cls, value: str) -> str:
        normalized = _normalized_line(value).lower().replace("_", "-")
        if not _DIRECTION_ID_RE.fullmatch(normalized):
            raise ValueError("direction_id must be a lowercase kebab-case identifier")
        return normalized

    @field_validator(
        "title",
        "bottleneck",
        "causal_mechanism",
        "first_implementation",
        "selection_reason",
    )
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return _normalized_line(value)

    @field_validator(
        "likely_files",
        "expected_immediate_signals",
        "acceptable_neutral_results",
        "roadmap",
        "risks",
        "local_checks",
    )
    @classmethod
    def _normalize_sequence(cls, values: Sequence[str]) -> tuple[str, ...]:
        normalized = tuple(_normalized_line(value) for value in values)
        if any(not value for value in normalized):
            raise ValueError("direction list fields cannot contain blank values")
        if len(set(normalized)) != len(normalized):
            raise ValueError("direction list fields cannot contain duplicates")
        return normalized

    @property
    def content_hash(self) -> str:
        return canonical_json_hash(self.model_dump(mode="json"))


class SeedDirectionOverlap(BaseModel):
    """One unordered pair in the selected portfolio overlap matrix."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    direction_a: Annotated[str, Field(min_length=3, max_length=64)]
    direction_b: Annotated[str, Field(min_length=3, max_length=64)]
    overlap_score: Annotated[float, Field(ge=0.0, le=1.0)]
    shared_surface: Annotated[str, Field(min_length=3, max_length=500)]
    mechanism_distinction: Annotated[str, Field(min_length=10, max_length=800)]

    @field_validator("direction_a", "direction_b")
    @classmethod
    def _normalize_direction_id(cls, value: str) -> str:
        return _normalized_line(value).lower().replace("_", "-")

    @field_validator("shared_surface", "mechanism_distinction")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return _normalized_line(value)

    @property
    def pair(self) -> tuple[str, str]:
        return tuple(sorted((self.direction_a, self.direction_b)))  # type: ignore[return-value]


class RejectedSeedDirection(BaseModel):
    """A proposed direction removed while curating the final portfolio."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    title: Annotated[str, Field(min_length=3, max_length=120)]
    causal_mechanism: Annotated[str, Field(min_length=10, max_length=1200)]
    duplicate_of_direction_id: Annotated[str | None, Field(max_length=64)] = None
    rejection_reason: Annotated[str, Field(min_length=10, max_length=800)]

    @field_validator("title", "causal_mechanism", "rejection_reason")
    @classmethod
    def _normalize_text(cls, value: str) -> str:
        return _normalized_line(value)

    @field_validator("duplicate_of_direction_id")
    @classmethod
    def _normalize_optional_id(cls, value: str | None) -> str | None:
        normalized = _normalized_line(value).lower().replace("_", "-")
        return normalized or None


class SeedPortfolioDraft(BaseModel):
    """Structured output expected from the one campaign-level planner call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    directions: Annotated[
        tuple[SeedDirection, ...],
        Field(min_length=1, max_length=MAX_SEED_PORTFOLIO_DIRECTIONS),
    ]
    pairwise_overlaps: Annotated[
        tuple[SeedDirectionOverlap, ...],
        Field(max_length=MAX_SEED_PORTFOLIO_PAIRWISE_OVERLAPS),
    ] = ()
    rejected_directions: Annotated[
        tuple[RejectedSeedDirection, ...],
        Field(min_length=1, max_length=128),
    ]
    curation_summary: Annotated[str, Field(min_length=10, max_length=1600)]

    @field_validator("curation_summary")
    @classmethod
    def _normalize_summary(cls, value: str) -> str:
        return _normalized_line(value)


class SeedRootEvidence(BaseModel):
    """Bounded evaluator-visible root evidence supplied to the portfolio owner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluation_summary: Annotated[str | None, Field(max_length=2000)] = None
    metrics: Annotated[tuple[dict[str, Any], ...], Field(max_length=128)] = ()
    diagnostics: Annotated[tuple[str, ...], Field(max_length=32)] = ()


class SeedPortfolioArtifact(BaseModel):
    """Content-addressed, replayable campaign artifact persisted by the scheduler."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = SEED_PORTFOLIO_SCHEMA_VERSION
    request_fingerprint: Annotated[str, Field(min_length=64, max_length=64)]
    portfolio_hash: Annotated[str, Field(min_length=64, max_length=64)]
    configured_direction_count: Annotated[
        int, Field(ge=1, le=MAX_SEED_PORTFOLIO_DIRECTIONS)
    ]
    direction_count: Annotated[int, Field(ge=1, le=MAX_SEED_PORTFOLIO_DIRECTIONS)]
    root_commit_hash: Annotated[str, Field(min_length=1, max_length=64)]
    campaign_program_hash: Annotated[str | None, Field(max_length=64)] = None
    objective_contract: tuple[dict[str, str], ...]
    objective_contract_fingerprint: Annotated[str, Field(min_length=64, max_length=64)]
    input_evidence_fingerprints: dict[str, str]
    model_route: dict[str, Any]
    reasoning_effort: Annotated[str, Field(min_length=1, max_length=32)]
    directions: tuple[SeedDirection, ...]
    pairwise_overlaps: tuple[SeedDirectionOverlap, ...]
    rejected_directions: tuple[RejectedSeedDirection, ...]
    curation_summary: str


@dataclass(frozen=True, slots=True)
class SeedPortfolioPlanningRequest:
    """All stable inputs that determine one portfolio-planning request."""

    configured_direction_count: int
    direction_count: int
    root_commit_hash: str
    campaign_program_hash: str | None
    campaign_title: str | None
    goal: str
    constraints: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]
    notes: tuple[str, ...]
    objective_contract: tuple[dict[str, str], ...]
    objective_contract_fingerprint: str
    root_evidence: SeedRootEvidence
    input_evidence_fingerprints: dict[str, str]
    model_route: dict[str, Any]
    reasoning_effort: str
    max_pairwise_overlap: float

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SEED_PORTFOLIO_SCHEMA_VERSION,
            "configured_direction_count": self.configured_direction_count,
            "direction_count": self.direction_count,
            "root_commit_hash": self.root_commit_hash,
            "campaign_program_hash": self.campaign_program_hash,
            "campaign_title": self.campaign_title,
            "goal": self.goal,
            "constraints": list(self.constraints),
            "acceptance_criteria": list(self.acceptance_criteria),
            "notes": list(self.notes),
            "objective_contract": list(self.objective_contract),
            "objective_contract_fingerprint": self.objective_contract_fingerprint,
            "root_evidence": self.root_evidence.model_dump(mode="json"),
            "input_evidence_fingerprints": dict(self.input_evidence_fingerprints),
            "model_route": dict(self.model_route),
            "reasoning_effort": self.reasoning_effort,
            "max_pairwise_overlap": self.max_pairwise_overlap,
        }

    @property
    def request_fingerprint(self) -> str:
        return canonical_json_hash(self.fingerprint_payload())


@dataclass(frozen=True, slots=True)
class SeedPortfolioPlanningResponse:
    draft: SeedPortfolioDraft
    prompt: str
    raw_output: str
    prompt_sha256: str
    output_sha256: str
    attempts: int
    duration_seconds: float
    usage_events: tuple[LLMUsageEventPayload, ...] = field(default_factory=tuple)


def validate_seed_portfolio_draft(
    draft: SeedPortfolioDraft,
    *,
    expected_direction_count: int,
    max_pairwise_overlap: float,
) -> SeedPortfolioDraft:
    """Apply global diversity and completeness policy to a structured draft."""

    expected = max(
        1,
        min(MAX_SEED_PORTFOLIO_DIRECTIONS, int(expected_direction_count)),
    )
    if len(draft.directions) != expected:
        raise SeedPortfolioValidationError(
            "portfolio direction count does not match the requested slate "
            f"(expected={expected} observed={len(draft.directions)})"
        )

    direction_ids = tuple(direction.direction_id for direction in draft.directions)
    if len(set(direction_ids)) != len(direction_ids):
        raise SeedPortfolioValidationError("portfolio direction IDs must be unique")
    titles = tuple(direction.title.casefold() for direction in draft.directions)
    if len(set(titles)) != len(titles):
        raise SeedPortfolioValidationError("portfolio direction titles must be unique")
    mechanisms = tuple(
        direction.causal_mechanism.casefold() for direction in draft.directions
    )
    if len(set(mechanisms)) != len(mechanisms):
        raise SeedPortfolioValidationError(
            "portfolio causal mechanisms must be distinct"
        )

    expected_pairs = {
        tuple(sorted((left, right)))
        for index, left in enumerate(direction_ids)
        for right in direction_ids[index + 1 :]
    }
    observed_pairs: dict[tuple[str, str], SeedDirectionOverlap] = {}
    for overlap in draft.pairwise_overlaps:
        pair = overlap.pair
        if overlap.direction_a == overlap.direction_b:
            raise SeedPortfolioValidationError(
                "pairwise overlap entries cannot compare a direction with itself"
            )
        if pair not in expected_pairs:
            raise SeedPortfolioValidationError(
                f"pairwise overlap references unknown direction pair {pair!r}"
            )
        if pair in observed_pairs:
            raise SeedPortfolioValidationError(
                f"pairwise overlap contains duplicate pair {pair!r}"
            )
        if float(overlap.overlap_score) > float(max_pairwise_overlap):
            raise SeedPortfolioValidationError(
                "selected directions exceed the pairwise overlap policy "
                f"(pair={pair!r} score={overlap.overlap_score:g} "
                f"maximum={float(max_pairwise_overlap):g})"
            )
        observed_pairs[pair] = overlap
    missing_pairs = expected_pairs - set(observed_pairs)
    if missing_pairs:
        raise SeedPortfolioValidationError(
            f"pairwise overlap matrix is incomplete (missing={sorted(missing_pairs)!r})"
        )

    selected_ids = set(direction_ids)
    for rejected in draft.rejected_directions:
        duplicate_of = rejected.duplicate_of_direction_id
        if duplicate_of is not None and duplicate_of not in selected_ids:
            raise SeedPortfolioValidationError(
                "rejected direction duplicate_of_direction_id must reference a "
                f"selected direction (observed={duplicate_of!r})"
            )

    if expected >= 2:
        intents = {direction.admission_intent for direction in draft.directions}
        required = {IMMEDIATE_EVIDENCE_LANE, EXPLORATORY_STEPPING_STONE_LANE}
        if not required.issubset(intents):
            raise SeedPortfolioValidationError(
                "portfolio must reserve directions for both seed admission intents"
            )
    return draft


def materialize_seed_portfolio(
    request: SeedPortfolioPlanningRequest,
    draft: SeedPortfolioDraft,
) -> SeedPortfolioArtifact:
    """Attach complete input provenance and compute the artifact content hash."""

    payload: dict[str, Any] = {
        "schema_version": SEED_PORTFOLIO_SCHEMA_VERSION,
        "request_fingerprint": request.request_fingerprint,
        "configured_direction_count": request.configured_direction_count,
        "direction_count": request.direction_count,
        "root_commit_hash": request.root_commit_hash,
        "campaign_program_hash": request.campaign_program_hash,
        "objective_contract": list(request.objective_contract),
        "objective_contract_fingerprint": request.objective_contract_fingerprint,
        "input_evidence_fingerprints": dict(request.input_evidence_fingerprints),
        "model_route": dict(request.model_route),
        "reasoning_effort": request.reasoning_effort,
        **draft.model_dump(mode="json"),
    }
    portfolio_hash = canonical_json_hash(payload)
    return SeedPortfolioArtifact.model_validate(
        {**payload, "portfolio_hash": portfolio_hash}
    )


class SeedPortfolioPlanner:
    """Invoke one repository-aware planner for the complete seed slate."""

    def __init__(
        self,
        settings: Settings | None = None,
        backend: AgentBackend | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.max_attempts = max(
            1,
            int(getattr(self.settings, "worker_seed_portfolio_max_attempts", 2)),
        )
        if backend is not None:
            self.backend = backend
        else:
            backend_ref = str(
                getattr(self.settings, "worker_seed_portfolio_backend", "") or ""
            ).strip()
            if not backend_ref:
                raise SeedPortfolioPlanningError(
                    "WORKER_SEED_PORTFOLIO_BACKEND must be configured"
                )
            if backend_ref == _DEFAULT_SEED_PORTFOLIO_BACKEND:
                from loreley.core.worker.agent.backends import (
                    build_kilocode_seed_portfolio_backend,
                )

                self.backend = build_kilocode_seed_portfolio_backend(self.settings)
            else:
                self.backend = load_agent_backend(
                    backend_ref,
                    label="seed portfolio backend",
                )

    def plan(
        self,
        request: SeedPortfolioPlanningRequest,
        *,
        working_dir: Path,
    ) -> SeedPortfolioPlanningResponse:
        prompt = self._render_prompt(request)
        task = AgentTask(
            name="seed-portfolio",
            prompt=prompt,
            phase="seed_portfolio",
        )
        draft, invocation, attempts = run_agent_task(
            backend=self.backend,
            task=task,
            working_dir=Path(working_dir).expanduser().resolve(),
            max_attempts=self.max_attempts,
            coerce_result=lambda item: self._coerce_draft(request, item),
            retryable_exceptions=(SeedPortfolioPlanningError,),
            error_cls=SeedPortfolioPlanningError,
            error_message=(
                "Seed portfolio planner could not produce a valid structured "
                f"portfolio after {self.max_attempts} attempt(s)."
            ),
        )
        return SeedPortfolioPlanningResponse(
            draft=draft,
            prompt=prompt,
            raw_output=invocation.stdout,
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            output_sha256=hashlib.sha256(
                (invocation.stdout or "").encode("utf-8")
            ).hexdigest(),
            attempts=attempts,
            duration_seconds=invocation.duration_seconds,
            usage_events=tuple(invocation.usage_events or ()),
        )

    def _coerce_draft(
        self,
        request: SeedPortfolioPlanningRequest,
        invocation: AgentInvocation,
    ) -> SeedPortfolioDraft:
        try:
            payload = _extract_portfolio_json(invocation.stdout)
            draft = SeedPortfolioDraft.model_validate(payload)
            return validate_seed_portfolio_draft(
                draft,
                expected_direction_count=request.direction_count,
                max_pairwise_overlap=request.max_pairwise_overlap,
            )
        except (SeedPortfolioValidationError, ValueError, TypeError) as exc:
            raise SeedPortfolioPlanningError(
                f"seed portfolio structured output is invalid: {exc}"
            ) from exc

    @staticmethod
    def _render_prompt(request: SeedPortfolioPlanningRequest) -> str:
        evidence = request.root_evidence.model_dump(mode="json")
        schema = SeedPortfolioDraft.model_json_schema()
        contract = {
            "root_commit_hash": request.root_commit_hash,
            "campaign_program_hash": request.campaign_program_hash,
            "campaign_title": request.campaign_title,
            "goal": request.goal,
            "constraints": list(request.constraints),
            "acceptance_criteria": list(request.acceptance_criteria),
            "notes": list(request.notes),
            "optimization_objectives": list(request.objective_contract),
            "root_evidence": evidence,
            "configured_direction_count": request.configured_direction_count,
            "direction_count": request.direction_count,
            "maximum_selected_pairwise_overlap": request.max_pairwise_overlap,
        }
        return (
            "You are Loreley's campaign-level seed portfolio planner. Inspect the "
            "repository with the available tools, identify credible engineering "
            "mechanisms, and own diversity across the entire initial slate.\n\n"
            "This is a read-only planning task. Do not modify, create, or delete "
            "repository files; do not create commits, branches, tags, or pushes; and "
            "do not install or download anything.\n\n"
            "Propose more directions than the final budget internally. Remove near-"
            "duplicates before returning the selected directions. Differences must be "
            "causal mechanisms or evolutionary routes, not wording alone. Each selected "
            "direction must be implementation-ready for one independent coding job. "
            "Do not run Loreley's evaluator or any framework-managed full benchmark.\n\n"
            "Keep the inspection and final artifact compact. Prefer a small number of "
            "targeted repository reads over an exhaustive survey, and begin synthesis "
            "once you have enough evidence to distinguish the requested mechanisms. "
            "For each selected direction, name at most four likely files; give exactly "
            "one immediate signal, one acceptable neutral result, one risk, and one "
            "local check; and give exactly two roadmap steps. Keep bottleneck, causal "
            "mechanism, first implementation, and selection reason to a few precise "
            "sentences each. Keep pairwise shared-surface and distinction explanations "
            "to one short sentence each, return no more than three rejected directions, "
            "and keep the curation summary to one paragraph.\n\n"
            "Reserve at least one immediate-evidence intent and one exploratory-"
            "stepping-stone intent when the direction count is at least two. Neutral "
            "short-term results are valid for exploratory directions after correctness "
            "and other hard gates pass. Optimization-objective trade-offs belong to "
            "the ordinary Pareto/QD archive policy, not this planner.\n\n"
            "Return exactly one JSON object and no Markdown fences or commentary. The "
            "object must conform to the supplied JSON Schema. Include every unordered "
            "selected-direction pair exactly once in pairwise_overlaps, and include at "
            "least one rejected direction.\n\n"
            "Campaign contract:\n"
            f"{json.dumps(contract, indent=2, sort_keys=True)}\n\n"
            "Required output JSON Schema:\n"
            f"{json.dumps(schema, indent=2, sort_keys=True)}"
        )


def _extract_portfolio_json(raw_output: str) -> Mapping[str, Any]:
    raw = str(raw_output or "").strip()
    if not raw:
        raise ValueError("planner returned empty output")
    candidates = [raw]
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidates.append("\n".join(lines).strip())
    start = raw.find("{")
    end = raw.rfind("}")
    if 0 <= start < end:
        candidates.append(raw[start : end + 1])

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        located = _locate_portfolio_payload(payload)
        if located is not None:
            return located

    # Kilo's formatted stdout can contain tool-call or tool-result JSON before
    # the planner's final response. Slicing from the first opening brace to the
    # last closing brace joins those independent values into invalid JSON. Scan
    # each possible JSON value instead so the final structured response remains
    # recoverable from otherwise noisy CLI output.
    decoder = json.JSONDecoder()
    cursor = 0
    while cursor < len(raw):
        object_start = raw.find("{", cursor)
        array_start = raw.find("[", cursor)
        starts = tuple(index for index in (object_start, array_start) if index >= 0)
        if not starts:
            break
        start = min(starts)
        try:
            payload, end = decoder.raw_decode(raw, start)
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        located = _locate_portfolio_payload(payload)
        if located is not None:
            return located
        cursor = max(end, start + 1)
    raise ValueError("planner output did not contain a portfolio JSON object")


def _locate_portfolio_payload(
    value: object, *, depth: int = 0
) -> Mapping[str, Any] | None:
    if depth > 6:
        return None
    if isinstance(value, Mapping):
        if "directions" in value and "rejected_directions" in value:
            return value
        for key in ("output", "result", "response", "content", "message"):
            nested = value.get(key)
            if isinstance(nested, str):
                try:
                    decoded = json.loads(nested)
                except json.JSONDecodeError:
                    continue
                located = _locate_portfolio_payload(decoded, depth=depth + 1)
            else:
                located = _locate_portfolio_payload(nested, depth=depth + 1)
            if located is not None:
                return located
        for nested in value.values():
            located = _locate_portfolio_payload(nested, depth=depth + 1)
            if located is not None:
                return located
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            located = _locate_portfolio_payload(nested, depth=depth + 1)
            if located is not None:
                return located
    return None


@dataclass(frozen=True, slots=True)
class SeedAdmissionDecision:
    lane: str
    reason: str
    directed_fractional_deltas: tuple[tuple[str, float], ...]

    @property
    def admitted(self) -> bool:
        """Every evaluator-valid portfolio seed proceeds to ordinary QD policy."""

        return True


def resolve_seed_portfolio_direction_count(settings: Settings) -> int:
    """Resolve the stable campaign-level direction count from configured caps."""

    configured = max(
        1,
        min(
            MAX_SEED_PORTFOLIO_DIRECTIONS,
            int(getattr(settings, "mapelites_seed_portfolio_direction_count", 8)),
        ),
    )
    per_island_target = max(
        0,
        int(getattr(settings, "mapelites_seed_population_size", 0) or 0),
        int(
            getattr(
                settings,
                "mapelites_feature_normalization_warmup_samples",
                0,
            )
            or 0
        ),
    )
    raw_total_cap = getattr(settings, "scheduler_max_total_jobs", None)
    total_cap = configured if raw_total_cap is None else max(0, int(raw_total_cap))
    return min(configured, per_island_target, total_cap)


def classify_seed_admission(
    *,
    objective_contract: ObjectiveContract,
    baseline_metrics: Sequence[Mapping[str, Any] | object],
    candidate_metrics: Sequence[Mapping[str, Any] | object],
    immediate_min_improvement_fraction: float = 0.0,
) -> SeedAdmissionDecision:
    """Label one evaluator-valid seed without replacing ordinary QD admission."""

    baseline = objective_contract.resolve(baseline_metrics)
    candidate = objective_contract.resolve(candidate_metrics)
    min_improvement = max(0.0, float(immediate_min_improvement_fraction))
    deltas: list[tuple[str, float]] = []
    for spec, baseline_score, candidate_score in zip(
        objective_contract.specs,
        baseline.scores,
        candidate.scores,
    ):
        scale = max(abs(float(baseline_score)), 1.0)
        fraction = (float(candidate_score) - float(baseline_score)) / scale
        if not math.isfinite(fraction):
            raise SeedPortfolioValidationError(
                f"non-finite admission delta for objective {spec.name!r}"
            )
        deltas.append((spec.name, fraction))

    improved = tuple(name for name, delta in deltas if delta > min_improvement)
    if improved:
        return SeedAdmissionDecision(
            lane=IMMEDIATE_EVIDENCE_LANE,
            reason=(
                "Observed directed improvement above the predeclared threshold for "
                f"optimization objective(s): {', '.join(improved)}. This is a "
                "provenance label; ordinary Pareto/QD policy decides archive admission."
            ),
            directed_fractional_deltas=tuple(deltas),
        )
    return SeedAdmissionDecision(
        lane=EXPLORATORY_STEPPING_STONE_LANE,
        reason=(
            "No optimization objective had an observed directed improvement above "
            "the predeclared threshold. The evaluator-valid seed proceeds to ordinary "
            "Pareto/QD policy with its distinct direction provenance."
        ),
        directed_fractional_deltas=tuple(deltas),
    )


__all__ = [
    "EXPLORATORY_STEPPING_STONE_LANE",
    "IMMEDIATE_EVIDENCE_LANE",
    "MAX_SEED_PORTFOLIO_DIRECTIONS",
    "MAX_SEED_PORTFOLIO_PAIRWISE_OVERLAPS",
    "SEED_PORTFOLIO_SCHEMA_VERSION",
    "RejectedSeedDirection",
    "SeedAdmissionDecision",
    "SeedDirection",
    "SeedDirectionOverlap",
    "SeedPortfolioArtifact",
    "SeedPortfolioDraft",
    "SeedPortfolioError",
    "SeedPortfolioPlanner",
    "SeedPortfolioPlanningError",
    "SeedPortfolioPlanningRequest",
    "SeedPortfolioPlanningResponse",
    "SeedPortfolioValidationError",
    "SeedRootEvidence",
    "canonical_json_hash",
    "classify_seed_admission",
    "materialize_seed_portfolio",
    "resolve_seed_portfolio_direction_count",
    "validate_seed_portfolio_draft",
]
