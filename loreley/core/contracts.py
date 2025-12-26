"""Typed, size-bounded contracts for Loreley's hot-path data flow.

These models define the *consumable* fields that flow through planning/coding/UI.
Raw prompts, logs, and other large evidence must be stored as artifacts instead
of being embedded in primary database rows.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "ArtifactRef",
    "CommitCard",
    "EvolutionJobSpec",
    "MetricBrief",
    "clamp_text",
    "normalize_single_line",
]


def normalize_single_line(text: str) -> str:
    """Normalize arbitrary text into a single trimmed line.

    Notes:
        - Collapses all whitespace (including newlines) into single spaces.
        - Does not enforce length budgets; callers should clamp separately.
    """

    return " ".join((text or "").split()).strip()


def clamp_text(text: str, max_chars: int) -> str:
    """Clamp text to a maximum character budget using an ellipsis when needed."""

    if max_chars <= 0:
        return ""
    value = (text or "").strip()
    if len(value) <= max_chars:
        return value
    if max_chars == 1:
        return "…"
    return f"{value[: max_chars - 1]}…"


SubjectText = Annotated[str, Field(min_length=1, max_length=72)]
ChangeSummaryText = Annotated[str, Field(min_length=1, max_length=512)]
EvaluationSummaryText = Annotated[str, Field(max_length=512)]
FilePathText = Annotated[str, Field(min_length=1, max_length=256)]
HighlightText = Annotated[str, Field(min_length=1, max_length=200)]
ShortTagText = Annotated[str, Field(min_length=1, max_length=64)]


class MetricBrief(BaseModel):
    """Small, stable metric snippet suitable for prompts and UIs."""

    model_config = ConfigDict(frozen=True)

    name: Annotated[str, Field(min_length=1, max_length=128)]
    value: float
    unit: Annotated[str | None, Field(max_length=32)] = None
    higher_is_better: bool | None = None
    summary: Annotated[str | None, Field(max_length=120)] = None


class CommitCard(BaseModel):
    """Lightweight, stable commit representation for inspiration and UI."""

    model_config = ConfigDict(frozen=True)

    commit_hash: Annotated[str, Field(min_length=1, max_length=64)]
    parent_commit_hash: Annotated[str | None, Field(max_length=64)] = None
    experiment_id: UUID | None = None
    island_id: Annotated[str | None, Field(max_length=64)] = None
    job_id: UUID | None = None

    subject: SubjectText
    change_summary: ChangeSummaryText
    key_files: list[FilePathText] = Field(default_factory=list, max_length=20)
    highlights: Annotated[list[HighlightText], Field(min_length=1, max_length=8)]
    evaluation_summary: EvaluationSummaryText | None = None

    @field_validator("subject")
    @classmethod
    def _validate_subject(cls, value: str) -> str:
        subject = normalize_single_line(value)
        if "```" in subject:
            raise ValueError("subject must not contain code fences")
        if subject.startswith("{") or subject.startswith("["):
            raise ValueError("subject must not start with JSON")
        return subject


class ArtifactRef(BaseModel):
    """Filesystem references to cold-path artifacts (prompts, raw outputs, logs)."""

    model_config = ConfigDict(frozen=True)

    planning_prompt_path: Annotated[str | None, Field(max_length=1024)] = None
    planning_raw_output_path: Annotated[str | None, Field(max_length=1024)] = None
    planning_plan_json_path: Annotated[str | None, Field(max_length=1024)] = None

    coding_prompt_path: Annotated[str | None, Field(max_length=1024)] = None
    coding_raw_output_path: Annotated[str | None, Field(max_length=1024)] = None
    coding_execution_json_path: Annotated[str | None, Field(max_length=1024)] = None

    evaluation_json_path: Annotated[str | None, Field(max_length=1024)] = None
    evaluation_logs_path: Annotated[str | None, Field(max_length=1024)] = None


class EvolutionJobSpec(BaseModel):
    """Size-bounded job specification used to schedule and re-run work."""

    model_config = ConfigDict(frozen=True)

    goal: Annotated[str, Field(min_length=1, max_length=512)]
    constraints: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(
        default_factory=list,
        max_length=20,
    )
    acceptance_criteria: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(
        default_factory=list,
        max_length=20,
    )
    notes: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(
        default_factory=list,
        max_length=20,
    )
    tags: list[ShortTagText] = Field(default_factory=list, max_length=20)

    base_commit_hash: Annotated[str, Field(min_length=1, max_length=64)]
    inspiration_commit_hashes: list[Annotated[str, Field(min_length=1, max_length=64)]] = Field(
        default_factory=list,
        max_length=10,
    )

    sampling_strategy: Annotated[str | None, Field(max_length=64)] = None
    sampling_radius_used: Annotated[int | None, Field(ge=0)] = None
    sampling_initial_radius: Annotated[int | None, Field(ge=0)] = None
    sampling_fallback_inspirations: Annotated[int | None, Field(ge=0)] = None
    iteration_hint: Annotated[str | None, Field(max_length=256)] = None

    priority: int = 0


