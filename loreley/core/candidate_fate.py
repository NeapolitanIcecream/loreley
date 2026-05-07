"""QD-aware operator-facing candidate fate derivation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from loreley.core.contracts import clamp_text, normalize_single_line

__all__ = [
    "CANDIDATE_FATE_LABELS",
    "CandidateFate",
    "derive_candidate_fate",
]

REASON_MAX_CHARS = 180

ELITE_INSERTED = "elite_inserted"
ELITE_REPLACED = "elite_replaced"
ELITE_RETAINED = "elite_retained"
VALID_NOT_ELITE = "valid_not_elite"
VALID_NOT_CONSIDERED = "valid_not_considered"
CANDIDATE_FAILED = "candidate_failed"
REPAIR_PENDING = "repair_pending"
POLICY_FAILED = "policy_failed"
DISCARDED_FOR_SAMPLING = "discarded_for_sampling"
UNKNOWN = "unknown"

CANDIDATE_FATE_LABELS = frozenset(
    {
        ELITE_INSERTED,
        ELITE_REPLACED,
        ELITE_RETAINED,
        VALID_NOT_ELITE,
        VALID_NOT_CONSIDERED,
        CANDIDATE_FAILED,
        REPAIR_PENDING,
        POLICY_FAILED,
        DISCARDED_FOR_SAMPLING,
        UNKNOWN,
    }
)

_REPAIR_PENDING_STATES = {"eligible", "scheduled", "repairing"}
_DISCARDED_STATES = {"discarded", "quarantined"}
_NON_CANDIDATE_FAILURE_OUTCOMES = {
    "evaluator_failed",
    "infrastructure_failed",
    "inconclusive",
}


@dataclass(frozen=True, slots=True)
class CandidateFate:
    """Operator-facing QD fate label plus a bounded explanation."""

    label: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {
            "candidate_fate_label": self.label,
            "candidate_fate_reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class _CandidateFateState:
    job_status: str
    ingestion_status: str
    ingestion_status_code: int | None
    ingestion_message: str | None
    cell_index: int | None
    evaluation_status: str
    archive_status: str
    repair_state: str
    lifecycle_status: str
    failure_stage: str
    failure_kind: str
    failure_summary: str
    commit_hash: str
    current_archive_member: bool

    @classmethod
    def from_rows(
        cls,
        *,
        job: Any | None,
        candidate: Any | None,
        current_archive_cell_index: int | None,
        current_archive_member: bool,
    ) -> "_CandidateFateState":
        ingestion_cell_index = _int_attr(job, "ingestion_cell_index")
        cell_index = (
            current_archive_cell_index
            if current_archive_cell_index is not None
            else ingestion_cell_index
        )
        return cls(
            job_status=_lower_attr(job, "status"),
            ingestion_status=_lower_attr(job, "ingestion_status"),
            ingestion_status_code=_int_attr(job, "ingestion_status_code"),
            ingestion_message=_first_text_attr(job, "ingestion_message", "ingestion_reason"),
            cell_index=cell_index,
            evaluation_status=_lower_attr(candidate, "evaluation_status"),
            archive_status=_lower_attr(candidate, "archive_status"),
            repair_state=_lower_attr(candidate, "repair_state"),
            lifecycle_status=_lower_attr(candidate, "lifecycle_status"),
            failure_stage=_lower_attr(candidate, "failure_stage"),
            failure_kind=_lower_attr(candidate, "failure_kind"),
            failure_summary=_text_attr(candidate, "failure_summary"),
            commit_hash=(
                _text_attr(candidate, "commit_hash")
                or _text_attr(job, "result_commit_hash")
                or _text_attr(job, "candidate_commit_hash")
            ),
            current_archive_member=current_archive_member,
        )

    @property
    def candidate_passed(self) -> bool:
        return self.evaluation_status == "passed" or (
            self.job_status == "succeeded" and bool(self.commit_hash)
        )


def derive_candidate_fate(
    *,
    job: Any | None = None,
    candidate: Any | None = None,
    current_archive_cell_index: int | None = None,
    current_archive_member: bool = False,
) -> CandidateFate:
    """Derive an ADR 0051 candidate fate from existing persisted state.

    The result is presentation state only. It does not replace lower-level job,
    evaluation, repair, or archive lifecycle fields.
    """

    state = _CandidateFateState.from_rows(
        job=job,
        candidate=candidate,
        current_archive_cell_index=current_archive_cell_index,
        current_archive_member=current_archive_member,
    )
    for classifier in (
        _discarded_fate,
        _repair_fate,
        _failure_fate,
        _archive_acceptance_fate,
        _current_archive_fate,
    ):
        fate = classifier(state)
        if fate is not None:
            return fate

    if not state.candidate_passed:
        return _not_passed_fate(state)
    return _passed_non_elite_fate(state)


def _discarded_fate(state: _CandidateFateState) -> CandidateFate | None:
    if state.lifecycle_status in _DISCARDED_STATES:
        return _fate(
            DISCARDED_FOR_SAMPLING,
            f"Candidate lifecycle_status={state.lifecycle_status}; excluded from default future sampling.",
        )
    if state.repair_state in _DISCARDED_STATES:
        return _fate(
            DISCARDED_FOR_SAMPLING,
            f"Candidate repair_state={state.repair_state}; excluded from default future sampling.",
        )
    return None


def _repair_fate(state: _CandidateFateState) -> CandidateFate | None:
    if state.repair_state not in _REPAIR_PENDING_STATES:
        return None
    return _fate(
        REPAIR_PENDING,
        _join_reason(
            f"Candidate repair_state={state.repair_state}.",
            _failure_reason(failure_stage=state.failure_stage, failure_kind=state.failure_kind),
        ),
    )


def _failure_fate(state: _CandidateFateState) -> CandidateFate | None:
    if state.failure_stage == "policy":
        return _fate(
            POLICY_FAILED,
            _join_reason(
                "Campaign or evaluator policy rejected the candidate.",
                _failure_reason(failure_stage=state.failure_stage, failure_kind=state.failure_kind),
                state.failure_summary,
            ),
        )
    if state.evaluation_status == "candidate_failed":
        return _fate(
            CANDIDATE_FAILED,
            _join_reason(
                "Candidate evaluation did not pass.",
                _failure_reason(failure_stage=state.failure_stage, failure_kind=state.failure_kind),
                state.failure_summary,
            ),
        )
    if state.evaluation_status in _NON_CANDIDATE_FAILURE_OUTCOMES:
        return _fate(
            UNKNOWN,
            f"Evaluation outcome={state.evaluation_status}; candidate-owned fate is not established.",
        )
    return None


def _archive_acceptance_fate(state: _CandidateFateState) -> CandidateFate | None:
    status_code = state.ingestion_status_code
    if state.ingestion_status != "succeeded" or status_code is None or status_code <= 0:
        return None
    if status_code == 2:
        return _fate(
            ELITE_INSERTED,
            _archive_acceptance_reason(
                "Candidate entered an empty archive niche.",
                status_code=status_code,
                cell_index=state.cell_index,
                current_archive_member=state.current_archive_member,
            ),
        )
    return _fate(
        ELITE_REPLACED,
        _archive_acceptance_reason(
            "Candidate improved an occupied archive niche.",
            status_code=status_code,
            cell_index=state.cell_index,
            current_archive_member=state.current_archive_member,
        ),
    )


def _current_archive_fate(state: _CandidateFateState) -> CandidateFate | None:
    if not state.current_archive_member:
        return None
    return _fate(
        ELITE_RETAINED,
        _cell_reason("Candidate is a current archive elite.", cell_index=state.cell_index),
    )


def _not_passed_fate(state: _CandidateFateState) -> CandidateFate:
    if state.job_status in {"pending", "queued", "running"}:
        return _fate(UNKNOWN, f"Job status={state.job_status}; no completed candidate fate yet.")
    if state.job_status == "failed":
        return _fate(UNKNOWN, "Job failed before a passing candidate fate was recorded.")
    return _fate(UNKNOWN, "No passing evaluation or archive decision is recorded.")


def _passed_non_elite_fate(state: _CandidateFateState) -> CandidateFate:
    if state.archive_status == "rejected" or state.ingestion_status == "skipped":
        return _fate(
            VALID_NOT_ELITE,
            _join_reason(
                "Candidate passed evaluation but did not enter the archive.",
                state.ingestion_message,
            ),
        )
    if state.archive_status == "member":
        return _fate(
            VALID_NOT_ELITE,
            "Candidate passed evaluation and has archive_status=member, but is not a current archive elite.",
        )
    if state.ingestion_status == "failed":
        return _fate(
            VALID_NOT_CONSIDERED,
            _join_reason(
                "Candidate passed evaluation but archive ingestion did not complete.",
                state.ingestion_message,
            ),
        )
    return _fate(
        VALID_NOT_CONSIDERED,
        "Candidate passed evaluation but archive insertion has not been recorded.",
    )


def _fate(label: str, reason: str) -> CandidateFate:
    safe_label = label if label in CANDIDATE_FATE_LABELS else UNKNOWN
    safe_reason = clamp_text(normalize_single_line(reason), REASON_MAX_CHARS)
    return CandidateFate(label=safe_label, reason=safe_reason or "No candidate fate reason is available.")


def _archive_acceptance_reason(
    summary: str,
    *,
    status_code: int,
    cell_index: int | None,
    current_archive_member: bool,
) -> str:
    parts = [
        summary,
        f"archive status_code={status_code}.",
    ]
    if cell_index is not None:
        parts.append(f"cell={cell_index}.")
    if not current_archive_member:
        parts.append("Current archive membership was not observed.")
    return _join_reason(*parts)


def _cell_reason(summary: str, *, cell_index: int | None) -> str:
    if cell_index is None:
        return summary
    return f"{summary} cell={cell_index}."


def _failure_reason(*, failure_stage: str, failure_kind: str) -> str | None:
    fields: list[str] = []
    if failure_stage:
        fields.append(f"stage={failure_stage}")
    if failure_kind:
        fields.append(f"kind={failure_kind}")
    if not fields:
        return None
    return "Failure " + " ".join(fields) + "."


def _join_reason(*parts: str | None) -> str:
    values = [normalize_single_line(part or "") for part in parts]
    return " ".join(value for value in values if value)


def _text_attr(obj: Any | None, name: str) -> str:
    value = _raw_attr(obj, name)
    if value is None:
        return ""
    return normalize_single_line(str(value))


def _first_text_attr(obj: Any | None, *names: str) -> str | None:
    for name in names:
        value = _text_attr(obj, name)
        if value:
            return value
    return None


def _lower_attr(obj: Any | None, name: str) -> str:
    value = _raw_attr(obj, name)
    if value is None:
        return ""
    value = getattr(value, "value", value)
    return normalize_single_line(str(value)).lower()


def _int_attr(obj: Any | None, name: str) -> int | None:
    value = _raw_attr(obj, name)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _raw_attr(obj: Any | None, name: str) -> Any | None:
    if obj is None:
        return None
    return getattr(obj, name, None)
