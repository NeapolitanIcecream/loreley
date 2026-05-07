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

    job_status = _lower_attr(job, "status")
    ingestion_status = _lower_attr(job, "ingestion_status")
    ingestion_status_code = _int_attr(job, "ingestion_status_code")
    ingestion_message = _first_text_attr(job, "ingestion_message", "ingestion_reason")
    ingestion_cell_index = _int_attr(job, "ingestion_cell_index")
    cell_index = current_archive_cell_index if current_archive_cell_index is not None else ingestion_cell_index

    evaluation_status = _lower_attr(candidate, "evaluation_status")
    archive_status = _lower_attr(candidate, "archive_status")
    repair_state = _lower_attr(candidate, "repair_state")
    lifecycle_status = _lower_attr(candidate, "lifecycle_status")
    failure_stage = _lower_attr(candidate, "failure_stage")
    failure_kind = _lower_attr(candidate, "failure_kind")
    failure_summary = _text_attr(candidate, "failure_summary")

    commit_hash = (
        _text_attr(candidate, "commit_hash")
        or _text_attr(job, "result_commit_hash")
        or _text_attr(job, "candidate_commit_hash")
    )

    if lifecycle_status in _DISCARDED_STATES:
        return _fate(
            DISCARDED_FOR_SAMPLING,
            f"Candidate lifecycle_status={lifecycle_status}; excluded from default future sampling.",
        )
    if repair_state in _DISCARDED_STATES:
        return _fate(
            DISCARDED_FOR_SAMPLING,
            f"Candidate repair_state={repair_state}; excluded from default future sampling.",
        )

    if repair_state in _REPAIR_PENDING_STATES:
        return _fate(
            REPAIR_PENDING,
            _join_reason(
                f"Candidate repair_state={repair_state}.",
                _failure_reason(failure_stage=failure_stage, failure_kind=failure_kind),
            ),
        )

    if failure_stage == "policy":
        return _fate(
            POLICY_FAILED,
            _join_reason(
                "Campaign or evaluator policy rejected the candidate.",
                _failure_reason(failure_stage=failure_stage, failure_kind=failure_kind),
                failure_summary,
            ),
        )

    if evaluation_status == "candidate_failed":
        return _fate(
            CANDIDATE_FAILED,
            _join_reason(
                "Candidate evaluation did not pass.",
                _failure_reason(failure_stage=failure_stage, failure_kind=failure_kind),
                failure_summary,
            ),
        )

    if evaluation_status in _NON_CANDIDATE_FAILURE_OUTCOMES:
        return _fate(
            UNKNOWN,
            f"Evaluation outcome={evaluation_status}; candidate-owned fate is not established.",
        )

    if ingestion_status == "succeeded" and ingestion_status_code is not None and ingestion_status_code > 0:
        if ingestion_status_code == 2:
            return _fate(
                ELITE_INSERTED,
                _archive_acceptance_reason(
                    "Candidate entered an empty archive niche.",
                    status_code=ingestion_status_code,
                    cell_index=cell_index,
                    current_archive_member=current_archive_member,
                ),
            )
        return _fate(
            ELITE_REPLACED,
            _archive_acceptance_reason(
                "Candidate improved an occupied archive niche.",
                status_code=ingestion_status_code,
                cell_index=cell_index,
                current_archive_member=current_archive_member,
            ),
        )

    if current_archive_member:
        return _fate(
            ELITE_RETAINED,
            _cell_reason("Candidate is a current archive elite.", cell_index=cell_index),
        )

    candidate_passed = evaluation_status == "passed" or (
        job_status == "succeeded" and bool(commit_hash)
    )
    if not candidate_passed:
        if job_status in {"pending", "queued", "running"}:
            return _fate(UNKNOWN, f"Job status={job_status}; no completed candidate fate yet.")
        if job_status == "failed":
            return _fate(UNKNOWN, "Job failed before a passing candidate fate was recorded.")
        return _fate(UNKNOWN, "No passing evaluation or archive decision is recorded.")

    if archive_status == "rejected" or ingestion_status == "skipped":
        return _fate(
            VALID_NOT_ELITE,
            _join_reason(
                "Candidate passed evaluation but did not enter the archive.",
                ingestion_message,
            ),
        )

    if archive_status == "member":
        return _fate(
            VALID_NOT_ELITE,
            "Candidate passed evaluation and has archive_status=member, but is not a current archive elite.",
        )

    if ingestion_status == "failed":
        return _fate(
            VALID_NOT_CONSIDERED,
            _join_reason(
                "Candidate passed evaluation but archive ingestion did not complete.",
                ingestion_message,
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
