"""Evaluator-scoped identities for equivalent candidate states."""

from __future__ import annotations

import hashlib
import json

from loreley.core.contracts import clamp_text, normalize_single_line

__all__ = ["evaluation_identity_key", "normalize_candidate_identity"]


def normalize_candidate_identity(value: object) -> str | None:
    """Return a bounded evaluator-provided identity, if one was supplied."""

    if value is None:
        return None
    return clamp_text(normalize_single_line(str(value)), 512) or None


def evaluation_identity_key(
    *,
    candidate_identity: object,
    evaluator_name: object,
    evaluator_version: object,
    campaign_program_hash: object,
    measurement_contract_fingerprint: object = None,
) -> str | None:
    """Hash the search identity independently of measurement protocol details.

    ``measurement_contract_fingerprint`` remains an accepted keyword for API
    compatibility, but it deliberately does not affect archive/search identity.
    The phased measurement cache includes it separately.
    """

    del measurement_contract_fingerprint

    identity = normalize_candidate_identity(candidate_identity)
    if identity is None:
        return None
    payload = {
        "campaign_program_hash": _bounded(campaign_program_hash, 64),
        "candidate_identity": identity,
        "evaluator_name": _bounded(evaluator_name, 128),
        "evaluator_version": _bounded(evaluator_version, 128),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bounded(value: object, limit: int) -> str:
    if value is None:
        return ""
    return clamp_text(normalize_single_line(str(value)), limit)
