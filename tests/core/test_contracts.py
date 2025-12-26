from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from loreley.core.contracts import CommitCard, EvolutionJobSpec


def _minimal_commit_card(**overrides: Any) -> CommitCard:
    payload: dict[str, Any] = {
        "commit_hash": "a" * 40,
        "subject": "Fix: keep contracts small",
        "change_summary": "Limit payload sizes for hot-path data.",
        "highlights": ["Enforce list and string budgets."],
    }
    payload.update(overrides)
    return CommitCard(**payload)


def _minimal_job_spec(**overrides: Any) -> EvolutionJobSpec:
    payload: dict[str, Any] = {
        "goal": "Improve the system without exceeding contract budgets.",
        "base_commit_hash": "b" * 40,
    }
    payload.update(overrides)
    return EvolutionJobSpec(**payload)


def test_commit_card_key_files_rejects_overflow() -> None:
    _minimal_commit_card(key_files=["a.py"] * 20)
    with pytest.raises(ValidationError):
        _minimal_commit_card(key_files=["a.py"] * 21)


@pytest.mark.parametrize(
    ("field_name", "max_len"),
    [
        ("constraints", 20),
        ("acceptance_criteria", 20),
        ("notes", 20),
        ("tags", 20),
        ("inspiration_commit_hashes", 10),
    ],
)
def test_evolution_job_spec_list_fields_reject_overflow(field_name: str, max_len: int) -> None:
    _minimal_job_spec(**{field_name: ["x"] * max_len})
    with pytest.raises(ValidationError):
        _minimal_job_spec(**{field_name: ["x"] * (max_len + 1)})


@pytest.mark.parametrize(
    "bad_value",
    [
        "",
        "a" * 201,
    ],
)
def test_evolution_job_spec_constraints_elements_are_bounded(bad_value: str) -> None:
    with pytest.raises(ValidationError):
        _minimal_job_spec(constraints=[bad_value])


def test_evolution_job_spec_tags_elements_are_bounded() -> None:
    with pytest.raises(ValidationError):
        _minimal_job_spec(tags=[""])
    with pytest.raises(ValidationError):
        _minimal_job_spec(tags=["a" * 65])


def test_evolution_job_spec_inspiration_commit_hashes_elements_are_bounded() -> None:
    with pytest.raises(ValidationError):
        _minimal_job_spec(inspiration_commit_hashes=[""])
    with pytest.raises(ValidationError):
        _minimal_job_spec(inspiration_commit_hashes=["a" * 65])


def test_commit_card_subject_normalizes_whitespace() -> None:
    card = _minimal_commit_card(subject="Hello \n  world")
    assert card.subject == "Hello world"


@pytest.mark.parametrize("bad_subject", ["```python", "{ \"a\": 1 }", "[1, 2, 3]"])
def test_commit_card_subject_rejects_code_fences_and_json_like_prefix(bad_subject: str) -> None:
    with pytest.raises(ValidationError):
        _minimal_commit_card(subject=bad_subject)


