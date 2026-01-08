from __future__ import annotations

import pytest

from loreley.scheduler.startup_approval import validate_repo_state_eligible_files_approval


def test_validate_repo_state_eligible_files_approval_requires_count() -> None:
    with pytest.raises(ValueError, match="requires explicit approval"):
        validate_repo_state_eligible_files_approval(
            observed_eligible_files=3,
            approved_count=None,
            root_commit="deadbeef",
        )


def test_validate_repo_state_eligible_files_approval_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="approval mismatch"):
        validate_repo_state_eligible_files_approval(
            observed_eligible_files=3,
            approved_count=2,
            root_commit="deadbeef",
        )


def test_validate_repo_state_eligible_files_approval_accepts_match() -> None:
    validate_repo_state_eligible_files_approval(
        observed_eligible_files=3,
        approved_count=3,
        root_commit="deadbeef",
    )


