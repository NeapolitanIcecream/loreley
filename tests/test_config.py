from __future__ import annotations

import pytest
from pydantic import ValidationError

from tests.support import TestSettings


def test_campaign_program_approve_policy_is_rejected_until_implemented() -> None:
    """Regression: approve policy was accepted while no approval flow existed."""

    with pytest.raises(ValidationError) as exc_info:
        TestSettings(CAMPAIGN_PROGRAM_CHANGE_POLICY="approve")

    errors = exc_info.value.errors()
    assert errors[0]["loc"] == ("CAMPAIGN_PROGRAM_CHANGE_POLICY",)
