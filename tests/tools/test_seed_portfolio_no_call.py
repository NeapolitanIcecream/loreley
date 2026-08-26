from __future__ import annotations

import json
from pathlib import Path

from tools.replay_seed_portfolio_no_call import build_replay_report


def test_seed_portfolio_no_call_replay_passes_every_acceptance_check() -> None:
    report = build_replay_report()

    assert report["checks"]
    assert all(report["checks"].values())
    assert len(set(report["prompt_sha256"])) == 2


def test_checked_in_seed_portfolio_replay_report_matches_current_code() -> None:
    report_path = (
        Path(__file__).resolve().parents[2]
        / "reports"
        / "seed-portfolio-no-call-replay.json"
    )

    persisted = json.loads(report_path.read_text(encoding="utf-8"))

    assert persisted == build_replay_report()
