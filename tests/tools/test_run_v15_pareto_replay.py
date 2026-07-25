from __future__ import annotations

import pytest

from tools.run_v15_pareto_replay import replay_policies, summarize_runtime


def test_summarize_runtime_reports_sample_and_bootstrap_interval() -> None:
    result = summarize_runtime([1.0, 2.0, 3.0, 4.0, 5.0])

    assert result["sample_count"] == 5
    assert result["p50_ms"] == 3.0
    assert result["p05_ms"] == pytest.approx(1.2)
    assert result["p95_ms"] == pytest.approx(4.8)
    assert result["median_bootstrap_95ci_low_ms"] <= 3.0
    assert result["median_bootstrap_95ci_high_ms"] >= 3.0


def test_replay_keeps_tradeoff_then_removes_it_for_real_dominator() -> None:
    measurements = {
        "baseline": {
            "quality_sum_radii": 0.25,
            "runtime_p50_ms": 0.002,
        },
        "deepseek": {
            "quality_sum_radii": 2.43,
            "runtime_p50_ms": 0.1,
        },
        "mini": {
            "quality_sum_radii": 2.44,
            "runtime_p50_ms": 0.001,
        },
    }

    result = replay_policies(measurements)

    assert result["pair_pareto_front"] == ["baseline", "deepseek"]
    assert result["pair_scalar_primary_retained"] == ["deepseek"]
    assert result["pair_tradeoff_observed"] is True
    assert result["front_after_dominator"] == ["mini"]
    assert result["dominator_collapsed_front"] is True
