from __future__ import annotations

from pathlib import Path

from loreley.core.map_elites.objectives import ObjectiveSpec
from loreley.preflight import check_campaign_program, check_seed_portfolio_planner


def _write_program(repo_root: Path, *, name: str, direction: str) -> None:
    (repo_root / "loreley.program.md").write_text(
        "\n".join(
            (
                "# Test campaign",
                "",
                "## Goal",
                "Improve the implementation.",
                "",
                "## Primary metric",
                f"name: {name}",
                f"direction: {direction}",
            )
        ),
        encoding="utf-8",
    )


def test_campaign_program_preflight_fails_on_primary_objective_name_mismatch(
    tmp_path: Path,
    settings,
) -> None:
    _write_program(tmp_path, name="throughput_geomean", direction="higher_is_better")
    settings.scheduler_repo_root = str(tmp_path)
    settings.mapelites_objectives = (
        ObjectiveSpec(name="compression_lower_95", direction="max"),
    )

    result = check_campaign_program(settings)

    assert result.status == "fail"
    assert "throughput_geomean" in result.details
    assert "compression_lower_95" in result.details


def test_campaign_program_preflight_fails_on_primary_objective_direction_mismatch(
    tmp_path: Path,
    settings,
) -> None:
    _write_program(tmp_path, name="latency_ms", direction="higher_is_better")
    settings.scheduler_repo_root = str(tmp_path)
    settings.mapelites_objectives = (ObjectiveSpec(name="latency_ms", direction="min"),)

    result = check_campaign_program(settings)

    assert result.status == "fail"
    assert "lower_is_better" in result.details


def test_campaign_program_preflight_accepts_matching_primary_objective(
    tmp_path: Path,
    settings,
) -> None:
    _write_program(tmp_path, name="compression_lower_95", direction="higher_is_better")
    settings.scheduler_repo_root = str(tmp_path)
    settings.mapelites_objectives = (
        ObjectiveSpec(name="compression_lower_95", direction="max"),
    )

    result = check_campaign_program(settings)

    assert result.status == "ok"


def test_seed_portfolio_preflight_enforces_sol_model_pin(settings) -> None:
    settings.mapelites_seed_portfolio_enabled = True
    settings.worker_seed_portfolio_model = "openai/gpt-5.6-luna"

    results = check_seed_portfolio_planner(settings)

    assert results[0].status == "fail"
    assert "gpt-5.6-sol" in results[0].details


def test_seed_portfolio_preflight_reports_opt_in_bounded_count(settings) -> None:
    disabled = check_seed_portfolio_planner(settings)

    assert disabled[0].status == "ok"
    assert "disabled (opt-in)" in disabled[0].details
    assert "directions=8 effective=8 hard_max=16" in disabled[0].details

    settings.mapelites_seed_portfolio_enabled = True
    settings.worker_seed_portfolio_backend = "tests.fake:portfolio_backend"
    enabled = check_seed_portfolio_planner(settings)

    assert enabled[0].status == "ok"
    assert "directions=8 effective=8 hard_max=16" in enabled[0].details
    assert "max_unsuccessful_attempts=2" in enabled[0].details
