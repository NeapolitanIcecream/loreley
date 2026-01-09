from __future__ import annotations

from loreley.config import Settings
from loreley.core.experiment_config import apply_experiment_config_snapshot


def test_apply_experiment_config_snapshot_overrides_behavior_keys(settings: Settings) -> None:
    settings.mapelites_dimensionality_target_dims = 4
    settings.mapelites_default_island_id = "main"
    settings.worker_evaluator_timeout_seconds = 1
    settings.app_name = "Loreley"

    snapshot = {
        "mapelites_dimensionality_target_dims": 2,
        "mapelites_default_island_id": "circle_packing",
        "worker_evaluator_timeout_seconds": 900,
        # Non-behavior keys must not override the base settings.
        "app_name": "ShouldNotApply",
    }

    effective = apply_experiment_config_snapshot(base_settings=settings, snapshot=snapshot)

    assert effective.mapelites_dimensionality_target_dims == 2
    assert effective.mapelites_default_island_id == "circle_packing"
    assert effective.worker_evaluator_timeout_seconds == 900
    assert effective.app_name == "Loreley"

    # Ensure we do not mutate the base settings object.
    assert settings.mapelites_dimensionality_target_dims == 4
    assert settings.mapelites_default_island_id == "main"
    assert settings.worker_evaluator_timeout_seconds == 1


def test_apply_experiment_config_snapshot_restores_non_finite_float_markers(settings: Settings) -> None:
    settings.mapelites_archive_threshold_min = 0.0

    snapshot = {
        "mapelites_archive_threshold_min": {"__float__": "-inf"},
    }

    effective = apply_experiment_config_snapshot(base_settings=settings, snapshot=snapshot)
    assert effective.mapelites_archive_threshold_min == float("-inf")


def test_apply_experiment_config_snapshot_overrides_worker_evolution_and_trajectory(settings: Settings) -> None:
    settings.worker_evolution_global_goal = "Goal A"
    settings.worker_planning_trajectory_block_size = 8

    snapshot = {
        "worker_evolution_global_goal": "Goal B",
        "worker_planning_trajectory_block_size": 16,
    }

    effective = apply_experiment_config_snapshot(base_settings=settings, snapshot=snapshot)
    assert effective.worker_evolution_global_goal == "Goal B"
    assert effective.worker_planning_trajectory_block_size == 16

