import pytest

from tests.support import TestSettings


def _settings(**overrides):
    values = {
        "OPENAI_API_KEY": "test",
        "MAPELITES_ADMISSION_POLICY": "fresh_comparison",
        "MAPELITES_OBJECTIVES": ({"name": "cost", "direction": "min"},),
        "MAPELITES_ISLANDS": ("main",),
        "MAPELITES_DIMENSION_REDUCTION_REFIT_INTERVAL": 0,
        "MAPELITES_MIGRATION_INTERVAL_JOBS": 0,
        "WORKER_EVALUATOR_MAX_CONCURRENCY": 1,
    }
    values.update(overrides)
    return TestSettings(**values)


def test_policy_is_opt_in_and_part_of_export():
    assert TestSettings().mapelites_admission_policy == "pareto"
    settings = _settings()
    assert settings.mapelites_comparison_confidence == .95
    assert "fresh_comparison" in str(settings.export_safe())


@pytest.mark.parametrize("override", [
    {"MAPELITES_ISLANDS": ("one", "two")},
    {"MAPELITES_DIMENSION_REDUCTION_REFIT_INTERVAL": 10},
    {"MAPELITES_MIGRATION_INTERVAL_JOBS": 10},
    {"WORKER_EVALUATOR_MAX_CONCURRENCY": 2},
    {"WORKER_EVALUATOR_MAX_CONCURRENCY": None},
    {"MAPELITES_OBJECTIVES": ({"name": "cost", "direction": "min"},
                             {"name": "memory", "direction": "min"})},
    {"MAPELITES_COMPARISON_CONFIDENCE": 1},
    {"MAPELITES_COMPARISON_CONFIDENCE": float("nan")},
])
def test_invalid_comparison_policy_fails_at_configuration(override):
    with pytest.raises(ValueError):
        _settings(**override)
