from __future__ import annotations

import pytest

from loreley.config import Settings
from loreley.core.experiments import ExperimentError, build_experiment_config_snapshot


def test_settings_does_not_require_embedding_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAPELITES_CODE_EMBEDDING_DIMENSIONS", raising=False)

    settings = Settings(_env_file=None)
    assert settings.mapelites_code_embedding_dimensions is None


def test_experiment_snapshot_requires_embedding_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAPELITES_CODE_EMBEDDING_DIMENSIONS", raising=False)

    settings = Settings(mapelites_code_embedding_dimensions=None, _env_file=None)
    with pytest.raises(ExperimentError):
        build_experiment_config_snapshot(settings)


