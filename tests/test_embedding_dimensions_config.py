from __future__ import annotations

import pytest

from loreley.config import Settings
from loreley.core.map_elites.file_embedding_cache import build_file_embedding_cache


def test_settings_does_not_require_embedding_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAPELITES_CODE_EMBEDDING_DIMENSIONS", raising=False)

    settings = Settings(_env_file=None)
    assert settings.mapelites_code_embedding_dimensions is None


def test_file_embedding_cache_requires_embedding_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MAPELITES_CODE_EMBEDDING_DIMENSIONS", raising=False)

    settings = Settings(mapelites_code_embedding_dimensions=None, _env_file=None)
    with pytest.raises(ValueError):
        build_file_embedding_cache(settings=settings, backend="memory")


