from __future__ import annotations

from loreley.config import Settings
from loreley.preflight import check_embedding_dimensions


def test_check_embedding_dimensions_missing(monkeypatch) -> None:
    monkeypatch.delenv("MAPELITES_CODE_EMBEDDING_DIMENSIONS", raising=False)
    settings = Settings(_env_file=None)
    result = check_embedding_dimensions(settings)
    assert result.status == "fail"


def test_check_embedding_dimensions_positive(monkeypatch) -> None:
    monkeypatch.setenv("MAPELITES_CODE_EMBEDDING_DIMENSIONS", "8")
    settings = Settings(_env_file=None)
    result = check_embedding_dimensions(settings)
    assert result.status == "ok"


