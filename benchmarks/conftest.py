from __future__ import annotations

from pathlib import Path
import sys
from typing import Generator

import pytest
from loguru import logger
from pydantic_settings import SettingsConfigDict

from loreley.config import Settings

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class BenchmarkSettings(Settings):
    """Settings variant for benchmarks.

    Benchmarks should not depend on a developer's local `.env` file.
    """

    # Prevent pytest from attempting to collect this as a test class.
    __test__ = False

    model_config = SettingsConfigDict(
        env_file=None,
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Generator[BenchmarkSettings, None, None]:
    """Return a fresh Settings instance for each benchmark test."""

    monkeypatch.setenv("MAPELITES_CODE_EMBEDDING_DIMENSIONS", "8")
    monkeypatch.setenv("EXPERIMENT_ID", "benchmark")
    yield BenchmarkSettings()


@pytest.fixture(autouse=True)
def _disable_loreley_logs() -> Generator[None, None, None]:
    """Avoid log overhead during tight benchmark loops."""

    logger.disable("loreley")
    yield
    logger.enable("loreley")

