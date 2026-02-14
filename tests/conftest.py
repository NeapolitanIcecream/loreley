from __future__ import annotations

from pathlib import Path
from typing import Any, Generator

import pytest
from loguru import logger

import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Many modules may initialise Settings (via get_settings()) during import. The embedding
# dimensionality is now optional at process startup, so tests should pass explicit
# settings when they exercise the embedding pipeline.

from tests.support import TestSettings


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Generator[TestSettings, None, None]:
    """Return a fresh Settings instance for each test.

    Tests can freely mutate fields on this object without affecting others.
    """

    monkeypatch.setenv("MAPELITES_CODE_EMBEDDING_DIMENSIONS", "8")
    monkeypatch.setenv("EXPERIMENT_ID", "test")
    yield TestSettings()


@pytest.fixture
def captured_logs() -> Generator[list[dict[str, Any]], None, None]:
    """In-memory loguru sink for asserting diagnostic signals in tests.

    Each captured record is a dict with ``level``, ``message`` and ``module``
    keys.  ``module`` comes from the ``logger.bind(module=...)`` convention
    used across the codebase (see observability SKILL).
    """

    records: list[dict[str, Any]] = []

    def _sink(message: Any) -> None:
        record = message.record
        extra = dict(record["extra"])
        records.append(
            {
                "level": record["level"].name,
                "message": record["message"],
                "module": extra.get("module"),
                "extra": extra,
            }
        )

    handler_id = logger.add(_sink, level="DEBUG")
    yield records
    logger.remove(handler_id)


