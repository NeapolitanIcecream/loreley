from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest

import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Many modules may initialise Settings (via get_settings()) during import. The embedding
# dimensionality is now optional at process startup, so tests should pass explicit
# settings when they exercise the embedding pipeline.

from loreley.config import Settings


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Generator[Settings, None, None]:
    """Return a fresh Settings instance for each test.

    Tests can freely mutate fields on this object without affecting others.
    """

    monkeypatch.setenv("MAPELITES_CODE_EMBEDDING_DIMENSIONS", "8")
    yield Settings(mapelites_code_embedding_dimensions=8, _env_file=None)


