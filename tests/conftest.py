from __future__ import annotations

from pathlib import Path
from typing import Generator

import os
import pytest

import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Settings are loaded from environment in many modules at import time (via get_settings()).
# Tests must provide required env vars up-front to avoid collection-time failures.
os.environ.setdefault("MAPELITES_CODE_EMBEDDING_DIMENSIONS", "8")

from loreley.config import Settings


@pytest.fixture
def settings() -> Generator[Settings, None, None]:
    """Return a fresh Settings instance for each test.

    Tests can freely mutate fields on this object without affecting others.
    """

    yield Settings(mapelites_code_embedding_dimensions=8)


