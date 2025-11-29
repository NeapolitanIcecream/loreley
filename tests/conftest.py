from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest

import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loreley.config import Settings


@pytest.fixture
def settings() -> Generator[Settings, None, None]:
    """Return a fresh Settings instance for each test.

    Tests can freely mutate fields on this object without affecting others.
    """

    yield Settings()


