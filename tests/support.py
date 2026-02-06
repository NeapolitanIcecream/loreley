from __future__ import annotations

from pydantic_settings import SettingsConfigDict

from loreley.config import Settings


class TestSettings(Settings):
    """Settings variant for tests.

    Tests should not depend on a developer's local `.env` file. This subclass
    disables `env_file` loading and relies on environment variables only.
    """

    model_config = SettingsConfigDict(
        env_file=None,
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )

