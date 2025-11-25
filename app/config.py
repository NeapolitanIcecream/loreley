from __future__ import annotations

from functools import lru_cache
from typing import Any
from urllib.parse import quote_plus

from loguru import logger
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console

console = Console()
log = logger.bind(module="config")


class Settings(BaseSettings):
    """Centralised application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="Loreley", alias="APP_NAME")
    environment: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    database_url: str | None = Field(default=None, alias="DATABASE_URL")
    db_scheme: str = Field(default="postgresql+psycopg", alias="DB_SCHEME")
    db_host: str = Field(default="localhost", alias="DB_HOST")
    db_port: int = Field(default=5432, alias="DB_PORT")
    db_username: str = Field(default="postgres", alias="DB_USER")
    db_password: str = Field(default="postgres", alias="DB_PASSWORD")
    db_name: str = Field(default="loreley", alias="DB_NAME")
    db_pool_size: int = Field(default=10, alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=20, alias="DB_MAX_OVERFLOW")
    db_pool_timeout: int = Field(default=30, alias="DB_POOL_TIMEOUT")
    db_echo: bool = Field(default=False, alias="DB_ECHO")

    metrics_retention_days: int = Field(default=30, alias="METRICS_RETENTION_DAYS")

    @computed_field(return_type=str)
    @property
    def database_dsn(self) -> str:
        """Return a SQLAlchemy compatible DSN."""
        if self.database_url:
            return self.database_url

        username = quote_plus(self.db_username)
        password = quote_plus(self.db_password)
        return (
            f"{self.db_scheme}://{username}:{password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    def export_safe(self) -> dict[str, Any]:
        """Return non-sensitive settings for debugging/logging."""
        return {
            "app_name": self.app_name,
            "environment": self.environment,
            "db_host": self.db_host,
            "db_port": self.db_port,
            "db_name": self.db_name,
            "db_pool_size": self.db_pool_size,
            "db_max_overflow": self.db_max_overflow,
            "db_pool_timeout": self.db_pool_timeout,
            "db_echo": self.db_echo,
        }


@lru_cache
def get_settings() -> Settings:
    """Load and cache application settings."""
    settings = Settings()
    console.log(
        f"[bold green]Loaded settings[/] env={settings.environment!r} "
        f"db_host={settings.db_host!r}",
    )
    log.info("Settings initialised: {}", settings.export_safe())
    return settings
