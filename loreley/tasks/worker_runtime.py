"""Spawn-safe Dramatiq bootstrap for ``loreley worker --processes``."""

from __future__ import annotations

from rich.console import Console

from loreley.config import get_settings
from loreley.entrypoints import (
    _apply_dramatiq_prefetch_settings,
    configure_process_logging,
)
from loreley.tasks.broker import setup_broker
from loreley.tasks.workers import build_evolution_job_worker_actor

settings = get_settings()
console = Console()
configure_process_logging(settings=settings, console=console, role="worker")
_apply_dramatiq_prefetch_settings(settings=settings, console=console)
broker = setup_broker(settings=settings)
run_evolution_job = build_evolution_job_worker_actor(
    settings=settings,
    broker=broker,
)

__all__ = ["broker", "run_evolution_job"]
