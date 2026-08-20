from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote_plus, urlparse, urlunparse

from loguru import logger
from pydantic import AliasChoices, Field, PositiveInt, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console

from loreley.core.map_elites.objectives import ObjectiveContract, ObjectiveSpec

console = Console()
log = logger.bind(module="config")

_LARGE_REPO_PROFILE_ALIASES = {"large-repo-1m-30k", "large_repo_1m_30k"}
_MAX_ISLAND_ID_LENGTH = 64
_LARGE_REPO_PROFILE_DEFAULTS: dict[str, object] = {
    "scheduler_poll_interval_seconds": 15.0,
    "tasks_worker_time_limit_seconds": 0,
    "worker_planning_timeout_seconds": 1200,
    "worker_coding_timeout_seconds": 3600,
    "worker_evaluator_timeout_seconds": 3600,
    "mapelites_preprocess_max_file_size_kb": 256,
    "mapelites_chunk_target_lines": 120,
    "mapelites_chunk_min_lines": 40,
    "mapelites_chunk_overlap_lines": 12,
    "mapelites_code_embedding_batch_size": 64,
    "mapelites_dimensionality_min_fit_samples": 128,
    "mapelites_dimensionality_history_size": 8192,
    "mapelites_dimensionality_refit_interval": 250,
    "mapelites_feature_normalization_warmup_samples": 128,
    "mapelites_seed_population_size": 32,
    "mapelites_sampler_inspiration_count": 4,
    "mapelites_sampler_neighbor_max_radius": 4,
    "mapelites_sampler_fallback_sample_size": 32,
}


def _normalized_profile_name(profile: object) -> str:
    raw = str(profile or "").strip()
    return raw.lower() if raw else "default"


def _profile_derived_defaults(profile: object) -> dict[str, object]:
    if _normalized_profile_name(profile) in _LARGE_REPO_PROFILE_ALIASES:
        return dict(_LARGE_REPO_PROFILE_DEFAULTS)
    return {}


def _mask_secret(value: str | None) -> str | None:
    """Return a constant marker for present secrets."""
    normalized = (value or "").strip()
    if not normalized:
        return None
    return "***"


def _sanitize_url(raw: str) -> str:
    """Best-effort redaction for credential-bearing URLs."""
    value = (raw or "").strip()
    if not value:
        return value
    parsed = urlparse(value)
    if not parsed.scheme:
        return value
    netloc = parsed.netloc
    if "@" in netloc:
        netloc = netloc.rsplit("@", 1)[1]
    safe = parsed._replace(netloc=netloc, query="", fragment="")
    return urlunparse(safe)


def _sanitize_sqlalchemy_dsn(raw: str) -> str:
    """Strip credential-bearing DSN fields for logs and masked exports."""
    try:
        from sqlalchemy.engine.url import make_url
    except Exception:
        return _sanitize_url(raw)

    try:
        make_url(raw)
        return _sanitize_url(raw)
    except Exception:
        return _sanitize_url(raw)


class Settings(BaseSettings):
    """Centralised application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )

    app_name: str = Field(default="Loreley", alias="APP_NAME")
    environment: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    profile: str = Field(default="default", alias="LORELEY_PROFILE")
    logs_base_dir: str | None = Field(
        default=None,
        alias="LOGS_BASE_DIR",
    )
    loreley_agent_api_token: str | None = Field(
        default=None,
        alias="LORELEY_AGENT_API_TOKEN",
    )
    loreley_api_write_token: str | None = Field(
        default=None,
        alias="LORELEY_API_WRITE_TOKEN",
    )

    # OpenAI-compatible API configuration
    openai_api_key: str | None = Field(
        default=None,
        alias="OPENAI_API_KEY",
        validation_alias=AliasChoices("OPENAI_API_KEY", "LORELEY_LLM_API_KEY"),
    )
    openai_base_url: str | None = Field(
        default=None,
        alias="OPENAI_BASE_URL",
        validation_alias=AliasChoices("OPENAI_BASE_URL", "LORELEY_LLM_BASE_URL"),
    )
    openai_dynamic_api_key_provider: str | None = Field(
        default=None,
        alias="OPENAI_DYNAMIC_API_KEY_PROVIDER",
    )
    openai_dynamic_api_key_ttl_seconds: int | None = Field(
        default=None,
        alias="OPENAI_DYNAMIC_API_KEY_TTL_SECONDS",
    )
    openai_dynamic_api_key_refresh_skew_seconds: int | None = Field(
        default=None,
        alias="OPENAI_DYNAMIC_API_KEY_REFRESH_SKEW_SECONDS",
    )
    openai_api_spec: Literal["responses", "chat_completions"] = Field(
        default="responses",
        alias="OPENAI_API_SPEC",
    )
    llm_usage_tracking_enabled: bool = Field(
        default=True,
        alias="LLM_USAGE_TRACKING_ENABLED",
    )
    llm_usage_pricing_path: str | None = Field(
        default=None,
        alias="LLM_USAGE_PRICING_PATH",
    )
    llm_usage_pricing_json: str | None = Field(
        default=None,
        alias="LLM_USAGE_PRICING_JSON",
    )

    database_url: str | None = Field(default=None, alias="DATABASE_URL")
    db_scheme: str = Field(default="postgresql+psycopg", alias="DB_SCHEME")
    db_host: str = Field(default="localhost", alias="DB_HOST")
    db_port: int = Field(default=5432, alias="DB_PORT")
    db_username: str = Field(default="loreley", alias="DB_USER")
    db_password: str = Field(default="loreley", alias="DB_PASSWORD")
    db_name: str = Field(default="loreley", alias="DB_NAME")
    db_pool_size: int = Field(default=10, alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=20, alias="DB_MAX_OVERFLOW")
    db_pool_timeout: int = Field(default=30, alias="DB_POOL_TIMEOUT")
    db_echo: bool = Field(default=False, alias="DB_ECHO")
    db_auto_migrate: bool = Field(default=True, alias="DB_AUTO_MIGRATE")
    db_migration_lock_timeout_seconds: PositiveInt = Field(
        default=30,
        alias="DB_MIGRATION_LOCK_TIMEOUT_SECONDS",
    )

    metrics_retention_days: int = Field(default=30, alias="METRICS_RETENTION_DAYS")

    tasks_redis_url: str | None = Field(default=None, alias="TASKS_REDIS_URL")
    tasks_redis_host: str = Field(default="localhost", alias="TASKS_REDIS_HOST")
    tasks_redis_port: int = Field(default=6379, alias="TASKS_REDIS_PORT")
    tasks_redis_db: int = Field(default=0, alias="TASKS_REDIS_DB")
    tasks_redis_password: str | None = Field(default=None, alias="TASKS_REDIS_PASSWORD")
    tasks_queue_prefetch: int = Field(default=1, alias="TASKS_QUEUE_PREFETCH")
    tasks_delay_queue_prefetch: int = Field(default=1, alias="TASKS_DELAY_QUEUE_PREFETCH")
    tasks_worker_max_retries: int = Field(default=0, alias="TASKS_WORKER_MAX_RETRIES")
    tasks_worker_time_limit_seconds: int = Field(
        default=3600,
        alias="TASKS_WORKER_TIME_LIMIT_SECONDS",
    )

    # Experiment / evolution configuration
    experiment_id: uuid.UUID | str | None = Field(
        default=None,
        alias="EXPERIMENT_ID",
    )
    mapelites_experiment_root_commit: str | None = Field(
        default=None,
        alias="MAPELITES_EXPERIMENT_ROOT_COMMIT",
    )
    # Experiment-scoped, pinned ignore rules used by repo-state embeddings.
    #
    # In the env-only settings model, the scheduler pins these values once at
    # process startup by reading `.gitignore` + `.loreleyignore` from the
    # configured root commit and storing the combined ignore text in
    # Settings for the lifetime of the process.
    #
    # They remain optional at process startup so local tools/tests can construct
    # Settings without a repository context.
    mapelites_repo_state_ignore_text: str | None = Field(
        default=None,
        alias="MAPELITES_REPO_STATE_IGNORE_TEXT",
    )

    scheduler_repo_root: str | None = Field(
        default=None,
        alias="SCHEDULER_REPO_ROOT",
    )
    scheduler_poll_interval_seconds: float = Field(
        default=30.0,
        alias="SCHEDULER_POLL_INTERVAL_SECONDS",
    )
    scheduler_max_unfinished_jobs: int = Field(
        default=4,
        alias="SCHEDULER_MAX_UNFINISHED_JOBS",
    )
    scheduler_max_total_jobs: int | None = Field(
        default=None,
        alias="SCHEDULER_MAX_TOTAL_JOBS",
    )
    scheduler_max_unique_evaluation_identities: int | None = Field(
        default=None,
        ge=1,
        alias="SCHEDULER_MAX_UNIQUE_EVALUATION_IDENTITIES",
    )
    scheduler_schedule_batch_size: int = Field(
        default=2,
        alias="SCHEDULER_SCHEDULE_BATCH_SIZE",
    )
    scheduler_dispatch_batch_size: int = Field(
        default=4,
        alias="SCHEDULER_DISPATCH_BATCH_SIZE",
    )
    scheduler_ingest_batch_size: int = Field(
        default=2,
        alias="SCHEDULER_INGEST_BATCH_SIZE",
    )
    scheduler_startup_approve: bool = Field(
        default=False,
        alias="SCHEDULER_STARTUP_APPROVE",
    )
    scheduler_stale_running_reclaim_batch_size: int = Field(
        default=32,
        alias="SCHEDULER_STALE_RUNNING_RECLAIM_BATCH_SIZE",
    )
    scheduler_stale_running_max_recovery_attempts: int = Field(
        default=3,
        alias="SCHEDULER_STALE_RUNNING_MAX_RECOVERY_ATTEMPTS",
    )
    campaign_program_change_policy: Literal["locked", "auto"] = Field(
        default="locked",
        alias="CAMPAIGN_PROGRAM_CHANGE_POLICY",
    )
    baseline_bootstrap_policy: Literal["required", "warn"] = Field(
        default="required",
        alias="BASELINE_BOOTSTRAP_POLICY",
    )

    failed_candidate_repair_enabled: bool = Field(
        default=False,
        alias="FAILED_CANDIDATE_REPAIR_ENABLED",
    )
    failed_candidate_repair_mode: Literal["rebase_from_nearest_viable"] = Field(
        default="rebase_from_nearest_viable",
        alias="FAILED_CANDIDATE_REPAIR_MODE",
    )
    # Deprecated compatibility setting. MVP scheduling is one-generation only:
    # original failed candidates with failed_depth=0 and no repair source.
    failed_candidate_repair_max_depth: int = Field(
        default=1,
        alias="FAILED_CANDIDATE_REPAIR_MAX_DEPTH",
    )
    failed_candidate_repair_max_attempts: int = Field(
        default=1,
        alias="FAILED_CANDIDATE_REPAIR_MAX_ATTEMPTS",
    )
    failed_candidate_repair_normal_jobs_per_token: int = Field(
        default=9,
        alias="FAILED_CANDIDATE_REPAIR_NORMAL_JOBS_PER_TOKEN",
    )
    failed_candidate_repair_max_tokens: int = Field(
        default=3,
        alias="FAILED_CANDIDATE_REPAIR_MAX_TOKENS",
    )
    failed_candidate_repair_max_active_jobs: int = Field(
        default=1,
        alias="FAILED_CANDIDATE_REPAIR_MAX_ACTIVE_JOBS",
    )
    failed_candidate_repair_max_jobs_per_tick: int = Field(
        default=1,
        alias="FAILED_CANDIDATE_REPAIR_MAX_JOBS_PER_TICK",
    )
    failed_candidate_repair_failure_kinds: str = Field(
        default="validation_failed,test_failed,typecheck_failed,lint_failed",
        alias="FAILED_CANDIDATE_REPAIR_FAILURE_KINDS",
    )
    failed_candidate_repair_agent_feedback_mode: str = Field(
        default="diagnostic_capsule",
        alias="FAILED_CANDIDATE_REPAIR_AGENT_FEEDBACK_MODE",
    )
    failed_candidate_repair_max_diff_bytes: int = Field(
        default=65_536,
        alias="FAILED_CANDIDATE_REPAIR_MAX_DIFF_BYTES",
    )
    failed_candidate_repair_max_diagnostic_bytes: int = Field(
        default=16_384,
        alias="FAILED_CANDIDATE_REPAIR_MAX_DIAGNOSTIC_BYTES",
    )
    worker_evaluator_rework_enabled: bool = Field(
        default=True,
        alias="WORKER_EVALUATOR_REWORK_ENABLED",
    )
    worker_evaluator_rework_max_attempts: int = Field(
        default=1,
        alias="WORKER_EVALUATOR_REWORK_MAX_ATTEMPTS",
    )
    worker_evaluator_rework_failure_kinds: str = Field(
        default="compile,typecheck,lint,test,validation",
        alias="WORKER_EVALUATOR_REWORK_FAILURE_KINDS",
    )
    worker_evaluator_rework_max_seconds: int = Field(
        default=1800,
        alias="WORKER_EVALUATOR_REWORK_MAX_SECONDS",
    )

    worker_repo_remote_url: str | None = Field(
        default=None,
        alias="WORKER_REPO_REMOTE_URL",
    )
    worker_repo_branch: str = Field(
        default="main",
        alias="WORKER_REPO_BRANCH",
    )
    worker_repo_worktree: str = Field(
        default_factory=lambda: str(Path.home() / ".cache" / "loreley" / "worker-repo"),
        alias="WORKER_REPO_WORKTREE",
    )
    worker_repo_worktree_randomize: bool = Field(
        default=False,
        alias="WORKER_REPO_WORKTREE_RANDOMIZE",
    )
    worker_repo_worktree_random_suffix_len: int = Field(
        default=8,
        alias="WORKER_REPO_WORKTREE_RANDOM_SUFFIX_LEN",
    )
    worker_repo_git_bin: str = Field(
        default="git",
        alias="WORKER_REPO_GIT_BIN",
    )
    worker_repo_fetch_depth: int | None = Field(
        default=None,
        alias="WORKER_REPO_FETCH_DEPTH",
    )
    worker_repo_clean_excludes: list[str] = Field(
        default_factory=lambda: [".venv", ".uv", ".python-version"],
        alias="WORKER_REPO_CLEAN_EXCLUDES",
    )
    worker_scope_gate_cleanup_paths: str = Field(
        default="",
        alias="WORKER_SCOPE_GATE_CLEANUP_PATHS",
    )
    worker_repo_enable_lfs: bool = Field(
        default=True,
        alias="WORKER_REPO_ENABLE_LFS",
    )
    worker_repo_job_branch_ttl_hours: int = Field(
        default=168,
        alias="WORKER_REPO_JOB_BRANCH_TTL_HOURS",
    )
    worker_job_lease_ttl_seconds: int = Field(
        default=1800,
        alias="WORKER_JOB_LEASE_TTL_SECONDS",
    )
    worker_job_heartbeat_interval_seconds: int = Field(
        default=60,
        alias="WORKER_JOB_HEARTBEAT_INTERVAL_SECONDS",
    )
    worker_processes: PositiveInt = Field(
        default=1,
        alias="WORKER_PROCESSES",
    )

    worker_planning_codex_bin: str = Field(
        default="codex",
        alias="WORKER_PLANNING_CODEX_BIN",
    )
    worker_planning_codex_profile: str | None = Field(
        default=None,
        alias="WORKER_PLANNING_CODEX_PROFILE",
    )
    worker_planning_codex_model: str | None = Field(
        default=None,
        alias="WORKER_PLANNING_CODEX_MODEL",
    )
    worker_planning_max_attempts: int = Field(
        default=2,
        alias="WORKER_PLANNING_MAX_ATTEMPTS",
    )
    worker_planning_timeout_seconds: int = Field(
        default=900,
        alias="WORKER_PLANNING_TIMEOUT_SECONDS",
    )
    worker_planning_extra_env: dict[str, str] = Field(
        default_factory=dict,
        alias="WORKER_PLANNING_EXTRA_ENV",
    )
    worker_coding_codex_bin: str = Field(
        default="codex",
        alias="WORKER_CODING_CODEX_BIN",
    )
    worker_coding_codex_profile: str | None = Field(
        default=None,
        alias="WORKER_CODING_CODEX_PROFILE",
    )
    worker_coding_codex_model: str | None = Field(
        default=None,
        alias="WORKER_CODING_CODEX_MODEL",
    )
    worker_coding_max_attempts: int = Field(
        default=2,
        alias="WORKER_CODING_MAX_ATTEMPTS",
    )
    worker_coding_timeout_seconds: int = Field(
        default=1800,
        alias="WORKER_CODING_TIMEOUT_SECONDS",
    )
    worker_coding_extra_env: dict[str, str] = Field(
        default_factory=dict,
        alias="WORKER_CODING_EXTRA_ENV",
    )
    worker_planning_backend: str | None = Field(
        default="loreley.core.worker.agent.backends.kilocode_cli:kilocode_planning_backend",
        alias="WORKER_PLANNING_BACKEND",
    )
    worker_coding_backend: str | None = Field(
        default="loreley.core.worker.agent.backends.kilocode_cli:kilocode_coding_backend",
        alias="WORKER_CODING_BACKEND",
    )
    worker_cursor_model: str = Field(
        default="gpt-5.2-high",
        alias="WORKER_CURSOR_MODEL",
    )
    worker_cursor_force: bool = Field(
        default=True,
        alias="WORKER_CURSOR_FORCE",
    )
    worker_kilocode_bin: str = Field(
        default="kilo",
        alias="WORKER_KILOCODE_BIN",
    )
    worker_kilocode_mode: str | None = Field(
        default=None,
        alias="WORKER_KILOCODE_MODE",
    )
    worker_kilocode_agent: str | None = Field(
        default=None,
        alias="WORKER_KILOCODE_AGENT",
    )
    worker_kilocode_model: str | None = Field(
        default=None,
        alias="WORKER_KILOCODE_MODEL",
    )
    worker_kilocode_planning_model: str | None = Field(
        default=None,
        alias="WORKER_KILOCODE_PLANNING_MODEL",
    )
    worker_kilocode_coding_model: str | None = Field(
        default=None,
        alias="WORKER_KILOCODE_CODING_MODEL",
    )
    worker_kilocode_variant: str | None = Field(
        default=None,
        alias="WORKER_KILOCODE_VARIANT",
    )
    worker_kilocode_pure: bool = Field(
        default=False,
        alias="WORKER_KILOCODE_PURE",
    )
    worker_kilocode_json_output: bool = Field(
        default=False,
        alias="WORKER_KILOCODE_JSON_OUTPUT",
    )
    worker_kilocode_usage_db_path: str | None = Field(
        default=None,
        alias="WORKER_KILOCODE_USAGE_DB_PATH",
    )
    worker_kilocode_state_root: str | None = Field(
        default=None,
        alias="WORKER_KILOCODE_STATE_ROOT",
    )
    worker_kilocode_provider_config_mode: Literal[
        "auto", "config", "legacy_env", "native", "none"
    ] = Field(
        default="auto",
        alias="WORKER_KILOCODE_PROVIDER_CONFIG_MODE",
    )
    worker_kilocode_openai_api_spec: Literal["responses", "chat_completions"] | None = Field(
        default=None,
        alias="WORKER_KILOCODE_OPENAI_API_SPEC",
    )
    worker_kilocode_openai_base_url: str | None = Field(
        default=None,
        alias="WORKER_KILOCODE_OPENAI_BASE_URL",
    )
    worker_kilocode_openai_api_key: str | None = Field(
        default=None,
        alias="WORKER_KILOCODE_OPENAI_API_KEY",
    )
    worker_kilocode_openai_model: str | None = Field(
        default=None,
        alias="WORKER_KILOCODE_OPENAI_MODEL",
    )
    worker_evaluator_plugin: str | None = Field(
        default=None,
        alias="WORKER_EVALUATOR_PLUGIN",
    )
    worker_evaluator_version: str | None = Field(
        default=None,
        alias="WORKER_EVALUATOR_VERSION",
    )
    worker_evaluator_python_paths: list[str] = Field(
        default_factory=list,
        alias="WORKER_EVALUATOR_PYTHON_PATHS",
    )
    worker_evaluator_timeout_seconds: int = Field(
        default=900,
        alias="WORKER_EVALUATOR_TIMEOUT_SECONDS",
    )
    worker_evaluator_max_metrics: int = Field(
        default=64,
        alias="WORKER_EVALUATOR_MAX_METRICS",
    )
    worker_evaluator_max_concurrency: int | None = Field(
        default=None,
        ge=1,
        alias="WORKER_EVALUATOR_MAX_CONCURRENCY",
    )
    worker_evaluator_slot_poll_seconds: float = Field(
        default=0.25,
        gt=0.0,
        alias="WORKER_EVALUATOR_SLOT_POLL_SECONDS",
    )
    worker_evaluator_measurement_max_json_bytes: PositiveInt = Field(
        default=8_388_608,
        alias="WORKER_EVALUATOR_MEASUREMENT_MAX_JSON_BYTES",
    )
    worker_evaluation_artifacts_enabled: bool = Field(
        default=True,
        alias="WORKER_EVALUATION_ARTIFACTS_ENABLED",
    )
    worker_evaluation_agent_feedback_mode: Literal["disabled", "manifest", "summary", "path"] = Field(
        default="summary",
        alias="WORKER_EVALUATION_AGENT_FEEDBACK_MODE",
    )
    worker_evaluation_agent_feedback_max_artifacts: int = Field(
        default=4,
        alias="WORKER_EVALUATION_AGENT_FEEDBACK_MAX_ARTIFACTS",
    )
    worker_evaluation_agent_feedback_max_diagnostics: int = Field(
        default=3,
        alias="WORKER_EVALUATION_AGENT_FEEDBACK_MAX_DIAGNOSTICS",
    )
    worker_evaluation_agent_feedback_max_chars: int = Field(
        default=2000,
        alias="WORKER_EVALUATION_AGENT_FEEDBACK_MAX_CHARS",
    )
    worker_evaluation_artifact_max_bytes: int = Field(
        default=10_485_760,
        alias="WORKER_EVALUATION_ARTIFACT_MAX_BYTES",
    )
    worker_evaluation_artifact_agent_path_max_bytes: int = Field(
        default=1_048_576,
        alias="WORKER_EVALUATION_ARTIFACT_AGENT_PATH_MAX_BYTES",
    )
    worker_evaluation_artifact_allowed_mime_types: list[str] = Field(
        default_factory=lambda: [
            "text/plain",
            "application/json",
            "image/svg+xml",
            "image/png",
            "text/html",
            "application/octet-stream",
        ],
        alias="WORKER_EVALUATION_ARTIFACT_ALLOWED_MIME_TYPES",
    )
    worker_evaluation_artifact_agent_path_mime_types: list[str] = Field(
        default_factory=lambda: [
            "text/plain",
            "application/json",
            "image/svg+xml",
            "text/html",
        ],
        alias="WORKER_EVALUATION_ARTIFACT_AGENT_PATH_MIME_TYPES",
    )
    # Global evolution objective used to guide planning and coding agents.
    # This should be a stable, plain-language description of what the
    # autonomous worker is trying to achieve across all evolution jobs.
    worker_evolution_global_goal: str = Field(
        default=(
            "Continuously improve the repository while keeping tests passing, "
            "maintaining code quality, and respecting project conventions."
        ),
        alias="WORKER_EVOLUTION_GLOBAL_GOAL",
    )
    worker_evolution_commit_author: str = Field(
        default="Loreley Worker",
        alias="WORKER_EVOLUTION_COMMIT_AUTHOR",
    )
    worker_evolution_commit_email: str = Field(
        default="worker@loreley.local",
        alias="WORKER_EVOLUTION_COMMIT_EMAIL",
    )
    # Planning-time inspiration trajectory rollups (LCA-aware).
    worker_planning_trajectory_block_size: int = Field(
        default=8,
        alias="WORKER_PLANNING_TRAJECTORY_BLOCK_SIZE",
    )
    worker_planning_trajectory_max_chunks: int = Field(
        default=3,
        alias="WORKER_PLANNING_TRAJECTORY_MAX_CHUNKS",
    )
    worker_planning_trajectory_max_raw_steps: int = Field(
        default=6,
        alias="WORKER_PLANNING_TRAJECTORY_MAX_RAW_STEPS",
    )
    worker_planning_trajectory_summary_model: str | None = Field(
        default=None,
        alias="WORKER_PLANNING_TRAJECTORY_SUMMARY_MODEL",
    )
    worker_planning_trajectory_summary_provider_mode: Literal[
        "global_openai",
        "custom",
    ] = Field(
        default="global_openai",
        alias="WORKER_PLANNING_TRAJECTORY_SUMMARY_PROVIDER_MODE",
    )
    worker_planning_trajectory_summary_api_key: str | None = Field(
        default=None,
        alias="WORKER_PLANNING_TRAJECTORY_SUMMARY_API_KEY",
    )
    worker_planning_trajectory_summary_base_url: str | None = Field(
        default=None,
        alias="WORKER_PLANNING_TRAJECTORY_SUMMARY_BASE_URL",
    )
    worker_planning_trajectory_summary_api_spec: Literal[
        "responses",
        "chat_completions",
    ] = Field(
        default="responses",
        alias="WORKER_PLANNING_TRAJECTORY_SUMMARY_API_SPEC",
    )
    worker_planning_trajectory_summary_thinking: Literal[
        "provider_default",
        "enabled",
        "disabled",
    ] = Field(
        default="provider_default",
        alias="WORKER_PLANNING_TRAJECTORY_SUMMARY_THINKING",
    )
    worker_planning_trajectory_summary_reasoning_effort: Literal[
        "provider_default",
        "high",
        "max",
    ] = Field(
        default="provider_default",
        alias="WORKER_PLANNING_TRAJECTORY_SUMMARY_REASONING_EFFORT",
    )
    worker_planning_trajectory_summary_temperature: float = Field(
        default=0.0,
        alias="WORKER_PLANNING_TRAJECTORY_SUMMARY_TEMPERATURE",
    )
    worker_planning_trajectory_summary_max_output_tokens: int = Field(
        default=256,
        alias="WORKER_PLANNING_TRAJECTORY_SUMMARY_MAX_OUTPUT_TOKENS",
    )
    worker_planning_trajectory_summary_max_retries: int = Field(
        default=3,
        alias="WORKER_PLANNING_TRAJECTORY_SUMMARY_MAX_RETRIES",
    )
    worker_planning_trajectory_summary_retry_backoff_seconds: float = Field(
        default=2.0,
        alias="WORKER_PLANNING_TRAJECTORY_SUMMARY_RETRY_BACKOFF_SECONDS",
    )
    worker_planning_trajectory_summary_max_chars: int = Field(
        default=800,
        alias="WORKER_PLANNING_TRAJECTORY_SUMMARY_MAX_CHARS",
    )

    mapelites_preprocess_max_file_size_kb: int = Field(
        default=512,
        alias="MAPELITES_PREPROCESS_MAX_FILE_SIZE_KB",
    )
    mapelites_preprocess_allowed_extensions: list[str] = Field(
        default_factory=lambda: [
            ".py",
            ".pyi",
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".go",
            ".rs",
            ".java",
            ".kt",
            ".swift",
            ".m",
            ".mm",
            ".c",
            ".cc",
            ".cpp",
            ".cxx",
            ".cs",
            ".h",
            ".hpp",
            ".php",
            ".rb",
            ".scala",
            ".sql",
            ".sh",
        ],
        alias="MAPELITES_PREPROCESS_ALLOWED_EXTENSIONS",
    )
    mapelites_preprocess_allowed_filenames: list[str] = Field(
        default_factory=lambda: ["Dockerfile", "Makefile"],
        alias="MAPELITES_PREPROCESS_ALLOWED_FILENAMES",
    )
    mapelites_preprocess_excluded_globs: list[str] = Field(
        default_factory=lambda: [
            "tests/**",
            "__pycache__/**",
            "node_modules/**",
            "build/**",
            "dist/**",
            ".git/**",
        ],
        alias="MAPELITES_PREPROCESS_EXCLUDED_GLOBS",
    )
    mapelites_preprocess_max_blank_lines: int = Field(
        default=2,
        alias="MAPELITES_PREPROCESS_MAX_BLANK_LINES",
    )
    mapelites_preprocess_tab_width: int = Field(
        default=4,
        alias="MAPELITES_PREPROCESS_TAB_WIDTH",
    )
    mapelites_preprocess_strip_comments: bool = Field(
        default=True,
        alias="MAPELITES_PREPROCESS_STRIP_COMMENTS",
    )
    mapelites_preprocess_strip_block_comments: bool = Field(
        default=True,
        alias="MAPELITES_PREPROCESS_STRIP_BLOCK_COMMENTS",
    )

    mapelites_chunk_target_lines: int = Field(
        default=80,
        alias="MAPELITES_CHUNK_TARGET_LINES",
    )
    mapelites_chunk_min_lines: int = Field(
        default=20,
        alias="MAPELITES_CHUNK_MIN_LINES",
    )
    mapelites_chunk_overlap_lines: int = Field(
        default=8,
        alias="MAPELITES_CHUNK_OVERLAP_LINES",
    )
    mapelites_chunk_max_chunks_per_file: int = Field(
        default=64,
        alias="MAPELITES_CHUNK_MAX_CHUNKS_PER_FILE",
    )
    mapelites_chunk_boundary_keywords: list[str] = Field(
        default_factory=lambda: [
            "def ",
            "class ",
            "async def ",
            "fn ",
            "function ",
            "impl ",
            "struct ",
            "interface ",
            "module ",
            "export ",
        ],
        alias="MAPELITES_CHUNK_BOUNDARY_KEYWORDS",
    )
    mapelites_repo_state_embedding_max_line_chars: int = Field(
        default=16000,
        alias="MAPELITES_REPO_STATE_EMBEDDING_MAX_LINE_CHARS",
    )
    mapelites_repo_state_embedding_max_chunk_chars: int = Field(
        default=65536,
        alias="MAPELITES_REPO_STATE_EMBEDDING_MAX_CHUNK_CHARS",
    )

    mapelites_code_embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="MAPELITES_CODE_EMBEDDING_MODEL",
    )
    mapelites_local_hash_embedding_acknowledged: bool = Field(
        default=False,
        alias="MAPELITES_LOCAL_HASH_EMBEDDING_ACKNOWLEDGED",
    )
    # Fixed embedding dimensionality for the entire experiment lifecycle.
    #
    # In the env-only settings model, this must be provided via environment
    # variables and kept consistent across long-running processes.
    mapelites_code_embedding_dimensions: PositiveInt | None = Field(
        default=None,
        alias="MAPELITES_CODE_EMBEDDING_DIMENSIONS",
    )
    mapelites_code_embedding_batch_size: int = Field(
        default=12,
        alias="MAPELITES_CODE_EMBEDDING_BATCH_SIZE",
    )
    mapelites_code_embedding_provider_only: tuple[str, ...] = Field(
        default=(),
        alias="MAPELITES_CODE_EMBEDDING_PROVIDER_ONLY",
    )
    mapelites_code_embedding_allow_fallbacks: bool = Field(
        default=True,
        alias="MAPELITES_CODE_EMBEDDING_ALLOW_FALLBACKS",
    )
    mapelites_code_embedding_require_parameters: bool = Field(
        default=False,
        alias="MAPELITES_CODE_EMBEDDING_REQUIRE_PARAMETERS",
    )
    mapelites_code_embedding_data_collection: Literal["allow", "deny"] = Field(
        default="allow",
        alias="MAPELITES_CODE_EMBEDDING_DATA_COLLECTION",
    )
    mapelites_code_embedding_max_retries: int = Field(
        default=3,
        alias="MAPELITES_CODE_EMBEDDING_MAX_RETRIES",
    )
    mapelites_code_embedding_retry_backoff_seconds: float = Field(
        default=2.0,
        alias="MAPELITES_CODE_EMBEDDING_RETRY_BACKOFF_SECONDS",
    )
    mapelites_dimensionality_target_dims: int = Field(
        default=4,
        alias="MAPELITES_DIMENSION_REDUCTION_TARGET_DIMS",
    )
    mapelites_dimensionality_min_fit_samples: int = Field(
        default=32,
        alias="MAPELITES_DIMENSION_REDUCTION_MIN_FIT_SAMPLES",
    )
    mapelites_dimensionality_history_size: int = Field(
        default=4096,
        alias="MAPELITES_DIMENSION_REDUCTION_HISTORY_SIZE",
    )
    mapelites_dimensionality_refit_interval: int = Field(
        default=50,
        alias="MAPELITES_DIMENSION_REDUCTION_REFIT_INTERVAL",
    )
    # Seed used for PCA randomness (e.g. randomized SVD) to keep projections reproducible.
    mapelites_dimensionality_seed: int = Field(
        default=0,
        alias="MAPELITES_DIMENSION_REDUCTION_SEED",
    )
    mapelites_dimensionality_penultimate_normalize: bool = Field(
        default=True,
        alias="MAPELITES_DIMENSION_REDUCTION_PENULTIMATE_NORMALIZE",
    )
    mapelites_feature_truncation_k: float = Field(
        default=3.0,
        alias="MAPELITES_FEATURE_TRUNCATION_K",
    )
    mapelites_feature_normalization_warmup_samples: int = Field(
        default=0,
        alias="MAPELITES_FEATURE_NORMALIZATION_WARMUP_SAMPLES",
    )
    mapelites_archive_cells_per_dim: int = Field(
        default=32,
        alias="MAPELITES_ARCHIVE_CELLS_PER_DIM",
    )
    mapelites_islands: tuple[str, ...] = Field(
        default=("main",),
        alias="MAPELITES_ISLANDS",
    )
    mapelites_objectives: tuple[ObjectiveSpec, ...] = Field(
        default=(ObjectiveSpec(name="composite_score", direction="max"),),
        alias="MAPELITES_OBJECTIVES",
    )
    mapelites_pareto_front_max_size: PositiveInt = Field(
        default=8,
        alias="MAPELITES_PARETO_FRONT_MAX_SIZE",
    )
    mapelites_pareto_epsilon: float = Field(
        default=1.0e-9,
        alias="MAPELITES_PARETO_EPSILON",
    )
    mapelites_migration_interval_jobs: int = Field(
        default=50,
        ge=0,
        alias="MAPELITES_MIGRATION_INTERVAL_JOBS",
    )
    mapelites_feature_clip: bool = Field(
        default=True,
        alias="MAPELITES_FEATURE_CLIP",
    )
    # Emit one INFO-level ingest log every N ingests to control hot-path log volume.
    mapelites_ingest_info_log_every: PositiveInt = Field(
        default=20,
        alias="MAPELITES_INGEST_INFO_LOG_EVERY",
    )
    mapelites_sampler_inspiration_count: int = Field(
        default=3,
        alias="MAPELITES_SAMPLER_INSPIRATION_COUNT",
    )
    # Deterministic RNG seed used by the MAP-Elites job sampler.
    mapelites_sampler_seed: int = Field(
        default=0,
        alias="MAPELITES_SAMPLER_SEED",
    )
    mapelites_sampler_recipe_cooldown_jobs: int = Field(
        default=64,
        ge=0,
        alias="MAPELITES_SAMPLER_RECIPE_COOLDOWN_JOBS",
    )
    mapelites_sampler_max_resample_attempts: PositiveInt = Field(
        default=32,
        alias="MAPELITES_SAMPLER_MAX_RESAMPLE_ATTEMPTS",
    )
    mapelites_sampler_neighbor_radius: int = Field(
        default=1,
        alias="MAPELITES_SAMPLER_NEIGHBOR_RADIUS",
    )
    mapelites_sampler_neighbor_max_radius: int = Field(
        default=3,
        alias="MAPELITES_SAMPLER_NEIGHBOR_MAX_RADIUS",
    )
    mapelites_sampler_fallback_sample_size: int = Field(
        default=8,
        alias="MAPELITES_SAMPLER_FALLBACK_SAMPLE_SIZE",
    )
    mapelites_sampler_default_priority: int = Field(
        default=0,
        alias="MAPELITES_SAMPLER_DEFAULT_PRIORITY",
    )
    mapelites_seed_population_size: int = Field(
        default=16,
        alias="MAPELITES_SEED_POPULATION_SIZE",
    )

    def model_post_init(self, __context: Any) -> None:
        """Apply derived defaults that depend on other fields."""

        _apply_profile_defaults(self)
        _validate_map_elites_contract(self)
        _randomize_worker_repo_worktree(self)
        _apply_map_elites_projection_defaults(self)

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

    def export_safe(self, *, mask_secrets: bool = True) -> dict[str, Any]:
        """Return effective settings for debugging/logging."""
        return _build_safe_export_payload(self, mask_secrets=mask_secrets)

    def effective_fingerprint(self) -> str:
        """Return a stable fingerprint of the masked effective settings export."""

        payload = self.export_safe(mask_secrets=True)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _apply_profile_defaults(settings: Settings) -> None:
    for field, value in _profile_derived_defaults(settings.profile).items():
        if field not in settings.model_fields_set:
            object.__setattr__(settings, field, value)


def _validate_map_elites_contract(settings: Settings) -> None:
    island_ids = tuple(
        str(island_id or "").strip()
        for island_id in settings.mapelites_islands
    )
    if not island_ids or any(not island_id for island_id in island_ids):
        raise ValueError("MAPELITES_ISLANDS must contain at least one non-empty island ID.")
    if any(len(island_id) > _MAX_ISLAND_ID_LENGTH for island_id in island_ids):
        raise ValueError(
            "MAPELITES_ISLANDS island IDs cannot exceed "
            f"{_MAX_ISLAND_ID_LENGTH} characters."
        )
    if len(set(island_ids)) != len(island_ids):
        raise ValueError("MAPELITES_ISLANDS island IDs must be unique.")
    object.__setattr__(settings, "mapelites_islands", island_ids)

    # Construction performs the cross-entry validation that Pydantic's
    # element validation cannot express (non-empty and unique names).
    ObjectiveContract(settings.mapelites_objectives)
    pareto_epsilon = float(settings.mapelites_pareto_epsilon)
    if not 0.0 <= pareto_epsilon < float("inf"):
        raise ValueError("MAPELITES_PARETO_EPSILON must be finite and non-negative.")


def _randomize_worker_repo_worktree(settings: Settings) -> None:
    if not settings.worker_repo_worktree_randomize:
        return
    suffix_len = max(
        1,
        min(32, int(settings.worker_repo_worktree_random_suffix_len or 0)),
    )
    suffix = uuid.uuid4().hex[:suffix_len]
    base = Path(settings.worker_repo_worktree).expanduser()
    object.__setattr__(
        settings,
        "worker_repo_worktree",
        str(base.parent / f"{base.name}-pid-{os.getpid()}-{suffix}"),
    )


def _apply_map_elites_projection_defaults(settings: Settings) -> None:
    if "mapelites_dimensionality_min_fit_samples" not in settings.model_fields_set:
        seed_population = int(settings.mapelites_seed_population_size or 0)
        if seed_population > 0:
            object.__setattr__(
                settings,
                "mapelites_dimensionality_min_fit_samples",
                max(2, seed_population),
            )

    min_fit = int(settings.mapelites_dimensionality_min_fit_samples)
    warmup = max(
        min_fit,
        int(settings.mapelites_feature_normalization_warmup_samples or 0),
    )
    object.__setattr__(
        settings,
        "mapelites_feature_normalization_warmup_samples",
        warmup,
    )
    truncation_k = float(settings.mapelites_feature_truncation_k)
    object.__setattr__(
        settings,
        "mapelites_feature_truncation_k",
        truncation_k if truncation_k > 0.0 else 3.0,
    )


def _safe_export_secret(value: str | None, *, mask_secrets: bool) -> str | None:
    normalized = (value or "").strip() or None
    if normalized is None:
        return None
    if mask_secrets:
        return _mask_secret(normalized)
    return normalized


def _safe_export_url(value: str | None, *, mask_secrets: bool) -> str | None:
    normalized = (value or "").strip() or None
    if normalized is None:
        return None
    if mask_secrets:
        return _sanitize_url(normalized)
    return normalized


def _safe_export_database_dsn(settings: Settings, *, mask_secrets: bool) -> str:
    if mask_secrets:
        return _sanitize_sqlalchemy_dsn(settings.database_dsn)
    return settings.database_dsn


def _safe_export_task_names(experiment_id: object) -> dict[str, str | None]:
    from loreley.naming import (
        DEFAULT_TASKS_QUEUE_PREFIX,
        DEFAULT_TASKS_REDIS_NAMESPACE_PREFIX,
        safe_namespace_or_none,
    )

    exp_ns = safe_namespace_or_none(experiment_id)
    return {
        "tasks_redis_namespace": (
            f"{DEFAULT_TASKS_REDIS_NAMESPACE_PREFIX}.{exp_ns}" if exp_ns else None
        ),
        "tasks_queue_name": f"{DEFAULT_TASKS_QUEUE_PREFIX}.{exp_ns}" if exp_ns else None,
    }


def _safe_export_worker_payload(settings: Settings, *, mask_secrets: bool) -> dict[str, Any]:
    return {
        "worker_repo_worktree": settings.worker_repo_worktree,
        "worker_repo_remote_url": _safe_export_url(
            settings.worker_repo_remote_url,
            mask_secrets=mask_secrets,
        ),
        "worker_repo_branch": settings.worker_repo_branch,
        "worker_repo_fetch_depth": settings.worker_repo_fetch_depth,
        "worker_scope_gate_cleanup_paths": settings.worker_scope_gate_cleanup_paths,
        "worker_repo_enable_lfs": settings.worker_repo_enable_lfs,
        "worker_repo_job_branch_ttl_hours": settings.worker_repo_job_branch_ttl_hours,
        "worker_job_lease_ttl_seconds": settings.worker_job_lease_ttl_seconds,
        "worker_job_heartbeat_interval_seconds": settings.worker_job_heartbeat_interval_seconds,
        "worker_processes": settings.worker_processes,
        "worker_planning_backend": settings.worker_planning_backend,
        "worker_planning_codex_model": settings.worker_planning_codex_model,
        "worker_planning_max_attempts": settings.worker_planning_max_attempts,
        "worker_planning_timeout_seconds": settings.worker_planning_timeout_seconds,
        "worker_coding_backend": settings.worker_coding_backend,
        "worker_coding_codex_model": settings.worker_coding_codex_model,
        "worker_coding_max_attempts": settings.worker_coding_max_attempts,
        "worker_coding_timeout_seconds": settings.worker_coding_timeout_seconds,
        "worker_cursor_model": settings.worker_cursor_model,
        "worker_kilocode_mode": settings.worker_kilocode_mode,
        "worker_kilocode_agent": settings.worker_kilocode_agent,
        "worker_kilocode_model": settings.worker_kilocode_model,
        "worker_kilocode_planning_model": settings.worker_kilocode_planning_model,
        "worker_kilocode_coding_model": settings.worker_kilocode_coding_model,
        "worker_kilocode_variant": settings.worker_kilocode_variant,
        "worker_kilocode_pure": settings.worker_kilocode_pure,
        "worker_kilocode_json_output": settings.worker_kilocode_json_output,
        "worker_kilocode_usage_db_path": settings.worker_kilocode_usage_db_path,
        "worker_kilocode_state_root": settings.worker_kilocode_state_root,
        "worker_kilocode_provider_config_mode": settings.worker_kilocode_provider_config_mode,
        "worker_kilocode_openai_api_spec": settings.worker_kilocode_openai_api_spec,
        "worker_kilocode_openai_base_url": _safe_export_secret(
            settings.worker_kilocode_openai_base_url,
            mask_secrets=mask_secrets,
        ),
        "worker_kilocode_openai_api_key": _safe_export_secret(
            settings.worker_kilocode_openai_api_key,
            mask_secrets=mask_secrets,
        ),
        "worker_kilocode_openai_model": settings.worker_kilocode_openai_model,
        "worker_evaluator_plugin": settings.worker_evaluator_plugin,
        "worker_evaluator_version": settings.worker_evaluator_version,
        "worker_evaluator_timeout_seconds": settings.worker_evaluator_timeout_seconds,
        "worker_evaluator_max_metrics": settings.worker_evaluator_max_metrics,
        "worker_evaluator_max_concurrency": settings.worker_evaluator_max_concurrency,
        "worker_evaluator_slot_poll_seconds": settings.worker_evaluator_slot_poll_seconds,
        "worker_evaluator_measurement_max_json_bytes": (
            settings.worker_evaluator_measurement_max_json_bytes
        ),
        "worker_evolution_global_goal": settings.worker_evolution_global_goal,
        "worker_evolution_commit_author": settings.worker_evolution_commit_author,
        "worker_evolution_commit_email": settings.worker_evolution_commit_email,
        "worker_planning_trajectory_summary_provider_mode": (
            settings.worker_planning_trajectory_summary_provider_mode
        ),
        "worker_planning_trajectory_summary_api_key": _safe_export_secret(
            settings.worker_planning_trajectory_summary_api_key,
            mask_secrets=mask_secrets,
        ),
        "worker_planning_trajectory_summary_base_url": _safe_export_secret(
            settings.worker_planning_trajectory_summary_base_url,
            mask_secrets=mask_secrets,
        ),
        "worker_planning_trajectory_summary_api_spec": (
            settings.worker_planning_trajectory_summary_api_spec
        ),
        "worker_planning_trajectory_summary_thinking": (
            settings.worker_planning_trajectory_summary_thinking
        ),
        "worker_planning_trajectory_summary_reasoning_effort": (
            settings.worker_planning_trajectory_summary_reasoning_effort
        ),
        "worker_planning_trajectory_summary_model": (
            settings.worker_planning_trajectory_summary_model
        ),
    }


def _build_safe_export_payload(settings: Settings, *, mask_secrets: bool) -> dict[str, Any]:
    from loreley.core.model_routes import resolve_effective_routes

    task_names = _safe_export_task_names(settings.experiment_id)
    return {
        "app_name": settings.app_name,
        "environment": settings.environment,
        "log_level": settings.log_level,
        "profile": settings.profile,
        "logs_base_dir": settings.logs_base_dir,
        "loreley_agent_api_token": _safe_export_secret(
            settings.loreley_agent_api_token,
            mask_secrets=mask_secrets,
        ),
        "loreley_api_write_token": _safe_export_secret(
            settings.loreley_api_write_token,
            mask_secrets=mask_secrets,
        ),
        "openai_api_spec": settings.openai_api_spec,
        "openai_base_url": _safe_export_secret(
            settings.openai_base_url,
            mask_secrets=mask_secrets,
        ),
        "openai_api_key": _safe_export_secret(settings.openai_api_key, mask_secrets=mask_secrets),
        "openai_dynamic_api_key_provider": settings.openai_dynamic_api_key_provider,
        "openai_dynamic_api_key_ttl_seconds": settings.openai_dynamic_api_key_ttl_seconds,
        "openai_dynamic_api_key_refresh_skew_seconds": (
            settings.openai_dynamic_api_key_refresh_skew_seconds
        ),
        "llm_usage_tracking_enabled": settings.llm_usage_tracking_enabled,
        "llm_usage_pricing_path": settings.llm_usage_pricing_path,
        "llm_usage_pricing_json_configured": bool(settings.llm_usage_pricing_json),
        "mapelites_experiment_root_commit": settings.mapelites_experiment_root_commit,
        "database_dsn": _safe_export_database_dsn(settings, mask_secrets=mask_secrets),
        "db_scheme": settings.db_scheme,
        "db_host": settings.db_host,
        "db_port": settings.db_port,
        "db_name": settings.db_name,
        "db_password": _safe_export_secret(settings.db_password, mask_secrets=mask_secrets),
        "db_pool_size": settings.db_pool_size,
        "db_max_overflow": settings.db_max_overflow,
        "db_pool_timeout": settings.db_pool_timeout,
        "db_echo": settings.db_echo,
        "db_auto_migrate": settings.db_auto_migrate,
        "db_migration_lock_timeout_seconds": settings.db_migration_lock_timeout_seconds,
        "tasks_redis_url": _safe_export_url(settings.tasks_redis_url, mask_secrets=mask_secrets),
        "tasks_redis_host": settings.tasks_redis_host,
        "tasks_redis_port": settings.tasks_redis_port,
        "tasks_redis_db": settings.tasks_redis_db,
        "tasks_redis_password": _safe_export_secret(
            settings.tasks_redis_password,
            mask_secrets=mask_secrets,
        ),
        **task_names,
        "tasks_queue_prefetch": settings.tasks_queue_prefetch,
        "tasks_delay_queue_prefetch": settings.tasks_delay_queue_prefetch,
        "tasks_worker_max_retries": settings.tasks_worker_max_retries,
        "tasks_worker_time_limit_seconds": settings.tasks_worker_time_limit_seconds,
        "experiment_id": str(settings.experiment_id) if settings.experiment_id else None,
        "effective_routes": resolve_effective_routes(settings),
        "scheduler_repo_root": settings.scheduler_repo_root,
        "scheduler_poll_interval_seconds": settings.scheduler_poll_interval_seconds,
        **_safe_export_worker_payload(settings, mask_secrets=mask_secrets),
        "mapelites_code_embedding_model": settings.mapelites_code_embedding_model,
        "mapelites_code_embedding_dimensions": settings.mapelites_code_embedding_dimensions,
        "mapelites_code_embedding_provider_only": list(
            settings.mapelites_code_embedding_provider_only
        ),
        "mapelites_code_embedding_allow_fallbacks": (
            settings.mapelites_code_embedding_allow_fallbacks
        ),
        "mapelites_code_embedding_require_parameters": (
            settings.mapelites_code_embedding_require_parameters
        ),
        "mapelites_code_embedding_data_collection": (
            settings.mapelites_code_embedding_data_collection
        ),
        "mapelites_local_hash_embedding_acknowledged": (
            settings.mapelites_local_hash_embedding_acknowledged
        ),
        "mapelites_repo_state_embedding_max_line_chars": (
            settings.mapelites_repo_state_embedding_max_line_chars
        ),
        "mapelites_repo_state_embedding_max_chunk_chars": (
            settings.mapelites_repo_state_embedding_max_chunk_chars
        ),
        "mapelites_dimensionality_target_dims": settings.mapelites_dimensionality_target_dims,
        "mapelites_archive_cells_per_dim": settings.mapelites_archive_cells_per_dim,
        "mapelites_islands": list(settings.mapelites_islands),
        "mapelites_objectives": resolve_objective_contract(settings).as_payload(),
        "mapelites_pareto_front_max_size": settings.mapelites_pareto_front_max_size,
        "mapelites_pareto_epsilon": settings.mapelites_pareto_epsilon,
        "mapelites_migration_interval_jobs": settings.mapelites_migration_interval_jobs,
        "mapelites_sampler_inspiration_count": settings.mapelites_sampler_inspiration_count,
        "mapelites_sampler_seed": settings.mapelites_sampler_seed,
        "mapelites_sampler_recipe_cooldown_jobs": (
            settings.mapelites_sampler_recipe_cooldown_jobs
        ),
        "mapelites_sampler_max_resample_attempts": (
            settings.mapelites_sampler_max_resample_attempts
        ),
        "mapelites_sampler_neighbor_radius": settings.mapelites_sampler_neighbor_radius,
        "mapelites_sampler_neighbor_max_radius": (
            settings.mapelites_sampler_neighbor_max_radius
        ),
        "mapelites_sampler_fallback_sample_size": (
            settings.mapelites_sampler_fallback_sample_size
        ),
        "mapelites_sampler_default_priority": settings.mapelites_sampler_default_priority,
        "mapelites_seed_population_size": settings.mapelites_seed_population_size,
        "scheduler_max_unfinished_jobs": settings.scheduler_max_unfinished_jobs,
        "scheduler_dispatch_batch_size": settings.scheduler_dispatch_batch_size,
        "scheduler_schedule_batch_size": settings.scheduler_schedule_batch_size,
        "scheduler_ingest_batch_size": settings.scheduler_ingest_batch_size,
        "scheduler_max_total_jobs": settings.scheduler_max_total_jobs,
        "scheduler_max_unique_evaluation_identities": (
            settings.scheduler_max_unique_evaluation_identities
        ),
        "scheduler_startup_approve": settings.scheduler_startup_approve,
        "scheduler_stale_running_reclaim_batch_size": (
            settings.scheduler_stale_running_reclaim_batch_size
        ),
        "scheduler_stale_running_max_recovery_attempts": (
            settings.scheduler_stale_running_max_recovery_attempts
        ),
        "campaign_program_change_policy": settings.campaign_program_change_policy,
        "baseline_bootstrap_policy": settings.baseline_bootstrap_policy,
        "failed_candidate_repair_enabled": settings.failed_candidate_repair_enabled,
        "failed_candidate_repair_mode": settings.failed_candidate_repair_mode,
        "failed_candidate_repair_max_depth": settings.failed_candidate_repair_max_depth,
        "failed_candidate_repair_max_attempts": settings.failed_candidate_repair_max_attempts,
        "failed_candidate_repair_normal_jobs_per_token": (
            settings.failed_candidate_repair_normal_jobs_per_token
        ),
        "failed_candidate_repair_max_tokens": settings.failed_candidate_repair_max_tokens,
        "failed_candidate_repair_max_active_jobs": (
            settings.failed_candidate_repair_max_active_jobs
        ),
        "failed_candidate_repair_max_jobs_per_tick": (
            settings.failed_candidate_repair_max_jobs_per_tick
        ),
        "failed_candidate_repair_failure_kinds": (
            settings.failed_candidate_repair_failure_kinds
        ),
        "failed_candidate_repair_agent_feedback_mode": (
            settings.failed_candidate_repair_agent_feedback_mode
        ),
        "failed_candidate_repair_max_diff_bytes": settings.failed_candidate_repair_max_diff_bytes,
        "failed_candidate_repair_max_diagnostic_bytes": (
            settings.failed_candidate_repair_max_diagnostic_bytes
        ),
        "worker_evaluator_rework_enabled": settings.worker_evaluator_rework_enabled,
        "worker_evaluator_rework_max_attempts": settings.worker_evaluator_rework_max_attempts,
        "worker_evaluator_rework_failure_kinds": (
            settings.worker_evaluator_rework_failure_kinds
        ),
        "worker_evaluator_rework_max_seconds": settings.worker_evaluator_rework_max_seconds,
    }


@lru_cache
def get_settings() -> Settings:
    """Load and cache application settings."""
    settings = Settings()  # type: ignore[call-arg]  # Loaded from environment via pydantic-settings.
    console.log(
        f"[bold green]Loaded settings[/] env={settings.environment!r} "
        f"db_host={settings.db_host!r}",
    )
    log.info("Settings initialised: {}", settings.export_safe())
    return settings


def resolve_default_island_id(settings: Settings | None = None) -> str:
    """Return the first configured island, used by single-island call sites."""

    base_settings = settings or get_settings()
    island_ids = tuple(base_settings.mapelites_islands)
    if not island_ids:
        raise ValueError("MAPELITES_ISLANDS must contain at least one island ID.")
    return island_ids[0]


def resolve_objective_contract(settings: Settings | None = None) -> ObjectiveContract:
    """Return the validated ordered optimization objective contract."""

    base_settings = settings or get_settings()
    return ObjectiveContract(base_settings.mapelites_objectives)
