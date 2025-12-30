from __future__ import annotations

from functools import lru_cache
import uuid
from pathlib import Path
from typing import Any, Literal
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
    logs_base_dir: str | None = Field(
        default=None,
        alias="LOGS_BASE_DIR",
    )

    # OpenAI-compatible API configuration
    openai_api_key: str | None = Field(
        default=None,
        alias="OPENAI_API_KEY",
    )
    openai_base_url: str | None = Field(
        default=None,
        alias="OPENAI_BASE_URL",
    )
    openai_api_spec: Literal["responses", "chat_completions"] = Field(
        default="responses",
        alias="OPENAI_API_SPEC",
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

    metrics_retention_days: int = Field(default=30, alias="METRICS_RETENTION_DAYS")

    tasks_redis_url: str | None = Field(default=None, alias="TASKS_REDIS_URL")
    tasks_redis_host: str = Field(default="localhost", alias="TASKS_REDIS_HOST")
    tasks_redis_port: int = Field(default=6379, alias="TASKS_REDIS_PORT")
    tasks_redis_db: int = Field(default=0, alias="TASKS_REDIS_DB")
    tasks_redis_password: str | None = Field(default=None, alias="TASKS_REDIS_PASSWORD")
    tasks_redis_namespace: str = Field(default="loreley", alias="TASKS_REDIS_NAMESPACE")
    tasks_queue_name: str = Field(default="loreley.evolution", alias="TASKS_QUEUE_NAME")
    tasks_worker_max_retries: int = Field(default=0, alias="TASKS_WORKER_MAX_RETRIES")
    tasks_worker_time_limit_seconds: int = Field(
        default=3600,
        alias="TASKS_WORKER_TIME_LIMIT_SECONDS",
    )

    # Experiment / evolution configuration
    mapelites_experiment_root_commit: str | None = Field(
        default=None,
        alias="MAPELITES_EXPERIMENT_ROOT_COMMIT",
    )
    mapelites_embedding_mode: Literal["repo_state"] = Field(
        default="repo_state",
        alias="MAPELITES_EMBEDDING_MODE",
    )
    mapelites_file_embedding_cache_backend: Literal["db", "memory"] | None = Field(
        default="db",
        alias="MAPELITES_FILE_EMBEDDING_CACHE_BACKEND",
    )
    mapelites_repo_state_max_files: int | None = Field(
        default=None,
        alias="MAPELITES_REPO_STATE_MAX_FILES",
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
    worker_repo_job_branch_prefix: str = Field(
        default="evolution/job",
        alias="WORKER_REPO_JOB_BRANCH_PREFIX",
    )
    worker_repo_enable_lfs: bool = Field(
        default=True,
        alias="WORKER_REPO_ENABLE_LFS",
    )
    worker_repo_job_branch_ttl_hours: int = Field(
        default=168,
        alias="WORKER_REPO_JOB_BRANCH_TTL_HOURS",
    )

    worker_planning_codex_bin: str = Field(
        default="codex",
        alias="WORKER_PLANNING_CODEX_BIN",
    )
    worker_planning_codex_profile: str | None = Field(
        default=None,
        alias="WORKER_PLANNING_CODEX_PROFILE",
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
    worker_planning_schema_path: str | None = Field(
        default=None,
        alias="WORKER_PLANNING_SCHEMA_PATH",
    )
    worker_coding_codex_bin: str = Field(
        default="codex",
        alias="WORKER_CODING_CODEX_BIN",
    )
    worker_coding_codex_profile: str | None = Field(
        default=None,
        alias="WORKER_CODING_CODEX_PROFILE",
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
    worker_coding_schema_path: str | None = Field(
        default=None,
        alias="WORKER_CODING_SCHEMA_PATH",
    )
    worker_planning_backend: str | None = Field(
        default=None,
        alias="WORKER_PLANNING_BACKEND",
    )
    worker_coding_backend: str | None = Field(
        default=None,
        alias="WORKER_CODING_BACKEND",
    )
    worker_cursor_model: str = Field(
        default="gpt-5.1-codex-max-high",
        alias="WORKER_CURSOR_MODEL",
    )
    worker_cursor_force: bool = Field(
        default=True,
        alias="WORKER_CURSOR_FORCE",
    )
    worker_planning_codex_schema_mode: Literal[
        "auto",
        "native",
        "prompt",
        "none",
    ] = Field(
        default="auto",
        alias="WORKER_PLANNING_CODEX_SCHEMA_MODE",
    )
    worker_planning_validation_mode: Literal["strict", "lenient", "none"] = Field(
        default="lenient",
        alias="WORKER_PLANNING_VALIDATION_MODE",
    )
    worker_coding_codex_schema_mode: Literal[
        "auto",
        "native",
        "prompt",
        "none",
    ] = Field(
        default="auto",
        alias="WORKER_CODING_CODEX_SCHEMA_MODE",
    )
    worker_coding_validation_mode: Literal["strict", "lenient", "none"] = Field(
        default="lenient",
        alias="WORKER_CODING_VALIDATION_MODE",
    )
    worker_evaluator_plugin: str | None = Field(
        default=None,
        alias="WORKER_EVALUATOR_PLUGIN",
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
    worker_evolution_commit_model: str = Field(
        default="gpt-4.1-mini",
        alias="WORKER_EVOLUTION_COMMIT_MODEL",
    )
    worker_evolution_commit_temperature: float = Field(
        default=0.2,
        alias="WORKER_EVOLUTION_COMMIT_TEMPERATURE",
    )
    worker_evolution_commit_max_output_tokens: int = Field(
        default=128,
        alias="WORKER_EVOLUTION_COMMIT_MAX_OUTPUT_TOKENS",
    )
    worker_evolution_commit_max_retries: int = Field(
        default=3,
        alias="WORKER_EVOLUTION_COMMIT_MAX_RETRIES",
    )
    worker_evolution_commit_retry_backoff_seconds: float = Field(
        default=2.0,
        alias="WORKER_EVOLUTION_COMMIT_RETRY_BACKOFF_SECONDS",
    )
    worker_evolution_commit_author: str = Field(
        default="Loreley Worker",
        alias="WORKER_EVOLUTION_COMMIT_AUTHOR",
    )
    worker_evolution_commit_email: str = Field(
        default="worker@loreley.local",
        alias="WORKER_EVOLUTION_COMMIT_EMAIL",
    )
    worker_evolution_commit_subject_max_chars: int = Field(
        default=72,
        alias="WORKER_EVOLUTION_COMMIT_SUBJECT_MAX_CHARS",
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
    worker_planning_trajectory_summary_prompt_signature: str = Field(
        default="v1",
        alias="WORKER_PLANNING_TRAJECTORY_SUMMARY_PROMPT_SIGNATURE",
    )

    mapelites_preprocess_max_files: int = Field(
        default=6,
        alias="MAPELITES_PREPROCESS_MAX_FILES",
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

    mapelites_code_embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="MAPELITES_CODE_EMBEDDING_MODEL",
    )
    mapelites_code_embedding_dimensions: int | None = Field(
        default=None,
        alias="MAPELITES_CODE_EMBEDDING_DIMENSIONS",
    )
    mapelites_code_embedding_batch_size: int = Field(
        default=12,
        alias="MAPELITES_CODE_EMBEDDING_BATCH_SIZE",
    )
    mapelites_code_embedding_max_chunks_per_commit: int = Field(
        default=512,
        alias="MAPELITES_CODE_EMBEDDING_MAX_CHUNKS_PER_COMMIT",
    )
    mapelites_code_embedding_max_retries: int = Field(
        default=3,
        alias="MAPELITES_CODE_EMBEDDING_MAX_RETRIES",
    )
    mapelites_code_embedding_retry_backoff_seconds: float = Field(
        default=2.0,
        alias="MAPELITES_CODE_EMBEDDING_RETRY_BACKOFF_SECONDS",
    )

    mapelites_summary_model: str = Field(
        default="gpt-4.1-mini",
        alias="MAPELITES_SUMMARY_MODEL",
    )
    mapelites_summary_temperature: float = Field(
        default=0.2,
        alias="MAPELITES_SUMMARY_TEMPERATURE",
    )
    mapelites_summary_max_output_tokens: int = Field(
        default=512,
        alias="MAPELITES_SUMMARY_MAX_OUTPUT_TOKENS",
    )
    mapelites_summary_source_char_limit: int = Field(
        default=6000,
        alias="MAPELITES_SUMMARY_SOURCE_CHAR_LIMIT",
    )
    mapelites_summary_max_retries: int = Field(
        default=3,
        alias="MAPELITES_SUMMARY_MAX_RETRIES",
    )
    mapelites_summary_retry_backoff_seconds: float = Field(
        default=2.0,
        alias="MAPELITES_SUMMARY_RETRY_BACKOFF_SECONDS",
    )
    mapelites_summary_embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="MAPELITES_SUMMARY_EMBEDDING_MODEL",
    )
    mapelites_summary_embedding_dimensions: int | None = Field(
        default=None,
        alias="MAPELITES_SUMMARY_EMBEDDING_DIMENSIONS",
    )
    mapelites_summary_embedding_batch_size: int = Field(
        default=16,
        alias="MAPELITES_SUMMARY_EMBEDDING_BATCH_SIZE",
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
    mapelites_archive_learning_rate: float = Field(
        default=1.0,
        alias="MAPELITES_ARCHIVE_LEARNING_RATE",
    )
    mapelites_archive_threshold_min: float = Field(
        default=float("-inf"),
        alias="MAPELITES_ARCHIVE_THRESHOLD_MIN",
    )
    mapelites_archive_epsilon: float = Field(
        default=1e-6,
        alias="MAPELITES_ARCHIVE_EPSILON",
    )
    mapelites_archive_qd_score_offset: float = Field(
        default=0.0,
        alias="MAPELITES_ARCHIVE_QD_SCORE_OFFSET",
    )
    mapelites_default_island_id: str = Field(
        default="main",
        alias="MAPELITES_DEFAULT_ISLAND_ID",
    )
    mapelites_fitness_metric: str = Field(
        default="composite_score",
        alias="MAPELITES_FITNESS_METRIC",
    )
    mapelites_fitness_higher_is_better: bool = Field(
        default=True,
        alias="MAPELITES_FITNESS_HIGHER_IS_BETTER",
    )
    mapelites_fitness_floor: float = Field(
        default=-1.0e6,
        alias="MAPELITES_FITNESS_FLOOR",
    )
    mapelites_feature_clip: bool = Field(
        default=True,
        alias="MAPELITES_FEATURE_CLIP",
    )
    mapelites_sampler_inspiration_count: int = Field(
        default=3,
        alias="MAPELITES_SAMPLER_INSPIRATION_COUNT",
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

        if self.worker_repo_worktree_randomize:
            suffix_len = int(self.worker_repo_worktree_random_suffix_len or 0)
            suffix_len = max(1, min(32, suffix_len))
            suffix = uuid.uuid4().hex[:suffix_len]
            base = Path(self.worker_repo_worktree).expanduser()
            randomized = base.parent / f"{base.name}-{suffix}"
            object.__setattr__(self, "worker_repo_worktree", str(randomized))

        min_fit = int(self.mapelites_dimensionality_min_fit_samples)
        warmup = int(self.mapelites_feature_normalization_warmup_samples or 0)
        if warmup <= 0:
            warmup = min_fit
        warmup = max(min_fit, warmup)
        object.__setattr__(
            self,
            "mapelites_feature_normalization_warmup_samples",
            warmup,
        )
        truncation_k = float(self.mapelites_feature_truncation_k)
        if truncation_k <= 0.0:
            truncation_k = 3.0
        object.__setattr__(self, "mapelites_feature_truncation_k", truncation_k)

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
            "mapelites_experiment_root_commit": self.mapelites_experiment_root_commit,
            "db_host": self.db_host,
            "db_port": self.db_port,
            "db_name": self.db_name,
            "db_pool_size": self.db_pool_size,
            "db_max_overflow": self.db_max_overflow,
            "db_pool_timeout": self.db_pool_timeout,
            "db_echo": self.db_echo,
            "tasks_redis_host": self.tasks_redis_host,
            "tasks_redis_port": self.tasks_redis_port,
            "tasks_queue_name": self.tasks_queue_name,
            "worker_repo_worktree": self.worker_repo_worktree,
            "worker_repo_branch": self.worker_repo_branch,
            "worker_repo_fetch_depth": self.worker_repo_fetch_depth,
            "scheduler_max_unfinished_jobs": self.scheduler_max_unfinished_jobs,
            "scheduler_dispatch_batch_size": self.scheduler_dispatch_batch_size,
            "scheduler_schedule_batch_size": self.scheduler_schedule_batch_size,
            "scheduler_max_total_jobs": self.scheduler_max_total_jobs,
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
