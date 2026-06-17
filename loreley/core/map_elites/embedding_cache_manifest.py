"""Compatibility manifests and import helpers for repo-state file embeddings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any
from urllib.parse import urlsplit, urlunsplit
import uuid

from loguru import logger
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from loreley.config import Settings
from loreley.db.base import get_engine, session_scope
from loreley.db.models import EmbeddingCacheManifest, MapElitesFileEmbeddingCache

REPO_STATE_FILE_EMBEDDING_CACHE_KIND = "repo_state_file_embedding"
REPO_STATE_FILE_EMBEDDING_MANIFEST_SCHEMA_VERSION = 1
REPO_STATE_FILE_EMBEDDING_ALGORITHM_VERSION = "v1"
FILE_CHUNK_AGGREGATION_ALGORITHM_VERSION = "weighted_average_v1"
_DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_BATCH_SIZE = 500

log = logger.bind(module="map_elites.embedding_cache_manifest")


class EmbeddingCacheManifestError(RuntimeError):
    """Raised when embedding cache compatibility cannot be established."""


@dataclass(frozen=True, slots=True)
class EmbeddingCacheImportResult:
    """Summary of an embedding cache import operation."""

    source_rows: int
    inserted_rows: int
    already_present_rows: int
    skipped_rows: int
    fingerprint: str
    source_manifest: str
    target_manifest: str
    source_dsn: str
    target_dsn: str

    def as_dict(self) -> dict[str, object]:
        return {
            "source_rows": self.source_rows,
            "inserted_rows": self.inserted_rows,
            "already_present_rows": self.already_present_rows,
            "skipped_rows": self.skipped_rows,
            "fingerprint": self.fingerprint,
            "source_manifest": self.source_manifest,
            "target_manifest": self.target_manifest,
            "source_dsn": self.source_dsn,
            "target_dsn": self.target_dsn,
        }


@dataclass(frozen=True, slots=True)
class EmbeddingCacheAttestationResult:
    """Summary of an operator attestation operation."""

    fingerprint: str
    manifest_source: str
    cache_rows: int
    embedding_model: str | None
    dimensions: int | None
    dsn: str

    def as_dict(self) -> dict[str, object]:
        return {
            "fingerprint": self.fingerprint,
            "manifest_source": self.manifest_source,
            "cache_rows": self.cache_rows,
            "embedding_model": self.embedding_model,
            "dimensions": self.dimensions,
            "dsn": self.dsn,
        }


def build_repo_state_file_embedding_manifest_payload(settings: Settings) -> dict[str, Any]:
    """Return the canonical payload used to fingerprint repo-state file embeddings."""

    dimensions = _current_embedding_dimensions(settings)
    requested_model = _current_embedding_model(settings)

    return {
        "kind": REPO_STATE_FILE_EMBEDDING_CACHE_KIND,
        "schema_version": REPO_STATE_FILE_EMBEDDING_MANIFEST_SCHEMA_VERSION,
        "embedding_model": requested_model,
        "embedding_provider": _embedding_provider_payload(settings, requested_model=requested_model),
        "embedding_dimensions": dimensions,
        "preprocess": _preprocess_fingerprint_payload(settings),
        "chunk": _chunk_fingerprint_payload(settings),
        "repo_state_embedding_filter": _repo_state_embedding_filter_payload(settings),
        "algorithm": _algorithm_fingerprint_payload(),
    }


def _current_embedding_dimensions(settings: Settings) -> int:
    dimensions = int(getattr(settings, "mapelites_code_embedding_dimensions", 0) or 0)
    if dimensions > 0:
        return dimensions
    raise EmbeddingCacheManifestError(
        "MAPELITES_CODE_EMBEDDING_DIMENSIONS must be configured before "
        "embedding cache manifests can be created."
    )


def _current_embedding_model(settings: Settings) -> str:
    requested_model = str(getattr(settings, "mapelites_code_embedding_model", "") or "").strip()
    if requested_model:
        return requested_model
    raise EmbeddingCacheManifestError(
        "MAPELITES_CODE_EMBEDDING_MODEL must be configured before "
        "embedding cache manifests can be created."
    )


def _preprocess_fingerprint_payload(settings: Settings) -> dict[str, object]:
    return {
        "allowed_extensions": _normalize_extensions(
            getattr(settings, "mapelites_preprocess_allowed_extensions", ()) or ()
        ),
        "allowed_filenames": _sorted_clean_strings(
            getattr(settings, "mapelites_preprocess_allowed_filenames", ()) or ()
        ),
        "excluded_globs": _normalize_excluded_globs(
            getattr(settings, "mapelites_preprocess_excluded_globs", ()) or ()
        ),
        "max_file_size_kb": int(getattr(settings, "mapelites_preprocess_max_file_size_kb", 0) or 0),
        "strip_comments": bool(getattr(settings, "mapelites_preprocess_strip_comments", False)),
        "strip_block_comments": bool(
            getattr(settings, "mapelites_preprocess_strip_block_comments", False)
        ),
        "max_blank_lines": int(getattr(settings, "mapelites_preprocess_max_blank_lines", 0) or 0),
        "tab_width": int(getattr(settings, "mapelites_preprocess_tab_width", 0) or 0),
    }


def _chunk_fingerprint_payload(settings: Settings) -> dict[str, object]:
    return {
        "target_lines": int(getattr(settings, "mapelites_chunk_target_lines", 0) or 0),
        "min_lines": int(getattr(settings, "mapelites_chunk_min_lines", 0) or 0),
        "overlap_lines": int(getattr(settings, "mapelites_chunk_overlap_lines", 0) or 0),
        "max_chunks_per_file": int(getattr(settings, "mapelites_chunk_max_chunks_per_file", 0) or 0),
        "boundary_keywords": _normalize_chunk_boundary_keywords(
            getattr(settings, "mapelites_chunk_boundary_keywords", ()) or (),
        ),
    }


def _repo_state_embedding_filter_payload(settings: Settings) -> dict[str, object]:
    return {
        "max_line_chars": int(
            getattr(settings, "mapelites_repo_state_embedding_max_line_chars", 0) or 0
        ),
        "max_chunk_chars": int(
            getattr(settings, "mapelites_repo_state_embedding_max_chunk_chars", 0) or 0
        ),
    }


def _algorithm_fingerprint_payload() -> dict[str, str]:
    return {
        "repo_state_file_embedding": REPO_STATE_FILE_EMBEDDING_ALGORITHM_VERSION,
        "file_chunk_aggregation": FILE_CHUNK_AGGREGATION_ALGORITHM_VERSION,
    }


def repo_state_file_embedding_fingerprint(settings: Settings) -> str:
    """Return the current repo-state file embedding compatibility fingerprint."""

    return fingerprint_payload(build_repo_state_file_embedding_manifest_payload(settings))


def fingerprint_payload(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 fingerprint for a manifest payload."""

    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def ensure_current_repo_state_file_embedding_manifest(
    *,
    settings: Settings,
    session: Session | None = None,
) -> EmbeddingCacheManifest:
    """Ensure the active DB has a manifest compatible with current settings."""

    if session is not None:
        return _ensure_current_manifest_in_session(session=session, settings=settings)

    with session_scope() as owned_session:
        return _ensure_current_manifest_in_session(session=owned_session, settings=settings)


def attest_repo_state_file_embedding_cache(
    *,
    settings: Settings,
    dsn: str,
    expected_fingerprint: str | None = None,
) -> EmbeddingCacheAttestationResult:
    """Write an operator-attested manifest for a legacy cache database."""

    engine = _create_external_engine(dsn)
    safe_dsn = sanitize_dsn(dsn)
    try:
        factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
        with factory() as session:
            _create_manifest_table_if_missing(session)
            result = _attest_in_session(
                session=session,
                settings=settings,
                safe_dsn=safe_dsn,
                expected_fingerprint=expected_fingerprint,
            )
            session.commit()
            return result
    except Exception:
        log.exception("Embedding cache attestation failed dsn={}", safe_dsn)
        raise
    finally:
        engine.dispose()


def import_repo_state_file_embedding_cache_from_dsn(
    *,
    settings: Settings,
    source_dsn: str,
) -> EmbeddingCacheImportResult:
    """Import compatible source file embeddings into the current target DB."""

    source_engine = _create_external_engine(source_dsn)
    target_engine = get_engine()
    source_safe = sanitize_dsn(source_dsn)
    target_safe = sanitize_dsn(settings.database_dsn)
    source_factory = sessionmaker(bind=source_engine, expire_on_commit=False, future=True)
    target_factory = sessionmaker(bind=target_engine, expire_on_commit=False, future=True)
    try:
        with source_factory() as source_session, target_factory() as target_session:
            result = import_repo_state_file_embedding_cache(
                settings=settings,
                source_session=source_session,
                target_session=target_session,
                source_dsn=source_safe,
                target_dsn=target_safe,
            )
            target_session.commit()
            return result
    except Exception:
        log.exception(
            "Embedding cache import failed source_dsn={} target_dsn={}",
            source_safe,
            target_safe,
        )
        raise
    finally:
        source_engine.dispose()


def import_repo_state_file_embedding_cache(
    *,
    settings: Settings,
    source_session: Session,
    target_session: Session,
    source_dsn: str = "source",
    target_dsn: str = "target",
) -> EmbeddingCacheImportResult:
    """Import compatible source file embeddings into a target session."""

    source_manifest = _load_manifest(source_session)
    if source_manifest is None:
        raise EmbeddingCacheManifestError(
            "Source embedding cache manifest is missing; run embedding-cache attest first."
        )
    target_manifest = _ensure_current_manifest_in_session(
        session=target_session,
        settings=settings,
    )
    if source_manifest.fingerprint != target_manifest.fingerprint:
        raise EmbeddingCacheManifestError(
            "Embedding cache fingerprint mismatch "
            f"(source={source_manifest.fingerprint} target={target_manifest.fingerprint})."
        )

    expected_model = _expected_stored_model(source_manifest.payload)
    expected_dimensions = _expected_dimensions(source_manifest.payload)
    source_rows = 0
    inserted_rows = 0
    skipped_rows = 0
    last_sha: str | None = None

    while True:
        rows = _load_source_cache_batch(source_session, last_sha=last_sha, batch_size=_BATCH_SIZE)
        if not rows:
            break
        values: list[dict[str, object]] = []
        for row in rows:
            source_rows += 1
            values.append(
                _validated_cache_row_values(
                    row=row,
                    expected_model=expected_model,
                    expected_dimensions=expected_dimensions,
                )
            )
        inserted_rows += _insert_cache_rows(target_session, values)
        last_sha = str(getattr(rows[-1], "blob_sha", "") or "").strip() or last_sha

    already_present_rows = max(source_rows - inserted_rows - skipped_rows, 0)
    result = EmbeddingCacheImportResult(
        source_rows=source_rows,
        inserted_rows=inserted_rows,
        already_present_rows=already_present_rows,
        skipped_rows=skipped_rows,
        fingerprint=str(source_manifest.fingerprint),
        source_manifest=str(source_manifest.source),
        target_manifest=str(target_manifest.source),
        source_dsn=source_dsn,
        target_dsn=target_dsn,
    )
    log.info(
        "Embedding cache import complete source_rows={} inserted_rows={} already_present_rows={} "
        "skipped_rows={} fingerprint={} source_manifest={} target_manifest={}",
        result.source_rows,
        result.inserted_rows,
        result.already_present_rows,
        result.skipped_rows,
        result.fingerprint,
        result.source_manifest,
        result.target_manifest,
    )
    return result


def sanitize_dsn(raw_dsn: str) -> str:
    """Return a log-safe DSN string."""

    try:
        make_url(str(raw_dsn))
        parts = urlsplit(str(raw_dsn))
    except Exception:
        return "<invalid-dsn>"
    if not parts.scheme:
        return "<invalid-dsn>"

    host = parts.hostname
    netloc = ""
    if host:
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        netloc = host
        if parts.port is not None:
            netloc = f"{netloc}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def _ensure_current_manifest_in_session(
    *,
    session: Session,
    settings: Settings,
) -> EmbeddingCacheManifest:
    payload = build_repo_state_file_embedding_manifest_payload(settings)
    fingerprint = fingerprint_payload(payload)
    existing = _load_manifest(session)
    if existing is not None:
        if str(existing.fingerprint) != fingerprint:
            raise EmbeddingCacheManifestError(
                "Embedding cache manifest is incompatible with current settings "
                f"(stored={existing.fingerprint} current={fingerprint})."
            )
        return existing

    cache_rows = _file_cache_row_count(session)
    if cache_rows > 0:
        raise EmbeddingCacheManifestError(
            "Embedding cache manifest is missing but file cache rows already exist; "
            "run embedding-cache attest before using or importing this cache."
        )

    manifest = EmbeddingCacheManifest(
        id=uuid.uuid4(),
        cache_kind=REPO_STATE_FILE_EMBEDDING_CACHE_KIND,
        fingerprint=fingerprint,
        payload=payload,
        source="generated",
    )
    session.add(manifest)
    session.flush()
    log.info("Generated embedding cache manifest fingerprint={}", fingerprint)
    return manifest


def _attest_in_session(
    *,
    session: Session,
    settings: Settings,
    safe_dsn: str,
    expected_fingerprint: str | None,
) -> EmbeddingCacheAttestationResult:
    existing = _load_manifest(session)
    if existing is not None:
        raise EmbeddingCacheManifestError(
            f"Embedding cache manifest already exists for {REPO_STATE_FILE_EMBEDDING_CACHE_KIND}."
        )

    payload = build_repo_state_file_embedding_manifest_payload(settings)
    fingerprint = fingerprint_payload(payload)
    expected = str(expected_fingerprint or "").strip()
    if expected and expected != fingerprint:
        raise EmbeddingCacheManifestError(
            f"Expected fingerprint {expected} does not match current settings fingerprint {fingerprint}."
        )

    combos = _file_cache_model_dimension_groups(session)
    cache_rows = sum(int(row["count"]) for row in combos)
    expected_model = _expected_stored_model(payload)
    expected_dimensions = _expected_dimensions(payload)
    row_model: str | None = None
    row_dimensions: int | None = None
    if len(combos) > 1:
        raise EmbeddingCacheManifestError(
            "Legacy embedding cache contains multiple embedding_model/dimensions combinations; "
            "cannot attest safely."
        )
    if combos:
        row_model = str(combos[0]["embedding_model"] or "")
        row_dimensions = int(combos[0]["dimensions"] or 0)
        if row_model != expected_model or row_dimensions != expected_dimensions:
            raise EmbeddingCacheManifestError(
                "Legacy embedding cache model/dimensions do not match current settings "
                f"(cache_model={row_model!r} cache_dims={row_dimensions} "
                f"expected_model={expected_model!r} expected_dims={expected_dimensions})."
            )

    manifest = EmbeddingCacheManifest(
        id=uuid.uuid4(),
        cache_kind=REPO_STATE_FILE_EMBEDDING_CACHE_KIND,
        fingerprint=fingerprint,
        payload=payload,
        source="operator_attested",
    )
    session.add(manifest)
    session.flush()
    log.warning(
        "Operator attested legacy embedding cache dsn={} cache_rows={} fingerprint={}",
        safe_dsn,
        cache_rows,
        fingerprint,
    )
    return EmbeddingCacheAttestationResult(
        fingerprint=fingerprint,
        manifest_source="operator_attested",
        cache_rows=cache_rows,
        embedding_model=row_model,
        dimensions=row_dimensions,
        dsn=safe_dsn,
    )


def _load_manifest(session: Session) -> EmbeddingCacheManifest | None:
    if not _manifest_table_exists(session):
        return None
    stmt = select(EmbeddingCacheManifest).where(
        EmbeddingCacheManifest.cache_kind == REPO_STATE_FILE_EMBEDDING_CACHE_KIND,
    )
    return session.execute(stmt).scalar_one_or_none()


def _manifest_table_exists(session: Session) -> bool:
    bind = session.get_bind()
    if bind is None:
        return True
    dialect_name = str(getattr(getattr(bind, "dialect", None), "name", "") or "")
    if dialect_name != "postgresql":
        return True
    result = session.execute(
        text("SELECT to_regclass('embedding_cache_manifests') IS NOT NULL")
    ).scalar_one()
    return bool(result)


def _create_manifest_table_if_missing(session: Session) -> None:
    session.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache_manifests (
                id UUID PRIMARY KEY,
                cache_kind VARCHAR(64) NOT NULL,
                fingerprint VARCHAR(64) NOT NULL,
                payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                source VARCHAR(64) NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now(),
                updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
            )
            """,
        )
    )
    session.execute(
        text(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM pg_constraint
                    WHERE conname = 'uq_embedding_cache_manifests_cache_kind'
                      AND conrelid = 'embedding_cache_manifests'::regclass
                ) THEN
                    ALTER TABLE embedding_cache_manifests
                    ADD CONSTRAINT uq_embedding_cache_manifests_cache_kind UNIQUE (cache_kind);
                END IF;
            END $$;
            """,
        )
    )


def _file_cache_row_count(session: Session) -> int:
    return int(
        session.execute(
            select(func.count(MapElitesFileEmbeddingCache.blob_sha)),
        ).scalar_one()
        or 0
    )


def _file_cache_model_dimension_groups(session: Session) -> list[dict[str, object]]:
    rows = session.execute(
        select(
            MapElitesFileEmbeddingCache.embedding_model,
            MapElitesFileEmbeddingCache.dimensions,
            func.count(MapElitesFileEmbeddingCache.blob_sha).label("count"),
        ).group_by(
            MapElitesFileEmbeddingCache.embedding_model,
            MapElitesFileEmbeddingCache.dimensions,
        )
    ).mappings()
    return [dict(row) for row in rows]


def _load_source_cache_batch(
    session: Session,
    *,
    last_sha: str | None,
    batch_size: int,
) -> list[MapElitesFileEmbeddingCache]:
    stmt = select(MapElitesFileEmbeddingCache).order_by(
        MapElitesFileEmbeddingCache.blob_sha.asc()
    )
    if last_sha:
        stmt = stmt.where(MapElitesFileEmbeddingCache.blob_sha > str(last_sha))
    stmt = stmt.limit(max(1, int(batch_size)))
    return list(session.execute(stmt).scalars().all())


def _validated_cache_row_values(
    *,
    row: MapElitesFileEmbeddingCache,
    expected_model: str,
    expected_dimensions: int,
) -> dict[str, object]:
    blob_sha = str(getattr(row, "blob_sha", "") or "").strip()
    if not blob_sha:
        raise EmbeddingCacheManifestError("Embedding cache row has an empty blob_sha.")
    embedding_model = str(getattr(row, "embedding_model", "") or "")
    if embedding_model != expected_model:
        raise EmbeddingCacheManifestError(
            "Embedding cache row has unexpected embedding_model "
            f"(blob_sha={blob_sha} expected={expected_model!r} got={embedding_model!r})."
        )
    dimensions = int(getattr(row, "dimensions", 0) or 0)
    if dimensions != expected_dimensions:
        raise EmbeddingCacheManifestError(
            "Embedding cache row has unexpected dimensions "
            f"(blob_sha={blob_sha} expected={expected_dimensions} got={dimensions})."
        )
    vector = [float(value) for value in (getattr(row, "vector", None) or [])]
    if not vector:
        raise EmbeddingCacheManifestError(
            f"Embedding cache row has an empty vector (blob_sha={blob_sha})."
        )
    if len(vector) != expected_dimensions:
        raise EmbeddingCacheManifestError(
            "Embedding cache row vector length does not match dimensions "
            f"(blob_sha={blob_sha} expected={expected_dimensions} got={len(vector)})."
        )
    return {
        "blob_sha": blob_sha,
        "embedding_model": embedding_model,
        "dimensions": dimensions,
        "vector": vector,
    }


def _insert_cache_rows(session: Session, values: Sequence[Mapping[str, object]]) -> int:
    payload = [dict(value) for value in values if value]
    if not payload:
        return 0
    stmt = pg_insert(MapElitesFileEmbeddingCache).values(payload)
    stmt = stmt.on_conflict_do_nothing(index_elements=["blob_sha"])
    result = session.execute(stmt)
    rowcount = int(getattr(result, "rowcount", 0) or 0)
    return max(rowcount, 0)


def _expected_stored_model(payload: Mapping[str, Any]) -> str:
    provider = payload.get("embedding_provider")
    if isinstance(provider, Mapping):
        requested = str(provider.get("requested_model") or "").strip()
        if requested:
            return requested
    model = str(payload.get("embedding_model") or "").strip()
    if model:
        return model
    raise EmbeddingCacheManifestError("Embedding cache manifest does not define an embedding model.")


def _expected_dimensions(payload: Mapping[str, Any]) -> int:
    dimensions = int(payload.get("embedding_dimensions") or 0)
    if dimensions <= 0:
        raise EmbeddingCacheManifestError(
            "Embedding cache manifest does not define positive embedding dimensions."
        )
    return dimensions


def _embedding_provider_payload(
    settings: Settings,
    *,
    requested_model: str,
) -> dict[str, str]:
    lowered_model = requested_model.lower()
    if lowered_model.startswith("local-hash"):
        return {
            "provider": "local-hash",
            "base_url": "local-hash://builtin",
            "requested_model": requested_model,
            "resolved_model": requested_model,
        }

    base_url = _normalized_base_url(
        str(getattr(settings, "openai_base_url", "") or "").strip() or _DEFAULT_OPENAI_BASE_URL
    )
    return {
        "provider": _provider_name_from_base_url(base_url),
        "base_url": base_url,
        "requested_model": requested_model,
        "resolved_model": requested_model,
    }


def _normalized_base_url(raw_url: str) -> str:
    parsed = urlsplit(raw_url)
    if not parsed.scheme or not parsed.netloc:
        parsed = urlsplit(f"https://{raw_url}")
    scheme = (parsed.scheme or "https").lower()
    hostname = (parsed.hostname or "").lower()
    port = f":{parsed.port}" if parsed.port is not None else ""
    path = parsed.path.rstrip("/") or ""
    return urlunsplit((scheme, f"{hostname}{port}", path, "", ""))


def _provider_name_from_base_url(base_url: str) -> str:
    host = urlsplit(base_url).hostname or ""
    host = host.lower()
    if host.endswith("openai.com"):
        return "openai"
    if "openrouter" in host:
        return "openrouter"
    return "openai-compatible"


def _normalize_extensions(values: Sequence[object]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        text_value = str(value or "").strip().lower()
        if not text_value:
            continue
        if not text_value.startswith("."):
            text_value = f".{text_value}"
        normalized.append(text_value)
    return sorted(set(normalized))


def _normalize_excluded_globs(values: Sequence[object]) -> list[str]:
    expanded: list[str] = []
    for value in values:
        cleaned = str(value or "").strip().replace("\\", "/")
        if cleaned.startswith("./"):
            cleaned = cleaned[2:]
        cleaned = cleaned.lstrip("/")
        if not cleaned:
            continue
        variants = {cleaned}
        if "/" in cleaned and not cleaned.startswith("**/"):
            variants.add(f"**/{cleaned}")
        expanded.extend(variants)
    return sorted(set(expanded))


def _sorted_clean_strings(values: Sequence[object], *, lower: bool = False) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        text_value = str(value or "").strip()
        if not text_value:
            continue
        if lower:
            text_value = text_value.lower()
        cleaned.append(text_value)
    return sorted(set(cleaned))


def _normalize_chunk_boundary_keywords(values: Sequence[object]) -> list[str]:
    return sorted({str(value).lower() for value in values if value})


def _create_external_engine(dsn: str) -> Engine:
    try:
        return create_engine(str(dsn), pool_pre_ping=True, future=True)
    except SQLAlchemyError:
        raise
    except Exception as exc:
        raise EmbeddingCacheManifestError(f"Invalid database DSN: {exc}") from exc
