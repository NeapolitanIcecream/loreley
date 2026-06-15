from __future__ import annotations

from types import SimpleNamespace

import pytest

import loreley.core.map_elites.embedding_cache_manifest as ecm
from tests.support import TestSettings


class _FakeSession:
    def __init__(self) -> None:
        self.added: list[object] = []
        self.flushed = False

    def add(self, value: object) -> None:
        self.added.append(value)

    def flush(self) -> None:
        self.flushed = True


def _settings(**overrides: object) -> TestSettings:
    defaults = {
        "EXPERIMENT_ID": "embedding-cache-test",
        "MAPELITES_EXPERIMENT_ROOT_COMMIT": "deadbeef",
        "MAPELITES_CODE_EMBEDDING_MODEL": "text-embedding-3-small",
        "MAPELITES_CODE_EMBEDDING_DIMENSIONS": 8,
    }
    defaults.update(overrides)
    return TestSettings(**defaults)


def test_fingerprint_includes_sanitized_provider_identity() -> None:
    settings = _settings(
        OPENAI_BASE_URL="https://user:secret@openrouter.ai/api/v1?token=hidden",
        MAPELITES_CODE_EMBEDDING_MODEL="openai/text-embedding-3-small",
    )

    payload = ecm.build_repo_state_file_embedding_manifest_payload(settings)

    provider = payload["embedding_provider"]
    assert provider == {
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "requested_model": "openai/text-embedding-3-small",
        "resolved_model": "openai/text-embedding-3-small",
    }
    assert "secret" not in ecm.fingerprint_payload(payload)


def test_sanitize_dsn_strips_userinfo_and_query_secrets() -> None:
    raw = "postgresql+psycopg://alice:secret@db.internal:5432/loreley?sslpassword=token"

    safe = ecm.sanitize_dsn(raw)

    assert safe == "postgresql+psycopg://db.internal:5432/loreley"
    assert "alice" not in safe
    assert "secret" not in safe
    assert "sslpassword" not in safe
    assert "token" not in safe


def test_ensure_manifest_generates_only_when_target_cache_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession()
    monkeypatch.setattr(ecm, "_load_manifest", lambda _session: None)
    monkeypatch.setattr(ecm, "_file_cache_row_count", lambda _session: 0)

    manifest = ecm.ensure_current_repo_state_file_embedding_manifest(
        settings=_settings(),
        session=session,  # type: ignore[arg-type]
    )

    assert manifest.source == "generated"
    assert manifest.cache_kind == ecm.REPO_STATE_FILE_EMBEDDING_CACHE_KIND
    assert session.added == [manifest]
    assert session.flushed is True


def test_ensure_manifest_refuses_to_bless_existing_cache_without_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ecm, "_load_manifest", lambda _session: None)
    monkeypatch.setattr(ecm, "_file_cache_row_count", lambda _session: 1)

    with pytest.raises(ecm.EmbeddingCacheManifestError, match="file cache rows already exist"):
        ecm.ensure_current_repo_state_file_embedding_manifest(
            settings=_settings(),
            session=_FakeSession(),  # type: ignore[arg-type]
        )


def test_import_reuses_only_manifest_compatible_valid_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    payload = ecm.build_repo_state_file_embedding_manifest_payload(settings)
    fingerprint = ecm.fingerprint_payload(payload)
    source_manifest = SimpleNamespace(
        fingerprint=fingerprint,
        payload=payload,
        source="operator_attested",
    )
    target_manifest = SimpleNamespace(
        fingerprint=fingerprint,
        payload=payload,
        source="generated",
    )
    batches = [
        [
            SimpleNamespace(
                blob_sha="sha1",
                embedding_model="text-embedding-3-small",
                dimensions=8,
                vector=[1.0] * 8,
            ),
            SimpleNamespace(
                blob_sha="sha2",
                embedding_model="text-embedding-3-small",
                dimensions=8,
                vector=[2.0] * 8,
            ),
        ],
        [],
    ]
    inserted_payloads: list[list[dict[str, object]]] = []

    monkeypatch.setattr(ecm, "_load_manifest", lambda _session: source_manifest)
    monkeypatch.setattr(
        ecm,
        "_ensure_current_manifest_in_session",
        lambda **_kwargs: target_manifest,
    )
    monkeypatch.setattr(ecm, "_load_source_cache_batch", lambda *_args, **_kwargs: batches.pop(0))

    def _fake_insert(_session, values):  # type: ignore[no-untyped-def]
        batch = [dict(value) for value in values]
        inserted_payloads.append(batch)
        return 1

    monkeypatch.setattr(ecm, "_insert_cache_rows", _fake_insert)

    result = ecm.import_repo_state_file_embedding_cache(
        settings=settings,
        source_session=object(),  # type: ignore[arg-type]
        target_session=object(),  # type: ignore[arg-type]
        source_dsn="postgresql://source",
        target_dsn="postgresql://target",
    )

    assert result.source_rows == 2
    assert result.inserted_rows == 1
    assert result.already_present_rows == 1
    assert result.source_manifest == "operator_attested"
    assert result.target_manifest == "generated"
    assert inserted_payloads[0][0]["blob_sha"] == "sha1"


def test_import_rejects_bad_source_row_before_insert(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings()
    payload = ecm.build_repo_state_file_embedding_manifest_payload(settings)
    fingerprint = ecm.fingerprint_payload(payload)
    manifest = SimpleNamespace(
        fingerprint=fingerprint,
        payload=payload,
        source="operator_attested",
    )

    monkeypatch.setattr(ecm, "_load_manifest", lambda _session: manifest)
    monkeypatch.setattr(ecm, "_ensure_current_manifest_in_session", lambda **_kwargs: manifest)
    monkeypatch.setattr(
        ecm,
        "_load_source_cache_batch",
        lambda *_args, **_kwargs: [
            SimpleNamespace(
                blob_sha="sha1",
                embedding_model="other-model",
                dimensions=8,
                vector=[1.0] * 8,
            )
        ],
    )
    monkeypatch.setattr(
        ecm,
        "_insert_cache_rows",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not insert")),
    )

    with pytest.raises(ecm.EmbeddingCacheManifestError, match="unexpected embedding_model"):
        ecm.import_repo_state_file_embedding_cache(
            settings=settings,
            source_session=object(),  # type: ignore[arg-type]
            target_session=object(),  # type: ignore[arg-type]
        )


def test_attestation_rejects_mixed_legacy_cache_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _FakeSession()
    monkeypatch.setattr(ecm, "_load_manifest", lambda _session: None)
    monkeypatch.setattr(
        ecm,
        "_file_cache_model_dimension_groups",
        lambda _session: [
            {"embedding_model": "text-embedding-3-small", "dimensions": 8, "count": 1},
            {"embedding_model": "other", "dimensions": 8, "count": 1},
        ],
    )

    with pytest.raises(ecm.EmbeddingCacheManifestError, match="multiple embedding_model"):
        ecm._attest_in_session(  # noqa: SLF001 - session-level contract avoids external DB setup
            session=session,  # type: ignore[arg-type]
            settings=_settings(),
            safe_dsn="postgresql://legacy",
            expected_fingerprint=None,
        )
