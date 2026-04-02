"""Runtime OpenAI-compatible API key resolution.

Supports either:
- static `OPENAI_API_KEY` / `LORELEY_LLM_API_KEY`
- a dynamic zero-arg provider that returns a token string

Dynamic providers are process-local and share one cached token for internal SDK
calls, while agent subprocess launches fetch one-shot tokens on demand.
"""

from __future__ import annotations

import inspect
from importlib import import_module
import threading
import time
from typing import Any, Callable

from loguru import logger

from loreley.config import Settings, get_settings

log = logger.bind(module="core.openai_auth")

__all__ = [
    "DynamicOpenAIKeyConfigurationError",
    "DynamicOpenAIKeyManager",
    "DynamicOpenAIKeyUnavailableError",
    "build_internal_openai_client_kwargs",
    "dynamic_openai_auth_enabled",
    "get_agent_openai_api_key",
    "get_dynamic_openai_key_manager",
    "get_internal_openai_api_key",
    "load_dynamic_openai_api_key_provider",
    "reset_dynamic_openai_key_managers",
    "validate_dynamic_openai_auth_settings",
    "validate_dynamic_openai_provider_ref",
]

Provider = Callable[[], str]

_MANAGER_REGISTRY: dict[tuple[str, int, int], "DynamicOpenAIKeyManager"] = {}
_MANAGER_REGISTRY_LOCK = threading.Lock()


class DynamicOpenAIKeyConfigurationError(ValueError):
    """Raised when dynamic API key settings are invalid."""


class DynamicOpenAIKeyUnavailableError(RuntimeError):
    """Raised when no usable dynamic API key can be obtained."""


def _required_param_count(callable_obj: Any) -> int:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return 0

    required = 0
    for param in signature.parameters.values():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.default is not inspect._empty:
            continue
        required += 1
    return required


def _assert_zero_arg_callable(callable_obj: Any, *, label: str) -> None:
    required = _required_param_count(callable_obj)
    if required > 0:
        raise DynamicOpenAIKeyConfigurationError(
            f"{label} must be a zero-argument callable.",
        )


def _callable_is_async(callable_obj: Any) -> bool:
    if inspect.iscoroutinefunction(callable_obj) or inspect.isasyncgenfunction(callable_obj):
        return True

    call_impl = getattr(callable_obj, "__call__", None)
    return bool(
        call_impl
        and (
            inspect.iscoroutinefunction(call_impl)
            or inspect.isasyncgenfunction(call_impl)
        )
    )


def _assert_sync_callable(callable_obj: Any, *, label: str) -> None:
    if _callable_is_async(callable_obj):
        raise DynamicOpenAIKeyConfigurationError(
            f"{label} must be a synchronous callable.",
        )


def _find_instance_call_implementation(cls: type[Any]) -> Any | None:
    for candidate in getattr(cls, "__mro__", ()):
        namespace = getattr(candidate, "__dict__", None) or {}
        if "__call__" in namespace:
            return namespace["__call__"]
    return None


def _assert_zero_arg_callable_instance_class(cls: type[Any], *, label: str) -> None:
    call_impl = _find_instance_call_implementation(cls)
    if call_impl is None:
        raise DynamicOpenAIKeyConfigurationError(
            f"{label} class instances must be callable.",
        )

    descriptor = call_impl
    if isinstance(descriptor, staticmethod):
        callable_obj = descriptor.__func__
        offset = 0
    elif isinstance(descriptor, classmethod):
        callable_obj = descriptor.__func__
        offset = 1
    else:
        callable_obj = descriptor
        offset = 1

    if inspect.iscoroutinefunction(callable_obj) or inspect.isasyncgenfunction(callable_obj):
        raise DynamicOpenAIKeyConfigurationError(
            f"{label} class instances must expose a synchronous zero-argument __call__ method.",
        )

    required = _required_param_count(callable_obj)
    effective_required = max(0, required - offset)
    if effective_required > 0:
        raise DynamicOpenAIKeyConfigurationError(
            f"{label} class instances must expose a zero-argument __call__ method.",
        )


def _split_reference(ref: str) -> tuple[str, str]:
    if ":" in ref:
        module_name, attr_path = ref.split(":", 1)
        if not module_name or not attr_path:
            raise DynamicOpenAIKeyConfigurationError(
                f"Invalid dynamic API key provider reference {ref!r}. "
                "Use 'module:attr' or 'module.attr'.",
            )
        return module_name, attr_path
    module_name, _, attr_path = ref.rpartition(".")
    if not module_name or not attr_path:
        raise DynamicOpenAIKeyConfigurationError(
            f"Invalid dynamic API key provider reference {ref!r}. "
            "Use 'module:attr' or 'module.attr'.",
        )
    return module_name, attr_path


def _import_reference(ref: str) -> Any:
    module_name, attr_path = _split_reference(ref)
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        raise DynamicOpenAIKeyConfigurationError(
            f"Could not import dynamic API key provider module {module_name!r}.",
        ) from exc
    except ValueError as exc:
        raise DynamicOpenAIKeyConfigurationError(
            f"Invalid dynamic API key provider reference {ref!r}. "
            "Use 'module:attr' or 'module.attr'.",
        ) from exc

    target: Any = module
    for part in attr_path.split("."):
        if not part:
            raise DynamicOpenAIKeyConfigurationError(
                f"Invalid dynamic API key provider attribute reference {attr_path!r}.",
            )
        try:
            target = getattr(target, part)
        except AttributeError as exc:
            raise DynamicOpenAIKeyConfigurationError(
                f"Module {module_name!r} does not expose attribute {attr_path!r}.",
            ) from exc
    return target


def validate_dynamic_openai_provider_ref(ref: str) -> None:
    normalized = str(ref or "").strip()
    if not normalized:
        raise DynamicOpenAIKeyConfigurationError(
            "OPENAI_DYNAMIC_API_KEY_PROVIDER is not set.",
        )

    target = _import_reference(normalized)
    if inspect.isclass(target):
        _assert_zero_arg_callable(target, label="OPENAI_DYNAMIC_API_KEY_PROVIDER")
        _assert_zero_arg_callable_instance_class(
            target,
            label="OPENAI_DYNAMIC_API_KEY_PROVIDER",
        )
        return
    if not callable(target):
        raise DynamicOpenAIKeyConfigurationError(
            "OPENAI_DYNAMIC_API_KEY_PROVIDER must resolve to a callable object.",
        )
    _assert_sync_callable(target, label="OPENAI_DYNAMIC_API_KEY_PROVIDER")
    _assert_zero_arg_callable(target, label="OPENAI_DYNAMIC_API_KEY_PROVIDER")


def load_dynamic_openai_api_key_provider(ref: str) -> Provider:
    validate_dynamic_openai_provider_ref(ref)
    target = _import_reference(ref)
    if inspect.isclass(target):
        instance = target()
        if not callable(instance):
            raise DynamicOpenAIKeyConfigurationError(
                "OPENAI_DYNAMIC_API_KEY_PROVIDER class instances must be callable.",
            )
        _assert_sync_callable(instance, label="OPENAI_DYNAMIC_API_KEY_PROVIDER")
        _assert_zero_arg_callable(instance, label="OPENAI_DYNAMIC_API_KEY_PROVIDER")
        return instance
    if not callable(target):
        raise DynamicOpenAIKeyConfigurationError(
            "OPENAI_DYNAMIC_API_KEY_PROVIDER must resolve to a callable object.",
        )
    _assert_sync_callable(target, label="OPENAI_DYNAMIC_API_KEY_PROVIDER")
    return target


def _provider_ref(settings: Settings) -> str | None:
    value = str(getattr(settings, "openai_dynamic_api_key_provider", "") or "").strip()
    return value or None


def _ttl_seconds(settings: Settings) -> int | None:
    raw = getattr(settings, "openai_dynamic_api_key_ttl_seconds", None)
    if raw is None:
        return None
    value = int(raw)
    if value <= 0:
        raise DynamicOpenAIKeyConfigurationError(
            "OPENAI_DYNAMIC_API_KEY_TTL_SECONDS must be a positive integer.",
        )
    return value


def _refresh_skew_seconds(settings: Settings, *, ttl_seconds: int) -> int:
    raw = getattr(settings, "openai_dynamic_api_key_refresh_skew_seconds", None)
    if raw is None:
        value = min(60, max(1, ttl_seconds // 10))
    else:
        value = int(raw)
    if value <= 0:
        raise DynamicOpenAIKeyConfigurationError(
            "OPENAI_DYNAMIC_API_KEY_REFRESH_SKEW_SECONDS must be a positive integer.",
        )
    if value >= ttl_seconds:
        raise DynamicOpenAIKeyConfigurationError(
            "OPENAI_DYNAMIC_API_KEY_REFRESH_SKEW_SECONDS must be smaller than "
            "OPENAI_DYNAMIC_API_KEY_TTL_SECONDS.",
        )
    return value


def dynamic_openai_auth_enabled(settings: Settings | None = None) -> bool:
    active = settings or get_settings()
    return _provider_ref(active) is not None


def validate_dynamic_openai_auth_settings(settings: Settings) -> tuple[str, int, int] | None:
    provider_ref = _provider_ref(settings)
    ttl_raw = getattr(settings, "openai_dynamic_api_key_ttl_seconds", None)
    skew_raw = getattr(settings, "openai_dynamic_api_key_refresh_skew_seconds", None)
    if provider_ref is None and ttl_raw is None and skew_raw is None:
        return None
    if provider_ref is None:
        raise DynamicOpenAIKeyConfigurationError(
            "OPENAI_DYNAMIC_API_KEY_PROVIDER must be set when "
            "OPENAI_DYNAMIC_API_KEY_TTL_SECONDS or "
            "OPENAI_DYNAMIC_API_KEY_REFRESH_SKEW_SECONDS is configured.",
        )
    ttl_seconds = _ttl_seconds(settings)
    if ttl_seconds is None:
        raise DynamicOpenAIKeyConfigurationError(
            "OPENAI_DYNAMIC_API_KEY_TTL_SECONDS must be set when "
            "OPENAI_DYNAMIC_API_KEY_PROVIDER is configured.",
        )
    skew_seconds = _refresh_skew_seconds(settings, ttl_seconds=ttl_seconds)
    validate_dynamic_openai_provider_ref(provider_ref)
    return provider_ref, ttl_seconds, skew_seconds


class DynamicOpenAIKeyManager:
    """Process-local shared-token manager for dynamic OpenAI-compatible auth."""

    def __init__(
        self,
        *,
        provider: Provider,
        provider_ref: str,
        ttl_seconds: int,
        refresh_skew_seconds: int | None = None,
        monotonic: Callable[[], float] | None = None,
        start_refresh_thread: bool = True,
    ) -> None:
        self._provider = provider
        self.provider_ref = str(provider_ref).strip() or "<dynamic-provider>"
        self.ttl_seconds = int(ttl_seconds)
        if self.ttl_seconds <= 0:
            raise DynamicOpenAIKeyConfigurationError(
                "OPENAI_DYNAMIC_API_KEY_TTL_SECONDS must be a positive integer.",
            )
        if refresh_skew_seconds is None:
            refresh_skew_seconds = min(60, max(1, self.ttl_seconds // 10))
        self.refresh_skew_seconds = int(refresh_skew_seconds)
        if self.refresh_skew_seconds <= 0 or self.refresh_skew_seconds >= self.ttl_seconds:
            raise DynamicOpenAIKeyConfigurationError(
                "OPENAI_DYNAMIC_API_KEY_REFRESH_SKEW_SECONDS must be positive and smaller than the TTL.",
            )
        self._monotonic = monotonic or time.monotonic
        self._start_refresh_thread = bool(start_refresh_thread)
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._refresh_thread: threading.Thread | None = None
        self._shared_token: str | None = None
        self._shared_expires_at: float | None = None
        self._shared_refresh_at: float | None = None
        self._last_refresh_error: str | None = None

    def close(self) -> None:
        self._stop_event.set()
        thread = self._refresh_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def get_shared_token(self) -> str:
        with self._lock:
            self._ensure_refresh_thread_locked()
        token = self.refresh_if_due()
        if not token:
            raise DynamicOpenAIKeyUnavailableError(
                "Dynamic OpenAI API key provider returned an empty shared token.",
            )
        return token

    def get_agent_token(self) -> str:
        token = self._fetch_one_shot_token()
        if not token:
            raise DynamicOpenAIKeyUnavailableError(
                "Dynamic OpenAI API key provider returned an empty agent token.",
            )
        return token

    def refresh_if_due(self, *, force: bool = False) -> str | None:
        with self._lock:
            now = float(self._monotonic())
            if (
                not force
                and self._shared_refresh_at is not None
                and now < self._shared_refresh_at
            ):
                if (
                    self._shared_token
                    and self._shared_expires_at is not None
                    and now < self._shared_expires_at
                ):
                    return self._shared_token
                if self._last_refresh_error:
                    raise DynamicOpenAIKeyUnavailableError(self._last_refresh_error)
            try:
                return self._fetch_shared_token_locked(now=now)
            except Exception as exc:
                error_message = str(exc)
                retry_in = self._schedule_failed_refresh_retry_locked(now=now)
                self._last_refresh_error = error_message
                if (
                    self._shared_token
                    and self._shared_expires_at is not None
                    and now < self._shared_expires_at
                ):
                    remaining = max(0.0, float(self._shared_expires_at - now))
                    log.warning(
                        "Dynamic OpenAI API key refresh failed; reusing cached key "
                        "provider_ref={} expires_in_seconds={:.1f} retry_in_seconds={:.1f} error={}",
                        self.provider_ref,
                        remaining,
                        retry_in,
                        exc,
                    )
                    return self._shared_token
                log.warning(
                    "Dynamic OpenAI API key is unavailable provider_ref={} "
                    "retry_in_seconds={:.1f} error={}",
                    self.provider_ref,
                    retry_in,
                    exc,
                )
                raise DynamicOpenAIKeyUnavailableError(error_message) from exc

    def _ensure_refresh_thread_locked(self) -> None:
        if not self._start_refresh_thread:
            return
        if self._refresh_thread is not None and self._refresh_thread.is_alive():
            return
        thread = threading.Thread(
            target=self._refresh_loop,
            name="dynamic-openai-key-refresh",
            daemon=True,
        )
        thread.start()
        self._refresh_thread = thread

    def _refresh_loop(self) -> None:
        while not self._stop_event.is_set():
            timeout = 0.5
            with self._lock:
                if self._shared_refresh_at is not None:
                    timeout = max(0.1, float(self._shared_refresh_at - self._monotonic()))
            if self._stop_event.wait(timeout):
                return
            try:
                self.refresh_if_due(force=True)
            except DynamicOpenAIKeyUnavailableError:
                # Foreground callers surface the hard failure once the cached token expires.
                continue

    def _fetch_shared_token_locked(self, *, now: float) -> str:
        token = self._coerce_token(self._provider())
        first_fetch = self._shared_token is None
        self._shared_token = token
        self._shared_expires_at = now + float(self.ttl_seconds)
        self._shared_refresh_at = self._shared_expires_at - float(self.refresh_skew_seconds)
        self._last_refresh_error = None
        log.info(
            "Updated shared dynamic OpenAI API key provider_ref={} ttl_seconds={} "
            "refresh_skew_seconds={} mode={}",
            self.provider_ref,
            self.ttl_seconds,
            self.refresh_skew_seconds,
            "initial" if first_fetch else "refresh",
        )
        return token

    def _schedule_failed_refresh_retry_locked(self, *, now: float) -> float:
        retry_in = min(
            30.0,
            max(1.0, float(self.refresh_skew_seconds) / 4.0),
        )
        if self._shared_expires_at is not None:
            remaining = max(0.0, float(self._shared_expires_at - now))
            if remaining > 0:
                retry_in = min(remaining, retry_in)
        self._shared_refresh_at = now + retry_in
        return retry_in

    def _fetch_one_shot_token(self) -> str:
        token = self._coerce_token(self._provider())
        log.info(
            "Fetched one-shot dynamic OpenAI API key for agent provider_ref={}",
            self.provider_ref,
        )
        return token

    @staticmethod
    def _coerce_token(value: Any) -> str:
        if inspect.isawaitable(value):
            close = getattr(value, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            raise DynamicOpenAIKeyUnavailableError(
                "Dynamic OpenAI API key provider returned an awaitable; async providers are not supported.",
            )
        token = str(value or "").strip()
        if not token:
            raise DynamicOpenAIKeyUnavailableError(
                "Dynamic OpenAI API key provider returned an empty token.",
            )
        return token


def get_dynamic_openai_key_manager(settings: Settings | None = None) -> DynamicOpenAIKeyManager:
    active = settings or get_settings()
    validated = validate_dynamic_openai_auth_settings(active)
    if validated is None:
        raise DynamicOpenAIKeyConfigurationError(
            "Dynamic OpenAI API key provider is not configured.",
        )
    provider_ref, ttl_seconds, skew_seconds = validated
    key = (provider_ref, ttl_seconds, skew_seconds)
    with _MANAGER_REGISTRY_LOCK:
        manager = _MANAGER_REGISTRY.get(key)
        if manager is None:
            manager = DynamicOpenAIKeyManager(
                provider=load_dynamic_openai_api_key_provider(provider_ref),
                provider_ref=provider_ref,
                ttl_seconds=ttl_seconds,
                refresh_skew_seconds=skew_seconds,
            )
            _MANAGER_REGISTRY[key] = manager
        return manager


def reset_dynamic_openai_key_managers() -> None:
    with _MANAGER_REGISTRY_LOCK:
        managers = list(_MANAGER_REGISTRY.values())
        _MANAGER_REGISTRY.clear()
    for manager in managers:
        manager.close()


def get_internal_openai_api_key(settings: Settings | None = None) -> str | None:
    active = settings or get_settings()
    if dynamic_openai_auth_enabled(active):
        return get_dynamic_openai_key_manager(active).get_shared_token()
    value = str(getattr(active, "openai_api_key", "") or "").strip()
    return value or None


def get_agent_openai_api_key(settings: Settings | None = None) -> str | None:
    active = settings or get_settings()
    worker_specific = str(getattr(active, "worker_kilocode_openai_api_key", "") or "").strip()
    if worker_specific:
        return worker_specific
    if dynamic_openai_auth_enabled(active):
        return get_dynamic_openai_key_manager(active).get_agent_token()
    value = str(getattr(active, "openai_api_key", "") or "").strip()
    return value or None


def build_internal_openai_client_kwargs(settings: Settings | None = None) -> dict[str, object]:
    active = settings or get_settings()
    kwargs: dict[str, object] = {}
    api_key = get_internal_openai_api_key(active)
    if api_key:
        kwargs["api_key"] = api_key
    base_url = str(getattr(active, "openai_base_url", "") or "").strip()
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs
