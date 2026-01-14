from __future__ import annotations

import inspect
from importlib import import_module
from typing import Any, cast

from loreley.core.worker.agent.contracts import AgentBackend


def _split_backend_reference(ref: str) -> tuple[str, str]:
    """Split a backend reference into module and attribute path."""
    if ":" in ref:
        module_name, attr_path = ref.split(":", 1)
        return module_name, attr_path
    module_name, _, attr_path = ref.rpartition(".")
    if not module_name or not attr_path:
        raise RuntimeError(
            f"Invalid agent backend reference {ref!r}. Use 'module:attr' or 'module.attr'.",
        )
    return module_name, attr_path


def _import_backend_target(module_name: str, attr_path: str) -> Any:
    """Import the target object for a backend reference."""
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Could not import agent backend module {module_name!r}.",
        ) from exc
    target: Any = module
    for part in attr_path.split("."):
        if not part:
            raise RuntimeError(
                f"Invalid agent backend attribute reference {attr_path!r}.",
            )
        try:
            target = getattr(target, part)
        except AttributeError as exc:
            raise RuntimeError(
                f"Module {module_name!r} does not expose attribute {attr_path!r}.",
            ) from exc
    return target


def load_agent_backend(ref: str, *, label: str) -> AgentBackend:
    """Resolve and instantiate an AgentBackend from a dotted reference.

    The reference can point to:
    - an already-instantiated backend object exposing a ``run(...)`` method
    - a class implementing the ``AgentBackend`` protocol (constructed with no arguments)
    - a callable factory that returns a backend instance when called with no arguments
    """
    module_name, attr_path = _split_backend_reference(ref)
    target = _import_backend_target(module_name, attr_path)

    # Already-instantiated backend instance.
    # Avoid treating classes as instances even though they expose a callable ``run`` attribute.
    if not inspect.isclass(target) and hasattr(target, "run") and callable(
        getattr(target, "run")
    ):
        return cast(AgentBackend, target)

    # Class or factory function returning a backend instance.
    if callable(target):
        instance = target()
        if hasattr(instance, "run") and callable(getattr(instance, "run")):
            return cast(AgentBackend, instance)
        raise RuntimeError(
            f"Resolved {label} {ref!r} callable did not return a valid AgentBackend "
            "(missing callable 'run' method).",
        )

    raise RuntimeError(
        f"Resolved {label} {ref!r} is not a valid AgentBackend "
        "(object must expose a callable 'run' method).",
    )


__all__ = ["load_agent_backend"]

