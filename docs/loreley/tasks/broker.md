# loreley.tasks.broker

Helpers for configuring the Dramatiq Redis broker used by Loreley workers.

## Public API

- **`build_redis_broker(settings: Settings | None = None) -> RedisBroker`**  
  Constructs a `RedisBroker` instance from the `Settings` object. It prefers `TASKS_REDIS_URL` when set and otherwise falls back to the individual `TASKS_REDIS_HOST`, `TASKS_REDIS_PORT`, `TASKS_REDIS_DB`, and `TASKS_REDIS_PASSWORD` fields, always attaching `TASKS_REDIS_NAMESPACE` as the Dramatiq namespace.

- **`setup_broker(settings: Settings | None = None) -> RedisBroker`**  
  Wraps `build_redis_broker()` and calls `dramatiq.set_broker(...)` so that Dramatiq actors use the configured Redis broker. It logs a sanitised representation of the Redis connection (scheme, host, port, and DB index) along with the logical namespace, explicitly avoiding logging any credentials from `TASKS_REDIS_URL` or `TASKS_REDIS_PASSWORD`.

- **`broker`**  
  A module-level `RedisBroker` instance created eagerly by calling `setup_broker()` at import time. Importing `loreley.tasks.broker` is therefore sufficient to configure the global Dramatiq broker; this side effect is relied on by `loreley.tasks.workers` when running worker processes.


