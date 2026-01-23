# loreley.tasks.broker

Helpers for configuring the Dramatiq Redis broker used by Loreley workers.

## Public API

- **`build_redis_broker(settings: Settings | None = None) -> RedisBroker`**  
  Constructs a `RedisBroker` instance from the `Settings` object. It prefers `TASKS_REDIS_URL` when set and otherwise falls back to the individual `TASKS_REDIS_HOST`, `TASKS_REDIS_PORT`, `TASKS_REDIS_DB`, and `TASKS_REDIS_PASSWORD` fields. The broker namespace is derived from `EXPERIMENT_ID` (not configurable) so multiple experiments can share a Redis instance safely.

- **`setup_broker(settings: Settings | None = None) -> RedisBroker`**  
  Wraps `build_redis_broker()` and calls `dramatiq.set_broker(...)` so that Dramatiq actors use the configured Redis broker. It logs a sanitised representation of the Redis connection (scheme, host, port, and DB index) along with the logical namespace, explicitly avoiding logging any credentials from `TASKS_REDIS_URL` or `TASKS_REDIS_PASSWORD`.

- **`broker`**  
  A module-level reference to the most recently configured `RedisBroker`. It is set when `setup_broker()` is called; importing `loreley.tasks.broker` does not configure Dramatiq on its own.


