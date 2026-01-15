# ADR 0022: Reuse UI API client connections via Streamlit resource cache

Date: 2026-01-15

Context: The Streamlit UI instantiated short-lived API clients frequently, preventing httpx connection pooling and adding avoidable per-request overhead; large error bodies could also produce noisy UI messages.
Decision: Cache `LoreleyAPIClient` instances with `st.cache_resource` and enable `HttpClient(reuse_connections=True)`; truncate non-2xx response bodies in `HttpCallError` messages.
Constraints: Keep the implementation synchronous; avoid new runtime dependencies; preserve existing UI error semantics (status code + message).
Consequences: UI calls reuse persistent HTTP connections, reducing overhead; error messages remain readable; timeouts keep a minimum clamp to avoid pathological values.

