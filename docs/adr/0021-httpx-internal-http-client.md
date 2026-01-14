# ADR 0021: Centralize HTTP calls on httpx via an internal client module

Date: 2026-01-14

Context: UI-to-UI-API calls used ad-hoc `urllib.request.urlopen`, duplicating timeout/redirect handling and making HTTP behavior harder to test consistently.
Decision: Introduce `loreley.net.http` (sync httpx wrapper) and refactor the UI API client and the UI API reachability check to use it, with injectable transports for unit tests.
Constraints: Keep the wrapper minimal and synchronous; preserve existing UI error semantics (status code + message) and avoid adding new external test dependencies.
Consequences: HTTP behavior is consistent across call sites, easier to extend (timeouts/headers/transports), and covered by fast deterministic tests.

