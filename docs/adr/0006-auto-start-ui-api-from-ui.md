# ADR 0006: Auto-start UI API from Streamlit UI entrypoint

Date: 2026-01-06

Context: The Streamlit UI depends on the read-only UI API. Requiring users to start two processes manually adds friction and often leads to a confusing "API not reachable" experience.
Decision: The UI entrypoint (`run_ui`) probes `GET /api/v1/health` and, when the configured API base URL is a local HTTP address, automatically starts the UI API as a subprocess.
Decision: Auto-start is only attempted for `http://127.0.0.1:<port>`, `http://localhost:<port>`, and `http://[::1]:<port>` (no path prefix); non-local and non-HTTP base URLs never trigger auto-start.
Decision: If the UI started the API subprocess, it owns its lifecycle and stops it when the UI exits (SIGTERM, Ctrl+C, or normal termination).
Consequences: Local development becomes a single-command workflow; remote deployments keep explicit control over the API process; startup errors surface early with clear console messages and logs per role.


