# Unreleased

These notes cover changes merged after `v0.7.8-alpha`.

- Added dynamic OpenAI-compatible API key support via `OPENAI_DYNAMIC_API_KEY_PROVIDER`.
  Internal SDK calls now share a process-local cached token that refreshes before expiry, while Kilocode planning/coding subprocesses fetch a fresh one-shot token immediately before each launch.
- Scheduler/worker preflight now accepts dynamic OpenAI auth configuration, validates provider/TTL wiring without calling the provider, and warns when a dynamic token TTL is shorter than configured Kilocode planning/coding timeouts.
