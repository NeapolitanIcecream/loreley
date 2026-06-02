# Unreleased

These notes cover changes merged after `v0.8.2-alpha`.

## Changed

- Refreshed Kilocode CLI compatibility for current Kilo releases. Loreley now
  discovers the installed `kilo run` command surface during worker preflight,
  uses isolated `KILO_CONFIG_CONTENT` provider config by default for
  `WORKER_KILOCODE_OPENAI_*`, and keeps `legacy_env` / `none` modes for pinned
  older CLIs or persisted Kilo auth/config.
- Kilocode usage tracking now resolves the DB path through `kilo db path` when
  `WORKER_KILOCODE_USAGE_DB_PATH` is unset and records explicit unavailable
  usage events when the DB is missing or the schema no longer matches.
  Preflight now fails when usage tracking is enabled but the installed Kilo
  binary does not expose `kilo run --title`, because tracked worker commands use
  that flag to correlate Kilo sessions with Loreley jobs.
