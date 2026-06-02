# Proposal: Refresh Kilo CLI compatibility

## Status

Implemented (phase 1)

## Date

2026-06-02

## Implementation Update

Implemented on 2026-06-02 as a warning-first compatibility refresh:

- Verified `@kilocode/cli@latest` is still `7.3.16`; the local installed
  `kilo` is `7.3.1`.
- Verified the local `kilo run --help` exposes `--auto`, `--agent`, `--model`,
  `--format`, `--title`, and `--variant`.
- Verified the local `kilo db path` command resolves the SQLite DB path.
- Verified `KILO_CONFIG_CONTENT` is consumed by `kilo debug config` and resolves
  `{env:LORELEY_KILO_OPENAI_API_KEY}` /
  `{env:LORELEY_KILO_OPENAI_BASE_URL}`.
- Added `WORKER_KILOCODE_PROVIDER_CONFIG_MODE=auto | config | legacy_env | none`.
  The default `auto` mode now uses isolated config content. `legacy_env` keeps
  the old `KILO_PROVIDER_TYPE` / `KILO_OPENAI_*` injection for pinned older
  Kilo versions, and `none` relies on Kilo's persisted auth/config.
- Added Kilo CLI capability parsing and worker preflight checks for command
  flags, provider injection verification, and usage DB availability.
- Changed usage DB resolution to prefer explicit path, then `kilo db path`, then
  the historical fallback. Missing tables/columns now produce unavailable usage
  events rather than failing a completed Kilo run.

## Context

Loreley currently uses `KilocodeCliBackend` as the default planning and coding
agent backend. The backend invokes `kilo run --auto` and optionally passes
`--agent`, `--model`, `--variant`, `--format json`, and `--title`.

Compatibility checks against current Kilo sources show a mixed result:

- `@kilocode/cli@latest` resolves to `7.3.16` on npm as of 2026-06-02.
- `kilo run --help` in `7.3.16` still exposes the flags Loreley depends on:
  `--auto`, `--agent`, `--model`, `--format`, `--title`, and `--variant`.
- `kilo db path` exists and reports the SQLite database path. The current local
  schema still has `session` and `message` tables with the columns used by
  Loreley's usage reader.
- The provider environment variables Loreley injects today are historical:
  `KILO_PROVIDER_TYPE`, `KILO_OPENAI_API_KEY`, `KILO_OPENAI_BASE_URL`, and
  `KILO_OPENAI_MODEL_ID`. Current Kilo documentation describes generic config
  and provider overrides instead, and a `kilo debug config` check did not show
  those historical variables being resolved.

Relevant references:

- Kilo CLI docs: https://kilo.ai/docs/code-with-ai/platforms/cli
- Kilo CLI command reference: https://kilo.ai/docs/code-with-ai/platforms/cli-reference
- Kilo repository: https://github.com/Kilo-Org/kilocode
- npm package: https://www.npmjs.com/package/@kilocode/cli

## Problem

Loreley's command path is still compatible with the newest stable Kilo CLI, but
provider injection is brittle. Deployments that rely on Kilo's own persisted
auth/config can continue to work. Deployments that expect
`WORKER_KILOCODE_OPENAI_*` or dynamic OpenAI keys to configure each subprocess
may silently launch Kilo without the intended provider settings.

The usage reader also works today, but it relies on a hard-coded default DB path
instead of Kilo's own `kilo db path` command. That makes it more fragile than it
needs to be as Kilo continues to evolve.

## Goals

- Keep `kilo run` as the default Loreley planning/coding backend.
- Preserve existing `WORKER_KILOCODE_*` environment names as Loreley's public
  configuration surface.
- Support current Kilo CLI releases without relying on undocumented historical
  provider variables.
- Keep dynamic per-subprocess API key resolution working without leaking keys to
  logs or persistent files.
- Make preflight distinguish "Kilo binary exists" from "Kilo backend is
  compatible enough to run with this configuration."
- Make usage tracking degrade to explicit unavailable usage events instead of
  failing worker runs when Kilo's DB shape changes.

## Non-Goals

- Do not replace Kilo with another default backend.
- Do not require users to authenticate Kilo interactively when Loreley has been
  given worker-specific API settings.
- Do not make Kilo JSON output the default. Kilo's `--format json` remains an
  event stream, not a final Markdown document.
- Do not depend on Kilo cloud organization selection unless the user explicitly
  supplies `KILO_ORG_ID` or equivalent environment.

## Proposed Design

### 1. Add Kilo capability discovery

Introduce a small internal capability object, for example `KiloCliCapabilities`,
resolved once per backend/preflight invocation:

- `version`: parsed from `kilo --version`
- `run_flags`: parsed from `kilo run --help`
- `supports_auto`
- `supports_agent`
- `supports_model`
- `supports_variant`
- `supports_title`
- `supports_format_json`
- `supports_db_path`
- `provider_config_mode`: detected or configured

Preflight should fail if required run flags are missing. It should warn, not
fail, when optional usage tracking or provider injection capabilities cannot be
confirmed.

### 2. Keep command construction stable

The current command shape should remain the default:

```text
kilo run --auto [--format json] [--agent AGENT] [--model MODEL] [--variant VARIANT] [--title TITLE] PROMPT
```

`--auto` still exists in current Kilo and is the right automation flag for
Loreley. Do not switch to `--dangerously-skip-permissions` by default. That flag
has different semantics and is unnecessary while `--auto` remains available.

### 3. Replace historical provider env injection with an adapter

Keep `WORKER_KILOCODE_OPENAI_API_KEY`, `WORKER_KILOCODE_OPENAI_BASE_URL`,
`WORKER_KILOCODE_OPENAI_MODEL`, and `WORKER_KILOCODE_OPENAI_API_SPEC` as
Loreley settings, but stop treating them as direct Kilo environment variable
names.

Add a provider adapter with explicit modes:

```text
WORKER_KILOCODE_PROVIDER_CONFIG_MODE=auto | config | legacy_env | none
```

Recommended defaults:

- `auto`: use the new config-based path when it is verified; fall back to
  `legacy_env` only for older Kilo versions that prove they resolve the legacy
  variables.
- `config`: force the new config-based path and fail preflight if it cannot be
  validated.
- `legacy_env`: preserve the old behavior for deployments pinned to old Kilo.
- `none`: do not inject provider settings; rely on Kilo's persisted auth/config.

The config-based path should generate isolated Kilo config for the subprocess
instead of mutating the user's global Kilo config. Use environment references for
secrets so API keys are not written to disk:

```json
{
  "$schema": "https://app.kilo.ai/config.json",
  "provider": {
    "openai": {
      "options": {
        "apiKey": "{env:LORELEY_KILO_OPENAI_API_KEY}",
        "baseURL": "{env:LORELEY_KILO_OPENAI_BASE_URL}"
      }
    }
  },
  "model": "openai/gpt-5.4"
}
```

Implementation should first verify the supported injection mechanism against the
real CLI. Candidate mechanisms, in order:

1. `KILO_CONFIG_CONTENT` if `kilo debug config` confirms it is loaded.
2. `KILO_CONFIG_FILES` or `KILO_CONFIG` if current Kilo accepts an explicit file.
3. A temporary config directory via `KILO_CONFIG_DIR`, with cleanup after the
   subprocess exits.

If no isolated config mechanism is supported, preflight should report that
worker-specific provider injection is unavailable for this Kilo version and
suggest either configuring Kilo itself or pinning `WORKER_KILOCODE_PROVIDER_CONFIG_MODE=none`.

### 4. Clarify Responses vs Chat Completions mapping

`WORKER_KILOCODE_OPENAI_API_SPEC` currently maps:

- `responses` -> `openai-responses`
- `chat_completions` -> `openai`

The new adapter should not assume this historical provider ID remains valid.
It should validate the selected provider through `kilo debug config` or an
equivalent dry-run config check. If Kilo no longer accepts `openai-responses`,
preflight should fail for `responses` mode with a targeted message instead of
silently launching an incorrectly configured subprocess.

### 5. Resolve usage DB path through Kilo

Change `_resolved_usage_db_path()` to prefer:

1. `WORKER_KILOCODE_USAGE_DB_PATH`
2. `kilo db path`
3. current fallback `~/.local/share/kilo/kilo.db`

Before querying, probe the schema for required tables and columns:

- `session.id`
- `session.title`
- `session.directory`
- `session.time_created`
- `session.time_updated`
- `message.id`
- `message.session_id`
- `message.time_created`
- `message.data`

If the schema is not compatible, return `unavailable_usage_event(...)` with a
reason that includes the missing table or column. Do not fail the agent
invocation after Kilo itself completed successfully.

### 6. Strengthen preflight

Extend `loreley doctor` / preflight checks for Kilo backends:

- Show Kilo binary path and parsed version.
- Check required `kilo run` flags.
- Check provider injection mode when worker-specific provider settings are set.
- Check `kilo db path` and schema when usage tracking is enabled.
- Warn when dynamic API key TTL is shorter than the planning/coding timeout, as
  current preflight already does.

Preflight output should make these states distinct:

- Kilo CLI missing.
- Kilo CLI present but command API incompatible.
- Kilo command API compatible but provider injection unverified.
- Kilo command/provider compatible but usage DB unavailable.

## Migration Plan

1. Add capability discovery and tests without changing runtime behavior.
2. Add `WORKER_KILOCODE_PROVIDER_CONFIG_MODE`, defaulting to `auto`.
3. Implement and validate the config-based provider adapter.
4. Update `KilocodeCliBackend._build_env()` to use the adapter.
5. Change usage DB path resolution to prefer `kilo db path`.
6. Update `env.example`, `docs/loreley/config.md`, release notes, and ADR 0034
   with the compatibility refresh.
7. Mark historical direct `KILO_OPENAI_*` injection as legacy in docs.

## Testing Plan

Unit tests:

- Command construction keeps the existing argument order and prompt handling.
- Capability parser recognizes the current `kilo run --help` output.
- Missing required flags produce preflight failures.
- Provider adapter does not leak API keys into logged command strings.
- `auto`, `config`, `legacy_env`, and `none` provider modes select the expected
  env/config behavior.
- Dynamic API keys are resolved immediately before launch.
- Explicit worker-specific API key still wins over inherited process env.
- Usage DB resolution prefers explicit path, then `kilo db path`, then fallback.
- Missing DB tables/columns produce unavailable usage events, not backend errors.

Optional real CLI contract tests:

```text
LORELEY_TEST_KILO_REAL_CLI=1 uv run pytest tests/core/worker/test_kilo_cli_contract.py
```

These tests should run `kilo --version`, `kilo run --help`, `kilo db path`, and
the chosen config injection probe against the installed CLI. CI can run them in
a scheduled job with `npx @kilocode/cli@latest` rather than every PR.

Regression tests:

- Existing Kilo backend tests continue passing.
- Planning/coding retry behavior still wraps Kilo failures in the right
  `PlanningError` / `CodingError`.
- Usage parser continues to extract `providerID`, `modelID`, `tokens`, cache
  tokens, reasoning tokens, and provider-reported cost from Kilo message rows.

## Rollout

Phase 1 should be warning-only for provider injection uncertainty. That gives
operators a clear signal without breaking deployments that rely on persisted
Kilo config.

Phase 2 should enable config-based provider injection by default in `auto` mode
once the probe is verified against current Kilo.

Phase 3 can deprecate `legacy_env` in docs, but should not remove it until at
least one release after the compatibility refresh. Some installations may still
pin older Kilo builds.

## Acceptance Criteria

- A fresh Loreley worker can launch `@kilocode/cli@latest` through
  `KilocodeCliBackend` using `WORKER_KILOCODE_*` settings.
- Worker-specific OpenAI-compatible gateway settings are applied to the Kilo
  subprocess without relying on `KILO_OPENAI_*`.
- Dynamic one-shot API keys still resolve per subprocess launch.
- Preflight gives actionable failures or warnings for command, provider, and
  usage DB compatibility.
- Existing Kilo usage tracking continues to work with the current SQLite schema.
- Existing users with Kilo already configured locally can choose
  `WORKER_KILOCODE_PROVIDER_CONFIG_MODE=none` and keep the old operational model.
