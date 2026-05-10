# Agent REST API

The agent REST facade exposes a narrow control surface for automation clients.
It lives next to the UI API under `/api/v1/agent` and wraps only existing
operator writes.

It does not add MCP, event streams, scheduler pause/resume, program activation,
PR bundles, `AGENTS.md`, or CLI commands.

## Authentication

Set `LORELEY_AGENT_API_TOKEN` to require bearer-token auth for all
`/api/v1/agent/*` routes:

```bash
LORELEY_AGENT_API_TOKEN=replace-me
```

Requests must then include:

```http
Authorization: Bearer replace-me
```

If `LORELEY_AGENT_API_TOKEN` is unset, agent routes stay open for local
development. Existing UI API routes keep their current auth behavior.

Agent-route errors use this JSON shape:

```json
{
  "error_code": "precondition_failed",
  "message": "Expected state mismatch for status.",
  "retryable": false,
  "resource": {"type": "job", "id": "00000000-0000-0000-0000-000000000000"},
  "suggested_next_actions": []
}
```

## Capability Discovery

Use capabilities before issuing writes:

```bash
curl http://127.0.0.1:8000/api/v1/agent/capabilities
```

The response includes the facade schema version, database schema version, auth
state, read resources, action whitelist, risk labels, dry-run support, required
parameters, and expected-state fields.

## Status And Next Actions

`GET /api/v1/agent/status` wraps `operator_status()` and adds:

- `health`: `healthy`, `actionable`, or `blocked`.
- `blocking_issues`: currently used for active campaign program mismatch.
- `safe_next_actions`: the same recommendations returned by
  `GET /api/v1/agent/next-actions`.

Triage is intentionally small:

- failed-stale jobs recommend `retry_failed_stale_jobs`;
- missing or failed baseline for the active campaign recommends
  `baseline_ensure`;
- eligible repair candidates with repair capacity recommend
  `repair_schedule_one`;
- active campaign program mismatch is blocking and does not recommend a write.

## Actions

All writes go through `POST /api/v1/agent/actions`.

Supported `action_type` values:

- `retry_job`
- `retry_failed_stale_jobs`
- `baseline_ensure`
- `repair_schedule_one`
- `repair_candidate_quarantine`
- `repair_candidate_discard`
- `repair_candidate_restore`

Example dry-run:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/agent/actions \
  -H 'Content-Type: application/json' \
  -d '{
    "action_type": "retry_job",
    "dry_run": true,
    "reason": "job exhausted stale lease recovery",
    "params": {"job_id": "00000000-0000-0000-0000-000000000000"},
    "expected_state": {"status": "failed", "recovery_count": 4}
  }'
```

Example execute with idempotency:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/agent/actions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer replace-me' \
  -d '{
    "action_type": "repair_schedule_one",
    "dry_run": false,
    "idempotency_key": "triage-2026-05-09T12:00Z-repair-1",
    "reason": "eligible repair candidate and capacity available",
    "params": {},
    "expected_state": {}
  }'
```

If `idempotency_key` is non-empty, repeat requests with the same
`action_type + idempotency_key` return the existing audit record and do not call
the underlying write service again.

`dry_run=true` validates parameters and focused current-state checks without
calling the write service. Execute requests call the existing service after
validation.

Focused v1 expected-state fields:

- Job retry: `status`, `lease_state`, `recovery_count`
- Repair candidate actions: `lifecycle_status`, `repair_state`,
  `active_repair_job_id`
- Baseline ensure: `campaign_program_hash`, `baseline_status`

## Action Audit Records

Every accepted action request writes an `agent_actions` audit record. Fetch it
with:

```bash
curl http://127.0.0.1:8000/api/v1/agent/actions/<action_id>
```

Records include the action id, status, dry-run flag, risk, preconditions,
result or structured error, and timestamps.

## Feedback

Use feedback endpoints to retrieve agent-safe evaluation evidence:

```bash
curl http://127.0.0.1:8000/api/v1/agent/jobs/<job_id>/feedback
curl http://127.0.0.1:8000/api/v1/agent/commits/<commit_hash>/feedback
```

These endpoints reuse the existing evaluation artifact helpers and
`build_agent_feedback_payload()`. They expose only artifacts marked
`agent_visible`; `human_only` and `hidden` artifacts are not included.

## Database Migration Note

This facade adds the `agent_actions` table and bumps
`INSTANCE_SCHEMA_VERSION` to `12`. Existing schema-version-5 databases should be
migrated before running this version:

```bash
uv run loreley db migrate
```
