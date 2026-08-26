# Exporting the evolution timeline

Loreley can export the internal facts needed to reconstruct job critical paths
and archive membership history without parsing logs or polling mutable rows.

```bash
uv run loreley timeline export > timeline.jsonl
```

Write directly to a file with `--output PATH`. The first JSON Lines record is
schema metadata; the remaining records are deterministically ordered by aware
UTC timestamp and stable event identity.

The export normalizes:

- append-only scheduler, worker, ingestion, and archive events;
- evaluator-attempt start/finish timing;
- evaluator resource request/acquire/release timing; and
- immutable job parent and inspiration selection.

Prompts, model output, credentials, evaluator logs, hidden corpus contents, and
artifact payloads are excluded. Existing `LLMUsageEvent` rows remain the source
of token and cost accounting; the timeline does not duplicate them.

## Strict completeness

```bash
uv run loreley timeline export --strict > timeline.jsonl
```

Strict mode exits non-zero and emits machine-readable issues to stderr when it
finds a post-boundary terminal event without a `job.run.started` event for the
same `(job_id, run_token)`, a terminal job without a terminal event, a stage
finish without its start, an unclassified active stage on a terminal run,
negative/impossible timing, or archive movement/removal without prior observed
membership. Pre-run cancellation remains exempt from the run-start
requirement. A reclaimed or failed run converts its unmatched stage starts into
explicit interrupted events; a higher-ordinal ingestion start similarly
converts the preceding unmatched attempt into `ingestion.interrupted`.

Databases migrated from schema 22 begin with a documented history boundary and
one `archive.member.initial_state` observation per member present at migration.
Loreley does not invent pre-migration planning, coding, evaluator, ingestion,
or archive entry times.
