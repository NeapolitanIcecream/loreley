# ADR 0046: Agent-visible evaluation artifacts

Date: 2026-04-29

Status: Draft

## Context

Loreley's worker loop currently carries evaluator feedback forward through compact
hot-path fields: a commit evaluation summary and structured metrics. The
evaluator can also return logs and arbitrary `extra` data, and the worker writes
evaluation JSON/log artifacts to disk, but those artifacts are cold-path audit
evidence. They are exposed through the UI/API for humans, not loaded into future
planning or coding prompts.

This is a weak feedback loop for benchmark-driven campaigns. Metrics can tell an
agent that a candidate improved or regressed, but they usually do not explain
why. Evaluation artifacts such as flamegraphs, profiler summaries, benchmark
JSON, failure cases, stderr excerpts, and memory reports can point directly at
the next promising change.

## Decision

Introduce evaluator-declared diagnostic artifacts as a first-class evaluation
output, with explicit control over what is visible to future agents.

The product surface should present this as evaluation feedback or diagnostic
evidence, not as raw file plumbing:

- commit and job detail pages show metrics, diagnosis, evidence, and an agent
  feedback preview;
- archive/commit list surfaces show lightweight indicators such as "has
  diagnostic evidence" and the top agent-visible diagnosis;
- operators can distinguish `agent_visible`, `human_only`, and hidden/internal
  evidence.

The technical contract should preserve the current hot/cold-path split:

- evaluator plugins may declare artifact metadata with key, kind, MIME type,
  path or generated payload reference, size, hash, summary, visibility, and
  optional extracted diagnostics;
- raw artifacts remain on disk/object storage and are referenced from the
  database;
- planning/coding prompts receive bounded summaries or a manifest by default,
  not raw artifact contents;
- direct path/URL exposure to agents is opt-in and gated by visibility, MIME
  allowlists, size limits, and path validation.

## Initial user stories

- As a campaign owner, I can open a commit and see why the evaluator rated it
  highly or poorly, including benchmark diagnostics and profiler evidence.
- As an evaluator author, I can attach flamegraphs, benchmark reports, logs, and
  failure cases while choosing which evidence is safe for agent consumption.
- As an operator, I can preview exactly what the next planning/coding agent will
  receive from evaluation evidence.
- As a system owner, I can configure artifact feedback policy as summary-only,
  manifest-only, or path-enabled without changing evaluator code.

## Implementation direction

The landing should be staged:

1. Extend `EvaluationResult` with typed artifact metadata while keeping existing
   evaluator payloads compatible.
2. Add a general persisted artifact model instead of relying only on fixed
   `JobArtifacts` columns.
3. Store evaluator-declared artifacts through the worker artifact store and
   expose them through the jobs/commits API.
4. Add a bounded artifact-feedback projection to commit planning context and
   render it in planning/coding prompts.
5. Add UI sections for evidence lists and agent feedback preview.
6. Add tests covering evaluator coercion, persistence, API exposure, prompt
   budget handling, visibility policy, and backward compatibility.

## Consequences

Agents receive richer, more actionable feedback while hot-path database rows and
prompts remain bounded. Evaluators become responsible for producing useful
diagnostic summaries when raw evidence is too large or unsuitable for direct
agent use.

The feature adds schema and policy surface area. The first implementation must
avoid silently leaking arbitrary files to agents and must keep raw profiler/log
payloads out of prompts unless explicitly allowed.
