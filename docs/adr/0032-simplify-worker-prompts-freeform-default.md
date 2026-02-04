# ADR 0032: Simplify worker prompts; default to free-form output

Date: 2026-02-03

Context: Loreley uses LLM prompts primarily to convey essential context, while evaluation handles verification and backends already optimize agent behavior.

Decision: Keep planning and coding prompts concise and non-prescriptive; default `WORKER_PLANNING_VALIDATION_MODE` and `WORKER_CODING_VALIDATION_MODE` to `"none"`; keep schema enforcement as an opt-in configuration.

Consequences: Prompt artifacts are shorter, free-form outputs are accepted by default, and structured JSON is only required when strict/lenient validation is explicitly configured.
