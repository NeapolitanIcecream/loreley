# ADR 0032: Simplify worker prompts; standardize on Markdown deliverables

Date: 2026-02-03

Context: Planning/coding agents already carry substantial reasoning complexity; strict machine-validated output formats increase failure modes and configuration surface area.

Decision: Keep planning and coding prompts concise and non-prescriptive. Standardize planning output as a Markdown plan document and coding output as a Markdown execution report, treated as plain text with best-effort summary extraction.

Consequences: Fewer configuration knobs, fewer formatting-related retries, and more readable artifacts; downstream consumers rely on summaries rather than strict schemas.
