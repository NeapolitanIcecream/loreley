# loreley.core.worker.commit_summary

Deterministic commit-message selection for evolution jobs.

## `build_commit_message`

`build_commit_message(job_id, plan, coding)` chooses the first safe, non-empty value from:

1. `coding.report.summary`;
2. `plan.summary`; and
3. `Evolution job <job_id>`.

Whitespace is normalized, but the git message is not truncated and no model is called. Persistence still projects the message into the bounded 72-character `CommitCard.subject`; the separate `change_summary` keeps the complete bounded worker summary, up to 800 characters.
