# ADR 0015: Experiment-attached workers and per-experiment task queues

Date: 2026-01-13

Context: Loreley assumes env-only settings are stable for the lifetime of a database. Sharing one worker process across multiple experiments would require per-job config switching or mutable runtime state, which is unnecessary complexity for long-running experiments.

Decision: Require evolution worker processes to attach to a single experiment UUID at startup (`WORKER_EXPERIMENT_ID`). The worker loads env settings once and builds settings-dependent runtime objects once; they MUST NOT change for the process lifetime.

Decision: Route evolution jobs via per-experiment Dramatiq queues derived from the configured prefix (`TASKS_QUEUE_NAME`): `"{TASKS_QUEUE_NAME}.{experiment_id.hex}"`.

Consequences: Worker deployments scale by starting workers per experiment queue; mismatched jobs are rejected; experiment behaviour is stable across restarts and environments.

