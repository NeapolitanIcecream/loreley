# ADR 0015: Experiment-attached workers and per-experiment task queues

Date: 2026-01-13

Context: Loreley locks experiment-scoped behaviour settings in `Experiment.config_snapshot`. Sharing one worker process across multiple experiments requires per-job config reloads or mutable runtime state, which is unnecessary complexity for reproducible long-running experiments.

Decision: Require evolution worker processes to attach to a single experiment UUID at startup (`WORKER_EXPERIMENT_ID`). The worker loads the experiment snapshot once and builds settings-dependent runtime objects once; they MUST NOT change for the process lifetime.

Decision: Route evolution jobs via per-experiment Dramatiq queues derived from the configured prefix (`TASKS_QUEUE_NAME`): `"{TASKS_QUEUE_NAME}.{experiment_id.hex}"`.

Consequences: Worker deployments scale by starting workers per experiment queue; mismatched jobs are rejected; experiment behaviour is stable across restarts and environments.

