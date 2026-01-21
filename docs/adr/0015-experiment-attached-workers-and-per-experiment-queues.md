# ADR 0015: Experiment-attached workers and per-experiment task queues

Date: 2026-01-13

Status: Superseded by ADR 0025.

Decision

Loreley attaches long-running processes to an explicit `EXPERIMENT_ID` and routes jobs via a per-experiment queue derived from `EXPERIMENT_ID` (not separately configurable).
See ADR 0025 for the current attachment model and constraints.

