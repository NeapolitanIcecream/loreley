# ADR 0026: Experiment ID as operational naming namespace

Date: 2026-01-21

Status: Accepted

## Context

Loreley runs multiple long-lived experiments concurrently on shared hosts. When Redis, filesystem paths, and remote branch namespaces are shared, configurable naming knobs introduce drift and cross-experiment interference.

## Decision

- `EXPERIMENT_ID` is the single source for experiment attachment and naming isolation.
- `EXPERIMENT_ID` may be a UUID or a short slug; slugs are mapped to stable UUIDs via uuid5.
- Redis broker namespace, Dramatiq queue name, worker job branch prefix, and logs/artifacts directories are derived from `EXPERIMENT_ID` and are not separately configurable.

## Consequences

- Operators run one environment file per experiment (DB + Redis URL + `EXPERIMENT_ID`).
- Redis broker namespace, Dramatiq queue name, and worker job branch prefix are not configurable; they are always derived from `EXPERIMENT_ID`.

