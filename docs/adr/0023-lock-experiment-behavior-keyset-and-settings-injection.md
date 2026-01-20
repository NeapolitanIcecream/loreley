# ADR 0023: Lock experiment behavior keyset and settings injection

Date: 2026-01-19

Status: Superseded by ADR 0024.

Decision

Loreley uses an env-only settings model and does not persist a settings snapshot
or inject `Settings` into agent backend factories. See ADR 0024 for the current
configuration model and backend constraints.

