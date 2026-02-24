# ADR 0045: Config profiles for large-repo campaigns

Date: 2026-02-24

Context: Loreley exposes many low-level knobs (MAP-Elites, timeouts, filters). In large repositories, safe defaults differ materially from small demos, but asking operators to tune dozens of variables is error-prone.

Decision: Introduce `LORELEY_PROFILE` as an optional preset selector. Profiles apply scaling-oriented defaults only when a field is not explicitly set via environment variables.

Consequences: The default behaviour remains unchanged (`LORELEY_PROFILE=default`). Operators can use `"large-repo-1m-30k"` to get a sensible baseline for ~1M LOC and ~30k iterations, while still overriding individual values as needed.

