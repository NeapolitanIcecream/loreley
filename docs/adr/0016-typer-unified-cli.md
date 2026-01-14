# ADR 0016: Use Typer for the unified Loreley CLI

Date: 2026-01-13

Context: Loreley had multiple argparse-based CLIs (`loreley/cli.py` and an embedded parser in `loreley.scheduler.main`), which duplicated flags, help text, and exit-code behavior and increased maintenance cost.

Decision: Use Typer as the single CLI framework for `loreley` and implement all commands (`doctor`, `scheduler`, `worker`, `api`, `ui`) as Typer subcommands with a shared global `--log-level` option.

Decision: Remove CLI parsing from `loreley.scheduler.main`; keep `main(settings=..., once=..., auto_approve=...)` as a reusable run function and route module execution through the unified CLI.

Consequences: CLI help output is consistent across commands, global options are shared, and the scheduler runtime is reusable without embedded CLI parsing.

