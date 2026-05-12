# Database schema commands

Use `loreley db` commands to inspect and upgrade a Loreley PostgreSQL database
without dropping experiment data.

These commands use the configured database from `DATABASE_URL` or the `DB_*`
settings. Migration and validation also check the single-tenant
`InstanceMetadata` marker against the current `EXPERIMENT_ID` and
`MAPELITES_EXPERIMENT_ROOT_COMMIT`.

## Check schema status

```bash
uv run loreley db current
```

Example output:

```text
schema_version=5 target=12 state=migratable needs_migration=true
```

Machine-readable output:

```bash
uv run loreley db current --json
```

Schema states:

- `fresh`: no Loreley instance marker exists yet.
- `current`: the database is already on the current schema.
- `migratable`: a native migration path exists.
- `future`: the database was created by a newer Loreley binary.
- `unsupported`: no native migration path exists.
- `damaged`: the schema marker or required tables are inconsistent.

## Migrate

Take a Postgres backup before upgrading a database you care about, then run:

```bash
uv run loreley db migrate
```

This command initializes an empty database or applies the native migration chain
from schema version 5 (`v0.7.9-alpha`) to the current schema. It uses a Postgres
advisory lock so concurrent startup or CLI migration attempts do not apply the
same migration chain twice.

Example output:

```text
from=5 to=12 applied=6,7,8,9,10,11,12 fresh=false
```

`uv run loreley db migrate` always runs the explicit migration path, even when
`DB_AUTO_MIGRATE=false`.

## Validate

```bash
uv run loreley db validate
```

Validation fails when the schema is missing required tables, indexes,
constraints, or the configured experiment identity does not match the database
marker.

## Automatic startup migration

API, scheduler, and worker startup call the same schema helper before marker
validation.

- With `DB_AUTO_MIGRATE=true`, startup may initialize a fresh database or
  migrate supported older schemas.
- With `DB_AUTO_MIGRATE=false`, startup fails until `uv run loreley db migrate`
  has completed.

`uv run loreley reset-db --yes` is still available for disposable local
databases, but it drops data. Use native migration for normal upgrades.
