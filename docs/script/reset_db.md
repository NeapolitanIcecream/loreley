# Resetting the database schema

Loreley intentionally does not ship migrations. For prototype workflows, the fastest path is to drop all tables and recreate the schema from ORM models.

!!! warning
    This operation is destructive. It will irreversibly drop **all** tables in the configured database and clear the experiment-scoped Redis namespace used by Dramatiq.

## Usage

```bash
uv run loreley reset-db --yes
```

## Notes

- The command uses the configured database in `loreley.config.Settings` (`DATABASE_URL` or `DB_*` fields).
- After recreating the schema, Loreley also clears Redis keys in the current experiment namespace.
- This is intended for development and local testing environments.
