# Resetting the database schema

`reset-db` is a destructive local/disposable fallback. Normal upgrades should use the native migration command:

```bash
uv run loreley db migrate
```

For non-destructive schema inspection and upgrades, see
[Database schema commands](db.md).

!!! warning
    This operation is destructive. It will irreversibly drop **all** tables in the configured database and clear the experiment-scoped Redis namespace used by Dramatiq.

## Usage

```bash
uv run loreley reset-db --yes
```

## Notes

- The command uses the configured database in `loreley.config.Settings` (`DATABASE_URL` or `DB_*` fields).
- After recreating the schema, the database includes the native migration audit table and managed indexes expected by `uv run loreley db validate`.
- After recreating the schema, Loreley also clears Redis keys in the current experiment namespace.
- This is intended for development and local testing environments where preserving experiment data is unnecessary.
