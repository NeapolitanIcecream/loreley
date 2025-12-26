## Resetting the database schema

Loreley intentionally does not ship migrations. For prototype workflows, the fastest path is to drop all tables and recreate the schema from ORM models.

!!! warning
    This operation is destructive. It will irreversibly drop **all** tables in the configured database.

## Usage

```bash
uv run python script/reset_db.py --yes
```

## Notes

- The script uses the configured database in `loreley.config.Settings` (`DATABASE_URL` or `DB_*` fields).
- This is intended for development and local testing environments.


