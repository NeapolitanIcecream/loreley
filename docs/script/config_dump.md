# Config dump

This command dumps the effective Loreley runtime configuration for troubleshooting or reproducibility.

## Usage

```bash
uv run loreley config dump
```

It prints the active configuration to standard output (JSON by default), including database, Redis, paths, and MAP-Elites settings. Secrets and credentials are masked by default.

## Options

- `--json`: Print effective settings as JSON (default).
- `--yaml`: Print effective settings as YAML.
- `--mask-secrets` / `--no-mask-secrets`: Enable or disable automatic redaction of credentials, URLs, and API keys. Masking is enabled by default.
