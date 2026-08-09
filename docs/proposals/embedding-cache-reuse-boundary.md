# Proposal: Minimal safe cross-experiment embedding cache reuse

## Background

Loreley's repository-state embedding pipeline computes file-level embeddings
for eligible repository files and aggregates them into commit-level repository
state embeddings. Bootstrapping the root of a large repository can process many
files, incur substantial API cost, and delay campaign startup.

The current implementation has two cache layers:

- File-level cache: `map_elites_file_embedding_cache`, keyed by `blob_sha` and
  storing a file embedding vector.
- Commit-level aggregate: `map_elites_repo_state_aggregates`, keyed by
  `commit_hash` and storing `sum_vector` and `file_count`.

Both caches belong to one Loreley experiment database. Smoke, canary, and long
runs commonly use separate databases. Even when those runs target the same
repository, root commit, and embedding configuration, a later run cannot reuse
file embeddings computed by an earlier run.

## Problem

A smoke run may compute the complete root repository-state embedding while
validating the pipeline. A later long run with a new database then pays again
to embed identical file contents.

This does not require a separate embedding subsystem. It can be addressed as an
extension of the existing file-level cache: let an experiment database import
file embeddings explicitly from another compatible experiment database.

## Goals

The minimal safe version addresses one operation:

> Import repository-state file embeddings from one database into another when
> compatibility is established by machine-checkable metadata or explicit
> operator attestation.

The implementation must:

- Reuse `blob_sha -> vector` rows from `map_elites_file_embedding_cache`.
- Validate an embedding-semantics fingerprint before reuse.
- Preserve existing rows in the target database.
- Leave the repository-state embedding pipeline unchanged.
- Keep cache semantics outside `CodeEmbedder`.
- Report a specific reason when reuse is rejected.
- Support explicit attestation for legacy caches without trusting them
  automatically.

## Non-goals

The MVP does not:

- Infer relationships among smoke, canary, and long runs.
- Find an upstream database by phase name, run directory, or symlink.
- Import `map_elites_repo_state_aggregates`.
- Reuse commit-level or repository-level aggregates.
- Add a global cache database, cache service, read-through cache, background
  synchronization, TTL, or eviction.
- Change every embedding path in Loreley.
- Make `CodeEmbedder` query database caches.
- Let a legacy database without a manifest participate automatically.
- Diagnose a miss reason for each file.
- Add a UI, dashboard, or metrics subsystem.

## Design principles

### Reuse only the file-level cache

The MVP imports this mapping across databases:

```text
blob_sha -> file embedding vector
```

The target database still performs its own root bootstrap. It enumerates
eligible files and aggregates the file vectors into the root commit aggregate.

This avoids most repeated embedding API work while keeping eligibility rules,
ignore rules, file counts, and aggregate semantics under the target
experiment's control.

### Do not reuse commit aggregates

A commit aggregate depends on:

- The eligible file set for the root or candidate commit.
- Ignore rules.
- Preprocessing configuration.
- Maximum file size.
- File-level vectors.
- The aggregation algorithm.
- The interpretation of the repository root or subpath.

Importing only `commit_hash -> sum_vector + file_count` can also leave the
target database without the old blob vectors needed for incremental updates.
When a file changes, ingestion must subtract the parent blob's vector before
adding the new blob's vector.

The MVP therefore imports no commit aggregate. After importing file vectors,
the target database bootstraps its own aggregate and continues incremental
ingestion using its local file cache and newly generated aggregates.

### Keep cache decisions upstream of `CodeEmbedder`

The repository-state pipeline already performs these steps:

1. Enumerate eligible files in a commit.
2. Read each file's `blob_sha`.
3. Query `map_elites_file_embedding_cache`.
4. Preprocess, chunk, and embed only cache misses.
5. Write newly generated vectors to the file cache.
6. Aggregate cached and new vectors into the commit embedding.

Cross-database import changes only the hit rate at step 3. `CodeEmbedder`
continues to call the embedding API for supplied chunks and does not participate
in database cache lookup.

## Embedding-semantics fingerprint

A file cache cannot be declared compatible from `blob_sha`, model name, and
dimensions alone. Identical content can produce a different file vector under
different preprocessing or chunking settings.

The MVP introduces `embedding_semantics_fingerprint` for source and target
compatibility checks. It contains at least:

- Embedding model.
- Embedding dimensions.
- A non-sensitive embedding provider identity:
  - Provider kind or name.
  - Normalized base URL origin and path, without credentials, query parameters,
    or tokens.
  - Resolved model ID when the provider maps a requested alias to another model.
- Preprocessing configuration:
  - Allowed extensions.
  - Allowed filenames.
  - Excluded globs.
  - Maximum file size.
  - Comment stripping.
  - Block-comment stripping.
  - Maximum consecutive blank lines.
  - Tab width.
- Chunking configuration:
  - Target lines.
  - Minimum lines.
  - Overlap lines.
  - Maximum chunks per file.
  - Boundary keywords.
- Repository-state file embedding algorithm version.
- File aggregation algorithm version.

The fingerprint does not contain the experiment ID because reuse across
experiments is its purpose.

Compute it from canonical JSON, for example:

```json
{
  "kind": "repo_state_file_embedding",
  "schema_version": 1,
  "embedding_provider": {
    "provider": "openai",
    "base_url": "https://api.openai.com/v1",
    "requested_model": "text-embedding-3-small",
    "resolved_model": "text-embedding-3-small"
  },
  "embedding_dimensions": 1536,
  "preprocess": {
    "allowed_extensions": [".py", ".ts"],
    "allowed_filenames": ["Dockerfile", "Makefile"],
    "excluded_globs": ["tests/**", "node_modules/**"],
    "max_file_size_kb": 512,
    "strip_comments": true,
    "strip_block_comments": true,
    "max_blank_lines": 2,
    "tab_width": 4
  },
  "chunk": {
    "target_lines": 80,
    "min_lines": 20,
    "overlap_lines": 8,
    "max_chunks_per_file": 64,
    "boundary_keywords": ["def ", "class ", "function "]
  },
  "algorithm": {
    "repo_state_file_embedding": "v1",
    "file_chunk_aggregation": "weighted_average_v1"
  }
}
```

The exact fields should match the implementation. Every setting or algorithm
that can change a file vector must be included.

Eligibility settings such as `allowed_extensions`, `allowed_filenames`, and
`excluded_globs` mainly determine whether a blob participates in the repository
state and do not necessarily change the vector of an individual blob. The MVP
still includes them to keep the compatibility boundary conservative. This may
reject some reuse that would be safe under a more detailed model.

## Data model

Add a small manifest table:

```text
embedding_cache_manifests
- id uuid primary key
- cache_kind varchar(64) not null unique
- fingerprint varchar(64) not null
- payload jsonb not null
- source varchar(64) not null
  -- generated | operator_attested
- created_at timestamptz not null
- updated_at timestamptz not null
```

The MVP supports one active manifest:

```text
cache_kind = 'repo_state_file_embedding'
```

`cache_kind` is unique. The MVP does not add an `active` flag. If manifest
history is needed later, add a separate history table or explicit active-state
semantics so startup and import never choose ambiguously among multiple rows.

When a fresh database or repository bootstrap has no manifest, fail closed:

- If `map_elites_file_embedding_cache` is empty, generate a manifest from the
  current settings and store it with `source='generated'`.
- If the file cache is non-empty, do not generate a manifest automatically.
  Fail and require operator attestation.

If a manifest exists but its fingerprint differs from the current settings,
startup or import fails and reports that the database uses incompatible
embedding semantics.

## Import command

Add an explicit command:

```bash
uv run loreley embedding-cache import \
  --source-dsn "$SMOKE_DATABASE_URL"
```

The current `DATABASE_URL` identifies the target database.

The command:

1. Connects to the source and target databases.
2. Reads the source manifest.
3. Reads the target manifest or generates it under the fail-closed rule.
4. Compares `cache_kind` and `fingerprint`.
5. Rejects a mismatch.
6. Reads source rows from `map_elites_file_embedding_cache`.
7. Validates each source row.
8. Inserts rows into the target with `ON CONFLICT DO NOTHING`.
9. Prints an import summary.

Row-level validation checks at least:

- `embedding_model` matches the expected stored model in the manifest payload,
  normally `embedding_provider.requested_model`. A legacy manifest may use a
  top-level `embedding_model` compatibility field.
- `dimensions` matches `embedding_dimensions` in the manifest payload.
- `vector` is non-empty.
- Vector length equals `dimensions`.

If any source row is invalid, the import fails before writing that batch. A
matching manifest alone is not sufficient to run `INSERT ... DO NOTHING` over
unchecked rows.

Example output:

```text
Embedding cache import complete
source_rows=125842
inserted_rows=118902
already_present_rows=6940
skipped_rows=0
fingerprint=4d3b...
source_manifest=generated
target_manifest=generated
```

## Legacy cache adoption

A legacy database may contain only:

- `blob_sha`
- `embedding_model`
- `dimensions`
- `vector`

These fields cannot prove compatibility because they omit preprocessing,
chunking, and algorithm versions.

The default behavior is:

- Reject an import when the source database has no manifest.
- Generate a target manifest from current settings only when the target cache
  is empty.
- Reject a target database that has no manifest but has cached rows.
- Require operator attestation before using a legacy source database.

### Attestation command

Provide an explicit command:

```bash
uv run loreley embedding-cache attest \
  --database-url "$OLD_DATABASE_URL" \
  --from-current-settings
```

The operator is declaring that the legacy file cache matches the embedding
semantics computed from the current settings.

The command:

1. Checks whether a manifest already exists.
2. Confirms that the cache contains only one model and dimensions combination.
3. Computes a fingerprint from the current settings.
4. Writes a manifest with `source='operator_attested'`.
5. Prints a warning that compatibility was declared by an operator rather than
   established automatically.

An explicit fingerprint form may also be supported:

```bash
uv run loreley embedding-cache attest \
  --database-url "$OLD_DATABASE_URL" \
  --fingerprint "$EXPECTED_FINGERPRINT"
```

The MVP does not derive a fingerprint from an old Loreley version number.

## Logging and observability

The MVP requires clear CLI and startup logs but no new UI or metrics service.

Import output includes:

- Sanitized source database identity, without usernames, passwords, tokens, or
  secret query parameters.
- Sanitized target database identity under the same rule.
- Source manifest origin.
- Target manifest origin.
- Fingerprint.
- Source row count.
- Inserted row count.
- Already-present row count.
- Skipped row count.
- The specific rejection reason on failure.

Add a root bootstrap cache summary:

```text
Repo-state root bootstrap cache summary commit=<sha> eligible_files=<n> unique_blobs=<n> hits=<n> misses=<n> embedded=<n> fingerprint=<hash>
```

After import, `misses` in the target database should be substantially lower
than without import and should approach zero for the same root and eligibility
scope. Continued high embedding traffic indicates an import or fingerprint
miss, or work on an embedding path outside this cache.

## Recovery from empty upstream responses

Root bootstrap on a large repository can issue many embedding batches. Some
OpenAI-compatible providers occasionally return HTTP 200 while the parsed SDK
response contains no embedding data. Handle this case with bounded retries at
the embedding-call boundary so one transient empty response does not terminate
scheduler startup.

The MVP:

- Recognizes only explicit empty-data cases:
  - The SDK raises `ValueError("No embedding data received")`.
  - Response `data` is missing or empty.
  - An individual response item has a missing or empty `embedding`.
- Converts these cases to a local retryable exception.
- Reuses `MAPELITES_CODE_EMBEDDING_MAX_RETRIES` and
  `MAPELITES_CODE_EMBEDDING_RETRY_BACKOFF_SECONDS`.
- Does not retry an unrelated `ValueError`.
- Records usage only after response validation; an invalid response produces no
  usage event.
- Preserves a clear terminal error such as
  `Embedding API returned no embedding data.` after retries are exhausted.

This is not a repository-state bootstrap checkpoint. The MVP does not change
transaction boundaries or commit successful batches incrementally. A provider
that keeps returning empty responses still causes bootstrap to fail. Incremental
cache checkpoints require a separate design.

## Correctness constraints

### Insert only

Import never overwrites a target cache row. Existing target rows take priority.
When the fingerprints match and a `blob_sha` already exists, ignore the source
row.

### Fail closed on fingerprint mismatch

Reject a mismatched fingerprint. Do not continue after a warning.

### Fail closed without a manifest

Reject a source without a manifest unless the operator first attests it.

Reject a target with cached rows but no manifest unless the operator first
attests it. Do not generate a manifest that assigns unknown legacy rows to the
current semantics.

### Fail closed on invalid source rows

Validate source model, dimensions, and vector shape before import. Any invalid
row fails the import so damaged or incompatible vectors do not enter the target
cache.

### Preserve runtime embedding semantics

Import only pre-populates the target file cache. Repository bootstrap and
incremental ingestion continue to determine hits and misses, embed missing
files, and aggregate commit vectors through the existing path.

## Incremental-update safety

This design preserves incremental updates because it does not import a commit
aggregate. The target database uses imported file vectors to build its own root
aggregate. Candidate ingestion then has a complete parent aggregate and can
read both old and new blob vectors from the target file cache.

When a new blob is absent from the imported cache, the existing path treats it
as a miss, calls `CodeEmbedder`, and stores the resulting vector locally.

File-cache import therefore reduces API calls without skipping any data needed
for incremental repository-state updates.

## Test plan

Cover these cases:

- Generate a target manifest before import when the target has no manifest and
  an empty cache.
- Reject a target with no manifest and a non-empty cache.
- Import source rows absent from the target when fingerprints match.
- Preserve existing target rows.
- Reject a source and target fingerprint mismatch.
- Reject a source database without a manifest.
- Reject a source row with an invalid model, dimensions, or vector length.
- Write an `operator_attested` manifest for a legacy source database.
- Reject attestation when a legacy cache mixes model or dimensions values.
- Enforce manifest uniqueness on `cache_kind`.
- Confirm that root bootstrap hits imported file vectors and reduces
  `cache_misses`.
- Confirm that `map_elites_repo_state_aggregates` is not imported.
- Retry a batch when the embedding SDK first returns
  `No embedding data received` and the next call succeeds.
- Exhaust the configured retries when response data remains absent and record
  no usage event for invalid responses.
- Do not retry an unrelated `ValueError`.

## Recommended implementation order

1. Add fingerprint computation and unit tests.
2. Add the `embedding_cache_manifests` table and migration.
3. Ensure that the target manifest exists and matches current settings before
   database initialization or repository-state bootstrap proceeds.
4. Add `embedding-cache attest`.
5. Add `embedding-cache import`.
6. Add the root bootstrap cache summary log.
7. Add cross-database import integration tests.

## Conclusion

This design extends the existing repository-state file cache. It does not add a
separate embedding subsystem.

The MVP performs explicit, auditable, fail-closed file-cache imports across
databases. It avoids repeated API calls without importing commit aggregates,
changing `CodeEmbedder`, or trusting a legacy cache automatically.
