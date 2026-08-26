# Unreleased

These notes cover changes merged after `v0.10.0-alpha`.

## Research release

- Publish the first Loreley preprint, *Loreley: Repository-Scale Program
  Evolution with Quality-Diversity Search*, as
  [arXiv:2608.19703](https://arxiv.org/abs/2608.19703). The paper reports a
  1,008-job matched Zstandard comparison of Loreley QD, Sequential Champion,
  and Independent Root search, alongside three earlier capability campaigns.
- Check in the paper source, bibliography, deterministic figure generator,
  formal records, evidence validators, and an arXiv v1 source manifest. The
  matched experiment observed archive retention and later reuse but did not
  establish an endpoint advantage for QD at the tested 48-job horizon.
- Update the public case-study package with the fixed Top-10 Zstandard holdout
  comparison, candidate patches, selection boundaries, and data-derived launch
  graphics. Rewrite the README around repository-scale Quality-Diversity
  search and the available evidence.

## Runtime and operator safety

- Add OpenRouter embedding-provider controls for provider selection,
  fallbacks, parameter support, and data-collection policy. Preserve
  provider-reported embedding cost when it is available.
- Preserve native DeepSeek routing in Kilo, isolate per-job Kilo state, disable
  unsupported headless delegation, and redact exact provider values from CLI
  output and persisted SQLite state.
- Mask credential-bearing gateway URLs in safe configuration exports and make
  the worker contract explicit that campaigns use only preinstalled tools.
- Align planning and coding context with the ordered archive objective contract,
  make explicit non-seed semantics authoritative, and reject campaign/archive
  primary-metric conflicts during preflight.
- Add an opt-in, persisted GPT-5.6 Sol seed portfolio per campaign. Bound the
  campaign slate independently of warmup/island count, cap unsuccessful
  direction attempts, reuse successful directions fairly, preserve provenance
  through evaluation/ingestion/lineage, and label evaluator-valid seeds as
  immediate or exploratory before ordinary Pareto/QD admission without another
  full evaluation.

## Compatibility

- Database schema version 23 adds the append-only `evolution_events` ledger and
  records the current archive as an explicit migration-boundary observation.
  Version 22 added `seed_portfolios`, `seed_directions`, and seed provenance;
  existing version-21 or version-22 databases migrate with
  `uv run loreley db migrate`.
- Add deterministic, sanitized `loreley timeline export` JSON Lines output over
  evolution events, evaluator attempts, and evaluator resource leases, with an
  optional strict completeness check.
- GitHub CI builds the source distribution and wheel, runs the test suite, and
  installs the wheel on Python 3.11, 3.12, and 3.13. Cremona must report no new
  structural regression before the next tag is created.
