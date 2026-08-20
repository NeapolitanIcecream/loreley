# Paper evidence

This directory contains the records used for the paper's numerical results.
The matched Zstandard study is the only controlled policy comparison. The
Python and earlier Zstandard files support capability cases with their own
selection qualifications; their effects are not pooled with the matched study.

## Matched Zstandard study

- `zstd_method_efficacy.json` is the compact analysis record used by the
  figures and presentation tables. It contains the three policy definitions,
  checkpoint summaries, primary contrasts, post-hoc secondary analyses,
  mechanism summaries, resources, and source digests.
- `zstd_formal_records.json` contains the full-precision public records behind
  the controlled result: 126 finalist groups, 439 validation evaluations, 126
  validation-to-holdout selections, 56 unique holdout evaluations, 21 final
  endpoints, the finalist-width sensitivity, and the public parent/inspiration
  graphs.
- `zstd_formal_treatment.json` records the frozen QD implementation treatment:
  task and ignore-text digests, agent routes and timeouts, prompt-context
  fields, warm-up and same-batch semantics, descriptor normalization and
  refits, archive settings, and inspiration-sampler fallback rules.
- `validate_zstd_method_efficacy.py` independently replays the complete
  validation winner rule for all 126 groups; checks the
  validation-to-holdout-to-endpoint mapping and checkpoint summaries; and
  recomputes primary, decompression, and combined-throughput effects from the
  full-precision endpoint records. It also checks the finalist-width
  sensitivity, public lineage and delayed-branch counts, resource totals, the
  included formal-record and treatment byte hashes, and consistency of
  digests recorded for private source artifacts.
- `build_zstd_formal_records.py` is the deterministic sanitizer used to export
  the public record from seven frozen internal files. It contains their exact
  expected SHA-256 values and fails before export if any input differs. Those
  private source files are not distributed, so the builder documents the
  export boundary rather than providing a standalone public rebuild.
- `../../tools/method_efficacy_experiment/zstd_target.py` releases the literal
  task and repository-state ignore text whose hashes are frozen in
  `zstd_formal_treatment.json`.

Run the public validation with:

```bash
python3 paper/evidence/validate_zstd_method_efficacy.py
```

The original finalist-width history is relevant only because it changes the
selection path. The paper gives one sentence in the main method and a numerical
sensitivity in the appendix. The
`zstd_top10_premeasurement_amendment.json` records the premeasurement ordering
conditions and is retained with the public evidence.

## Earlier capability cases

- `python_uncertainty.json` contains the retained `markdown-it-py` validation
  interval and the fixed-candidate `python-pathspec` replication.
- `python_qd_audit.json` contains the final archive counts for the two Python
  campaigns. These are descriptive archive summaries, not policy comparisons.
- `python_generation_cost_audit.json` reconstructs request-level proxy costs
  for the two Python campaigns; `generate_python_generation_cost_audit.py`
  regenerates that record.
- `campaign_roots.json` distinguishes upstream revisions from experiment roots.
- `zstd_candidate_split_records.json` contains every split reported for the
  earlier Zstandard candidate `fe39bee8` and records when each split was used.
- `zstd_registered_thresholds.json` records the selection gates used by that
  earlier campaign.
- `zstd_qd_audit.json` contains the earlier Zstandard archive and descriptor
  summaries used in the discussion.

## Evidence boundary

The public formal record is sufficient to replay finalist selection, endpoint
mapping, reported statistics, and the public lineage counts. It does not
contain candidate source, prompts, private filesystem paths, hidden corpora, or
provider state. Several archive diagnostics in the paper were computed from
seven retained private database dumps; they are explicitly reported as
descriptive aggregates and cannot be reconstructed from the public record.

The two DeepSeek dollar values in the capability cases are public-price proxy
estimates, not provider bills. All three capability cases retain the selection
and measurement qualifications stated in the paper.
