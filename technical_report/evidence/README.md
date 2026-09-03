# Evolution dynamics evidence

The two JSON files contain the sanitized timing data and derived summaries for
the seven-block Zstandard comparison:

- [zstd_quality_time.json](zstd_quality_time.json): 1,008 candidate-job timestamp
  records, 126 checkpoint makespans, and their joins to validation-selected
  holdout winners.
- [zstd_quality_time_summary.json](zstd_quality_time_summary.json): per-checkpoint
  results, paired search-time ratios, effective parallelism, and quality at each
  block's QD completion time.

The extraction and analysis code is
[timing_analysis.py](../../tools/method_efficacy_experiment/timing_analysis.py).
The winner identities and quality measurements are in
[zstd_formal_records.json](../../paper/evidence/zstd_formal_records.json).
Both JSON files preserve the exported experiment records without changing their
values.

Run the independent consistency check from the repository root:

    python3 technical_report/evidence/validate_dynamics.py

The validator recomputes durations and checkpoint makespans from timestamps,
rejoins the published holdout winners, and recomputes the summaries and
time-to-threshold figures. It does not rerun candidate generation or benchmarks.

Time starts at the first scheduled candidate and ends when all jobs through a
checkpoint have completed. Failed attempts consume time and job budget.
Validation and holdout evaluation time is excluded. The QD-completion deadline
is specific to each block; it is not a shared preassigned deadline. First-hit
counts include any checkpoint reaching +0.50%, whereas the endpoint table counts
only the winner at job 48.

The exports contain timestamps and categorical outcomes, not credentials,
service endpoints, agent prompts, or local filesystem paths.
