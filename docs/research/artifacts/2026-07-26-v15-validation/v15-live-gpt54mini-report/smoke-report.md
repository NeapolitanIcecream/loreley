# Circle-Packing Report (smoke)

- Experiment ID: `circle-packing-codex-gpt54-smoke-4w`
- Generated at: `2026-07-25T17:43:52.308008+00:00`
- Jobs: total=4 succeeded=4 failed=0
- Seed success rate: 1.0
- Non-seed success rate: None
- Best sum_radii: 2.5 commit='12ce3d5032238c88c5b02e8dd783fb24c5c5f232'
- Best runtime_p50_ms: 0.002415967173874378
- Archive occupied cells: 4 retained_elites=4 best_primary_value=2.5

## Timing

| metric | count | mean | p50 | p90 |
| --- | ---: | ---: | ---: | ---: |
| job_total_seconds | 4 | 152.86497431993484 | 151.06133151054382 | 207.7492292404175 |
| planning_seconds | 4 | 11.949506353994366 | 11.491377041500527 | 13.99948072489351 |
| coding_seconds | 4 | 136.21876265626634 | 134.68891372950748 | 192.78335906249706 |
| evaluator_seconds | 4 | 0.179028750048019 | 0.1778435205342248 | 0.1830770125496201 |
| runtime_p50_ms | 4 | 0.030780996894463897 | 0.004687521141022444 | 0.07969558937475087 |
| planning_attempts | 4 | 1.0 | 1.0 | 1.0 |
| coding_attempts | 4 | 1.0 | 1.0 | 1.0 |

## Worker Throughput

| worker | total | succeeded | failed | best_sum_radii | mean_job_seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| pid-98414 | 2 | 2 | 0 | 2.5 | 108.7837940454483 |
| pid-98415 | 2 | 2 | 0 | 2.5 | 196.9461545944214 |

## References

| label | status | commit | sum_radii | density | time_p50_ms | deterministic | error |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| root | available | `6dab191` | 0.2499999999999999 | 0.007551905417283154 | 0.0025420449674129486 | True |  |
| historical_best | available | `62d15a3` | 2.0035698039793197 | 0.48569511038584745 | 77.13329198304564 | True |  |
| current_best | available | `12ce3d5032238c88c5b02e8dd783fb24c5c5f232` | 2.5000000000000013 | 0.7723081940074908 | 0.0025420449674129486 | True |  |

## Objective Trajectory

| idx | worker | seed | sum_radii | density | total_seconds | elapsed_minutes |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | pid-98414 | yes | 2.4389662673771957 | 0.7187669712876036 | 132.1841950416565 | 2.204 |
| 2 | pid-98415 | yes | 2.4389662673796346 | 0.718766971289041 | 169.93846797943115 | 2.833 |
| 3 | pid-98414 | yes | 2.5 | 0.7723081940074908 | 85.38339304924011 | 3.651 |
| 4 | pid-98415 | yes | 2.5 | 0.7592182246175334 | 223.95384120941162 | 6.578 |
