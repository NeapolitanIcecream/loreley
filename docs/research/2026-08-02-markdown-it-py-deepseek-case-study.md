# Case study: Loreley found a 6.75% `markdown-it-py` speedup

Date: 2026-08-02
Scope: one preregistered campaign on `markdown-it-py` commit
`97aff4f564e02e24f8526d9e2cd7899c47f714a6`

## Result in one minute

Loreley combined eight small, human-written optimization seeds with 56
Kilo/DeepSeek evolution jobs. The selected candidate rendered a separate
28-document corpus **1.0675x as fast as the fixed root** while using slightly
less peak allocated memory. It passed correctness, packaging, CLI, public API,
and edit-scope checks. Every one of the 28 validation documents improved.

| Measure | Result |
|---|---:|
| Jobs | 64 total: 8 seeds + 56 evolution |
| Successful / failed | 54 / 10 |
| Best manual seed on training | 1.0323x |
| Winner on training | 1.0699x |
| Winner on separate validation corpus | **1.0675x** |
| Validation documents improved | 28 / 28 |
| Per-document speedup | 1.0071x min, 1.0675x median, 1.1715x max |
| Peak allocation | 3.4890 MiB winner vs. 3.4924 MiB root |
| Campaign wall time | 4.35 hours |
| Generation usage | 215.35M tokens, 3,792 requests |
| Recorded DeepSeek cost | $2.08 |

The winner was job 26, commit
`b10adb6fad0da2a9825c3d1525048fd7b177d773`. The training-to-validation gap
was only 0.24 percentage points. Four feasible candidates exceeded 1.06x on
training, so the outcome was not supported by one isolated training sample.

This is strong evidence for the fixed case study. It is not an estimate of
Loreley's average effect across repositories.

## What was tested

The campaign used the real Loreley scheduler, database, MAP-Elites archive,
sampler, worker, Kilo backend, and evaluator interface. Its fixed controls were:

- Kilo 7.4.16 with `deepseek-v4-flash` over Chat Completions for planning and
  coding;
- OpenAI `text-embedding-3-small` embeddings with 1,536 dimensions for archive
  diversity; no hash embedding participated;
- two islands, an algorithm limit of four unfinished jobs, four physical model
  workers, scheduler batches of four, and one serialized evaluator lane;
- one planning attempt, one coding attempt, no rework, and a 128-request guard
  per evolution job;
- a frozen training evaluator and a separate 28-document validation corpus that
  was used only after the winner was selected.

We did not spend another large arm on a baseline agent. The campaign reused a
previously frozen and measured root as its control. The 64-job endpoint was
chosen because the observed throughput projected a 128-job run beyond the
same-day completion deadline.

## How the winner evolved

The eight seeds covered different hot paths: HTML escaping, source
normalization, renderer attributes, token attribute conversion, inline
terminator caching, token-list iteration, inline HTML matching, and entity
matching. All eight were correct and faster than root on training; their gains
ranged from 0.15% to 3.23%.

Loreley records the winner as generation 4. Its primary ancestry is easy to
interpret:

| Generation | Job | Contribution |
|---:|---:|---|
| 1 | 7 | Avoid allocating a source suffix for inline HTML matching. |
| 2 | 12 | Add renderer and token-attribute fast paths. |
| 3 | 14 | Add no-op HTML escaping and faster renderer dispatch. |
| 4 | 26 | Add the normalization fast path and become the winner. |

Inspiration edges also brought in ideas derived from other seeds, including a
normalization branch explored on the other island. The final candidate changed
five files with 54 additions and 14 deletions. It was substantially stronger
than the best seed, which is the useful signal here: the search recombined
compatible ideas instead of merely selecting the initial winner.

## Validation

After model calls closed, the frozen winner and root were measured in an
interleaved, CPU-pinned container run. The geometric-mean throughput ratio was
1.067538. The candidate improved all 28 documents and reduced peak allocation
by 0.10%.

The validation also passed:

- output and semantic equality checks;
- the full upstream test suite;
- wheel build and installation;
- installed import and CLI smoke tests;
- package metadata, entry points, exported symbols, and module-surface
  comparisons;
- an edit-scope gate that rejected benchmark specialization and protected-file
  changes.

## Cost and throughput

The campaign completed 16 four-job waves at 14.7 terminal jobs per wall-clock
hour. Median job duration was 10.0 minutes. With four workers, the effective
wall-clock cost was about 4.1 minutes per terminal job. Host free memory stayed
between 42% and 49% before dispatch, so memory was not the limiting resource;
the single evaluator lane was retained to protect timing measurements.

DeepSeek returned HTTP 200 for all 3,792 recorded generation requests. Loreley
recorded 215,349,501 generation tokens and $2.0833 in provider cost. The archive
also used 199,343 embedding tokens across 66 external embedding events.

Ten jobs failed and remained part of the result: four reached the request
guard, four produced no effective change, one exhausted coding without a final
report, and one failed the release contract. These failures point to agent
efficiency and early-stopping opportunities, not provider instability.

## What the experiment changed in Loreley

Running the campaign exposed reusable integration defects. The accompanying
project changes:

- isolate Kilo state and usage databases by job and run token;
- account for descendant Kilo sessions and reasoning output tokens;
- keep rework invocations distinct while preserving their accumulated usage;
- propagate bounded token/request failure reason codes;
- support Kilo's `--pure` mode with preflight capability checks;
- fetch a requested candidate commit explicitly when a narrow clone refspec
  does not advertise it.

The purpose-built experiment harness supplied deterministic seed injection,
the benchmark evaluator, provider accounting, evidence freezing, and final
report assembly. It did not replace Loreley's search loop. It did use direct
database setup for the reused root baseline and deterministic seed ordering;
those controls are not presented as general product features. The one-off
harness and raw machine artifacts are intentionally not part of this change.

## Claim boundary

The defensible promotional statement is:

> In a preregistered 64-job `markdown-it-py` case study, Loreley combined eight
> small human-written seed ideas with Kilo/DeepSeek search to find a candidate
> that was 6.75% faster on a separate 28-document corpus, with no correctness,
> release, scope, or allocation regression.

This result has four important limits: it covers one repository and host, the
search began with human guidance, it reused an earlier root measurement instead
of running a new baseline arm, and the validation corpus was measured once
after selection. Replication on other repositories is required before making a
general effectiveness claim.
