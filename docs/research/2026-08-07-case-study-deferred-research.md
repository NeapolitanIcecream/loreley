# Deferred Research After the First Three Case Studies

Date: 2026-08-07

Status: research register. The items below do not enter the current framework
convergence proposal because the evidence does not yet select a general
solution. Each item requires a design decision or another experiment before
implementation.

## 1. Noisy objectives and archive correctness

### Observation

In Zstandard V19, the training Top-10 compression lower bounds spanned 0.276
percentage points, while their median point-to-lower-bound distance was 0.541
points. Fixed eight-round validation reduced that distance to 0.129 points, but
the validation winner had ranked tenth on training.

### Open question

A MAP-Elites archive treats stored objective values as fixed. With
heteroskedastic measurements, a favorable observation can displace an incumbent
and then influence future parent selection. Possible policies include repeated
incumbent/challenger measurement, confidence-bound dominance, Bayesian
shrinkage, indifference zones, and racing. They change search cost and archive
dynamics differently.

### Why deferred

The case study establishes local ranking instability but does not identify which
policy improves downstream search per unit of evaluator time. Adding confidence
fields without an admission policy would not solve the problem.

### Reopen when

A replay or controlled noisy benchmark compares at least two policies on elite
retention error, useful-candidate discovery, evaluator time, and asynchronous
behavior.

## 2. Finalist breadth and multiple selection

### Observation

The registered Zstandard Top 3 missed the later validation winner. Expanding to
Top 10 added about 39 minutes and changed the selected candidate. The Top-10
sample is training-selected and cannot estimate full-campaign
training-validation correlation.

### Open question

The next protocol could validate a fixed Top N, every candidate within an
effect-size band, or an adaptive racing set. A wider set improves winner recall
but increases validation cost and post-selection multiplicity.

### Why deferred

One run supports “Top 3 was too narrow” but cannot select a universal N or
distance threshold.

### Reopen when

Several campaigns report frontier width, validation rank changes, finalist
cost, and winner recall under candidate rules such as Top 10, Top 20, and a
0.003 effect-size band.

## 3. Archive geometry and embedding quality

### Observation

V19 used a 3D 4x4x4 grid and ended with 13 Pareto entries across 11 of 64
coordinates. Earlier PCA analysis found that the first two components explained
most observed variance, but that fact alone did not establish that a 2D archive
would search better.

### Open question

The useful dimension count, grid resolution, PCA refit schedule, and embedding
model may depend on repository size and candidate diversity. Coverage can be low
because the grid is too large, the candidate manifold is narrow, or the search
has not run long enough.

### Why deferred

Archive occupancy is a diagnostic, not an optimization outcome. Changing the
grid from one campaign's PCA variance would be post-hoc tuning.

### Reopen when

A same-candidate replay or replicated search compares 2D and 3D geometry on
sampling diversity, unique accepted identities, lineage depth, and final
validated quality.

## 4. Quality-diversity attribution

### Observation

The three case studies used Loreley's archive and sampler end to end. None ran a
same-budget root-independent or champion-sequential model arm. `python-pathspec`
shows an archived branch being revisited after 20 other jobs, but this does not
prove that a simpler search would miss the result.

### Open question

How much value comes from the QD archive, multi-generation editing, manual
seeds, model quality, or simply drawing many independent candidates?

### Why deferred

The current evidence is observational. Framework changes cannot resolve a
missing causal comparison.

### Reopen when

A preregistered study runs the same target, evaluator, seeds, models, token
budget, and physical concurrency across QD, champion-sequential, and
root-independent arms, with multiple search replicates.

## 5. Seed dependence and search ceiling

### Observation

The `markdown-it-py` and `python-pathspec` final candidates substantially beat
their best manual seeds. The registered Zstandard winner was manual seed 5.
The supplemental generation-4 candidate generalized to a fresh corpus but was
not compared head to head with the registered winner on one new corpus.

### Open question

How often does evolution improve a strong seed, and how much do seed quality and
directional diversity determine the search ceiling?

### Why deferred

Three hand-designed seed sets and one run per repository cannot separate seed
quality from search quality.

### Reopen when

One target is repeated with no seeds, weak seeds, strong seeds, and multiple
independent seed sets under the same search budget.

## 6. Search-run reproducibility

### Observation

The reports quantify benchmark sampling uncertainty for selected candidates.
They do not quantify variance across complete stochastic search runs.

### Open question

What are the success probability, time-to-first-useful-candidate distribution,
and final-quality variance at a fixed budget?

### Why deferred

Repeated searches require substantially more model spend and evaluator time.
One selected-candidate confidence interval is not a substitute.

### Reopen when

At least three independent runs use frozen models, prompts, seeds, evaluator,
and endpoints, and report both failures and unique candidate identities.

## 7. Cross-host and cross-architecture performance

### Observation

Zstandard results cover one Apple-silicon host, one compiler, single-threaded
levels 1/3/5, and generated corpus families. Source changes can compile
differently on x86-64 and other compilers.

### Open question

Does the measured gain persist across architectures, compilers, real corpora,
and multi-threaded operation, and is the patch maintainable upstream?

### Why deferred

No second controlled host or upstream review result is currently available.

### Reopen when

A frozen candidate set is replayed on at least one x86-64 host with an agreed
compiler matrix and real public corpora before any results are observed.

## 8. Evaluator-relevant identity beyond exact binaries

### Observation

Binary SHA-256 is effective for one fixed Zstandard build. Other targets may
need wheel identity, generated artifacts, query plans, hardware netlists,
semantic traces, or a tuple of outputs. Equal binaries also do not make
source-level portability or scope checks redundant.

### Open question

What constitutes safe reusable equality for each evaluator phase?

### Why deferred

No universal identity is correct. The framework proposal supplies an
evaluator-defined identity and phase-specific reuse contract, but the identity
definition remains target-owned.

### Reopen when

A second compiled or generated-artifact target exercises the public identity
contract and exposes a missing invariant.

## 9. Choosing evaluator concurrency

### Observation

Zstandard V19 showed that four calibrated lanes preserved paired root/candidate
bias while increasing throughput. Earlier experiments needed one lane, and a
macOS Background service caused severe absolute slowdown and precision
failures.

### Open question

How should a user choose `E` from calibration data, host load, and acceptable
precision without target-specific trial rules?

### Why deferred

The framework can enforce `E` independently, but no cross-workload rule maps a
calibration result to the right lane count.

### Reopen when

Several evaluators publish lane-count curves for wall speedup, absolute
throughput, paired bias, interval width, and failure rate.

## 10. Agent output-contract failures

### Observation

Headless Kilo policy removed interactive tool stalls. Some jobs still ended
without a valid plan, report, or effective repository change. The three case
studies retained these failures in their denominators.

### Open question

Structured output, a repair turn, or a different agent profile may recover more
jobs, but each option adds calls and can conceal model failure rates.

### Why deferred

The remaining failure rate has not been separated into model, prompt, parser,
and task-impossibility causes under a common provider.

### Reopen when

A failure replay measures recovery rate, added tokens, and candidate quality for
one bounded repair attempt versus terminal failure.

## 11. Provider-billed cost

### Observation

The DeepSeek reports contain provider-recorded generation cost. The GPT V19
route exposes Kilo catalog estimates and token/cache counters but no authoritative
provider invoice. Embedding routes are unpriced.

### Open question

Can the active provider expose billed usage by request or campaign without local
price reconstruction?

### Why deferred

This requires provider data, not a framework estimate. Reporting a catalog
number as billed spend would mislead users.

### Reopen when

The provider returns authoritative request cost or a campaign-level billing
export that can be reconciled with Kilo session IDs.

## 12. More experiments

Additional repository case studies, larger Zstandard searches, and another 64
or 128 jobs are not automatic next steps. A new experiment should answer one of
the questions above with a frozen comparison or materially broaden repository,
architecture, or evaluator coverage. Running more jobs only to increase the
headline job count does not resolve a current evidence gap.
