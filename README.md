# Loreley

[![CI](https://github.com/NeapolitanIcecream/loreley/actions/workflows/ci.yml/badge.svg)](https://github.com/NeapolitanIcecream/loreley/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-online-6563FF.svg)](https://neapolitanicecream.github.io/loreley/)
[![Paper](https://img.shields.io/badge/arXiv-2608.19703-b31b1b.svg)](https://arxiv.org/abs/2608.19703)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](https://www.python.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://github.com/NeapolitanIcecream/loreley/blob/main/LICENSE)

> Quality-Diversity program search over complete Git repositories.

Loreley uses planning and coding agents to propose repository-level commits. A
project-specific evaluator builds, verifies, and scores each candidate.
Candidates that pass may enter a persistent Quality-Diversity archive and
remain available as parents or inspirations for later jobs.

A Git commit records the source state and its ancestry. For compiled or
generated targets, the evaluator can define a separate measurement identity,
such as a release-binary hash, so equivalent artifacts do not consume another
benchmark run.

[Results](#results-from-three-repository-searches) ·
[Evolution dynamics](#evolution-dynamics-research) ·
[How it works](#how-loreley-works) ·
[Paper](https://arxiv.org/abs/2608.19703) ·
[Documentation](https://neapolitanicecream.github.io/loreley/) ·
[Design-partner intake](https://github.com/NeapolitanIcecream/loreley/issues/new?template=design-partner.yml)

![Loreley paper overview: repository-scale Quality-Diversity search, matched Zstandard comparison, and capability campaigns](https://raw.githubusercontent.com/NeapolitanIcecream/loreley/main/docs/marketing/assets/loreley-paper-overview.png)

The paper, [*Loreley: Repository-Scale Program Evolution with
Quality-Diversity Search*](https://arxiv.org/abs/2608.19703), reports the system,
three capability campaigns, and a matched Zstandard experiment comparing
Loreley QD with Sequential Champion and Independent Root search. The controlled
experiment used seven paired blocks and 48 physical candidate jobs per policy
and block (1,008 total). At 48 jobs, neither comparison established a QD
advantage. Read the [PDF](https://arxiv.org/pdf/2608.19703) or inspect the
[public experiment evidence](paper/evidence/zstd_method_efficacy.json).

## Evolution dynamics research

The four-page technical report follows candidate quality over wall-clock time
and successive generations. It compares the lineages of Independent Root,
Loreley QD, and Sequential Champion, then examines late-stage improvement and
reuse of retained branches.

Sequential Champion took **2.78× as long** as QD to finish 48 jobs (paired
geometric mean). At QD's completion time, QD led in all seven paired blocks.
QD's mean holdout gain also grew from **+0.47% at job 24 to +0.82% at job 48**;
the longer runs show retained branches producing later descendants.

[English report (PDF)](technical_report/loreley-evolution-dynamics-en.pdf) ·
[中文报告 (PDF)](technical_report/loreley-evolution-dynamics-zh.pdf) ·
[LaTeX and build instructions](technical_report/README.md) ·
[Timing data and validation](technical_report/evidence/README.md)

## Results from three repository searches

Loreley has been evaluated on fixed revisions of three existing repositories.
The studies completed 348 jobs: 310 succeeded and 38 failed. Each study used a
different workload and selection protocol, so the percentage results should not
be averaged or treated as expected performance on another repository.

![Results and selection status from three Loreley repository searches](https://raw.githubusercontent.com/NeapolitanIcecream/loreley/main/docs/marketing/assets/loreley-three-case-evidence.png)

| Repository | Measured result | Selection status |
| --- | --- | --- |
| [`markdown-it-py`](https://neapolitanicecream.github.io/loreley/research/2026-08-02-markdown-it-py-deepseek-case-study/) | Throughput +6.75% on a separate 28-document corpus; 28/28 documents improved | Winner frozen before validation |
| [`python-pathspec`](https://neapolitanicecream.github.io/loreley/research/2026-08-03-pathspec-deepseek-case-study/) | Throughput +25.14% across five reference workloads; 5/5 improved | Post-hoc selection after the registered candidate failed its allocation gate |
| [Zstandard](https://neapolitanicecream.github.io/loreley/research/2026-08-07-zstandard-gpt-v19-top10-validation-supplement/) | Validation-selected generation-4 candidate: compression throughput +1.173% on the original holdout and +0.891% on a newly sealed corpus | Original holdout previously opened at study level; fresh-corpus recipe chosen after candidate fixation |

For Zstandard, expanded validation selected generation-4 candidate `fe39bee8`.
Its own original-holdout score was unknown at selection; it later measured
+1.173% there and +0.891% on a newly generated and sealed corpus. The
[aggregate evidence report](https://neapolitanicecream.github.io/loreley/research/2026-08-07-loreley-case-study-evidence-report/)
contains the protocols, costs, failure counts, candidate selection records, and
claim limits for all three studies.

## How Loreley works

![Loreley searches repository states proposed by coding agents and accepted by an external evaluator](https://raw.githubusercontent.com/NeapolitanIcecream/loreley/main/docs/marketing/assets/loreley-search-loop.png)

1. **Define a campaign.** Fix the root commit, optimization goal, protected
   scope, evaluator, objectives, and job budget.
2. **Propose repository changes.** The scheduler selects retained commits as
   bases or inspirations. Planning and coding agents edit isolated Git
   worktrees and produce new commits.
3. **Evaluate each candidate.** The project evaluator builds the candidate,
   applies correctness and scope gates, measures the configured objectives, and
   may report an artifact identity.
4. **Retain useful states.** Loreley stores the commit, ancestry, metrics,
   artifacts, and terminal outcome. Passing candidates may enter the
   Quality-Diversity archive and be sampled by later jobs.

The evaluator controls what constitutes a valid improvement. It may call local
scripts, a C or C++ build, a Java benchmark, a container, a hardware testbed, or
a remote evaluation service. The evaluator plugin has a Python interface; the
target repository can use any implementation language.

## Quality-Diversity over repository states

Loreley derives behaviour descriptors from repository-state code embeddings.
File embeddings are cached by Git blob SHA, aggregated into commit vectors, and
can be reduced with PCA before placement in a MAP-Elites grid. Each occupied
cell retains a bounded Pareto front over the configured objectives.

Configured islands maintain separate archives and can exchange retained
candidates as inspirations. This keeps multiple valid branches available when
they occupy different behavioural niches or represent different objective
trade-offs.

The three capability case studies show multi-generation lineages, archive
retention, and later reuse of retained branches. The paper's separate matched
experiment did not establish that Quality-Diversity outperforms
root-independent sampling or sequential editing of a single champion at its
48-job horizon.

| Concern | Representation in Loreley |
| --- | --- |
| Reproducible source state | Complete Git commit |
| Candidate ancestry | Parent and inspiration edges between commits |
| Measurement identity | Evaluator-defined artifact, such as a binary, container image, or trace |
| Behavioural diversity | Repository-state embeddings and MAP-Elites cells |
| Multiple objectives | Bounded per-cell Pareto fronts |
| Persistent execution | PostgreSQL state, Redis/Dramatiq queues, schedulers, and worker processes |

## When a repository is a fit

A campaign needs:

- unattended build and correctness checks;
- at least one numerical objective with a known direction;
- an evaluator whose runtime and noise are compatible with the effect size of
  interest;
- enough safe evaluation parallelism for the proposed job budget;
- an isolated repository mirror, build environment, and authorized model
  access;
- enough engineering or business value to justify repeated model calls and
  evaluations.

The [design-partner brief](https://neapolitanicecream.github.io/loreley/marketing/loreley-design-partner-brief/)
describes evaluator calibration, candidate identity, data separation, budget
planning, and the non-confidential intake process.

## Run Loreley

Loreley is alpha software. A campaign requires Python 3.11+, Git worktrees,
PostgreSQL, Redis, the Kilocode CLI or another configured agent backend, an
OpenAI-compatible embedding endpoint, and an unattended evaluator plugin.

```bash
git clone https://github.com/NeapolitanIcecream/loreley.git
cd loreley
uv sync
docker compose up -d postgres redis
cp env.example .env
```

Configure the experiment ID, root commit, job cap, target repository remote,
evaluator plugin, embedding settings, objectives, and islands in `.env`. See the
[configuration guide](https://neapolitanicecream.github.io/loreley/loreley/config/)
for the complete contract.

Run the preflight checks, then start the scheduler and workers in separate
shells:

```bash
uv run loreley doctor --role all
uv run loreley scheduler
uv run loreley worker --processes 4
uv run loreley status
```

Before dispatching a new campaign, the scheduler scans the configured root
commit and requests operator approval. Non-interactive deployments can use
`--yes` or `SCHEDULER_STARTUP_APPROVE=true` after reviewing the scan settings.

The optional operator console is available with:

```bash
uv sync --extra ui
uv run loreley ui
```

## Documentation and evidence

| Resource | Contents |
| --- | --- |
| [Evolution dynamics research](technical_report/README.md) ([English PDF](technical_report/loreley-evolution-dynamics-en.pdf), [中文 PDF](technical_report/loreley-evolution-dynamics-zh.pdf)) | Four-page reports on wall-clock time, late-stage gains, and branch reuse, with reproducible figures and timing data |
| [Paper: arXiv:2608.19703](https://arxiv.org/abs/2608.19703) ([PDF](https://arxiv.org/pdf/2608.19703)) | Method, matched 1,008-job experiment, capability studies, and limitations |
| [Documentation home](https://neapolitanicecream.github.io/loreley/) | Architecture, configuration, CLI, and operations |
| [Scheduler and worker guides](https://neapolitanicecream.github.io/loreley/script/run_scheduler/) | Campaign startup and worker operation |
| [Three-case evidence report](https://neapolitanicecream.github.io/loreley/research/2026-08-07-loreley-case-study-evidence-report/) | Results, costs, failures, and evidence boundaries |
| [Candidate diff index](https://neapolitanicecream.github.io/loreley/marketing/candidates/) | Published source patches and artifact identities |
| [English essay](https://neapolitanicecream.github.io/loreley/marketing/2026-08-loreley-launch-article-en/) | Repository search model and the three studies |
| [中文文章](https://neapolitanicecream.github.io/loreley/marketing/2026-08-loreley-launch-article-zh/) | 《用编码智能体搜索真实代码仓库》 |
| [Architecture decisions](https://neapolitanicecream.github.io/loreley/adr/) | Accepted design records |
| [Release notes](https://github.com/NeapolitanIcecream/loreley/releases/tag/v0.10.0-alpha) | Current alpha release |

## Status and scope

The current release is `v0.10.0-alpha`. The three case studies demonstrate the
end-to-end system on their frozen repositories and evaluators. They do not
estimate Loreley's success rate or average effect on a new repository.

Loreley is licensed under the
[Apache License 2.0](https://github.com/NeapolitanIcecream/loreley/blob/main/LICENSE).

## Citation

```bibtex
@misc{chen2026loreley,
  title         = {Loreley: Repository-Scale Program Evolution with Quality-Diversity Search},
  author        = {Mohan Chen},
  year          = {2026},
  eprint        = {2608.19703},
  archiveprefix = {arXiv},
  primaryclass  = {cs.SE},
  url           = {https://arxiv.org/abs/2608.19703}
}
```

Machine-readable metadata are available in [`CITATION.cff`](CITATION.cff).
