# Searching Real Code Repositories with Coding Agents

*Results from 348 Loreley jobs on `markdown-it-py`, `python-pathspec`, and Zstandard*

Loreley performs evaluator-guided search over complete Git repositories. Planning and coding agents propose changes in isolated worktrees. A project-specific evaluator builds the result, applies correctness gates, and measures the configured objectives. Candidates that pass may enter a distributed quality-diversity archive and serve as parents or inspirations for later jobs.

A Git commit records the source state and its ancestry. For compiled or generated projects, the evaluator can provide a separate artifact identity, such as a release-binary hash, so that equivalent artifacts do not consume additional measurement budget.

We evaluated Loreley on fixed revisions of three repositories. The studies completed 348 jobs: 310 succeeded and 38 failed.

| Repository | Search budget | Active run time | Independent result | Selection status |
| --- | ---: | ---: | --- | --- |
| `markdown-it-py` | 64 jobs | 4.35 hours | throughput +6.75% | candidate frozen before validation |
| `python-pathspec` | 64 jobs | 3.91 hours | throughput +25.14% | post-hoc selection after allocation failure |
| Zstandard V19 | 220 jobs | 5.31 runner-hours | compression +1.019% | preregistered winner; manual seed |

The studies used different workloads and selection protocols. Their percentage results should not be averaged or treated as estimates of performance on other repositories.

![Results from the three Loreley case studies](assets/loreley-three-case-evidence.png)

## Search model and evaluator

The space of possible repository states is large. The states that build, pass the required tests, stay within the permitted edit scope, and improve the objective are sparse. Coding agents use the existing implementation, tests, names, types, call sites, profiles, and previous results to propose concrete commits. The evaluator determines which of those commits satisfy the experiment contract.

Each job starts from a commit retained by the search. A planning agent inspects the goal, repository, and earlier results. A coding agent edits an isolated Git worktree and creates a commit. The evaluator then builds, tests, and measures that worktree.

![Loreley repository search loop](assets/loreley-search-loop.png)

Evaluators use a small Python interface. The following simplified plugin returns a throughput measurement and the identity of the tested artifact:

```python
from loreley.core.worker.evaluator import EvalFail, EvalPass


def evaluate(context):
    result = build_test_and_benchmark(context.worktree)
    if not result.tests_passed:
        return EvalFail(kind="test", summary=result.failure)

    return EvalPass(
        summary="tests passed; benchmark completed",
        metrics={
            "name": "throughput",
            "value": result.throughput,
            "unit": "items/s",
            "higher_is_better": True,
        },
        candidate_identity=f"release-binary:{result.binary_sha256}",
    )
```

`build_test_and_benchmark()` may invoke a shell script, C or C++ build, Java benchmark, container, hardware testbed, or remote service. The Python interface does not restrict the implementation language of the target project.

Passing candidates are placed in a quality-diversity archive. The archive retains several high-performing candidates in different behavioral niches instead of maintaining only one incumbent. A later job can continue one retained lineage and receive an implementation from another lineage as inspiration.

## Case study 1: `markdown-it-py`

The `markdown-it-py` study used 64 jobs: 8 human-written seeds and 56 evolution jobs. The final candidate was frozen before the validation corpus was opened. On a separate corpus of 28 documents, it improved geometric-mean throughput by 6.75%. All 28 documents improved.

The final patch accumulated four generations of changes:

1. reduce string slicing in inline HTML parsing;
2. modify renderer dispatch and token-attribute hot paths;
3. reduce HTML-escaping and dispatch overhead;
4. add a normalization fast path.

The resulting diff changed five files, with 54 lines added and 14 removed. Each generation passed the evaluator before it became the parent of the next generation. The candidate was frozen before validation, so its selection was prospective. The four commits record compatible optimizations accumulating across generations.

## Case study 2: `python-pathspec`

The `python-pathspec` study used 64 jobs: 6 human-written seeds and 58 evolution jobs. The lineage that produced the reported candidate began at 0.9978× the root throughput. Later generations reached 1.0721×, 1.0866×, 1.1921×, and 1.2536× on the training workload.

The changes bound calls used in the hot loop, removed `groupdict()`, precomputed regular expressions, invoked `search()` directly, and flattened dispatch into pre-bound matcher tuples. Twenty jobs explored other candidates between generations 3 and 4. The archive retained the lineage during that interval.

![Primary lineages in the two Python studies](assets/loreley-case-lineages.png)

The final candidate improved throughput by 25.14% across five reference workloads. All five workloads improved, and the candidate passed the correctness, semantic, edit-scope, and allocation gates.

Candidate selection was post-hoc. The registered training winner failed the larger allocation shape used in reference evaluation, and the reported candidate was selected after that result was known. The 25.14% measurement applies to the five reference workloads, but it is not a prospective holdout result.

The experiment records the archive retaining the 0.9978× lineage and sampling it again in a later job. It does not compare quality-diversity with a single-champion or root-independent search under an equal budget.

## Case study 3: Zstandard V19

Zstandard V19 used 220 jobs: 8 human-written seeds and 212 evolution jobs. Of the 211 successful jobs, 167 produced distinct release binaries. The remaining 44 produced a binary that had already appeared.

The evaluator returned a release-binary SHA-256 as the candidate identity. After measurement caching was enabled, 19 repeated binaries reused an accepted result. Their median evaluator time was 21.6 seconds, compared with 186.7 seconds for jobs that ran the benchmark.

![Zstandard source identity, binary identity, and measured effects](assets/loreley-zstd-identity-results.png)

The preregistered Top-3 validation selected manual seed 5, a nine-line change in `hist.c` that unrolled a scalar histogram loop four bytes at a time:

```c
while ((size_t)(end - ip) >= 4) {
    count[ip[0]]++;
    count[ip[1]]++;
    count[ip[2]]++;
    count[ip[3]]++;
    ip += 4;
}
```

On the sealed holdout, the patch improved compression throughput by 1.019%, with a 95% confidence interval from +0.962% to +1.076%. Decompression changed by +0.010%, with an interval from -0.110% to +0.130%. Compressed size was unchanged, and peak RSS increased by 0.063 MiB.

Under the registered selection rule, the 212 evolution jobs did not outperform the manual seed.

The registered protocol validated the training Top 3. After that analysis was complete, a second protocol was registered for the remaining members of the training Top 10. A generation-4 candidate ranked tenth in training produced a 0.891% compression-throughput gain on a newly generated corpus, with a 95% interval from +0.522% to +1.261%. The registered winner and the follow-up winner were evaluated on different fresh corpora, so the two percentages are not a head-to-head comparison.

V19 separates three identities: the Git commit for source ancestry, the release binary for measurement reuse, and the evaluation report for a particular benchmark execution. The compression lower bounds of the training Top 10 spanned 0.276 percentage points, and the validation winner was ranked tenth in training. The Top-3 rule excluded that candidate from the registered validation.

## Resource accounting

The reported active times sum to 13.57 hours. The reports use slightly different campaign-time and active-runner definitions. These figures exclude experiment preparation, human analysis, and external waiting time.

The `markdown-it-py` and `python-pathspec` studies recorded DeepSeek generation costs of $2.0833 and $2.4856, respectively, for a combined $4.5689. Embeddings, hosts, and human work were not priced.

Zstandard V19 recorded a $60.2472 Kilo model-catalog estimate rather than a provider invoice. Its accounting basis differs from the two DeepSeek costs, so the three dollar figures should not be summed as an all-in project cost.

The [aggregate evidence report](../research/2026-08-07-loreley-case-study-evidence-report.md) contains the complete metrics, failure categories, token records, and selection status. The [candidate index](candidates/README.md) contains the four published source diffs.

## Relation to prior work

[FunSearch](https://www.nature.com/articles/s41586-023-06924-6) combined language-model generation, executable evaluation, and a database of previous programs within a human-provided program skeleton. [AlphaEvolve](https://arxiv.org/abs/2506.13131) expanded the editable code and supported multiple objectives and expensive external evaluation. Google later [reported](https://deepmind.google/blog/alphaevolve-impact/) that an AlphaEvolve-discovered Spanner LSM compaction heuristic reduced write amplification by 20%. In July 2026, Google [made AlphaEvolve available through Google Cloud](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/alphaevolve-on-cloud/).

Repository-scale systems published in 2025 and 2026 include:

- [SATLUTION](https://arxiv.org/abs/2509.07367), which modified a large C/C++ SAT solver over roughly 70 cycles with about 400 candidates per cycle, or approximately 28,000 candidate evaluations at a granularity comparable to a Loreley job;
- [ABCEvo](https://arxiv.org/abs/2604.15082), which connected agents to a million-line electronic-design-automation codebase with compilation, benchmark flows, and formal-equivalence checks;
- [CodeEvolve](https://arxiv.org/abs/2605.04677), which used runtime profiles to select optimization targets in Java and Apex;
- [HORIZON](https://arxiv.org/abs/2606.28279), which used Git worktrees and executable acceptance protocols to retain accepted engineering traces.

Loreley uses complete Git commits for source and ancestry, evaluator-defined artifact identities for measurement, and a distributed quality-diversity archive for parent and inspiration selection. The three studies reported here use 348 jobs, compared with approximately 28,000 candidate evaluations reported for SATLUTION.

## Evidence scope and next experiments

Across the three studies, Loreley generated and evaluated cross-file repository changes, and candidates passed independent performance evaluation. None of the studies compared quality-diversity search with simpler strategies under an equal budget.

Recent work reports that [independent sampling or sequential rewriting can match more elaborate search](https://arxiv.org/abs/2602.16805) on some code-evolution tasks. A separate [analysis of evolution traces](https://arxiv.org/abs/2605.20086) attributes some reported improvements to parameter tuning, reintroduced code, or evaluator overfitting.

A controlled comparison should hold the model, evaluator, and candidate-evaluation budget fixed across three strategies:

1. independent candidates sampled from the root;
2. sequential edits to the current champion;
3. Loreley's quality-diversity archive for parent and inspiration selection.

Each strategy requires repeated runs. Further work also includes Zstandard replication on x86-64 and a preregistered finalist policy that covers Top 10, an effect band, or adaptive racing.

## Integration requirements

Repository language is not the main constraint. A suitable evaluator must:

- build and run correctness checks without manual intervention;
- measure the target objective with known noise and runtime;
- define an identity for the artifact that consumes measurement budget;
- run with enough safe parallelism for the intended search budget.

Potential targets include compression and storage systems, database execution paths, compilers and EDA, SAT/SMT solvers, inference serving, and internal performance-critical code. The value of an expected improvement must justify the model and evaluation cost.

The [design-partner brief](loreley-design-partner-brief.md) lists the measurements needed to scope a run. Loreley is [available on GitHub](https://github.com/NeapolitanIcecream/loreley).
