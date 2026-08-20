# Loreley 发布文案包

> 内部工作文件，不作为对外页面。下方引用块可直接用于发布。

本文件中的数字受 [发布口径表](2026-08-loreley-launch-claim-sheet.md) 约束。发布时只替换链接和平台格式，不重新改写实验结论。

## 统一链接

- 项目：<https://github.com/NeapolitanIcecream/loreley>
- 中文长文：<https://neapolitanicecream.github.io/loreley/marketing/2026-08-loreley-launch-article-zh/>
- 英文长文：<https://neapolitanicecream.github.io/loreley/marketing/2026-08-loreley-launch-article-en/>
- 三案例证据：<https://neapolitanicecream.github.io/loreley/research/2026-08-07-loreley-case-study-evidence-report/>
- Zstandard：<https://neapolitanicecream.github.io/loreley/research/2026-08-07-zstandard-gpt-v19-case-study-report/>
- 候选源码 diff：<https://neapolitanicecream.github.io/loreley/marketing/candidates/>
- 合作说明：<https://neapolitanicecream.github.io/loreley/marketing/loreley-design-partner-brief/>
- 合作 intake：<https://github.com/NeapolitanIcecream/loreley/issues/new?template=design-partner.yml>

外部平台统一使用以上绝对 URL。

## 中文标题

首选：

> 用编码智能体搜索真实代码仓库

备选：

- Loreley：完整 Git 代码仓库上的评估器驱动搜索
- Loreley 的三个代码仓库实验
- 代码仓库搜索中的提交、二进制文件与评估器

不使用“首个仓库级演化系统”“自主重写任意项目”或“低成本自动优化所有代码库”。

## 300–500 字摘要

Loreley 在完整 Git 代码仓库上运行评估器驱动的搜索。规划智能体和编码智能体修改隔离的 worktree；项目评估器负责构建、正确性检查和性能测量；通过检查的候选提交可以进入质量-多样性档案库。对于编译型项目，评估器可以使用二进制文件哈希定义接受测量的产物身份。

我们在三个固定代码仓库版本上完成了 348 个任务。`markdown-it-py` 的候选方案在验证前冻结，并在独立的 28 文档语料库上将吞吐量提高 6.75%。`python-pathspec` 的四代谱系在 5 个参考工作负载上提高 25.14%，但候选方案是在内存分配检查结果揭示后事后选择的。Zstandard 的第 4 代候选方案 `fe39bee8` 由扩展验证选出，在原留出集上提高压缩吞吐量 1.173%，在新封存语料上提高 0.891%。选择时还不知道它的留出集分数，但该留出集已用于另一候选；新语料的生成方案则在候选确定后选择。原预登记 Top-3 协议的获胜方案仍是人工种子。

完整报告包含候选选择过程、评估协议、谱系、失败记录和成本口径。项目正在寻找拥有自动评估器、真实代码仓库和相应计算预算的设计合作伙伴。

完整文章与证据：<https://neapolitanicecream.github.io/loreley/marketing/2026-08-loreley-launch-article-zh/>

## 短帖

> Loreley 使用编码智能体和项目评估器搜索完整 Git 代码仓库。三个固定版本实验共运行 348 个任务：`markdown-it-py` 在独立语料库上 +6.75%；`python-pathspec` 在 5 个参考工作负载上 +25.14%，候选为事后选择；Zstandard 的第 4 代候选方案在原留出集和新封存语料上分别 +1.173% 和 +0.891%，两项测量各有明确的协议限制。报告包含选择过程、成本和限制：<https://neapolitanicecream.github.io/loreley/marketing/2026-08-loreley-launch-article-zh/>

## 极短帖

> Loreley 在三个真实代码仓库上完成了 348 个搜索任务。候选选择过程、失败记录、成本和源码 diff 均已公开：<https://neapolitanicecream.github.io/loreley/research/2026-08-07-loreley-case-study-evidence-report/>

## 知乎或博客导语

> Loreley 在完整 Git 代码仓库上运行评估器驱动的搜索。本文说明一次搜索任务如何产生并评估候选提交，并报告 `markdown-it-py`、`python-pathspec` 和 Zstandard 上共 348 个任务的结果。内容包括四代演化谱系、提交与二进制文件的身份差异、独立验证结果、资源使用和目前缺少的对照实验。

## 技术社区帖

标题：

> Loreley：完整 Git 代码仓库上的质量-多样性搜索

正文：

> Loreley 使用 Git 提交记录候选方案的源码和祖先关系。外部编码智能体修改隔离的 worktree，项目评估器执行构建、测试和性能测量。通过检查的候选方案可以进入 MAP-Elites/Pareto 档案库。对于编译型项目，评估器可以另行提供二进制文件哈希作为测量身份。
>
> 三个固定代码仓库版本的结果如下：
>
> - `markdown-it-py`：64 个任务；候选方案在验证前冻结，在独立的 28 文档语料库上将吞吐量提高 6.75%，28 份文档全部改善；
> - `python-pathspec`：64 个任务；四代谱系在 5 个参考工作负载上提高 25.14%，候选方案为事后选择；
> - Zstandard：220 个任务，产生 167 个不同的发布版二进制文件；第 4 代候选方案 `fe39bee8` 由扩展验证选出，在原留出集上 +1.173% (95% CI +1.102% 到 +1.245%)，在新封存语料上 +0.891% (95% CI +0.522% 到 +1.261%)。选择时不知道它的原留出集分数，但该留出集此前已打开；新语料生成方案在候选确定后选择。原预登记 Top-3 winner 仍是人工 seed。
>
> 三项结果使用不同的工作负载和选择协议，不能横向平均。统一报告包含失败、token、成本口径和证据范围。后续实验将按相同预算比较质量-多样性、从根版本独立采样和沿当前最优候选连续修改三种策略。
>
> 文章与报告：<https://neapolitanicecream.github.io/loreley/marketing/2026-08-loreley-launch-article-zh/>

## 图片使用

| 顺序 | 文件 | 用途 |
| ---: | --- | --- |
| 1 | [`loreley-three-case-evidence.png`](assets/loreley-three-case-evidence.png) | 长文结果总表之后；展示三项结果和选择状态 |
| 2 | [`loreley-search-loop.png`](assets/loreley-search-loop.png) | 搜索模型与评估器部分；说明智能体、评估器和档案库的关系 |
| 3 | [`loreley-case-lineages.png`](assets/loreley-case-lineages.png) | 两个 Python 案例；展示多代累积和档案库重新采样 |
| 4 | [`loreley-zstd-identity-results.png`](assets/loreley-zstd-identity-results.png) | Zstandard 案例；展示二进制身份与置信区间 |

图片均提供 SVG 和 1600×900 PNG。发布时保留卡片底部的 scope note，不单独截取百分比。

## 发布顺序

1. GitHub 和文档站先落地中英文长文、统一证据报告、设计合作伙伴说明与图片；
2. 发布完整中文文章，以文章链接作为中文长文入口；
3. 同日发布中文短帖和技术社区帖，只复用本文件中的结果句；
4. 发布英文长文和英文技术社区帖；
5. 收集问题并更新 FAQ，不直接改动已经发布的实验口径。

英文稿是信息结构和结论口径的母稿。中文稿只做保守翻译和中文句法校订，不增加过渡、总结、强调、比喻或叙事框架。

## 平台发布检查

- 使用“统一链接”中的公开 URL；
- 为每个平台上传原始 PNG，不依赖平台抓取本地相对路径；
- 不在发布帖中承诺回复时限、实验成功率或固定美元报价。

## English title

Primary:

> Searching Real Code Repositories with Coding Agents

Alternate:

- Loreley: Evaluator-Guided Search over Complete Git Repositories
- Three Repository-Scale Studies of Loreley
- Commits, Binaries, and Evaluators in Repository Search

## English summary

Loreley performs evaluator-guided search over complete Git repositories. Planning and coding agents edit isolated worktrees; project evaluators build, test, and benchmark each candidate; passing commits may enter a quality-diversity archive. Compiled targets can use a binary hash as the artifact identity that consumes measurement budget.

We completed 348 jobs on fixed revisions of three repositories. A `markdown-it-py` candidate frozen before validation improved throughput by 6.75% on a separate 28-document corpus. A four-generation `python-pathspec` candidate improved five reference workloads by 25.14%, but candidate selection was post-hoc after an allocation failure. Zstandard generation-4 candidate `fe39bee8` was selected on expanded validation and then measured +1.173% compression on the original holdout and +0.891% on a newly sealed corpus. Its own holdout score was unknown at selection, but that corpus had already been opened for another candidate; the new-corpus recipe was chosen after selection. The preregistered Top-3 result remains a manual seed.

The reports include candidate selection, evaluation protocols, lineage, failures, and cost semantics. We are looking for design partners with automated evaluators, valuable optimization targets, and appropriate compute budgets.

Article and evidence: <https://neapolitanicecream.github.io/loreley/marketing/2026-08-loreley-launch-article-en/>

## English short post

> Loreley performs evaluator-guided search over complete Git repositories. Three fixed-revision studies completed 348 jobs: `markdown-it-py` +6.75% after candidate freeze; `python-pathspec` +25.14% with post-hoc selection; and a validation-selected generation-4 Zstandard candidate at +1.173% on the original holdout and +0.891% on a newly sealed corpus. The Zstandard measurements have different protocol limits, documented with the reports, source diffs, and costs: <https://neapolitanicecream.github.io/loreley/marketing/2026-08-loreley-launch-article-en/>

## Hacker News or technical forum post

Title:

> Show HN: Loreley – evaluator-guided search over complete Git repositories

Body:

> Loreley records each candidate source state and ancestry as a Git commit. External coding agents modify isolated worktrees; a project-specific evaluator builds, tests, and measures the result; passing candidates may enter a MAP-Elites/Pareto archive. Compiled targets can provide a separate binary identity for measurement and caching.
>
> We completed three fixed-revision studies:
>
> - `markdown-it-py`: 64 jobs; a candidate frozen before validation was 6.75% faster on a separate 28-document corpus, with 28/28 documents improving.
> - `python-pathspec`: 64 jobs; a four-generation lineage was 25.14% faster on five reference workloads. Candidate selection was post-hoc after the registered winner failed an allocation gate.
> - Zstandard: 220 jobs and 167 distinct release binaries; generation-4 candidate `fe39bee8` was selected on expanded validation, then measured +1.173% compression on the original holdout (95% CI +1.102% to +1.245%) and +0.891% on a newly sealed corpus (95% CI +0.522% to +1.261%). Its holdout score was unknown at selection, but that corpus had been opened for another candidate; the new-corpus recipe was chosen after selection. The preregistered Top-3 winner remains a manual seed.
>
> The aggregate report includes failures, token records, cost semantics, and the scope of each result. The next controlled experiment will compare quality-diversity with same-budget root-independent and champion-sequential search.
>
> <https://neapolitanicecream.github.io/loreley/marketing/2026-08-loreley-launch-article-en/>
