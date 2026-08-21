# Loreley 发布文案包

> 内部工作文件，不作为对外页面。下方引用块可直接用于发布。

本文件中的数字受 [发布口径表](2026-08-loreley-launch-claim-sheet.md) 约束。发布时只替换平台格式，不重新改写实验结论。

## 统一链接

- 论文：<https://arxiv.org/abs/2608.19703>
- 项目：<https://github.com/NeapolitanIcecream/loreley>
- 中文长文：<https://neapolitanicecream.github.io/loreley/marketing/2026-08-loreley-launch-article-zh/>
- 英文长文：<https://neapolitanicecream.github.io/loreley/marketing/2026-08-loreley-launch-article-en/>
- 三案例证据：<https://neapolitanicecream.github.io/loreley/research/2026-08-07-loreley-case-study-evidence-report/>
- Zstandard 案例：<https://neapolitanicecream.github.io/loreley/research/2026-08-07-zstandard-gpt-v19-case-study-report/>
- 候选源码 diff：<https://neapolitanicecream.github.io/loreley/marketing/candidates/>
- 合作说明：<https://neapolitanicecream.github.io/loreley/marketing/loreley-design-partner-brief/>
- 合作 intake：<https://github.com/NeapolitanIcecream/loreley/issues/new?template=design-partner.yml>

外部平台统一使用以上绝对 URL。论文标题统一写作 *Loreley: Repository-Scale Program Evolution with Quality-Diversity Search*，引用编号统一写作 `arXiv:2608.19703`。

## 中文标题

论文首发：

> Loreley：在完整代码仓库上运行 Quality-Diversity 程序搜索

技术文章：

> 用编码智能体搜索真实代码仓库

备选：

- Loreley 的方法实验与三个代码仓库案例
- Coding agent、Git 仓库与外部 evaluator

不使用“首个仓库级演化系统”“自主重写任意项目”“QD 已优于简单搜索”或“低成本自动优化所有代码库”。

## 中文摘要

> Loreley 在完整 Git 仓库上运行 Quality-Diversity 程序搜索。Coding agent 在隔离 worktree 中修改代码，项目提供的 evaluator 负责构建、正确性检查和数值评测；通过检查的 Git commit 可以进入 archive，供后续任务继续修改或作为参考上下文。
>
> 论文包含一项 Zstandard 策略实验：Loreley QD、Sequential Champion 和 Independent Root 各在 7 个配对 block 中运行 48 个 candidate jobs，共 1,008 jobs。48-job 终点上，QD 相对 Sequential Champion 为 -0.135%（95% BCa 区间 -0.556% 至 +0.161%），相对 Independent Root 为 +0.320%（-0.082% 至 +0.686%）。两项比较均未建立 QD 优势；Sequential Champion 的终点均值和中位数最高。实验观察到 archive retention 和后续采样，但没有建立相应的终点收益。
>
> 论文还报告三项较早的 capability campaigns，共 348 jobs。`markdown-it-py` 的冻结候选在独立语料上提高吞吐量 6.75%；`python-pathspec` 的四代候选提高 25.14%，但属于事后选择；Zstandard 的 validation-selected 四代候选在原 holdout 和新封存 corpus 上分别提高 1.173% 和 0.891%，两项测量各有明确的选择与数据协议限制。这三项案例说明 Loreley 能找到通过 evaluator 的多代、多文件改进，不构成跨仓库平均收益或搜索策略优势的估计。
>
> 论文：<https://arxiv.org/abs/2608.19703>  项目与证据：<https://github.com/NeapolitanIcecream/loreley>

## 中文短帖

> Loreley 论文已公开：<https://arxiv.org/abs/2608.19703>
>
> 我们在完整 Git 仓库上比较了 Loreley QD、Sequential Champion 和 Independent Root：7 个 Zstandard 配对 block、每种策略每个 block 48 jobs，共 1,008 jobs。QD archive 的非 incumbent 分支确实被保留并重新使用，但 48-job 终点没有建立 QD 相对两种 baseline 的优势，Sequential Champion 的观测均值和中位数最高。
>
> 论文同时报告三个较早的 capability cases（348 jobs），包括两个四代 Python 改进和一个四代 Zstandard 改进。代码与证据：<https://github.com/NeapolitanIcecream/loreley>

## 中文极短帖

> Loreley 将 coding agent、完整 Git commit、外部 evaluator 和 Quality-Diversity archive 组成仓库级搜索系统。论文报告 1,008-job Zstandard 策略实验与三个较早的 capability cases：<https://arxiv.org/abs/2608.19703>

## 知乎或博客导语

> Loreley 把完整 Git 仓库作为候选状态，由 coding agent 修改、外部 evaluator 检查和评分，并用 Quality-Diversity archive 保留多条可继续搜索的分支。本文介绍系统设计、1,008-job Zstandard 策略实验，以及 `markdown-it-py`、`python-pathspec` 和另一版 Zstandard 上共 348 jobs 的 capability results。受控实验观察到了 archive retention 和后续采样，但没有在 48-job 终点建立 QD 对 Sequential Champion 或 Independent Root 的优势。

## 中文技术社区帖

标题：

> Loreley：完整 Git 代码仓库上的 Quality-Diversity 搜索

正文：

> Loreley 把每个候选源码状态和祖先关系记录为 Git commit。Coding agent 在隔离 worktree 中修改代码，项目 evaluator 执行构建、测试和数值评测；通过检查的候选可以进入 MAP-Elites/Pareto archive。Evaluator 是协议边界，可以调用任意语言的构建系统、容器、硬件测试台或远程服务。
>
> 论文的受控实验在同一 Zstandard revision 上比较三种在线策略。7 个配对 block 中，Loreley QD、Sequential Champion 和 Independent Root 各运行 48 个 physical candidate jobs，总计 1,008 jobs；三种策略使用相同 root、agent routes、evaluator 和 post-search winner rule。Validation 冻结候选后，由 agent 不可见的 holdout 测量终点。
>
> 在 48-job 终点：
>
> - QD 相对 Sequential Champion 为 -0.135%，95% BCa 区间为 -0.556% 至 +0.161%；
> - QD 相对 Independent Root 为 +0.320%，区间为 -0.082% 至 +0.686%；
> - 两项比较均未建立 QD 优势；Sequential Champion 的观测终点均值和中位数最高。
>
> 实验记录到 archive engagement：7 个最终 QD winners 中，4 个的 primary-parent ancestry 包含被 archive 保留的非 incumbent 状态；计入 inspiration context 后为 6 个。后一个计数只表示上下文被提供给 agent，不表示它造成了具体 edit。这些记录满足论文定义的 mechanism-engagement condition，但没有建立 holdout endpoint benefit。
>
> 论文另行报告三个较早的 capability campaigns，共 348 jobs：
>
> - `markdown-it-py`：64 jobs；冻结候选在独立 28 文档语料上提高吞吐量 6.75%，28/28 文档改善；
> - `python-pathspec`：64 jobs；四代候选在 5 个参考 workload 上提高 25.14%，候选为事后选择；
> - Zstandard：220 jobs；validation-selected 四代候选 `fe39bee8` 在原 holdout 上 +1.173%，在新封存 corpus 上 +0.891%。前者不是未触碰的 study-level holdout，后者的 corpus recipe 在候选冻结后确定。
>
> 三个案例使用不同 workload 和选择协议，不能横向平均。它们说明系统找到了通过各自 evaluator 的多代、多文件改进，不估计新仓库的平均效果。
>
> 论文：<https://arxiv.org/abs/2608.19703>
>
> 项目与公开证据：<https://github.com/NeapolitanIcecream/loreley>

## 图片使用

| 顺序 | 文件 | 用途 |
| ---: | --- | --- |
| 1 | [`loreley-paper-overview.png`](assets/loreley-paper-overview.png) | arXiv 与社区首发主图；同时展示方法循环、1,008-job endpoint、mechanism activity 和 capability cases |
| 2 | [`loreley-search-loop.png`](assets/loreley-search-loop.png) | 技术长文；说明 agent、evaluator 和 archive 的关系 |
| 3 | [`loreley-three-case-evidence.png`](assets/loreley-three-case-evidence.png) | capability results；展示三项结果和选择状态 |
| 4 | [`loreley-case-lineages.png`](assets/loreley-case-lineages.png) | 两个 Python 案例；展示多代累积和 archive 重新采样 |
| 5 | [`loreley-zstd-identity-results.png`](assets/loreley-zstd-identity-results.png) | 较早的 Zstandard 案例；展示 binary identity 与 Top-10 holdout 结果 |

现有宣传图均提供 SVG 和 1600×900 PNG。`loreley-paper-overview.png` 是论文首发默认图片；三案例总表与谱系图只用于解释 earlier capability campaigns。发布时保留 scope note，不单独截取百分比，也不把三案例图误标为 matched policy result。

## 发布顺序

1. 确认 GitHub、文档站和合作说明均包含 arXiv 链接；
2. 使用 `loreley-paper-overview.png` 发布中文短帖和中文技术社区帖；
3. 使用同一主图发布英文短帖和 Show HN/英文技术社区帖；
4. 在长文导语或置顶说明中补充论文与 1,008-job 受控实验入口；
5. 收集问题并维护 FAQ，实验口径只通过口径表统一更新。

## 平台发布检查

- 使用“统一链接”中的公开 URL；
- 区分 1,008-job matched policy experiment 与 348-job capability campaigns，不相加为一项实验；
- 同时报告 QD 与两个 baseline 的方向和不确定性，不只摘取 +0.320%；
- 不把 archive engagement 写成 endpoint benefit；
- 为每个平台上传原始图片，不依赖相对路径；
- 不承诺实验成功率、回复时限或固定美元报价。

## English title

Paper post:

> Loreley: Repository-Scale Program Evolution with Quality-Diversity Search

Technical article:

> Searching Real Code Repositories with Coding Agents

## English summary

> Loreley performs Quality-Diversity program search over complete Git repositories. Coding agents edit isolated worktrees; a project-supplied evaluator builds, verifies, and scores each candidate; passing commits may enter an archive and remain available as parents or context for later jobs.
>
> The paper reports a matched Zstandard policy experiment with seven paired blocks and 48 physical candidate jobs per policy and block: 1,008 jobs across Loreley QD, Sequential Champion, and Independent Root. At the 48-job endpoint, QD was 0.135% below Sequential Champion (95% BCa interval -0.556% to +0.161%) and 0.320% above Independent Root (-0.082% to +0.686%). Neither contrast established a QD advantage; Sequential Champion had the highest observed endpoint mean and median. The QD archive did retain and later resample non-incumbent states, but that engagement did not produce an established endpoint benefit.
>
> Three separate earlier capability campaigns completed 348 jobs. A frozen `markdown-it-py` candidate improved throughput by 6.75% on a separate corpus. A generation-4 `python-pathspec` candidate improved five reference workloads by 25.14%, with post-hoc selection. A validation-selected generation-4 Zstandard candidate measured +1.173% on the original holdout and +0.891% on a newly sealed corpus, with a stated protocol limitation for each measurement. These cases show multi-generation, multi-file improvements that passed their evaluators; they do not estimate average performance on a new repository or comparative policy efficacy.
>
> Paper: <https://arxiv.org/abs/2608.19703>  Code and evidence: <https://github.com/NeapolitanIcecream/loreley>

## English short post

> Loreley is now on arXiv: <https://arxiv.org/abs/2608.19703>
>
> We compared Loreley QD, Sequential Champion, and Independent Root on Zstandard in seven paired blocks, with 48 candidate jobs per policy and block (1,008 total). QD retained and reused non-incumbent repository states, but at 48 jobs it did not establish an endpoint advantage over either baseline; Sequential Champion had the highest observed mean and median.
>
> The paper also reports three earlier capability campaigns (348 jobs) that produced generation-4, multi-file improvements. Code and evidence: <https://github.com/NeapolitanIcecream/loreley>

## Hacker News or technical forum post

Title:

> Show HN: Loreley – Quality-Diversity search over complete Git repositories

Body:

> Loreley records candidate source states and ancestry as Git commits. Coding agents modify isolated worktrees; a project evaluator builds, tests, and measures each candidate; passing commits may enter a MAP-Elites/Pareto archive and remain available as future parents or context. The evaluator is a protocol boundary and can invoke builds, containers, hardware benches, or remote services in any language.
>
> The paper includes a matched Zstandard experiment comparing Loreley QD, Sequential Champion, and Independent Root. We ran seven paired blocks with 48 physical candidate jobs per policy and block, for 1,008 jobs. The policies shared the frozen root, agent routes, evaluator, candidate budget, and post-search selection rule. Validation fixed each winner before an agent-hidden holdout measured it.
>
> At 48 jobs, QD was 0.135% below Sequential Champion (95% BCa interval -0.556% to +0.161%) and 0.320% above Independent Root (-0.082% to +0.686%). Neither contrast established a QD advantage. Sequential Champion had the highest observed endpoint mean and median.
>
> Archive engagement did occur. Four of seven final QD winners had a retained non-incumbent in their primary-parent ancestry; the count was six when inspiration-context edges were included. Inspiration records context supplied to the agent, not a demonstrated causal edit. These diagnostics show that the configured QD mechanism was active, while the endpoint comparison shows no established benefit at the tested horizon.
>
> The paper separately reports three earlier capability campaigns (348 jobs):
>
> - `markdown-it-py`: 64 jobs; a candidate frozen before validation was 6.75% faster on a separate 28-document corpus, with 28/28 documents improving.
> - `python-pathspec`: 64 jobs; a generation-4 candidate was 25.14% faster on five reference workloads. Selection was post-hoc after the registered candidate failed an allocation gate.
> - Zstandard: 220 jobs; validation-selected generation-4 candidate `fe39bee8` measured +1.173% on the original holdout and +0.891% on a newly sealed corpus. The former was not an untouched study-level holdout; the latter corpus recipe was chosen after candidate fixation.
>
> Paper: <https://arxiv.org/abs/2608.19703>
>
> Code and evidence: <https://github.com/NeapolitanIcecream/loreley>
