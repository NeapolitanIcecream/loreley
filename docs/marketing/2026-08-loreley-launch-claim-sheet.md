# Loreley 发布口径表

日期：2026-08-07

> 内部工作文件，不作为对外页面。实验数字只从本文件链接的正式报告读取。

## 项目定位

中文：

> Loreley 是一个面向完整 Git 仓库的 quality-diversity 程序搜索系统。Coding agent 提出仓库级修改，外部 evaluator 负责构建、正确性检查和数值评测，系统用 Git 谱系和多样性 archive 保留可继续搜索的有效状态。

英文：

> Loreley is a quality-diversity program search system for complete Git repositories. Coding agents propose repository-level changes, external evaluators build, verify, and score them, and Loreley retains useful states through Git lineage and a diversity archive.

“把 AlphaEvolve 扩展到真实软件仓库”可以作为解释项目方向的标题或主线，不能写成首创声明。SATLUTION、ABCEvo、CodeEvolve 和 HORIZON 已经覆盖不同形式的仓库级代码演化。

## 三个案例

所有 speedup 都是 `candidate throughput / root throughput`。不要把 throughput 提升和固定工作量下的耗时下降混写。

| 案例 | 搜索与结果 | 选择状态 | 发布用法 |
| --- | --- | --- | --- |
| `markdown-it-py` | 64 jobs：8 seeds + 56 evolution；generation-4 winner 在独立 28 文档 corpus 上加速 **6.75%**，28/28 文档改善 | 预登记、validation 揭示前冻结的前瞻性结果 | 主要定量结果 |
| `python-pathspec` | 64 jobs：6 seeds + 58 evolution；generation-4 candidate 在 5 个 reference workloads 上加速 **25.14%**，5/5 改善 | 初始候选在 reference allocation gate 失败后改选；属于 post-hoc capability evidence | 解释多代谱系和 archive 重访 |
| Zstandard V19 | 220 jobs：8 seeds + 212 evolution；211 成功，167 个不同 release binaries | 主结果和补充结果必须分开 | 解释系统仓库、评测精度、binary identity 和负面结果边界 |

Zstandard V19 主结果：

- 预登记 winner 是 9 行人工 seed，不是 evolved candidate；
- sealed holdout compression throughput 提升 **1.019%**，95% CI 为 **+0.962% 到 +1.076%**；
- decompression 为 **+0.010%**，95% CI 为 **-0.110% 到 +0.130%**；
- compressed size 不变，peak RSS 增加 0.063 MiB；
- 结论是 modest-positive，不能写成 evolution 超过 strongest seed。

Zstandard V19 补充结果：

- 事后把 finalist 范围从 training Top 3 扩到 Top 10，再按预先封存的补充协议评测；
- training rank 10 的 generation-4 candidate `fe39bee8` 成为 validation winner；
- 在新生成、独立封存的 corpus 上，compression throughput 提升 **0.891%**，95% CI 为 **+0.522% 到 +1.261%**；
- 两个候选使用不同 fresh corpora，没有 head-to-head，补充结果不能改写预登记 winner。

正式来源：

- [三案例统一证据报告](../research/2026-08-07-loreley-case-study-evidence-report.md)
- [`markdown-it-py` 案例](../research/2026-08-02-markdown-it-py-deepseek-case-study.md)
- [`python-pathspec` 案例](../research/2026-08-03-pathspec-deepseek-case-study.md)
- [Zstandard V19 案例](../research/2026-08-07-zstandard-gpt-v19-case-study-report.md)
- [Zstandard V19 Top-10 补充](../research/2026-08-07-zstandard-gpt-v19-top10-validation-supplement.md)

## 聚合数字

- 348 terminal jobs；310 成功，38 失败；
- 三份报告记录的 campaign 或 active-runner time 合计 13.57 小时；这不是 all-in elapsed time；
- 三个固定仓库都产生通过各自 evaluator 的改进候选；
- 案例数量不足以估计对新仓库的成功率或平均收益。

## 成本口径

| 案例 | Token 记录 | 美元记录 | 解释 |
| --- | ---: | ---: | --- |
| `markdown-it-py` | 215.35M generation；0.20M embedding | $2.0833 | provider-recorded DeepSeek generation cost；embedding、主机和人工未计价 |
| `python-pathspec` | 241.63M generation；0.26M embedding | $2.4856 | provider-recorded DeepSeek generation cost；embedding、主机和人工未计价 |
| Zstandard V19 | 52.65M total，含 cached input 和 embedding | $60.2472 | Kilo model-catalog estimate，不是 provider bill；embedding 未计价 |

前两个 DeepSeek generation cost 可以相加为 $4.5689。Zstandard 的估算口径不同，三个数字不能相加成项目总现金花费。

## 可以公开表达

- Coding agent 能利用仓库语义和执行反馈，在稀疏的有效仓库状态之间提出可评测的修改。
- Loreley 把完整 Git commit、外部 evaluator、记录的 ancestry 和 quality-diversity archive 组织成持续搜索流程。
- evaluator 的 Python 接口是调度入口，不限制目标项目语言；它可以调用任意构建、测试、容器、硬件 benchmark 或远程评测系统。
- 三个固定案例覆盖 Python 库和 C 系统仓库，并分别提供前瞻性结果、谱系机制证据和可靠测量案例。
- 对编译型目标，source commit 不等于新的性能状态；V19 用 release-binary identity 去重，并对 19 个重复 binary 复用评测。
- 当前结果足以展示端到端能力和寻找 design partners，不足以证明 quality-diversity 优于简单搜索。

## 不能公开表达

- 首个、唯一或最大的仓库级代码演化系统；
- 在任意仓库上有效、平均收益为正或成功率已知；
- quality-diversity 已经优于 best-of-N、root-independent 或 champion-sequential；
- `python-pathspec` 是预登记确认性结果；
- Zstandard evolution 超过人工 seed、取得 2% 提升、跨架构成立或可以直接 upstream；
- 三案例百分比的平均值；
- 把三项美元记录相加为 all-in spend；
- Python 是唯一可接入语言；
- 用户需要先运行演示才能理解或采用项目。

## 固定短句

中文结果句：

> 在三个固定仓库案例中，Loreley 找到了通过独立 evaluator 的候选：`markdown-it-py` 的前瞻性验证提升 6.75%，`python-pathspec` 的 post-hoc capability result 提升 25.14%，Zstandard V19 的预登记 holdout compression 提升 1.019%。这些案例不能用于估计新仓库的平均收益。

英文结果句：

> Across three fixed-repository case studies, Loreley produced candidates that passed independent evaluation: a prospective 6.75% gain for `markdown-it-py`, a post-hoc 25.14% capability result for `python-pathspec`, and a preregistered 1.019% holdout compression gain for Zstandard V19. These cases do not estimate the average effect on a new repository.
