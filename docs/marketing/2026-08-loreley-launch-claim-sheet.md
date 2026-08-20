# Loreley 发布口径表

日期：2026-08-11

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
| `markdown-it-py` | 64 jobs：8 seeds + 56 evolution；generation-4 winner 在独立 28 文档 corpus 上加速 **6.75%**，28/28 文档改善 | endpoint 在模型调用前固定；candidate 在 validation 前冻结 | 主要定量结果 |
| `python-pathspec` | 64 jobs：6 seeds + 58 evolution；generation-4 candidate 在 5 个 reference workloads 上加速 **25.14%**，5/5 改善 | 初始候选在 reference allocation gate 失败后改选；属于 post-hoc capability evidence | 解释多代谱系和 archive 重访 |
| Zstandard | 220 jobs：8 seeds + 212 evolution；211 成功，167 个不同 release binaries；报告 generation-4 candidate `fe39bee8` 在每个 split 上的结果 | validation 用于选择；original holdout 在此前已打开；fresh-corpus recipe 在候选确定后选择 | 解释系统仓库、评测精度、binary identity 和协议边界 |

Zstandard 报告候选 `fe39bee8`：

- 事后把 finalist 范围从 training Top 3 扩到 Top 10，再按预先封存的补充协议评测；
- training rank 10 的 generation-4 candidate `fe39bee8` 成为 validation winner；
- expanded-validation compression throughput 提升 **1.234%**，95% CI 为 **+1.156% 到 +1.312%**；该 split 用于选择，区间没有做 selection adjustment；
- 选择 `fe39bee8` 时还不知道它在 original holdout 上的分数；后来测得 compression throughput 提升 **1.173%**，95% CI 为 **+1.102% 到 +1.245%**；
- original holdout 此前已用于预登记 Top-3 winner，因此这是 candidate-level out-of-selection evidence，不是 untouched study-level holdout；
- 在新生成、独立封存的 corpus 上，compression throughput 提升 **0.891%**，95% CI 为 **+0.522% 到 +1.261%**；生成规则和 seed 是在 candidate 确定后选择的；
- compressed size 在四个 split 上均不变；三个 split 的 worst cell 略低于 root，但在预登记的 `0.98` floor 之上。

预登记 Top-3 协议历史：

- 预登记 winner 是 9 行人工 seed `7b9aef38`，不是 evolved candidate；
- 它在当时封存的 holdout 上将 compression throughput 提升 **1.019%**，95% CI 为 **+0.962% 到 +1.076%**；
- 该结果仍是原协议的正式结论，但跨 split 表格统一报告 `fe39bee8`。

Zstandard Top-10 描述性结果：

- 随后的 fixed-Top-10 比较在新测量前固定了候选身份和顺序，复用 `7b9aef38` 的原报告，并在同一原 holdout 上对其余 9 个候选各测 12 轮；
- 9 项新 holdout 测量用时 4,404 秒，即 73.4 分钟本地评测，没有模型调用；
- 10 个候选全部符合 modest-positive 规则，压缩提升中位数为 **1.116%**，点估计范围为 **+0.856% 到 +1.239%**，所有压缩置信下界都高于 root；
- 按压缩置信下界进行描述性排名，generation-3 evolved candidate `5ee53426` 排第一，提升 **1.228%**，95% CI 为 **+1.125% 到 +1.330%**；`fe39bee8` 排第二，提升 **1.173%**，95% CI 为 **+1.102% 到 +1.245%**；
- Top-10 比较不是新的 blinded winner。

正式来源：

- [三案例统一证据报告](../research/2026-08-07-loreley-case-study-evidence-report.md)
- [`markdown-it-py` 案例](../research/2026-08-02-markdown-it-py-deepseek-case-study.md)
- [`python-pathspec` 案例](../research/2026-08-03-pathspec-deepseek-case-study.md)
- [Zstandard 案例](../research/2026-08-07-zstandard-gpt-v19-case-study-report.md)
- [Zstandard Top-10 补充](../research/2026-08-07-zstandard-gpt-v19-top10-validation-supplement.md)

## 聚合数字

- 348 terminal jobs；310 成功，38 失败；
- 三份报告记录的 campaign 或 active-runner time 合计 13.57 小时；这不是 all-in elapsed time；
- 三个固定仓库都产生通过各自 evaluator 的改进候选；
- 案例数量不足以估计对新仓库的成功率或平均收益。

## 成本口径

| 案例 | Token 记录 | 美元记录 | 解释 |
| --- | ---: | ---: | --- |
| `markdown-it-py` | 215.35M generation；0.20M embedding | $2.0833 | 代理日志按记录的公开价格计算；不是 provider bill；embedding、主机和人工未计价 |
| `python-pathspec` | 241.63M generation；0.26M embedding | $2.4856 | 代理日志按记录的公开价格计算；不是 provider bill；embedding、主机和人工未计价 |
| Zstandard | 52.65M total，含 cached input 和 embedding | $60.2472 | Kilo model-catalog estimate，不是 provider bill；embedding 未计价 |

前两个 DeepSeek generation cost 可以相加为 $4.5689。Zstandard 的估算口径不同，三个数字不能相加成项目总现金花费。

## 可以公开表达

- Coding agent 能利用仓库语义和执行反馈，在稀疏的有效仓库状态之间提出可评测的修改。
- Loreley 把完整 Git commit、外部 evaluator、记录的 ancestry 和 quality-diversity archive 组织成持续搜索流程。
- evaluator 的 Python 接口是调度入口，不限制目标项目语言；它可以调用任意构建、测试、容器、硬件 benchmark 或远程评测系统。
- 三个固定案例覆盖 Python 库和 C 系统仓库，并分别提供冻结候选验证、谱系机制证据和可靠测量案例。
- 对编译型目标，source commit 不等于新的性能状态；Zstandard 实验用 release-binary identity 去重，并对 19 个重复 binary 复用评测。
- Zstandard 的 validation-selected `fe39bee8` 在 original holdout 上提升 +1.173%，在新封存 corpus 上提升 +0.891%；公开表达必须同时说明两个协议边界。
- Zstandard 的 fixed Top 10 在原 holdout 上 10/10 获得正向压缩结果，中位数为 +1.116%；该比较必须标注 holdout 已揭示和 post-selection 边界。
- 按置信下界的描述性排名，Zstandard holdout 前两位均为 evolved candidates；这不构成新的盲测 winner 或候选间显著差异。
- 当前结果足以展示端到端能力和寻找 design partners，不足以证明 quality-diversity 优于简单搜索。

## 不能公开表达

- 首个、唯一或最大的仓库级代码演化系统；
- 在任意仓库上有效、平均收益为正或成功率已知；
- quality-diversity 已经优于 best-of-N、root-independent 或 champion-sequential；
- `python-pathspec` 是预登记确认性结果；
- 把 Zstandard 的描述性排名写成 evolution 在新盲测中超过人工 seed，或把 `5ee53426` 写成新 winner；
- Zstandard 取得 2% 提升、跨架构成立或可以直接 upstream；
- 三案例百分比的平均值；
- 把三项美元记录相加为 all-in spend；
- Python 是唯一可接入语言；
- 用户需要先运行演示才能理解或采用项目。

## 固定短句

中文结果句：

> 在三个固定仓库案例中，Loreley 找到了通过独立 evaluator 的候选：`markdown-it-py` 的冻结候选验证提升 6.75%，`python-pathspec` 的 post-hoc capability result 提升 25.14%，Zstandard 的 validation-selected evolved candidate 在原 holdout 和新封存 corpus 上分别提升 1.173% 和 0.891%。后两个结果各有明确的协议限制。这些案例不能用于估计新仓库的平均收益。

Zstandard 补测句：

> 在事后固定候选的比较中，Zstandard 训练阶段 Top 10 在原 holdout 上 10/10 获得正向压缩结果，中位数为 +1.116%。该 holdout 此前已为预登记 winner 揭示，因此这不是新的盲测 winner。

英文结果句：

> Across three fixed-repository case studies, Loreley produced candidates that passed independent evaluation: a 6.75% frozen-candidate validation gain for `markdown-it-py`, a post-hoc 25.14% capability result for `python-pathspec`, and a validation-selected evolved Zstandard candidate with +1.173% on the original holdout and +0.891% on a newly sealed corpus. Each Zstandard measurement has a stated protocol limitation. These cases do not estimate the average effect on a new repository.

Zstandard supplementary sentence:

> In a later post-selection fixed-candidate comparison, all ten Zstandard training finalists remained positive on the original holdout, with a median compression gain of 1.116%. That holdout had already been revealed for the preregistered winner, so this is not a new blinded winner claim.
