# Loreley 发布口径表

日期：2026-08-21

> 内部工作文件，不作为对外页面。实验数字只从论文、机器可读证据和本文件链接的正式报告读取。

## 规范入口

- 论文：<https://arxiv.org/abs/2608.19703>
- 代码与公开证据：<https://github.com/NeapolitanIcecream/loreley>
- 三案例统一证据报告：[2026-08-07-loreley-case-study-evidence-report.md](../research/2026-08-07-loreley-case-study-evidence-report.md)
- 论文源码与证据说明：[`paper/README.md`](../../paper/README.md)

论文标题统一写作 *Loreley: Repository-Scale Program Evolution with Quality-Diversity Search*。

## 项目定位

中文：

> Loreley 是一个面向完整 Git 仓库的 Quality-Diversity 程序搜索系统。Coding agent 提出仓库级修改，外部 evaluator 负责构建、正确性检查和数值评测，系统用 Git 谱系和多样性 archive 保留可继续搜索的有效状态。

英文：

> Loreley is a Quality-Diversity program search system for complete Git repositories. Coding agents propose repository-level changes, external evaluators build, verify, and score them, and Loreley retains useful states through Git lineage and a diversity archive.

“把 AlphaEvolve 扩展到真实软件仓库”可以解释项目方向，不能写成首创声明。SATLUTION、ABCEvo、CodeEvolve、HORIZON、RHO/HELIX 和 CktEvo 已覆盖不同形式的仓库级或多文件代码演化。

## 两类实验不得混写

论文包含两组目的不同的证据：

1. **Matched policy experiment**：在一个 Zstandard revision 上进行 1,008 jobs 的三策略配对比较，用于检验 Quality-Diversity search 的机制活动和相对终点效果。
2. **Earlier capability campaigns**：在 `markdown-it-py`、`python-pathspec` 和另一版 Zstandard 上完成 348 jobs，用于展示系统能否产生通过 evaluator 的多代、多文件改进。

不得把 1,008 和 348 jobs 合并成一项统一实验、一个成功率或一个跨仓库策略比较。受控实验的结果不能替三个 capability cases 追认因果优势；capability results 也不能替受控实验证明 QD 更优。

## 1,008-job Zstandard matched policy experiment

### 设计

- 7 个配对 block；
- Loreley QD、Sequential Champion、Independent Root 三种策略；
- 每种策略在每个 block 运行 48 个 physical candidate jobs；
- 每种策略共 336 jobs，总计 1,008 jobs；
- 948 个成功 candidate，60 个失败或无效 job；失败仍占用预算；
- 三种策略共享 frozen root、任务、agent routes、training evaluator、48-job 预算和 post-search winner rule；
- 三种策略保留各自的 parent 更新、online state、context 和原生并发，这是一项完整 policy 比较，不是单独 archive 开关的组件消融；
- validation 在每个 checkpoint 冻结 winner；agent 不可见的 holdout 只测量已冻结 candidate，不改变选择；
- primary endpoint 是 48-job holdout compression-throughput ratio。

### 主要结果

| 对比 | QD/control 配对效应 | 95% BCa 区间 | Holm `p` | 允许的结论 |
| --- | ---: | ---: | ---: | --- |
| QD vs Sequential Champion | **-0.135%** | **-0.556% 到 +0.161%** | `.547` | 未建立 QD 优势 |
| QD vs Independent Root | **+0.320%** | **-0.082% 到 +0.686%** | `.375` | 未建立 QD 优势 |

48-job endpoint 的描述性结果：

| 策略 | Mean gain | Median gain | Range | `>= +0.5%` blocks |
| --- | ---: | ---: | ---: | ---: |
| Independent Root | +0.502% | +0.412% | +0.195% 到 +1.331% | 2/7 |
| Loreley QD | +0.824% | +0.739% | +0.062% 到 +1.390% | 6/7 |
| Sequential Champion | +0.960% | +0.819% | +0.514% 到 +1.789% | 7/7 |

Sequential Champion 的观测终点均值和中位数最高。主要效应区间仍允许小幅正向或负向效果，因此既不能说 QD 已证明更优，也不能说三种策略已证明等价。该结论只适用于一个 target、一个 host、固定 agent routes 和 48-job horizon。

### Archive engagement

- 7/7 个最终 QD winners 都不是 root；
- 4/7 个最终 QD winners 的 primary-parent ancestry 包含一个在 admission 时不是 incumbent、但被 archive 保留并在后来作为 parent 使用的状态；
- 同时递归统计 primary-parent 和 inspiration edges 时为 6/7；
- inspiration edge 只表示 commit 被作为上下文提供给 agent，不表示 agent 使用了其中内容或该上下文造成了 edit。

允许表达“archive retention 和后续采样实际发生”或“QD 策略满足了本文定义的 mechanism-engagement condition”。不能由此推出 retained state 对 winner 有因果贡献或 QD 提高了 holdout endpoint；只有 matched endpoint comparison 检验后一个命题，而本次实验没有建立该收益。

### 成本

- 可归因 generation 与 embedding 记录合计 **$517.8877**；不是 provider invoice，local compute 未计价；
- Independent Root、Loreley QD、Sequential Champion 各 336 jobs，对应的记录金额分别为 $185.30、$140.05 和 $192.54；
- equal physical jobs 是冻结的主要公平性定义，不表示 equal dollars、tokens 或 wall time。

## 三个 earlier capability cases

所有 speedup 都是 `candidate throughput / root throughput`。不要把 throughput 提升和固定工作量下的耗时下降混写。

| 案例 | 搜索与结果 | 选择状态 | 发布用法 |
| --- | --- | --- | --- |
| `markdown-it-py` | 64 jobs：8 seeds + 56 evolution；generation-4 winner 在独立 28 文档 corpus 上加速 **6.75%**，28/28 文档改善 | endpoint 在模型调用前固定；candidate 在 validation 前冻结 | 最干净的 capability result |
| `python-pathspec` | 64 jobs：6 seeds + 58 evolution；generation-4 candidate 在 5 个 reference workloads 上加速 **25.14%**，5/5 改善 | 初始候选在 reference allocation gate 失败后改选；属于 post-hoc capability evidence | 多代谱系和 archive 重访 |
| Zstandard | 220 jobs：8 seeds + 212 evolution；211 成功，167 个不同 release binaries；generation-4 candidate `fe39bee8` 在每个 split 上均有记录 | validation 用于选择；original holdout 在此前已打开；fresh-corpus recipe 在候选确定后选择 | 成熟 C 仓库、测量精度和 binary identity |

三项 campaigns 合计 348 terminal jobs：310 成功、38 失败。三个结果都来自 generation-4 descendants，不是最终选中的人工 seed。它们展示系统 capability，不估计新仓库成功率、平均收益或 QD 相对 baseline 的效果。

### Zstandard capability candidate `fe39bee8`

- training rank 10 的 generation-4 candidate `fe39bee8` 是 expanded-validation winner；
- expanded-validation compression throughput 提升 **1.234%**，95% CI 为 **+1.156% 到 +1.312%**；该 split 用于选择，区间没有做 selection adjustment；
- 选择时还不知道它自己的 original-holdout 分数；后来测得提升 **1.173%**，95% CI 为 **+1.102% 到 +1.245%**；
- original holdout 此前已用于预登记 Top-3 winner，因此这是 candidate-level out-of-selection evidence，不是 untouched study-level holdout；
- 新生成并封存的 corpus 上提升 **0.891%**，95% CI 为 **+0.522% 到 +1.261%**；生成规则和 seed 在 candidate 确定后选择；
- compressed size 在四个 split 上均不变；三个 split 的 worst cell 略低于 root，但在预登记的 `0.98` floor 之上。

原预登记 Top-3 winner 是 9 行人工 seed `7b9aef38`。它在当时封存的 holdout 上提升压缩吞吐量 **1.019%**，95% CI 为 **+0.962% 到 +1.076%**。该结果仍是原协议的正式结论；论文跨 split 表格报告 validation-selected `fe39bee8`。

随后的 fixed-Top-10 original-holdout 比较是 post-selection sensitivity evidence：10/10 candidates 为正，compression gain 中位数 **1.116%**，点估计范围 **+0.856% 到 +1.239%**。按 compression lower bound 描述性排名，generation-3 candidate `5ee53426` 为 +1.228%，`fe39bee8` 为 +1.173%。这不是新的 blinded winner，也没有建立 candidates 之间的显著差异。

正式来源：

- [`markdown-it-py` 案例](../research/2026-08-02-markdown-it-py-deepseek-case-study.md)
- [`python-pathspec` 案例](../research/2026-08-03-pathspec-deepseek-case-study.md)
- [Zstandard 案例](../research/2026-08-07-zstandard-gpt-v19-case-study-report.md)
- [Zstandard Top-10 补充](../research/2026-08-07-zstandard-gpt-v19-top10-validation-supplement.md)

## Capability campaign 成本口径

| 案例 | Token 记录 | 美元记录 | 解释 |
| --- | ---: | ---: | --- |
| `markdown-it-py` | 215.35M generation；0.20M embedding | $2.0833 | 代理日志按记录的公开价格计算；不是 provider bill；embedding、主机和人工未计价 |
| `python-pathspec` | 241.63M generation；0.26M embedding | $2.4856 | 代理日志按记录的公开价格计算；不是 provider bill；embedding、主机和人工未计价 |
| Zstandard | 52.65M total，含 cached input 和 embedding | $60.2472 | Kilo model-catalog estimate；不是 provider bill；embedding 未计价 |

前两个 DeepSeek generation estimates 可以相加为 $4.5689。三项 capability 数字口径不同，不能相加成项目总现金花费；它们也不能与 matched experiment 的 attributed ledger 直接相加为 all-in spend。

## 可以公开表达

- Coding agent 能利用仓库语义和执行反馈提出跨文件、可构建、可评测的 repository edits。
- Loreley 把完整 Git commit、外部 evaluator、记录的 ancestry 和 Quality-Diversity archive 组织成持续搜索流程。
- evaluator 的 Python 接口是调度入口，不限制目标项目语言；它可以调用任意构建、测试、容器、硬件 benchmark 或远程评测系统。
- 1,008-job experiment 是同一 Zstandard target 上三种完整 online policies 的 matched comparison。
- 在 48-job endpoint，QD 相对 Sequential Champion 为 -0.135%，相对 Independent Root 为 +0.320%；两项区间均跨零，没有建立 QD 优势。
- Archive 保留和重新采样 non-incumbents 的机制活动被观察到，但没有建立 endpoint benefit。
- 三个 capability cases 共 348 jobs，产生了通过各自 evaluator 的 generation-4、multi-file candidates；每项结果必须保留自己的 selection qualification。
- 当前论文支持公开方法、实现、机制观察、受控阴性结果与 capability evidence，并支持寻找 design partners。

## 不能公开表达

- 首个、唯一或最大的仓库级代码演化系统；
- 在任意仓库上有效、平均收益为正或成功率已知；
- Quality-Diversity 已经优于 best-of-N、Independent Root 或 Sequential Champion；
- 三种策略已证明等价，或 Sequential Champion 已被普遍证明更优；
- 把 +0.320% 单独写成方法收益而省略 interval 和另一个 baseline；
- 把 4/7 或 6/7 archive-engagement 计数写成 retained states 的因果贡献；
- 把 1,008-job matched experiment 和 348-job capability campaigns 合并成一项实验、统一成功率或统一效果；
- `python-pathspec` 是预登记确认性结果；
- 把 Zstandard 的描述性排名写成 evolution 在新盲测中超过人工 seed，或把 `5ee53426` 写成新 winner；
- Zstandard 取得 2% 提升、跨架构成立或可以直接 upstream；
- 三案例百分比的平均值；
- 把不同口径的美元记录相加为 all-in spend；
- Python 是唯一可接入语言；
- 用户需要先运行 demo 才能理解或采用项目。

## 固定短句

中文方法句：

> 在一项 1,008-job Zstandard 配对实验中，Loreley QD 的 archive 确实保留并重新使用了非 incumbent 状态；但在 48-job endpoint，QD 相对 Sequential Champion 为 -0.135%（95% BCa 区间 -0.556% 至 +0.161%），相对 Independent Root 为 +0.320%（-0.082% 至 +0.686%），两项比较均未建立 QD 优势。

中文 capability 句：

> 在三个较早的固定仓库案例中，Loreley 找到了通过各自 evaluator 的 generation-4 candidates：`markdown-it-py` 的冻结候选验证提升 6.75%，`python-pathspec` 的 post-hoc capability result 提升 25.14%，Zstandard 的 validation-selected candidate 在原 holdout 和新封存 corpus 上分别提升 1.173% 和 0.891%。这些案例不能用于估计新仓库的平均收益。

English method sentence:

> In a 1,008-job matched Zstandard experiment, the Loreley QD archive retained and later reused non-incumbent states; however, at the 48-job endpoint QD was 0.135% below Sequential Champion (95% BCa interval -0.556% to +0.161%) and 0.320% above Independent Root (-0.082% to +0.686%), so neither contrast established a QD advantage.

English capability sentence:

> Three earlier fixed-repository campaigns produced generation-4 candidates that passed their evaluators: a 6.75% frozen-candidate validation gain for `markdown-it-py`, a post-hoc 25.14% capability result for `python-pathspec`, and a validation-selected Zstandard candidate with +1.173% on the original holdout and +0.891% on a newly sealed corpus. These cases do not estimate average performance on a new repository.
