# Loreley 推广与研究计划

日期：2026-08-03

最后更新：2026-08-08

状态：内部工作文档，不进入 MkDocs 站点。后续讨论、文献核查、案例选择和实验设计都更新到本文件。

## 当前目标

形成一套以案例证据和研究贡献为中心的推广计划，产出包括：

1. 一篇英文技术文章，以及只做保守翻译和中文句法校订的中文版本；
2. 一篇达到可公开挂在 arXiv 上的研究论文；
3. `markdown-it-py`、`python-pathspec` 和 Zstandard 三个案例的可审计叙事；
4. 一轮不等待论文对照实验的公开推广，包括文章、案例证据包和社区发布材料；
5. 面向拥有自动 evaluator 和计算预算的团队的合作入口。

公开可运行 Demo 不是当前前置条件。需要保留的是可浏览的候选 diff、演化谱系、评测协议、聚合结果、资源账本和 claim boundary。

## 当前推广判断

现有证据已经足够支持第一轮推广，不需要等待 Zstandard 再取得更高数字，也不需要先补完论文级对照实验。第一轮推广的目标是让目标用户理解 Loreley 能解决什么问题、三组案例分别证明什么，并找到愿意提供 evaluator、仓库和算力的设计合作方；它不承担证明 quality-diversity 优于所有简单搜索方法的任务。

当前已经具备完整的发布材料：

- 三案例统一证据报告：[2026-08-07-loreley-case-study-evidence-report.md](2026-08-07-loreley-case-study-evidence-report.md)；
- Zstandard V19 正式报告：[2026-08-07-zstandard-gpt-v19-case-study-report.md](2026-08-07-zstandard-gpt-v19-case-study-report.md)；
- Zstandard Top-10 与 fresh-confirmation 补充报告：[2026-08-07-zstandard-gpt-v19-top10-validation-supplement.md](2026-08-07-zstandard-gpt-v19-top10-validation-supplement.md)；
- README、文档首页和 MkDocs 导航已经指向三案例报告与 V19，不再以 V13 作为对外主结果。
- 统一口径表：[2026-08-loreley-launch-claim-sheet.md](../marketing/2026-08-loreley-launch-claim-sheet.md)；
- 英文长文：[2026-08-loreley-launch-article-en.md](../marketing/2026-08-loreley-launch-article-en.md)；
- 中文长文：[2026-08-loreley-launch-article-zh.md](../marketing/2026-08-loreley-launch-article-zh.md)；
- 摘要、短帖和社区文案：[2026-08-loreley-launch-copy-kit.md](../marketing/2026-08-loreley-launch-copy-kit.md)；
- 合作入口：[loreley-design-partner-brief.md](../marketing/loreley-design-partner-brief.md) 和公开 GitHub intake；
- 四张数据驱动证据图及 SVG 源文件：[marketing/assets](../marketing/assets/loreley-search-loop.png)。

Zstandard V19 的数值没有超过 V13，但它修复了 V13 暴露的主要证据问题。V19 按 release binary 统计和冻结候选，使用新语料、四条经校准的 evaluator lanes、显式模型路由和完整成本记录。V13 只保留为基础设施与 binary-equivalence 的历史材料，不计作第四个案例。

## 当前完成状态与下一步

第一轮 GitHub 与文档站材料已经发布。PR [#54](https://github.com/NeapolitanIcecream/loreley/pull/54) 包含中英文长文、四张图、候选 diff、统一证据入口、design-partner brief、公开 issue intake、README 和 MkDocs 导航。Pages 从发布提交 `018c144` 生成，`gh-pages` 提交为 `c66dc01`；[部署任务](https://github.com/NeapolitanIcecream/loreley/actions/runs/31249188262)成功。主页、中英文文章、三案例证据报告、候选 diff 索引和合作说明均完成线上 HTTP 与标题检查。

公开入口如下：

- 中文长文：<https://neapolitanicecream.github.io/loreley/marketing/2026-08-loreley-launch-article-zh/>
- 英文长文：<https://neapolitanicecream.github.io/loreley/marketing/2026-08-loreley-launch-article-en/>
- 三案例证据报告：<https://neapolitanicecream.github.io/loreley/research/2026-08-07-loreley-case-study-evidence-report/>
- 候选 diff：<https://neapolitanicecream.github.io/loreley/marketing/candidates/>
- 合作说明：<https://neapolitanicecream.github.io/loreley/marketing/loreley-design-partner-brief/>。

后续工作按以下顺序推进：

1. 先投中文渠道，再发布中文短帖、英文文章和英文社区帖；
2. 根据收到的问题补 FAQ，并筛选 design-partner intake；
3. 继续论文轨道，补 quality-diversity、champion-sequential、root-independent 三个同预算 arms，搜索重复、x86-64 复现和 finalist/noisy-objective 研究。

本轮没有执行外部平台发布。

## 已确认的定位

### 用户范围

Loreley 不按编程语言划分用户。当前 evaluator 使用 Python 接口接收 worktree 和返回结构化结果，但插件可以调用任意语言的构建、测试、容器、硬件 benchmark 或远程评测系统。

目标用户是满足下列条件的工程或研究团队：

- 有可自动运行的正确性门和数值目标；
- 有值得持续优化的代码库或算法系统；
- 一次有效改进具有足够高的工程或商业价值；
- 能提供与目标相称的评测算力。

### 小品文的核心 insight

真实软件仓库的程序状态空间看似无法搜索。能够构建、通过测试并带来实际改进的状态在这个空间中很稀疏。Coding agent 利用代码语义、仓库上下文和执行反馈提出有意义的跨文件修改，使搜索过程能够在这些稀疏的有效状态之间移动。

放在 AlphaEvolve 的延长线上，Loreley 的主张是：把 evaluator 驱动的程序演化扩展到真实软件仓库的尺度。Git commit 是候选的 source representation 和 ancestry 节点，完整仓库是搜索对象，搜索过程保留多条可继承的有效谱系。

这两个表述分别承担不同作用：前者解释为什么仓库级搜索可行，后者说明 Loreley 在相关工作脉络中的位置。

Zstandard 案例进一步修正了“Git commit 是候选状态”的含义：commit 是可复现的 source representation 和 ancestry 节点，不一定是独立的 evaluator-relevant 状态。V13 中 48 份 passed reports、6 个 Git trees 编译成同一个 ARM64 binary；V19 因此改用 evaluator 提供的 release-binary identity 做 archive admission、终点统计和 finalist freezing。V19 的 211 个成功 job 对应 167 个 binary，44 个成功 job 仍生成了已有 executable，其中启用 measurement cache 后的 19 个重复项直接复用已有评测。对于编译型目标，搜索空间更接近“仓库状态按 artifact 和 evaluator 行为等价关系取商”后的空间；commit、tree、artifact 和 evaluation identity 都要分别记录。

### SATLUTION 的量级口径

后续材料同时报告两个单位，不把 cycle 数当成唯一口径：

- 约 70 个仓库演化 cycle；
- 每个 cycle 包含约 400 个并行 candidate evaluation。按当前讨论中与 Loreley job 对齐的 evaluation 粒度计算，总量约为 28,000 次，因此可以表述为两三万次迭代。

材料必须在数字旁写出计数单位，避免把 cycle、代码候选、solver-instance evaluation 和完整 benchmark round 混为同一对象。

## 三个案例的作用

### `markdown-it-py`

- 64 个总 job，包括 8 个手工 seed 和 56 个模型驱动 evolution job；
- winner 在独立 28 文档验证集上加速 6.75%；
- 28 个文档全部改善；
- 适合作为当前最主要的独立验证证据。

### `python-pathspec`

- 64 个总 job，包括 6 个手工 seed 和 58 个模型驱动 evolution job；
- 最终验证 winner 在 reference workloads 上加速 25.14%；
- winner 是在最初 training pick 暴露 reference allocation 失败后确定的；
- 适合解释多代演化、archive 保留谱系和候选改进机制，不作为干净的前瞻性成功。

### Zstandard

- V19 运行 220 个物理 job，包括 8 个手工 seed 和 212 个 evolution job；211 个成功 job 对应 167 个不同 release binary；
- 预登记 winner 在 sealed holdout 上将单线程 compression throughput 提升 1.019%，95% 置信区间为 +0.962% 到 +1.076%；decompression 为 +0.010%，compressed size 不变，peak RSS 增加 0.063 MiB；
- 预登记 winner 是一个 9 行的人工 seed，不是后续 evolved candidate；因此主结果证明系统保留、排序和独立验证了有效候选，不证明 evolution 超过最强 seed；
- 保留原结论后进行的 Top-10 sensitivity follow-up 找到一个 generation-4 evolved candidate；它在新生成的 disjoint corpus 上取得 +0.891% compression，95% 置信区间为 +0.522% 到 +1.261%；
- Top-10 follow-up 属于事后扩大 finalist 数量后的独立补充，不能改写预登记 winner，也没有把两个候选放在同一 fresh corpus 上做 head-to-head；
- 适合提供 C 系统仓库、binary-aware search、分离 training/validation/holdout、评测不确定性和小幅性能改进统计测量的证据。

三个案例都只能支持固定案例中的系统能力，不能估计 Loreley 在任意仓库上的平均效果。

### 前两个案例的统一指标

以下 speedup 均为 `candidate throughput / root throughput`。例如 `1.0675x` 表示单位时间处理量增加 6.75%。若工作量固定，等价耗时下降为 `1 - 1 / speedup`，因此对应 6.33%，不能把两种百分比混用。

#### 性能与有效性

| 指标 | `markdown-it-py` | `python-pathspec` |
| --- | ---: | ---: |
| Frozen upstream revision | `97aff4f564e` | `6568072c2703` |
| Campaign | 64 jobs：8 seeds + 56 evolution | 64 jobs：6 seeds + 58 evolution |
| Terminal outcomes | 54 成功，10 失败，成功率 84.4% | 45 成功，19 失败，成功率 70.3% |
| Best manual seed，training | `1.032328x`，+3.23% throughput | `1.1227x`，+12.27% throughput |
| 最终候选，training | `1.069911x`，+6.99% | `1.2536x`，+25.36% |
| 最终候选，validation/reference | `1.067538x`，+6.75% | `1.2514x`，+25.14% |
| 固定工作量的等价耗时下降 | 6.33% | 20.09% |
| Training 到 validation/reference 的差距 | 0.237 percentage points | 约 0.21 percentage points |
| Validation/reference 覆盖 | 28 个文档，28/28 改善 | 5 个 reference scenarios，5/5 改善 |
| 单项范围 | `1.007149x` 到 `1.171532x` | `1.1550x` 到 `1.3673x` |
| Peak allocation | 3.488990 MiB，root 为 3.492437 MiB，下降 0.099% | 0.04354 MiB，低于 0.05 MiB gate |
| Correctness 与 scope | full tests、output/semantic、wheel、installed CLI/API、scope 全部通过 | 197 tests、276 skips、142 subtests、semantic、API、scope 全部通过 |
| 证据等级 | winner 在独立 28 文档 validation 揭示前冻结，达到预登记 strong outcome | 最终候选在初选 candidate 的 reference allocation 失败后确定，属于能力与机制证据 |

`markdown-it-py` 的独立 28 文档结果是当前最干净的结果。它同时满足预先冻结 winner、全部文档改善、内存不回退和 release surface 验证。

`markdown-it-py` 的 per-document speedup 最低为 `1.007149x`，raw evidence 记录的中位数为 `1.068380x`，最高为 `1.171532x`。这对应各文档 throughput 提升 0.71% 到 17.15%。

`python-pathspec` 的初选 candidate 在 training 为 `1.2633x`，在 reference 为 `1.2619x`，但 reference peak allocation 为 0.06472 MiB，超过 0.05 MiB gate。最终采用的 candidate 在 training 为 `1.2536x`、0.02942 MiB，在 reference 为 `1.2514x`、0.04354 MiB。最终 candidate 的性能和正确性结果成立，但 reference 已经揭示，因此不能将它表述为预登记的确认性成功。

| `python-pathspec` reference scenario | Final candidate speedup |
| --- | ---: |
| Compile 150 gitignore patterns | `1.3673x` |
| `GitIgnoreSpec` match，150 patterns | `1.2796x` |
| `PathSpec` match，150 patterns | `1.2384x` |
| `PathSpec` match，2 patterns | `1.1550x` |
| `PathSpec` match，40 patterns | `1.2265x` |
| Geometric mean | `1.2514x` |

当前两篇精简 case-study 都没有报告 primary speedup 的置信区间或搜索重复之间的方差。训练与 validation/reference 的点估计接近，说明候选没有出现明显的 aggregate generalization gap，但不能替代置信区间，也不能估计搜索方法的成功概率。

#### 搜索过程

| 指标 | `markdown-it-py` | `python-pathspec` |
| --- | ---: | ---: |
| Winner | job 26，generation 4 | job 38，generation 4 |
| Commit | `b10adb6fad0d` | `9d977f0a73d5` |
| 相对 best seed 的 training throughput 增量 | 3.64% | 11.66% |
| Final diff | 5 files，+54/-14 | 5 files，+127/-51 |
| Archive | 36 entries，18 island/cell coordinates | 28 entries，19 island/cell coordinates |
| 谱系证据 | 汇入 seeds 3、4、8 的 inspiration，形成四代组合优化 | generation-3 branch 在 archive 保留 20 个其他 job 后再次被采样并产生 winner |

这些数据证明最终候选明显强于最好的人工 seed，并展示 archive 保留和重访谱系的机制。两个实验都没有运行同预算的 root-independent 和 champion-sequential 模型 arm，因此不能用这两组数据单独证明 quality-diversity 优于简单搜索。

#### 花费与运行效率

| 指标 | `markdown-it-py` | `python-pathspec` |
| --- | ---: | ---: |
| 实验报告记录的 generation cost | $2.0833 | $2.4856 |
| 每个 evolution job 的平均 generation cost | $0.0372 | $0.0429 |
| 每个 scheduled job 的平均 generation cost | $0.0326 | $0.0388 |
| Embedding cash cost | 报告未提供 | 报告未提供 |
| Host 与人工成本 | 报告未货币化 | 报告未货币化 |
| Campaign wall time | 4.35 h | 3.91 h |
| Terminal throughput | 14.73 jobs/h | 16.4 jobs/h |
| Median job time | 10.02 min | 9.98 min |
| Generation requests | 3,792 | 3,977 |
| Generation tokens | 215,349,501 | 241,634,477 |
| Embedding tokens | 199,343 | 258,055 |

合计为 128 jobs，其中 14 个 manual seeds、114 个 evolution jobs；99 成功、29 失败；生成使用 7,769 requests 和 456,983,978 tokens；embedding 使用 457,398 tokens；两次 campaign wall-time 相加为 8.26 小时；实验报告记录的 generation cost 合计为 $4.5689，平均每个 evolution job 为 $0.0401，平均每个 scheduled job 为 $0.0357。两个仓库的 speedup 不能求算术平均，因为 workload、基线和指标分布不同。

当前可以直接统计的现金花费只有 generation cost。Embedding、主机、开发和人工 seed 没有报告金额，因此 `$4.5689` 不能写成两个实验的 all-in cost。

#### 发布前口径检查

1. 已在主报告解决：`python-pathspec` 的旧 evidence report 仍保留 `$105.57` proxy ledger，其中 `$105.00` 来自 42 个 job 各 `$2.50` 的 reservation fallback。对外链接的 final case-study 已使用 `$2.4856` generation cost；统一证据报告也注明它不是 all-in cost。
2. 仍需修正：`markdown-it-py` 的 raw evidence report 记录 per-document median 为 `1.068380x`，当前精简 case-study 表格写成 `1.0675x`。前者来自 raw container report；应把精简表格的 median 改为 `1.068380x`，不要把 geometric mean 当成 median。
3. 已统一叙事：`python-pathspec` 的旧 evidence report 将预登记 primary outcome 标为 invalid，final case-study 把后续通过全部 gate 的候选称为 final winner。公开口径为：“最终有效候选取得 25.14% reference throughput 提升，但它是在初选 candidate 的 reference allocation 失败并揭示 reference 后确定的，因此支持系统能力和演化机制，不作为干净的前瞻性成功。”

#### 当前可用的结论

- `markdown-it-py` 支持一条独立验证的定量 claim：在预登记 64-job 案例中，最终候选在独立 28 文档 corpus 上 throughput 提升 6.75%，28/28 文档改善，并通过 correctness、release、scope 和 allocation checks。
- `python-pathspec` 支持一条带选择过程限定的定量 claim：在 64-job 案例中，generation-4 最终有效候选在 disjoint reference workloads 上 throughput 提升 25.14%，5/5 scenarios 改善并通过所有 gates；该候选在初选结果揭示后确定。
- 两个案例共同支持“coding agent 可以沿多代 Git 谱系把人工 seed 演化成更强的仓库状态”。它们不支持跨仓库平均成功率、无人类 seed 的效果、upstream 可接受性或 quality-diversity 相对简单搜索的因果优势。

## Zstandard 第三个案例：V19 正式结果（2026-08-07）

正式报告见 [2026-08-07-zstandard-gpt-v19-case-study-report.md](2026-08-07-zstandard-gpt-v19-case-study-report.md)，Top-10 sensitivity 与 fresh confirmation 见 [2026-08-07-zstandard-gpt-v19-top10-validation-supplement.md](2026-08-07-zstandard-gpt-v19-top10-validation-supplement.md)。V19 使用 Zstandard upstream `82d322c4973d9e2968d94047a40892bc6d9a9bdf`，实验 root 为 `5b3fe474e4df572a7588be7abf3d8b6bd4b6010e`，只覆盖一台 Apple-silicon host 上的单线程 levels 1、3、5。

### 预登记主结果

以下 ratio 均为 `candidate throughput / root throughput`。

| Holdout 指标 | Ratio / delta | 95% 置信区间 |
| --- | ---: | ---: |
| Compression throughput | `1.01019x`，+1.019% | `1.00962–1.01076x`，+0.962% 到 +1.076% |
| Decompression throughput | `1.00010x`，+0.010% | `0.99890–1.00130x`，-0.110% 到 +0.130% |
| Combined throughput | `1.00513x`，+0.513% | `1.00441–1.00585x`，+0.441% 到 +0.585% |
| Worst measured cell | `0.99817x`，-0.183% | 未单独报告 |
| Maximum compressed-size ratio | `1.00000x` | 不适用 |
| Peak RSS delta | +0.063 MiB | 不适用 |

Holdout 使用 12 对交错 root/candidate 测量。Compression 和 combined throughput 的置信区间完全高于 `1.0x`；decompression 的区间跨过 `1.0x`，因此公开表述为中性。候选通过 upstream checks、release build、root/candidate 双向解码、compressed-size 和 RSS gates，conclusion audit 通过。

预登记的 strong rule 要求 compression point estimate 至少为 `1.02x`，且 lower 95% 至少为 `1.01x`。实际 point estimate 为 `1.01019x`，所以正式 outcome 是 modest-positive。固定工作量的等价压缩耗时下降约 1.01%，不能写成 1.019% latency reduction。

预登记 winner 为 commit `7b9aef38ecd4`、release binary `e7e9ef6b060f…`。它是人工 seed 5，在 `lib/compress/hist.c` 中把 scalar histogram update loop 按四字节展开，相对 root 为 8 insertions 和 1 deletion。它没有 corpus-specific logic、格式改动、新依赖或大范围重写。

### Search、identity 和运行效率

| 指标 | 结果 |
| --- | ---: |
| 物理 terminal jobs | 220：8 seeds + 212 evolution |
| 成功 / 失败 | 211 / 9，成功率 95.9% |
| 不同成功 release binaries | 167，占成功 job 的 79.1% |
| 生成已有 binary 的成功 job | 44，占成功 job 的 20.9% |
| Measurement cache 复用 | 19 个 job；启用前另有 25 个重复 binary 被重新测量 |
| 最终 archive | 13 entries，11/64 coordinates occupied |
| 前 128 jobs | 125 成功、3 失败、102 个 unique binaries；3.19 runner-hours |
| Efficient continuation | 92 个物理 job 新增 65 个 unique binaries；2.12 runner-hours |
| 完整 search | 5.31 runner-hours；41.4 physical jobs/hour |
| 真实评测的 median evaluator time | 186.7 s |
| Median end-to-end job time | 333.6 s |
| Cache reuse 的 median evaluator time | 21.6 s |

四条 evaluator lanes 先用 root/root 实验校准，并交错 root-first 与 candidate-first 顺序。四 lane 相对单 lane 提高约 4 倍 wall-clock throughput，aggregate root/root point 为 `0.99993x`，95% 区间为 `0.99705–1.00282x`。这证明该 host 上的四 lane 配置没有观察到足以解释 1% holdout gain 的系统性配对偏差；它不证明所有机器都适合四路并发。

9 个失败包括 4 个 upstream 或 candidate test failures、4 个 measurement-precision rejections，以及 1 个 compressed-size gate failure。失败和 precision-invalid 结果没有进入 archive 或 measurement cache。

V19 把 commit、Git tree 和 evaluator identity 分开记录。Archive 和 finalist freezing 按 release-binary identity 去重，因此 V13 中“同一 executable 获得多个 finalist slots”的问题没有重现。搜索仍会生成 evaluator-equivalent source states；44/211 的重复率说明去重能够节省评测预算，但不能替代 proposal diversity 的改进。

### Top-10 sensitivity 与 evolved candidate

预登记协议只验证 training Top 3，最终选中人工 seed `7b9aef38`。保留该结论后，实验又将固定 training 排名扩展到 Top 10：复用前三份 validation report，另外对 ranks 4–10 做 8-round validation。7 次新增 validation 共用 2,348 秒，约 39 分钟，不产生新模型调用。

Training rank 10 的 generation-4 candidate `fe39bee8` 在 validation 上成为新 winner：compression point 为 `1.01234x`，lower 95% 为 `1.01156x`。在测量它之前，另行冻结了一个新生成、与 training/validation/原 holdout 都不重叠的 16 MiB corpus 和 12-round confirmation protocol。结果为：

| Fresh-confirmation 指标 | Ratio / delta | 95% 置信区间 |
| --- | ---: | ---: |
| Compression throughput | `1.00891x`，+0.891% | `1.00522–1.01261x`，+0.522% 到 +1.261% |
| Decompression throughput | `0.99830x`，-0.170% | `0.99471–1.00191x`，-0.529% 到 +0.191% |
| Combined throughput | `1.00359x`，+0.359% | `1.00004–1.00716x`，+0.004% 到 +0.716% |
| Maximum compressed-size ratio | `1.00000x` | 不适用 |
| Peak RSS delta | +0.031 MiB | 不适用 |

`fe39bee8` 的四代谱系组合了 zero-literal fast path、compression hot-path 修改和八字节 histogram unroll，相对 root 改动 3 个文件、+33/-16。它证明 V19 的 evolution 产生了一个在新鲜语料上仍为正的候选，但不证明它优于人工 seed：两者没有在同一新语料上做冻结后的 head-to-head。

Top-10 分析还说明 training 的细粒度排序不稳定。Top-10 的 training compression lower bounds 只相差 0.276 percentage points，而 training estimate 的不确定性大于这个差距；validation winner 因此从 training rank 10 产生。未来 finalist 规则至少需要覆盖 Top 10，或者采用预先冻结的 effect band 和上限。这个方法问题进入论文轨道，不阻塞当前推广。

### Usage 和成本

V19 记录 52,653,004 tokens，包含 cached input 和 embeddings。424 个 Kilo generation sessions 的 catalog cost 覆盖完整：planning 为 $57.8499，coding 为 $2.3973，合计 $60.2472。这个数字是 Kilo model-catalog estimate，不是 provider-billed spend；303 个 embedding events 有 token 记录但没有价格，因此不能称为 all-provider 或 all-in cost。

V19 与前两个 DeepSeek 案例的美元记录语义不同。`markdown-it-py` 的 $2.0833 和 `python-pathspec` 的 $2.4856 是 provider-recorded generation cost，V19 的 $60.2472 是 catalog estimate，三者不能直接相加成项目总现金花费。三案例的统一成本表以 [2026-08-07-loreley-case-study-evidence-report.md](2026-08-07-loreley-case-study-evidence-report.md) 为准。

正式 Top-3 validation 共约 16.7 分钟，registered holdout 为 8.1 分钟；Top-10 新增 validation 为 39.1 分钟，fresh confirmation 为 8.1 分钟。后两项只使用本机评测，不产生模型 token。

### 在推广材料中的角色

Zstandard V19 不作为“最大提升”或“evolution 击败人工”的 headline。它承担四项证据职责：

- 证明 evaluator 可以接入非 Python 的成熟 C 系统仓库；
- 展示约 1% 改进所需的 paired measurement、置信区间、sealed data 和 correctness gates；
- 展示 commit/tree/binary identity 分层与 measurement reuse 的必要性；
- 用 Top-10 sensitivity 暴露训练排序和 finalist breadth 的限制，而不是只报告最好看的数字。

公开主句可以写成：“在一个固定 Zstandard revision 和一台 Apple-silicon host 上，Loreley 从 167 个不同 release binaries 中按预登记协议选出一个 9 行候选；它在 sealed holdout 上将单线程 compression throughput 提高 1.019%，95% 置信区间为 +0.962% 到 +1.076%，decompression 中性且 compressed size 不变。”

补充句可以写成：“保留原结论后的 Top-10 sensitivity follow-up 还找到一个 generation-4 candidate；它在另一个预先 sealed 的 fresh corpus 上取得 +0.891% compression，95% 置信区间为 +0.522% 到 +1.261%。”补充句必须保留 post-hoc shortlist expansion 和不同 corpus 的限定。

不能公开表述为：2% 改进、跨平台 Zstandard 加速、evolution 超过最强 seed、quality-diversity 优于简单搜索、upstream 可接受，或 220 个 job 都对应不同程序行为。

### 与旧版 V13 的关系

| 项目 | V13 | V19 |
| --- | --- | --- |
| 对外状态 | 历史基础设施证据 | 当前 Zstandard 主案例 |
| Holdout compression | +1.044%，95% CI +0.880% 到 +1.209% | +1.019%，95% CI +0.962% 到 +1.076% |
| 候选 identity | 3 个 source finalists 只有 2 个 binaries | 167 个 unique binaries；finalists 按 binary 冻结 |
| Search | 256 logical jobs，55 个 unique binaries | 220 physical jobs，167 个 unique binaries |
| Active runner time | 15.72 h | 5.31 h |
| 模型成本记录 | $4.8277 provider-reported DeepSeek generation | $60.2472 Kilo catalog estimate，embedding 未定价 |

两个 holdout 点估计只相差 0.025 percentage points，但使用不同 fresh corpus、模型路由和评测执行条件，不能把差值解释为 V19 退步或持平的精确估计。推广时只报告 V19；V13 用于解释 sampler restart、measurement isolation 和 binary identity 修复从何而来。

## Zstandard V13 历史记录（已由 V19 取代）

以下记录保留 2026-08-06 的审计过程，避免丢失为什么要重跑 V19 的决策依据。对外数字、三案例汇总和 README 均不再使用 V13。

历史报告见 [2026-08-06-zstandard-reliable-qd-case-study-report.md](2026-08-06-zstandard-reliable-qd-case-study-report.md)。结果对应 Zstandard `82d322c4973d9e2968d94047a40892bc6d9a9bdf` 上的一次 256-logical-job quality-diversity campaign，holdout-selected source commit 为 `893bf5cf9e02a703ff530116ce990c3f4dae6ad6`。后续审计发现该 commit、frozen finalist `c1f1852b` 和更早的 `817af317` 生成相同的 ARM64 executable；完整 campaign 中共有 48 份 passed reports、6 个 Git trees 对应该 binary。历史报告已经按该审计修正 winner lineage，并在开头标记由 V19 取代。

### 主要结果

以下 ratio 均为 `candidate throughput / root throughput`。Compression throughput 的 `1.010442x` 表示吞吐提高 1.044%；固定工作量的等价耗时下降约 1.033%。

| Holdout 指标 | Ratio / delta | 95% 置信区间 |
| --- | ---: | ---: |
| Compression throughput，levels 1/3/5 geometric mean | `1.010442x`，+1.044% | `1.008802–1.012085x`，+0.880% 到 +1.209% |
| Decompression throughput，levels 1/3/5 geometric mean | `0.999817x`，-0.018% | `0.998067–1.001570x`，-0.193% 到 +0.157% |
| Compression 与 decompression combined throughput | `1.005116x`，+0.512% | `1.003455–1.006780x`，+0.345% 到 +0.678% |
| 最差单项，level 1 decompression | `0.997747x`，-0.225% | 未单独报告 |
| Levels 1/3/5 compressed-size ratio | 均为 `1.000000x` | 不适用 |
| Peak RSS delta | +0.03125 MiB | 不适用 |

| Compression level | Compression throughput | Decompression throughput |
| ---: | ---: | ---: |
| 1 | +1.018% | -0.225% |
| 3 | +0.851% | +0.112% |
| 5 | +1.264% | +0.059% |

三个 level 的 compression point estimate 均为正；decompression 的主要回退集中在 level 1。报告没有为单个 cell 给出置信区间，因此只能对跨 levels 的 aggregate 作统计显著性表述。

Holdout 使用 12 对 root/candidate 交错测量，每个 benchmark 至少运行 3 秒。Compression 和 combined throughput 的置信区间均完全高于 `1.0x`，因此结果不能用纯测量噪声解释。Decompression 的置信区间跨过 `1.0x`，应表述为基本中性，不能宣称有改善。

这些置信区间只量化一台 host、一个 frozen binary、一个 holdout corpus 上 12 对测量的 sampling uncertainty；它们不包含重新运行搜索的方差、不同机器或架构的方差，也不估计 Loreley 找到该结果的成功概率。

预登记的 strong outcome 同时要求 compression point estimate 至少提高 2%，且 95% 置信区间下界至少提高 1%。实际 point estimate 为 1.044%，下界为 0.880%，因此正式结论是 modest-positive，不能简写成 strong success。

### 评测和选择协议

- Generation 使用 Kilo 上的 DeepSeek v4 Flash，code embedding 使用 OpenAI `text-embedding-3-small` 1,536 维；运行 4 个 agent workers、1 条 evaluator lane，最多 4 个 unfinished jobs；
- 搜索使用 3D `4×4×4` archive，每个 cell 保留 Pareto front，并从 7 个独立人工 seed directions 开始；
- Training、validation 和 holdout 各使用互不重叠的 sealed corpus；每个 split 含 8 个 2 MiB 文件，覆盖源代码、records、自然文本和 structured binary，共 16 MiB；
- 性能评测为单线程 compression/decompression levels 1、3、5；
- Agent 可以查看和修改 C 源码，但不能运行私有 build 或性能工具；build、upstream checks、cross-version decode、scope 和 benchmark 都由 evaluator 执行；
- 256 个逻辑 job 结束后，按 training compression lower 95% 冻结 3 个不同 Git tree；只在 validation 上比较这 3 个 finalist；
- 按冻结规则选出 `893bf5cf` 后，只将该 source commit 运行一次 holdout 流程，没有在 holdout 揭示后测试 runner-up；
- 模型调用在 hidden data 揭示前关闭；终点、排除项、三名 finalist、一个 validation winner 和一个 holdout target 均通过 conclusion audit。

三名 finalist 的排序变化如下：

| Candidate | ARM64 binary SHA-256 | Training compression | Validation compression | Validation lower 95% |
| --- | --- | ---: | ---: | ---: |
| `c1f1852b` | `2cddc94…` | +1.672% | +0.961% | +0.374% |
| `e1c2cfad` | `5e7d2d5…` | +1.477% | +1.024% | +0.923% |
| `893bf5cf` | `2cddc94…` | +1.141% | +1.133% | +1.013% |

Validation 为 `c1f1852b` 和 `893bf5cf` 分别构建的文件均为 ARM64 Mach-O，SHA-256 同为 `2cddc94eb7cbc1650c99237935e0bca7ae6416f61528f38647b5e0f2f2e0d391`，`cmp` 逐字节比较也一致。`893bf5cf` 相对 `c1f1852b` 的唯一修改位于 `#else /* !defined(__aarch64__) */` 分支，因此没有进入 M4 上的 binary。

两次 validation 给同一 executable 测得 +0.961% 和 +1.133%，差异来自分开运行的测量，而不是 source edit。原报告中“training 第三名比 training leader 泛化更好”的解释在当前 host 上不成立：三个 source finalist 实际只代表两个 executable，且同一 executable 获得了两次 validation 入场机会。Holdout 的 +1.044% 仍是一个独立、有效的 binary-level 结果，但 final source-commit selection 暴露了缺少 binary deduplication 的协议问题。

### 搜索过程和 source/binary 区分

| 指标 | 结果 |
| --- | ---: |
| 逻辑 campaign | 256 jobs：7 seeds + 249 evolution |
| 物理 terminal rows | 274，其中 18 个按冻结规则排除 |
| Passed candidate evaluations | 205，逻辑 job 的 80.08% |
| Archive-decision-qualified | 208，逻辑 job 的 81.25% |
| 失败或未形成有效评测 | 34 measurement-precision invalid；3 test failures；14 未到 evaluation |
| 可行候选多样性 | 205 个 commit；100 个 Git tree；55 个 binary |
| 重复率 | Tree 51.22%；binary 73.17% |
| 最终 archive | 12 source/tree entries；10 unique binaries；10/64 coordinates occupied |
| Holdout binary 的 passed training reports | 48 份，来自 6 个 Git trees |
| Effective ARM64 binary 首次出现 | `817af317`，按 terminal completion order 为第 71 个逻辑 job |
| 同 binary 的 frozen finalist | `c1f1852b`，第 118 个逻辑 job |
| Holdout-selected source commit 出现 | `893bf5cf`，第 245 个逻辑 job |
| 最早等价 source diff | `817af317` 相对 root：4 个 compression C 文件，+39/-33 |
| 较小的 frozen-finalist diff | `c1f1852b` 相对 root：4 个 compression C 文件，+33/-30 |
| Selected source diff | `893bf5cf` 相对 root：5 个 C 文件，+55/-40；新增部分未进入 ARM64 binary |

按 selected source ancestry，`893bf5cf` 是从 seed 算起的 generation 3 候选；前三步完成 fast hash-table filling 特化、跳过 zero-frequency entropy-cost 工作和合并 compression hot-path 修改。但 holdout binary 不是该谱系独有：`817af317` 所在的另一分支已在第 71 个逻辑 completion 产生相同 codegen，随后共有 6 个不同 Git trees 收敛到该 binary。

Generation-3 的最后一步以 `c1f1852b` 为 parent，并从 `e1c2cfad` 和 `74492608` 获取 inspiration，但其新增 decoder edit 只修改 non-AArch64 分支。它证明 archive inspiration 影响了后续 source proposal，不能证明该 inspiration 改善了本次评测的 executable。

前 128 个 job 已在第 71 个逻辑 completion 找到 holdout 所测 binary，并在第 118 个 completion 产生较小且进入 frozen finalists 的等价 source tree `c1f1852b`。后 128 个 job 没有提高 training 榜首，也没有产生最终使用的不同 ARM64 binary；它把 unique Git trees 从 51 增加到 100、unique binaries 从 33 增加到 55、archive entries 从 11 增加到 12、occupied coordinates 从 9 增加到 10。扩展阶段增加了搜索覆盖，但其 $2.6407 成本不能归因于最终 holdout 改进。

第 64 个逻辑 job 时尚未出现 SHA `2cddc94…`，第 71 个 completion 首次出现，第 128 个 checkpoint 已包含它。对于本次单次 run，64-job 预算会漏掉该 executable，128-job 预算足够找到它，256-job 预算没有改变最终 measured binary。这是一次 campaign trajectory 的事后事实，不能外推为其他仓库的通用预算规律。

205 个 passed candidate evaluations 只对应 100 个 Git tree 和 55 个 binary；tree duplicate rate 为 51.22%，binary duplicate rate 为 73.17%。其中 48 份 passed training reports，占全部 passed evaluations 的 23.4%，来自 6 个 Git trees，却对应同一个 holdout binary。它们的 training compression point estimate 从 +0.887% 到 +1.672%，中位数为 +1.068%，算术均值为 +1.083%，lower 95% 从 -0.101% 到 +1.277%。`c1f1852b` 的 +1.672% 正是这 48 次同 binary 测量中的最大值，因此 training leader 的 selection bias 可以直接观察到；这些事后聚合值只能用于诊断，不能替代预登记结果。

该现象与“有效 executable 状态稀疏”的项目 insight 相容，但不能单独证明该命题。它直接暴露出两个工程问题：archive admission、finalist freezing 和预算统计应优先按 binary 或 evaluator-relevant state 去重；training 排名也应聚合同一 binary 的重复测量，不能让不同 source wrapper 各自参与极值选择。

最终 archive 的 12 个 entries 对应 12 个 Git trees，但只有 10 个 binary。`c1f1852b` 与 `086aa1d4` 作为同一 binary 的两个 Pareto entries 共处 cell 22；`e1c2cfad` 与 `1ee2e0e3` 生成同一 binary，却分别占据 cells 24 和 25。这证明 source embedding 可以把 evaluator-equivalent 状态视为不同 behavior，并使 archive occupancy 高估 executable diversity。

本次只运行了 quality-diversity arm，没有运行同预算的 champion-sequential 或 root-independent arm。谱系和跨分支 inspiration 证明机制实际发生过，但不能据此得出 quality-diversity 优于简单搜索的因果结论。

### 花费、时间和运行事故

| 指标 | 结果 |
| --- | ---: |
| Kilo / DeepSeek generation cost | $4.8277；493 个 token-bearing events 成本覆盖完整 |
| 64-job checkpoint generation cost | $0.9160；此时尚未出现 holdout binary |
| 128-job checkpoint generation cost | $2.1870 |
| 每个逻辑 job 的平均 generation cost | $0.0189 |
| 每个 evolution job 的平均 generation cost | $0.0194 |
| 每个 unique Git tree 的平均 generation cost | $0.0483 |
| 每个 unique binary 的平均 generation cost | $0.0878 |
| Observed tokens | 286,838,265，包含 generation 与 embedding usage |
| Kilo reasoning output | 4,478,202 tokens；正式报告的展示表暂未单列 |
| OpenAI embedding usage | 126 events；1,167,501 tokens；报告未提供价格 |
| Active runner sessions | 合计 15.72 h，分布在 3 次 session |
| 前 64 jobs 的 active runner time | 3.23 h |
| 前 128 jobs 的 active runner time | 4.53 h，合计 2 次 runner sessions |
| 前 128 jobs 的 database elapsed time | 15.47 h，从首个 job scheduled 到第 128 个逻辑 job completed，包含暂停和恢复 |
| First-to-last database window | 31.1 h，包含暂停、事故分析和恢复 |
| Median / p90 job row lifetime | 17.9 / 27.7 min，包含排队 |
| 三名 finalist 的 validation | 合计 16.8 min |
| 单一 selected binary 的 holdout | 8.1 min |

128-job checkpoint 记录 $2.1870 generation cost 和 143,323,908 observed tokens。扩展到 256 个逻辑 job 后，新增 $2.6407 generation cost 和 143,514,357 observed tokens。三个案例合计 384 个逻辑 job、304 个成功评测和 80 个失败 job；实验报告记录的 generation cost 合计为 $9.3966。Embedding、主机、开发和人工 seed 仍未全部货币化，因此该数字不是 all-in cost。

运行中出现三类需要保留的负面证据：

1. Codex app restart 使继承的 standard file descriptors 失效，18 个物理 rows 被按预先冻结的 lifecycle exclusion 规则排除；
2. macOS `launchd` 的 background process policy 使 fixed-root throughput 降低 4.4–6.4 倍，并在修复前造成 11 个 extension precision failures；切换为 interactive policy 并加入 shared-agent/exclusive-measurement lock 后，extension 不再出现 precision failure；
3. Post-reveal audit 首次运行暴露了 audit tool 对预期 plaintext 的误报；失败记录和修复被保留，修正后的 audit 通过，selection 与 holdout 没有改变。

这些事故使 31.1 小时 wall-clock window 不能直接当作 evaluator 的固有耗时。它们同时说明高精度系统 benchmark 需要把进程调度状态、评测独占锁、失败分类和 prospective exclusions 纳入实验协议。

### V13 当时的证据角色（已废止）

| 案例 | 主要定量结果 | 选择与验证状态 | 当前证据角色 |
| --- | --- | --- | --- |
| `markdown-it-py` | 独立 28 文档 validation +6.75%，28/28 改善 | Winner 在 validation 揭示前冻结；未报告 CI | 当前最大且干净的前瞻性 Python 案例 |
| `python-pathspec` | Reference workloads +25.14%，5/5 改善 | 最终候选在初选 reference allocation failure 揭示后确定；未报告 CI | 能力、谱系和 archive 重访机制案例 |
| Zstandard V13 | Sealed holdout compression +1.044%，95% CI +0.880% 到 +1.209% | 三阶段 sealed split；只评测一个 holdout target；3 个 source finalists 只有 2 个 unique binaries | 历史 C 系统仓库结果；已由 V19 取代 |

V13 当时把语言无关 evaluator、C 系统 hot path、sealed holdout 和约 1% 小改进的可靠测量放进同一个案例。它提供的长期价值是失败分析：Git tree 不等于 evaluator-relevant program state，source-level diversity 可以在编译后消失。当前三案例证据角色以 V19 小节和统一证据报告为准。

### V13 的历史 claim boundary

V13 历史报告可以表述：在一个固定 Zstandard revision 和一台 Apple-silicon host 上，campaign 在第 71 个逻辑 completion 产生了一个 format-preserving 的 ARM64 binary；该 binary 在 sealed holdout 上将单线程 compression throughput 提高约 1%，decompression 基本中性。完整 256-job campaign 的 provider-reported generation cost 低于 $5。该段不进入当前推广主文。

V13 不能表述：generation-3 edit 带来了本次提升、后 128 个 job 改善了最终 executable、跨分支 inspiration 改善了本次结果、普遍的 Zstandard 加速、2% 改进、跨架构成立、upstream 可接受性，或 quality-diversity 相对简单搜索的优势。

V13 不再进入推广文章的三案例主表。它可以在技术附注中解释 candidate identity 修复；论文若研究 cross-architecture equivalence，可以在 x86-64 host 上按预先冻结的顺序重放 root、`817af317`、`c1f1852b` 和 `893bf5cf`，测试这些 ARM64-equivalent source trees 是否产生不同 binary 和性能。

不应重新打开 holdout 来比较 `817af317`、`c1f1852b` 与 `893bf5cf`：当前 host 上三者生成同一个 executable，已有 holdout 数据已经适用于该 binary。修订应在不追加 holdout selection 的前提下，把 canonical measured artifact 记为 binary SHA-256 `2cddc94…`；同时分别记录最早 observed source commit `817af317`、较小的 frozen-finalist source `c1f1852b` 和按原协议选择的 holdout source `893bf5cf`。

#### V13 已完成的历史勘误

1. `Winner and lineage` 应区分 holdout-selected source commit `893bf5cf`、frozen finalist `c1f1852b` 和最早 observed equivalent source `817af317`；相同 ARM64 binary 在第 71 个逻辑 completion 已出现。
2. Finalist 统计应写成 3 个 unique Git trees、2 个 unique ARM64 binaries；解释 validation 排名时不能把相同 binary 的两次测量差异归因于 source edit 或泛化。
3. `893bf5cf` 的最后一代 diff 只影响 non-AArch64 分支；删除“该 edit 改善了 Apple-silicon 结果”和“最后一次跨分支 inspiration 对 measured result 有贡献”的暗示。
4. Holdout 数字、置信区间、compressed-size、RSS、correctness、成本和 sealed-data 结论不受 binary equivalence 影响，可以保留。
5. 后 128 个 job 的资源可以计入完整 campaign ledger，但不能作为发现 holdout binary 的必要成本；该 binary 在第 71 个逻辑 completion 已出现。
6. Usage 表应增加 4,478,202 个 `reasoning_output_tokens`。当前表中的 input、cached input、output 和 embedding 合计为 282,360,063，与 286,838,265 total observed tokens 相差的部分正是 reasoning output；总 token 和 $4.8277 cost 本身不需要修改。

## 技术脉络研究要求

现有项目文档不能作为相关工作综述的权威来源。技术脉络需要从一手论文、官方研究页面和正式开源实现重新建立，至少覆盖 2025-08 至 2026-08 的新工作。

需要回答：

1. 从 FunSearch、AlphaEvolve 到最近系统，搜索对象、候选粒度和 evaluator 如何变化；
2. 最近一年在样本效率、搜索策略、仓库尺度、运行时反馈、基线比较和 trace 分析方面有哪些结果；
3. 哪些工作已经覆盖 MAP-Elites、islands、完整仓库或企业代码优化；
4. Loreley 可以主张的研究差异是什么，哪些旧表述需要撤回；
5. 最近的反面证据对论文实验设计提出了什么最低要求。

## 2025–2026 技术脉络初稿

以下按研究问题组织。时间和结论以一手论文或官方实现为准。

### 搜索对象从局部代码扩展到完整工程

| 工作 | 搜索对象与评测方式 | 对 Loreley 的影响 |
| --- | --- | --- |
| [FunSearch](https://www.nature.com/articles/s41586-023-06924-6)（Nature 2023） | 在人类提供的程序骨架中演化单个 Python 函数，用可执行 evaluator 筛选和积累候选 | 建立“LLM 生成程序加自动评测加进化选择”的起点，搜索对象仍是受控的局部函数 |
| [AlphaEvolve](https://arxiv.org/abs/2506.13131)（2025-06） | 从 FunSearch 的局部函数扩展到整文件、数百行程序和任意语言；以自动 evaluator 进行多指标、可能昂贵的评测 | 是最直接的思想起点，但“超越单函数”和“支持任意语言”都不能作为 Loreley 独有贡献 |
| [EvoGit](https://arxiv.org/abs/2506.02049)（2025-06） | 用 Git 分支、提交、合并组织多个 coding agent 和版本谱系；案例包括从零构建 Web 应用和 bin-packing meta-solver | Git commit、分支谱系和多 agent 演化已有直接先例 |
| [SATLUTION](https://arxiv.org/abs/2509.07367)（2025-09） | 演化完整 C/C++ SAT solver 仓库；约 70 个 cycle，每个 cycle 并行评测约 400 个 candidate | 证明高算力、仓库级、工业性能目标可以运行；量级应同时报告 cycle 与约 28,000 个、粒度接近 Loreley job 的 candidate evaluations |
| [ABCEvo](https://arxiv.org/abs/2604.15082)（DAC 2026） | 三类 agent 修改约 120 万行、4,000 多文件的 ABC 仓库；完整集成编译、八种 flow、多个 benchmark suite 和形式等价检查 | 已覆盖百万行完整仓库和严格 EDA evaluator，Loreley 不能声称首次或最大规模仓库演化 |
| [CodeEvolve](https://arxiv.org/abs/2605.04677)（2026-05） | 面向企业 Java/Apex，结合运行时 profiling、component graph、MCTS、构建测试和性能评测 | “企业代码”“跨语言”“运行时反馈”已有直接相关工作 |
| [HORIZON](https://arxiv.org/abs/2606.28279)（2026-06） | 把 Markdown harness 展开成完整硬件工程；在隔离 Git worktree 中生成、编译、验证并保存 accepted commit trace | Git worktree、仓库契约、evaluator 和提交轨迹均与 Loreley 高度重叠，不能单独构成新颖性 |

### 搜索方法从单一 champion 扩展到 archive 和开放式谱系

| 工作 | 方法变化 | 对 Loreley 的影响 |
| --- | --- | --- |
| [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954)（ICLR 2026） | 让 coding agent 修改自身代码，并维护由多条改进路径组成的 agent archive | 支持“多样化 stepping stones 比单一 champion 更重要”的研究问题，但其对象是 agent 自身而非任意目标仓库 |
| [ShinkaEvolve](https://openreview.net/forum?id=lKEdGCoDNC)（ICLR 2026） | 自适应选择 parent、模型和 prompt，并用 novelty rejection 提高样本效率 | Loreley 需要证明固定或学习到的 quality-diversity 机制相对于更简单 adaptive sampling 的增益 |
| [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve) | 开源 islands、MAP-Elites 和多指标程序演化实现 | islands 与 MAP-Elites 本身不是论文贡献；贡献必须落到仓库状态表示、行为描述符、调度或实证结果 |
| [Evolutionary Ensemble of Agents](https://arxiv.org/abs/2605.09018)（2026-05） | 联合演化代码解法和 agent guidance/skills，并按边际贡献调节 agent 组合 | 提示论文可以区分“搜索目标仓库”和“搜索 agent 策略”，当前 Loreley 主要解决前者 |

### Evaluator 从评分函数扩展到验证流水线

FunSearch 的 evaluator 主要对局部程序返回任务分数。AlphaEvolve 支持多指标和昂贵外部计算。SATLUTION、ABCEvo 和 HORIZON 进一步把完整构建、benchmark suite、形式验证、scope policy 和提交验收组合成流水线。仓库级系统的 evaluator 因而需要同时承担四项职责：过滤无效状态、计算优化目标、限制可修改范围、隔离训练反馈与最终验证。

Loreley 的 Python evaluator 接口只是调度入口。论文应把它表述为可调用任意构建、容器、硬件或远程评测系统的验证协议，并在第三案例中展示多阶段 gate、隐藏配置和 sealed holdout。

### 最近一年出现的反面证据和实验约束

- [Simple Baselines are Competitive with Code Evolution](https://arxiv.org/abs/2602.16805)（2026-02）显示，独立采样或顺序改写等简单方法在若干任务上可以匹配或超过复杂演化框架。Loreley 论文必须做同预算的 root-independent 和 champion-sequential 对照。
- [What Do Evolutionary Coding Agents Evolve?](https://arxiv.org/abs/2605.20086)（2026-05）指出，分数提升可能来自参数调整、重组、过拟合或重复引入已有代码。论文需要发布 ancestry、edit taxonomy、replay 和 holdout 结果，不能只展示 winner score。
- [HORIZON](https://arxiv.org/abs/2606.28279)明确观察到 reward hacking 和 over-solving 风险。第三案例应把可见 training feedback、隐藏或随机化 validation、最终 sealed holdout 分开。
- [Barbarians at the Gate: How AI is Upending Systems Research](https://arxiv.org/abs/2510.06189)（2025-10）把自动性能验证视为系统研究适合 agent-driven search 的关键条件，并覆盖负载均衡、MoE inference、SQL 和事务调度。这支持以 evaluator 能力定义目标用户。

### 从研究系统到企业服务

- Google DeepMind 在 [2026-05 的 AlphaEvolve 进展报告](https://deepmind.google/blog/alphaevolve-impact/)中称，AlphaEvolve 已成为其基础设施的常用工具。其中一个案例是改进 Google Spanner 的 LSM compaction heuristics，把 write amplification 降低 20%。报告还列出金融、半导体、物流和生命科学企业案例。
- Google 在 [2026-07-09](https://blog.google/innovation-and-ai/infrastructure-and-cloud/google-cloud/alphaevolve-on-cloud/)宣布 AlphaEvolve 通过 Gemini Enterprise Agent Platform 向全部 Google Cloud 客户正式开放。Evaluator 驱动的程序演化已经形成面向企业的托管产品。
- RocksDB 案例不能把“优化 LSM”本身写成新颖性。它的研究价值应来自公开可审计的完整仓库搜索、同预算搜索基线，以及 throughput、write amplification、tail latency 之间的多目标谱系。

### 当前可辩护的论文定位

“把 AlphaEvolve scale 到真实软件仓库”适合作为项目解释和文章主线，但不适合作为排他性的 priority claim。仓库级代码演化、Git 作为 substrate、企业代码和硬件工程都已有 2025–2026 的直接工作。

当前更可能成立的研究对象是“面向完整 Git 仓库的通用 quality-diversity 搜索系统”。需要由实验验证的组合差异包括：

- 把任意现有仓库和外部 evaluator 作为输入，而非为单个领域定制演化器；
- 以 commit 表示候选，同时保留 ancestry 与跨谱系 inspiration；
- 区分 source identity 与 evaluator-relevant identity，按 binary、产物、trace 或行为等价类聚合重复候选；
- 从仓库状态学习 behavior descriptor，维护受限 Pareto fronts、islands 和多条有效谱系；
- 在分布式异步 worker 上运行，并按 request、token、evaluation、device-hour 和 wall time 审计预算；
- 相对于独立采样和单 champion 顺序搜索，在相同预算下证明增益，并用 holdout 和 trace analysis 排除过拟合或重复发现。

上述条目是待证假设，不是当前已经成立的贡献声明。

## 第三个案例：选型和预实验记录

选型已经结束，Zstandard 正式实验的结果见上文。以下内容保留 RocksDB 与其他候选的筛选过程，以及 Zstandard 正式运行前的可行性测量；RocksDB 仍可作为需要企业存储资源的后续合作案例。

选型阶段的候选包括：

- 系统压缩库，例如 Zstandard；
- LSM 存储引擎，例如 RocksDB 的 compaction/write path；
- LLM inference serving，例如 vLLM 或 SGLang；
- EDA，例如 OpenROAD；
- SAT/SMT solver。

LSM 存储引擎是最初首选。筛选条件包括：

- clean build 与 incremental build 时间；
- correctness smoke gate 时间；
- 短 training benchmark 时间和噪声；
- 完整 validation benchmark 时间；
- 单机与并行环境下每天可完成的 candidate evaluation 数；
- 运行 256、1,024 和 28,000 次 evaluation 所需的 device-hours。

本机 throughput 初测结果见下。独占 Linux NVMe 上的 validity 测试仍是最终 go/no-go 条件；如果 I/O 噪声无法稳定排序，或者训练评测延迟过高，就撤回该推荐。

### RocksDB evaluator 可行性实测

测量日期为 2026-08-03。源码为 [RocksDB](https://github.com/facebook/rocksdb) `4b35e9966c821b7bf29de3b042f405f30acc635e`（`db_bench` 11.9.0）。机器为 14 核 Apple M4 Pro、24 GB RAM、本机 SSD、Apple Clang 21。该环境不是独占 Linux NVMe benchmark host，因此结果只回答吞吐可行性，不能支持 RocksDB 性能改进 claim。

#### 构建与正确性门

| 项目 | wall time |
| --- | ---: |
| clean release `db_bench` build，CMake、本地 gflags | 66.59 s |
| clean debug build，包含三个相关测试 binary | 84.66 s |
| 模拟修改五个 compaction/write `.cc` 后重建 release `db_bench` | 2.54 s |
| 同一修改后重建三个 debug test binary | 8.82 s |
| 运行 `compaction_job_test`，35 tests | 3.16 s |
| 运行 `db_write_test`，52 tests | 7.48 s |
| 运行 `write_controller_test`，4 tests | 0.64 s |

典型五文件 candidate 的增量 release build、debug test build 和 91 个相关测试合计约 22.6 秒。clean build 是 worker 镜像或缓存准备成本；触及高扇出公共 header 的 candidate 可能接近 clean-build 上界。

#### 10 秒 write/compaction proxy

命令使用固定 seed、4 个写线程、64 MB write buffer、无压缩，执行 10 秒 `fillrandom`，随后等待后台 compaction 完成。每次约 ingest 1.18–1.24 GB，并产生 3.96–4.02 GB cumulative compaction write。五次结果为：

| run | ops/s | 完整 wall time |
| ---: | ---: | ---: |
| 1 | 313,445 | 15.12 s |
| 2 | 322,100 | 15.22 s |
| 3 | 316,817 | 15.17 s |
| 4 | 307,507 | 15.16 s |
| 5 | 310,012 | 15.18 s |

ops/s 均值为 313,976，样本标准差约 5,743，CV 为 1.83%。单次 15 秒 proxy 可以淘汰明显退化，但不能可靠地区分 1–3% 改进。五次固定 workload 重复约需 75.9 秒，适合把约 5% 作为训练阶段的晋级阈值。

#### 当前耗时结论

- 单次 smoke：增量构建、91 个相关测试和一次 proxy，约 38 秒；
- 五次重复的 training evaluator：约 99 秒/典型 candidate；
- 触及广泛依赖、接近 clean rebuild 的 candidate：约 4 分钟上界；
- 按 99 秒计算，256、1,024 和 28,000 次 evaluation 分别需要约 7、28 和 770 device-hours；
- 28,000 次在 32 个真正独立的 worker/NVMe 上理想下界约 24 小时，在 64 个上约 12 小时。多个 I/O job 共用一块盘不能算有效并行。

RocksDB 通过了 evaluator throughput 的初步可行性检查：按 99 秒/候选计算，256–1,024 次需要 7–28 device-hours；28,000 次需要独立存储 worker 池。它还没有通过 evaluator validity 检查，因为当前数据库小于内存、运行仅 10 秒、没有读写混合、尾延迟、跨配置 holdout 或独占 Linux NVMe 数据。

#### 测量 1% 改进的时间预算

以下计算把“可靠”定义为双侧显著性水平 5%，并分别采用 80% 和 90% 检验功效。它使用当前单次短评测的 CV 1.829%，通过非中心 t 分布估算样本数。每次 benchmark 按 15.17 秒计算，并为 candidate 加一次 22.64 秒的增量构建和相关测试。

| 设计 | 80% power | 总时间 | 90% power | 总时间 |
| --- | ---: | ---: | ---: | ---: |
| 根版本均值已由大量历史测量固定 | 29 次 candidate run | 7.7 min | 38 次 candidate run | 10.0 min |
| 交错配对 A/B，假设相邻运行相关系数为 0.5 | 29 对，58 次 run | 15.0 min | 38 对，76 次 run | 19.6 min |
| baseline 与 candidate 完全独立 | 每组 54 次，108 次 run | 27.7 min | 每组 72 次，144 次 run | 36.8 min |

固定根版本均值的方案不处理机器温度、SSD 状态和后台负载造成的时间漂移。论文实验优先采用随机顺序的交错配对 A/B，并直接测量 pairwise difference 的方差。按相关系数 0.5 的暂定值，单个 workload cell 的 evaluator 会从约 99 秒增加到 15–20 分钟，约为当前的 9–12 倍。如果全部 28,000 次 evaluation 都使用该精度，需要约 7,000–9,200 device-hours；32 个独立 worker 需要约 9–12 天，64 个需要约 4.6–6 天。三个独立 workload cell 需要约 45–60 分钟/候选，尚未包含完整测试套件。

当前 CV 只来自五次重复。在独立正态噪声假设下，其标准差的 95% 区间约为 1.10%–5.26%，不足以锁定正式实验预算。样本数近似与 `CV² / effect²` 成正比；如果真实 CV 是 3%，上述时间约增加到 2.7 倍。正式实验前应完成一次基线噪声研究：至少 30–50 次重复，覆盖不同 seed、热状态和运行顺序，并比较 10、30、60 和 120 秒 workload 的 pairwise CV。

1% 精度不应施加到全部搜索 candidate。训练阶段保留 99 秒的多保真 gate，只对预先选定的少量 finalist 使用 15–20 分钟配对测量；最终 winner 再进入多个 workload、完整测试和 sealed holdout。这样可以避免对 28,000 次搜索 evaluation 全部支付高精度成本，也避免在 holdout 上反复挑选 winner。

建议采用分级 evaluator：

1. 所有 candidate 运行 scope gate、增量构建、91 个相关测试和一次短 proxy；
2. 超过预设阈值者追加四次重复，训练分数使用稳健聚合；
3. 晋级 candidate 在独占 Linux NVMe 上运行 30–60 秒的多个 write/compaction workload、不同 seeds 和未公开配置；
4. finalist 运行完整 310 个 CTest target、长时间 steady-state、读写混合、tail latency 和 sealed holdout。

第三、四阶段的实际耗时和噪声仍需在目标服务器上测量。在这之前，不能把 99 秒写成最终 evaluator 成本。

### 评测更快的第三案例候选

| 候选 | 速度依据 | 研究价值 | 主要问题 | 当前判断 |
| --- | --- | --- | --- | --- |
| Zstandard | 本机实测见下；纯内存 CPU benchmark | 成熟 C 系统库；压缩速度、解压速度、压缩率和内存构成天然 Pareto 问题 | 需要隐藏语料、防止针对 corpus 特化，并锁定编译配置 | 第三案例新首选 |
| SQLite | 官方以 Cachegrind 跑约 30,000 条 SQL，并称结果可重复到至少 7 位有效数字，可测 0.05%–0.1% microoptimization | 数据库内核、查询执行和存储系统，工业价值高 | Cachegrind 是 CPU proxy；完整正确性测试复杂；需在 x86 Linux 实测 wall time | 备选；精度优先时有价值 |
| DuckDB | [官方 benchmark runner](https://duckdb.org/docs/current/dev/benchmark)内置多次运行和正确性结果；[2024 年官方记录](https://duckdb.org/2024/06/26/benchmarks-over-time)称当时完整 suite 单次低于 35 秒 | 完整分析型数据库仓库，workload 和 operator 多样 | 构建成本较高；真实 wall time 仍需重复；当前 suite 已变化 | 硬核程度高于 Zstandard，实测后再决定 |
| CaDiCaL 等 SAT solver | 构建和功能测试快；官方含 API、CNF、proof、trace 和 model-based tests | 形式验证强，问题重要 | 性能必须覆盖大量难实例和 timeout；与 SATLUTION 叙事重叠 | 不作为提速方案 |

SQLite 的精度依据来自其[官方 CPU 测量说明](https://sqlite.org/cpu.html)。该页面同时指出 Cachegrind 只测 CPU proxy，不能覆盖真实 I/O latency。

### Zstandard evaluator 可行性实测

实验设计与实施交接见 [2026-08-03-zstandard-third-case-study-handoff.md](2026-08-03-zstandard-third-case-study-handoff.md)。该 handoff 将已决定的研究目标、正式实验的证据约束和仍需预实验决定的参数分开，供负责前两个案例的 agent 接手。

测量日期为 2026-08-03。源码为 [Zstandard](https://github.com/facebook/zstd) `82d322c4973d9e2968d94047a40892bc6d9a9bdf`，版本 1.6.0。仓库含 658 个 tracked files；275 个 C/C++ source/header 约 138,614 行。机器与 RocksDB 实测相同。

官方 CLI 的 [benchmark mode](https://github.com/facebook/zstd/blob/dev/programs/zstd.1.md)在内存中反复压缩和解压输入，`-i#` 可把每个阶段的最短时间设为 1 秒。官方仓库还维护 [automated benchmarking](https://github.com/facebook/zstd/blob/dev/tests/automated_benchmarking.py)，直接比较两个 build 的 compression/decompression speed，并把 1% 设为 regression threshold。

#### 本机耗时

| 项目 | wall time |
| --- | ---: |
| clean release build | 2.79 s |
| 修改一个核心 compression `.c` 后增量 build | 1.30 s |
| `make check` 基础正确性测试 | 23.20 s |
| 单 corpus、单 compression level、`-i1` benchmark | 2.07 s |
| 单 corpus、levels 1–5、`-i1` benchmark | 13.61 s |

测试 corpus 是仓库 `lib/` 下 103 个文件，共 3.31 MB。固定最终 release binary 后重复 15 次，结果为：

| 指标 | 均值 | 样本标准差 | CV |
| --- | ---: | ---: | ---: |
| compression speed | 608.27 MB/s | 2.90 MB/s | 0.477% |
| decompression speed | 1,826.29 MB/s | 10.25 MB/s | 0.561% |

按较差的 0.561% CV、双侧显著性水平 5% 计算，独立 baseline/candidate A/B 检测 1% 差异需要每组 7 次达到至少 80% power，每组 8 次达到至少 90% power。单 corpus、单 level 的完整 evaluator 加上 `make check` 约需 53–58 秒。Levels 1–5 全部测量约需 3.6–4.0 分钟。该估算采用独立样本，不依赖相邻 A/B 的正相关。

`make check` 生成的另一套构建配置与固定 release binary 之间出现过约 7% 的性能差异。正式 evaluator 必须固定 compiler、flags、link mode、CPU affinity 和 binary hash，并分别保存 baseline 与 candidate build。不能混用测试 binary 和性能 binary。

#### 案例设计

- 允许 agent 修改 `lib/` 和必要构建代码；保护 `programs/`、`tests/`、evaluator 与 corpus；
- training 使用公开 corpus groups 和多个 compression levels；validation 使用未公开的代码、JSON、文本和二进制 corpus；
- 目标至少包含 compression speed、decompression speed、compressed size 和 peak memory，直接形成 Pareto frontier；
- 正确性门包含 `make check`、round-trip、旧版本 decode compatibility 和 sanitizer/fuzzer 晋级测试；
- finalist 在 x86-64 与 ARM64 上运行交错 A/B，并报告跨架构和跨 corpus 的 holdout；
- 分析候选改进是参数调整、局部优化、新算法结构还是对 corpus 的特化。

Zstandard 比 RocksDB 更适合作为第三案例：单次高精度 evaluator 快约 4–15 倍，不需要独占 NVMe，且多目标结构更能检验 Loreley 的 quality-diversity 设计。RocksDB 保留为后续需要企业存储资源的合作案例。

## 论文的最低实验问题

现有三个案例都没有同预算的模型驱动对照组。论文至少需要比较：

1. Loreley 的 quality-diversity 搜索；
2. 只继续修改当前 champion 的顺序搜索；
3. 每次从 root 独立生成候选的 best-of-N。

预算需要同时报告 model requests、tokens、candidate evaluations、不同 evaluator identities、evaluator device-hours 和 wall time。每个 arm 需要预先固定 training、validation、holdout、正确性门、protected scope、seed policy 和 winner selection rule。

Zstandard V19 已完成三阶段数据隔离、binary-aware archive admission、unique-binary endpoint 和 finalist freezing，但只运行了 quality-diversity arm。Champion-sequential、root-independent 和跨架构复现仍是论文最低实验要求，不能用现有案例谱系代替。

论文还需要回答四个由 V19 暴露的问题：

1. 至少三次独立搜索能否复现 useful-candidate discovery，成功概率和 time-to-first-win 如何分布；
2. 无 seed、弱 seed 和当前人工 seed 条件下，evolution 对最终质量各贡献多少；
3. Top 10、effect-band 和 adaptive racing 哪种 finalist policy 能以可控成本避免 Top-3 miss；
4. noisy objectives 下的 archive admission 是否需要 incumbent/challenger remeasurement、confidence-bound dominance 或其他策略。

Tree-level identity 与 evaluator-state identity 已经成为系统设计选择。下一步不是再次事后折叠 V19 trace，而是在第二个 compiled 或 generated-artifact target 上验证 phased measurement reuse 和 evaluator-defined identity 的通用性。

## 推广材料

### 中英文文章

英文标题：

> Searching Real Code Repositories with Coding Agents

英文副标题：

> Results from 348 Loreley jobs on `markdown-it-py`, `python-pathspec`, and Zstandard

中文标题：

> 用编码智能体搜索真实代码仓库

中文副标题：

> Loreley 在 `markdown-it-py`、`python-pathspec` 和 Zstandard 上运行 348 个任务的结果

2026-08-08 的第四轮审阅参考了用户提供的直接翻译稿和两份独立 review。主要问题不是翻译腔、正式程度或句子是否足够自然，而是模型替作者增加了叙事与判断。统一编辑规则如下：

- 允许改句法，不允许增加修辞功能；
- 不增加用于宣布例子、总结例子、制造下一层问题或提示 takeaway 的连接句；
- 不把实验事实包装成“问题、案例、启示、更大问题”的叙事弧；
- 不把平实标题改成更抓人、更人格化的标题；
- 不为塑造作者声音增加态度、情绪、反问、比喻或 punchline；
- 技术词有成熟中文译法时使用中文，没有稳定译法时再保留英文；不维持一套刻意整齐的中英文混用 register；
- 证据等级、选择时间和不确定性说明属于实验结论，不按“AI connective tissue”删除；
- 英文母稿本身也执行同样检查，不能把英文中的 editorialization 原样传给中文。

当前写作流程：

1. 英文母稿先确定系统定义、实验设置、结果、证据范围和相关工作；
2. 逐句删除没有增加事实、方法、条件、证据边界或行动要求的 editorialization；
3. 中文保留英文的信息顺序和命题，只修不符合中文习惯的句法；
4. 核对两种语言中的结果、选择口径、成本、术语和链接。

当前正文结构：

1. 系统定义与三案例总表；
2. 搜索模型、评估器接口和质量-多样性档案库；
3. `markdown-it-py`、`python-pathspec` 和 Zstandard 三个正式案例部分；
4. 资源使用与成本；
5. FunSearch、AlphaEvolve、SATLUTION、ABCEvo、CodeEvolve 和 HORIZON；
6. 证据范围、同预算搜索基线和接入要求。

### 案例证据包

每个案例需要提供：

- root 与 selected source diff、canonical measured artifact hash，以及已知的 source/artifact equivalence class；
- 主要 ancestry 和 inspiration edges；
- training 与 validation 指标；
- evaluator、scope gate 和选择规则；
- 失败 candidate 统计；
- tokens、请求、成本和 wall time；
- 已知限制和不能宣称的结论。

统一文字证据包已经由 [2026-08-07-loreley-case-study-evidence-report.md](2026-08-07-loreley-case-study-evidence-report.md) 提供。谱系和核心统计已做成四张可浏览图片；四份 canonical source patches 已由 [candidate diff 索引](../marketing/candidates/README.md) 固化并从正式报告链接。GitHub/MkDocs 相对链接与移动端页面已经完成本地审计。

### 论文

论文暂定研究对象是“完整 Git 仓库上的 quality-diversity 程序演化”。题目和贡献声明等待最近一年技术脉络重建后再定，不沿用旧文档中的 first、production-grade 或 repository-scale novelty 表述。

## 决策记录

### 2026-08-03

- 取消把公开可运行 Demo 作为推广前置条件。
- 确认 evaluator 的 Python 接口不限制被优化项目的语言。
- 将现有两个案例分别定位为独立验证证据和演化机制证据。
- 修正 SATLUTION 的量级表述，保留 cycle 和 candidate evaluation 两种计数单位。
- 将小品文的核心 insight 改为 coding agent 对稀疏有效仓库状态的语义搜索。
- 决定重做最近一年技术脉络，不把现有项目文档视为权威综述。
- 将 RocksDB 列为第三案例候选并开始 evaluator 时延测量。
- 初步确认仓库级、Git-based、企业代码和硬件工程均已有直接相关工作，撤回任何依赖这些单点的 priority claim。
- 将候选论文定位收窄为“完整 Git 仓库上的通用 quality-diversity 搜索”，等待基线实验验证。
- 将 AlphaEvolve 的 Spanner LSM 案例和 2026-07 企业正式服务纳入 to-B 脉络；RocksDB 的价值改为公开、可审计、多目标的仓库级对照实验。
- 在 M4 Pro 上完成 RocksDB evaluator 吞吐初测：典型 candidate 的五重复 training evaluation 约 99 秒，短 workload CV 为 1.83%。
- 以当前噪声估算，可靠测量 1% 改进需要约 15–20 分钟的交错配对 A/B；该数字等待 30–50 次基线噪声研究校准。
- RocksDB 通过本机 throughput 初测；独占 Linux NVMe 上的 evaluator validity 测试仍是 go/no-go 条件。
- 比较 Zstandard、SQLite、DuckDB 和 CaDiCaL 后，确认 CPU-bound evaluator 更适合快速测量 1% 变化。
- Zstandard 本机单-level evaluator 约 53–58 秒，levels 1–5 约 3.6–4.0 分钟；将它调整为第三案例首选，RocksDB 改为后续企业合作案例。
- 形成 Zstandard 实验 handoff，要求先冻结协议和校准 evaluator，再运行 quality-diversity、champion-sequential 与 root-independent 三个同预算 arm；正式证据必须区分 training、hidden validation 和 sealed holdout。

### 2026-08-05

- 统一 `markdown-it-py` 与 `python-pathspec` 的 throughput、等价耗时、validation/reference、内存、搜索、资源和证据等级口径。
- 确认 `markdown-it-py` 是当前主要独立验证结果；`python-pathspec` 的 25.14% 结果用于能力与谱系机制叙事，并明确其候选在 reference 揭示后确定。
- 记录三项发布前数据清理：`python-pathspec` 的旧成本 proxy、`markdown-it-py` 的 per-document median 冲突，以及 `python-pathspec` outcome 命名差异。
- 将实验报告中的 generation cost 纳入统一指标：`markdown-it-py` 为 $2.0833，`python-pathspec` 为 $2.4856，合计 $4.5689；embedding、主机和人工成本未货币化。

### 2026-08-06

- Zstandard 256-logical-job campaign 完成，正式 outcome 为 modest-positive：sealed holdout compression throughput +1.044%，95% CI 为 +0.880% 到 +1.209%；decompression -0.018%，compressed size 不变，peak RSS +0.03125 MiB。
- Holdout-selected source commit `893bf5cf` 由后 128 个 job 产生，按 terminal completion order 是第 245 个逻辑 job；但它与第 118 个逻辑 job 的 `c1f1852b`、第 71 个逻辑 job 的 `817af317` 编译成逐字节相同的 ARM64 binary。扩展阶段没有发现最终使用的不同 executable，只将 unique trees 从 51 增加到 100、unique binaries 从 33 增加到 55。
- 记录 Zstandard generation cost $4.8277；三个案例报告的 generation cost 合计更新为 $9.3966，仍不包含未定价 embedding、host、开发和人工成本。
- 将 Zstandard 定位为当前 holdout 测量最完整的 C 系统仓库案例；它支持约 1% 的固定 binary-level claim，不支持 generation-3 edit 的效果、2%、跨架构或普遍 Zstandard 加速。
- 确认本次没有执行 champion-sequential 和 root-independent 对照，也没有完成 x86-64 复现；这些项目继续作为论文最低实验要求。
- 将 18 个 lifecycle exclusions、macOS background throttling 和 post-reveal audit 误报纳入运行可靠性记录，公开 wall time 时区分 15.72 小时 active runner sessions 与 31.1 小时 first-to-last database window。
- 追加 binary-equivalence 审计：三个 source finalists 只有两个 ARM64 binaries；正式 case-study 在发布前必须修正 winner lineage、validation 排名和后 128 个 job 价值的表述。
- 确认 holdout binary 在第 71 个逻辑 completion 首次出现；48 份 passed training reports、6 个 Git trees 生成该 binary，占全部 passed evaluations 的 23.4%。将 binary/evaluator-state deduplication 加入 archive admission、finalist freezing 和论文实验要求。
- 确认正式报告的 usage 表漏列 4,478,202 reasoning output tokens；286,838,265 total observed tokens 和 $4.8277 generation cost 不变。

### 2026-08-07

- 以 Zstandard V19 取代 V13 成为第三个对外案例；V13 只保留为 candidate identity、restart 和 measurement isolation 的历史基础设施证据，不计作第四个案例。
- V19 完成 220 个物理 job：211 成功、9 失败，成功结果对应 167 个不同 release binaries；完整 search 使用 5.31 runner-hours。
- 预登记 winner `7b9aef38` 是一个 9 行人工 seed。它在 sealed holdout 上取得 +1.019% compression，95% CI 为 +0.962% 到 +1.076%；decompression 中性、compressed size 不变，正式 outcome 为 modest-positive。
- 确认 V19 修复了 V13 的 finalist binary duplication：archive admission、终点统计和 finalist freezing 使用 evaluator-provided release-binary identity。44 个成功 job 仍生成已有 binary，其中启用 cache 后的 19 个复用已有 measurement。
- 保留预登记结论后进行 Top-10 sensitivity follow-up。Training rank 10 的 generation-4 candidate `fe39bee8` 成为 expanded validation winner，并在另一个预先 sealed 的 fresh corpus 上取得 +0.891% compression，95% CI 为 +0.522% 到 +1.261%。该补充不改写原 winner，也不是同 corpus head-to-head。
- 将 Top-3 finalist breadth 记为方法问题：未来至少验证 Top 10，或预登记 effect-band/adaptive-racing rule；noisy-objective archive policy 继续作为论文研究问题。
- 记录 V19 的 52,653,004 tokens 和 $60.2472 Kilo catalog estimate。该数字不是 provider bill，embedding 未定价，也不能与前两个案例的 provider-recorded DeepSeek cost 相加成 all-in spend。
- 确认现有证据足以先做第一轮推广，不等待更高 Zstandard 数字、搜索基线或跨架构结果。第一轮目标是发布中文文章、统一证据图包和 design-partner 入口；论文级对照实验在推广后继续。
- 修正 `markdown-it-py` 精简报告中的 per-document median：使用 raw evidence 的 `1.068380x`，不再把 `1.067538x` geometric mean 写成 median。
- 完成首发口径表、中英文长文、摘要与社区分发文案、design-partner brief、公开 intake 和四张 SVG/PNG 证据图；README、文档首页和 MkDocs 导航已增加首发入口。
- 固化 `markdown-it-py`、`python-pathspec`、Zstandard registered winner 和 Zstandard evolved follow-up 的 canonical source patches，并核对它们与实验 bare repository 的 `git diff` 输出逐字节一致。
- 严格 MkDocs 构建、首发相对链接、SVG/XML、issue-template YAML、PNG 尺寸和本机绝对路径审计通过；390 px 移动端渲染没有页面级水平溢出。
- 首发材料进入人工审阅。当前没有 commit、push、部署或对外发布。
- 人工审阅指出，面向读者的文章混入了“首创声明、主张边界、首发是否依赖 Demo”等内部编辑语言。材料重新分为两层：发布口径表、分发文案包和审阅索引明确标为内部工作文件；中英文文章、合作页、候选 diff 页面和图片只保留读者需要的事实、方法、限制与行动入口。
- 中英文文章删除审阅状态和“首发”元叙事，将相关工作段改为 Loreley 所解决的工程问题，将 Demo 辩护段改为 evaluator 接入流程，将结尾改为下一阶段的搜索策略对照实验。
- 三案例图片把 `Evidence`、`Boundary` 和 `evidence roles` 改为 `Run summary`、`Scope note` 和仓库搜索结果；README、文档首页与 MkDocs 导航同步改为长期可用的文章与合作入口。
- 第二轮审阅参考 Lilian Weng 的 harness 综述和 Jiayi Weng 的 Heuristic Learning 博客，确认现稿仍有“抽象命题先行、案例像报告摘要、缺少作者判断、文献按名称堆叠”的问题。
- 中英文文章改为案例驱动结构：从 `markdown-it-py` 四代谱系进入，展示 evaluator plugin 代码，用 `python-pathspec` 解释 archive 支线，用 Zstandard 解释 binary identity、measurement cache 和 finalist breadth，之后再补相关工作与研究计划。
- 删除“枚举所有程序”“持续提出和验证下一步”等假设性对照和抽象流程句；分发摘要、博客导语、README 标题与英文稿同步更新。
- 第三轮审阅认为案例驱动版本仍然生硬，尤其是以“AlphaEvolve 的基本循环很直接”开场，暴露了先写抽象中文框架再填内容的问题。旧稿废弃，改为先从头写英文，再按其叙事重写中文。
- 新英文稿以第一次 `markdown-it-py` 64-job 实验开场，让 Loreley 的设计从四代 patch、0.9978× 支线和 binary duplication 三个事实中逐步出现；AlphaEvolve 及后续工作移到三个案例之后。
- 新中文稿不逐句翻译英文，保留“25.14% 后面要带一个星号”“212 个 evolution jobs 没有打过 manual seed”等口语化但精确的句子。README、MkDocs 导航和发布文案包同步使用新标题。
- 第三轮稿件通过严格 MkDocs 构建、桌面与 390 px 移动端渲染检查；中英文页面均无页面级水平溢出，四张证据图均成功加载。

### 2026-08-08

- 对比用户提供的直接中文翻译稿与仓库中的中文重写稿，并阅读两份独立 AI-fluff review。确认主要问题是结构级 editorialization，而不是句子是否足够自然。
- 采用“允许改句法，不允许增加修辞功能”的翻译与校稿规则。轻微翻译腔可以保留；不新增过渡、takeaway、叙事弧、人格化标题或作者姿态。
- 英文母稿也按同一规则重写，不再把文章伪装成从实验出发的技术博客。新稿使用系统定义、结果总表、三个案例、资源成本、相关工作、证据范围和接入要求的正式结构。
- 中文稿以英文稿为信息结构来源，使用“编码智能体、评估器、候选方案、吞吐量、质量-多样性档案库、二进制文件”等中文术语，只保留没有稳定译法或作为代码标识的英文。
- 发布文案包、README 和 MkDocs 导航同步采用新标题与低 editorialization 表达。
- 第四轮稿件通过严格 MkDocs 构建、`git diff --check`、1440 px 桌面和 390 px 移动端渲染检查；中英文页面均无页面级水平溢出，四张证据图均成功加载。
- 发布分支 `codex/publish-loreley-case-studies` 通过两条 CI 测试和 Cremona 结构检查。图片生成器最初产生 7 个新的参数数量热点；将可选绘图参数收进 typed keyword options 后，干净发布快照的新增热点降为 0，生成图片不变。
- PR [#54](https://github.com/NeapolitanIcecream/loreley/pull/54) 收录 GitHub 与文档站材料。GitHub Pages 从提交 `018c144` 部署到 `gh-pages` 提交 `c66dc01`；[部署任务](https://github.com/NeapolitanIcecream/loreley/actions/runs/31249188262)成功。
- 线上核验覆盖主页、中英文文章、三案例证据报告、候选 diff 索引和合作说明。六个 URL 均返回 HTTP 200，中英文页面标题和 348-job 汇总出现在发布后的 HTML 中。
