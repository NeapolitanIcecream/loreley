# Loreley 第三个案例交接：Zstandard

日期：2026-08-03

状态：历史交接已完成。Zstandard V19 已成为第三个公开案例；本文保留最初的选型、测量和论文级实验要求。

执行结果见 [Zstandard V19 正式报告](2026-08-07-zstandard-gpt-v19-case-study-report.md)和 [Top-10 补充报告](2026-08-07-zstandard-gpt-v19-top10-validation-supplement.md)。V19 完成 220 个物理 job，其中 211 个成功，对应 167 个 release binaries。预先登记的 manual-seed winner 在 sealed holdout 上取得 +1.019% compression throughput；Top-10 补充在另一组新语料上确认了一个 generation-4 candidate 的 +0.891% 结果。本轮只运行 quality-diversity arm；champion-sequential、root-independent 和跨架构复现仍属于论文后续实验，不能从现有结果推断。

接收者：负责 `markdown-it-py` 与 `python-pathspec` 两个案例的实验 agent。

## 交接任务

请以 [Zstandard](https://github.com/facebook/zstd) 为 Loreley 的第三个案例，完成实验协议设计、evaluator 实现、预实验、正式运行和证据整理。

先提交可审查的实验协议和 evaluator 校准结果，再冻结正式实验。不要直接复用本文的本机初测参数启动大规模搜索。本文确定研究目标和证据要求，具体 corpus、compression level、运行时长、搜索 job 数和计算环境应由预实验决定。

## 已作决定

1. 第三个案例使用 Zstandard。RocksDB 不再是本轮首选，保留为需要独占 Linux NVMe 和更多企业资源的后续案例。
2. 本案例需要展示 Loreley 可以通过 evaluator 接入非 Python 项目。Loreley 的 Python evaluator 接口只负责调度，目标仓库和构建评测链路是 C/C++。
3. 本案例需要同时产生案例文章证据和论文实验证据。公开可运行 Demo 不是交付项。
4. 正式结果必须包含同预算搜索对照、隐藏 validation 和 sealed holdout。仅再运行一次 64-job Loreley campaign，不能补上现有两个案例缺失的搜索基线。
5. 正式 holdout 失败后不得改选 runner-up。可以预先登记一个有限 finalist 集合并一次性评测，但集合大小、选择规则和多重检验处理必须在揭示 holdout 前冻结。

## 为什么选 Zstandard

| 选择依据 | 已有证据 | 对实验的意义 |
| --- | --- | --- |
| 目标不同于前两个 Python 库 | Zstandard 是约 13.9 万行 C/C++ 的压缩系统库 | 第三个案例可以检验语言无关的 evaluator 接入，以及 agent 对底层性能代码的修改能力 |
| 正确改进受多项约束 | 修改需要保持解码正确性、格式兼容、API/ABI、压缩率和内存边界 | evaluator 需要同时检查性能与这些约束 |
| 评测速度较快 | 本机单 corpus、单 level 的 1 秒 benchmark 约 2.07 秒，`make check` 约 23.20 秒 | 可以把重复测量和搜索对照纳入预算 |
| 本机短评测噪声较低 | 固定 release binary 的 15 次重复中，压缩速度 CV 为 0.477%，解压速度 CV 为 0.561% | 1% 量级的候选有可能在一分钟级 evaluator 中被区分，仍需在正式 host 上重测 |
| 目标天然多维 | compression speed、decompression speed、compressed size 和 peak memory 可能互相制约 | 适合检验 Loreley 的 Pareto archive 和 quality-diversity 搜索，而非只比较单一 champion |
| 并行条件比存储系统简单 | 核心 benchmark 在内存中运行，不依赖每个 worker 独占 NVMe | 可以按 CPU core 扩展，但必须先测频率和内存带宽干扰 |

Zstandard 的难度来自成熟 C/C++ 代码、低层性能优化、格式兼容、跨语料泛化和多目标约束。不要把仓库总行数直接写成每个候选都进行了全仓库修改。正式报告应给出实际修改文件、调用路径和 diff 范围。

## 可复现的前期测量

前期测量使用以下环境：

- Zstandard commit：`82d322c4973d9e2968d94047a40892bc6d9a9bdf`；
- 仓库报告版本：1.6.0；
- 658 个 tracked files；
- 275 个 C/C++ source/header files，共约 138,614 行；
- 机器：14 核 Apple M4 Pro、24 GB RAM、Apple Clang 21；
- 测试 corpus：目标仓库 `lib/` 下 103 个文件，共 3,310,200 bytes。

| 操作 | 本机 wall time |
| --- | ---: |
| clean release build | 2.79 s |
| 修改 `lib/compress/zstd_compress.c` 后增量 build | 1.30 s |
| `make check` | 23.20 s |
| 单 corpus、level 1、`-i1 -T1` | 2.07 s |
| 单 corpus、levels 1 到 5、`-i1 -T1` | 13.61 s |

固定最终 release binary 后，level 1 的 15 次结果为：

| 指标 | 均值 | 样本标准差 | CV |
| --- | ---: | ---: | ---: |
| compression speed | 608.27 MB/s | 2.90 MB/s | 0.477% |
| decompression speed | 1,826.29 MB/s | 10.25 MB/s | 0.561% |

按较差的 0.561% CV、双侧显著性水平 5% 和独立 baseline/candidate 样本估算，检测 1% 差异需要每组 7 次达到至少 80% power，每组 8 次达到至少 90% power。由此得到的本机成本是：

- 单 corpus、单 level，包含一次 `make check` 和 build，约 53 到 58 秒；
- 单 corpus、levels 1 到 5，约 3.6 到 4.0 分钟；
- 只做一次 levels 1 到 5 的低精度 training pass，约 38 秒。

这些数字只证明 evaluator 具有进一步设计的价值。它们不证明正式 corpus、不同 levels、不同编译器和目标 host 上也有相同噪声。

前期测量还发现，`make check` 生成的另一套构建配置与原先 release binary 之间有约 7% 的性能差异。正确性测试和性能评测必须使用隔离的 build 产物。每次评测应保存 compiler、flags、link mode、CPU affinity、源码 commit 和 binary hash，不能让 `make check` 隐式替换待测 binary。

## 本案例需要回答的问题

正式协议至少回答四个问题：

1. Loreley 能否在冻结的 Zstandard revision 上找到通过正确性和兼容性验证的实际性能改进？
2. 改进能否从 training corpus 泛化到未公开的内容类型、文件大小和 compression levels？
3. Loreley 的 quality-diversity 搜索在相同模型和预算下，是否优于从 root 独立采样和持续修改 champion？
4. archive 是否保留了有用的多条谱系或 Pareto trade-off，还是最终增益可以由更简单的搜索解释？

前两个问题支持第三个案例的事实陈述，后两个问题支持论文对搜索方法的论证。任何一项都允许得到负结果；不能在看到结果后更换主要问题或胜负规则。

## 正式协议的硬约束

### 冻结对象与环境

协议应冻结并记录：

- upstream URL 和源码 commit。建议从已经完成预实验的 `82d322c4973d9e2968d94047a40892bc6d9a9bdf` 开始；如需更换，先说明原因并重做校准；
- compiler、版本、编译参数、link mode 和 CPU feature policy；
- performance binary 与 correctness binary 的独立构建方式；
- evaluator、容器或机器镜像、corpus manifest 和所有脚本的 hash；
- CPU governor、turbo policy、affinity、并发 lane、机器型号和操作系统；
- 模型、agent backend、prompt、request guard、timeout、embedding provider 和 embedding dimension；
- model request、token、cash、candidate evaluation、device-hour 和 wall-time 预算。

正式 benchmark 期间不得同时进行会改变 CPU 频率、缓存或内存带宽的构建和评测任务。一个机器是否能安全运行多个 benchmark lane，需要用 root/root 干扰实验决定，不能按核心数直接推算。

### 修改范围与防作弊

默认只允许 agent 修改 `lib/**` 中的产品源码和头文件。以下内容应保护：

- `programs/**`；
- `tests/**`；
- evaluator、corpus、结果解析器和实验配置；
- compiler flags、benchmark 参数和输出格式；
- `loreley.program.md`、`.loreleyignore` 及其他 experiment-control files。

如果预实验确认必须开放某个 build file，应逐个列入 allowlist，并增加检查，防止通过关闭安全检查、改变 CPU flags 或替换 benchmark binary 获得分数。

evaluator 需要拒绝以下候选：

- 修改受保护文件；
- 对 corpus 文件名、路径、已知摘要或输入字节模式做特判；
- 跳过输入、减少工作量或伪造 benchmark 输出；
- 改变公开 API、frame format 或兼容性而未被协议允许；
- 通过不一致的编译参数比较 root 与 candidate。

静态 scope gate 不能发现所有 corpus 特化，因此还必须使用未公开 corpus 和 edit audit。

### Corpus 分层

至少分成三层：

1. `training`：agent 可以接收分数的公开 corpus groups；
2. `validation`：不向 agent 提供原始 corpus，只用于按冻结规则晋级和选择 finalist；
3. `sealed holdout`：在搜索结束、候选与选择规则冻结后才运行一次。

三层不能只是同一 corpus 的不同随机切片。应覆盖代码、JSON 或结构化文本、普通文本和二进制数据，并覆盖不同文件大小。是否加入 dictionary、streaming、small-block 或 multithread 模式，由预实验根据时间和论文问题决定。

协议应记录每个 corpus 的来源、许可、内容 hash、字节数、文件数和泄漏检查。目标仓库自身的 `lib/` 可以用于 evaluator 开发，但不能成为正式实验的唯一 corpus。

`validation` 分数不得回流到模型。`sealed holdout` 的 corpus、聚合值和单项结果在 finalist 集合冻结前都不能查看。若要由同一 agent 保管 holdout，应采用独立加密或访问边界，并在证据中记录首次解封时间。

### 指标与胜负规则

每个 workload cell 至少保留以下原始值：

- compression throughput；
- decompression throughput；
- compressed bytes 或 compression ratio；
- peak memory；
- correctness、compatibility 和 scope-gate 状态；
- 每次重复的次序、时间戳、host 和 binary hash。

不能只保存一个综合分数。协议需要在搜索前定义：

- 一个 primary endpoint；
- 用于 Pareto archive 的目标向量；
- 每个 cell 可接受的最坏回退；
- compressed size 和 peak memory 的限制或 Pareto 处理方式；
- finalist 数量与选择函数；
- 多个候选、指标或 profile 同时检验时的多重比较处理；
- strong、modest-positive、negative 和 invalid 四类结论。

建议把 strong result 定义为：预先选择的 candidate 在 sealed holdout 的 primary endpoint 上达到预先规定的实际改进阈值，置信区间排除无改进，并通过所有正确性、兼容性、scope、压缩率、内存和单-cell 回退限制。若公开 claim 是“点估计改进至少 1%”，点估计应达到 1% 且置信区间下界高于零；若 claim 是“真实改进至少 1%”，置信区间下界也必须高于 1%。是否采用 1% 阈值，应由目标 host 的噪声研究确认。

Pareto trade-off 可以作为有价值的次要结果，但不能把压缩率明显下降换来的吞吐提升写成无条件加速。

### 正确性与兼容性门

每个 scored candidate 至少需要：

- clean 或可审计的增量 build；
- upstream `make check`；
- training corpora 的 compress/decompress round-trip；
- root 与 candidate 之间的交叉解码；
- scope gate 和 benchmark-output sanity check。

晋级 candidate 和 finalist 还需要协议选定的 medium/long tests、sanitizer、fuzzer、legacy decode 和 API/ABI checks。Zstandard 的官方 [TESTING.md](https://github.com/facebook/zstd/blob/82d322c4973d9e2968d94047a40892bc6d9a9bdf/TESTING.md)列出了 short、medium、long、sanitizer、fuzzer、legacy 和跨平台测试。不要把 `make check` 的 23 秒结果表述为完整上游验证。

内存门应在计划支持的最大输入 shape 上校准，并保留安全余量。`python-pathspec` 的初始 winner 因 training shape 太小而在 reference allocation gate 失败，本案例不能重复这一选择偏差。

### 测量协议

正式搜索前，先在目标 host 上完成基线噪声研究：

- 至少 30 到 50 次 root 重复；
- 比较不同 benchmark 最短时长、corpus、level、冷暖状态和运行时段；
- 运行 root/root 的随机交错或配对实验，测量 pairwise difference 方差；
- 检查单 lane 与多 lane 的频率、缓存和内存带宽干扰；
- 用实测方差重新计算 1% 或最终 primary threshold 所需重复数。

candidate 与 root 应采用随机化、交错的 A/B 顺序。报告原始重复值、效应量和置信区间，不以一次 CLI 汇总替代统计证据。若不同 corpus 或 level 的异方差明显，应按 cell 设计重复次数或使用预先登记的稳健聚合方法。

### 分级 evaluator

请在正式协议中给出具体阈值和时间预算。建议结构如下：

| 阶段 | 适用对象 | 目的 |
| --- | --- | --- |
| Gate 0 | 所有 candidate | scope、build、`make check`、round-trip 和输出 sanity check |
| Gate 1 | 通过 Gate 0 的 candidate | 低成本 training corpora 和 levels，向搜索返回分数 |
| Gate 2 | 按预先规则晋级的 candidate | 增加 repetitions、corpus groups 和最大输入 memory check，减少噪声误选 |
| Gate 3 | 冻结的 finalist 集合 | hidden validation、较完整测试和 edit audit；据此按冻结规则确定待确认 candidate |
| Gate 4 | 冻结后的待确认 candidate 或预登记集合 | sealed holdout、正式统计、兼容性、sanitizer/fuzzer，以及可用时的第二架构复验 |

Gate 1 可以是约 38 秒的单次 levels 1 到 5 评测，也可以更短；这只是初始设计点。需要用 false-promotion、false-rejection 和 evaluator queue throughput 决定。不要对全部搜索 candidate 支付最终 1% 精度，也不要让低精度 training score直接成为论文中的性能 claim。

### 搜索对照

至少实现三个同预算 arm：

1. Loreley quality-diversity search；
2. 只从当前 champion 继续修改的 sequential search；
3. 每次从 root 独立生成候选的 independent best-of-N。

三个 arm 应使用相同的目标仓库、模型、可见训练反馈、evaluator gates、总 model requests 或预先选择的等价预算，以及相同的成功和失败记账方式。除非协议给出另一种主要预算口径，至少同时报告 jobs、requests、tokens 和 evaluator device-hours。

manual seeds 可以不用。如果使用，三个 arm 必须获得相同的 seed 信息与 seed 预算，或者把 seeded 与 unseeded 明确拆成不同实验。不能只给 Loreley arm 提供人工优化方向。

三个 arm 应作为独立 campaign 运行，不能在一次 Loreley campaign 内把现有 archive 事后解释成基线。协议还要决定每个 arm 的独立 campaign replicate 数和随机种子。一个 campaign 内的 candidate jobs 共享搜索状态，不能充当算法效果的独立重复。如果预算只允许每个 arm 运行一次，结果只能作为描述性比较，不能据此给出搜索算法层面的显著性结论。每个 arm 的随机种子、停止规则、失败 job 和未完成 job 都要保留。

## 初步 evaluator 预算

下表只计算 evaluator device time，不含 agent 生成、排队、容器准备、晋级测试和最终 holdout。数字来自本机测量，每个搜索 arm 都需单独计算。

| evaluator 设计点 | 256 evaluations | 1,024 evaluations | 28,000 evaluations |
| --- | ---: | ---: | ---: |
| levels 1 到 5 单次 pass，约 38 s | 2.70 h | 10.81 h | 295.56 h |
| 单 level 的 1% 测量，约 53 到 58 s | 3.77 到 4.12 h | 15.08 到 16.50 h | 412.22 到 451.11 h |
| levels 1 到 5 的 1% 测量，约 3.6 到 4.0 min | 15.36 到 17.07 h | 61.44 到 68.27 h | 1,680 到 1,867 h |

三个同预算 arm 的总 evaluator 成本约为表中数字的三倍。正式设计应让大多数 candidate 使用 Gate 1，只让预先限定的比例进入后续 gate。

CPU-bound 不代表 device-hours 可以无损换成 wall time。只有在多 lane 干扰实验通过后，才能按独立物理核心或独立 host 估算并行加速。前两个案例约 10 分钟的模型 job 时长不能直接套用到 C/C++ 任务，agent 生成吞吐需要 pilot 实测。

正式 job 数目前未决定。请分别给出 pilot、可发表的最低预算和扩展预算，并说明每档预算能回答哪个问题。不要因为前两个案例各运行 64 jobs，就默认本案例也以 64 为终点。

## 实施顺序和交付物

### 阶段 A：协议设计

提交一份 preregistration-style 实验计划，至少包含：

- 冻结 revision 与环境；
- corpus 三层划分和许可；
- workload matrix；
- primary endpoint、Pareto objectives 和回退限制；
- 三个搜索 arm 的预算与公平性；
- 每个 arm 的独立 replicate 数、分析单位和随机种子；
- finalist、holdout 和多重检验规则；
- threat model、失败分类和 stop rules；
- 预计 device-hours、model usage、cash 和 wall time。

### 阶段 B：evaluator 和校准

实现并测试：

- hermetic build 和 binary provenance；
- scope/protected-file gate；
- correctness、round-trip 和 compatibility gates；
- benchmark parser 与原始数据存档；
- corpus access boundary；
- root/candidate 交错测量；
- fake candidate、known regression、no-op 和 benchmark-cheating fixtures；
- 30 到 50 次基线噪声报告；
- 不同 evaluator lanes 的干扰报告。

### 阶段 C：小规模 pilot

运行足以验证完整链路的小规模 pilot，检查：

- C/C++ agent 能否在 timeout 内完成有效修改；
- candidate build cache 是否正确失效；
- Gate 1 与 Gate 2 的排序一致性；
- evaluator 是否成为吞吐瓶颈；
- token、request、失败率和 wall time 是否支持计划预算；
- 模型是否出现 corpus 特化、测试修改或 benchmark 绕过行为。

pilot 结果只能用于冻结协议和预算，不能与正式 run 混合成确认性样本。

### 阶段 D：正式运行

冻结所有 hashes 和选择规则后，运行三个搜索 arm。保存完整 campaign database、candidate commits、ancestry、inspiration edges、模型 usage、evaluator raw runs、失败记录和环境事件。任何协议偏差都应先记入 deviation log，不得静默修复后继续计算同一确认性结果。

### 阶段 E：验证和证据包

模型调用停止后，冻结 finalist 集合，再依次执行 hidden validation 和 sealed holdout。交付：

- root、finalists 与报告候选的 commits 和 diffs；
- 每个 arm 的预算、有效 candidate 数、best score 和达到各 gate 的数量；
- primary endpoint 的 raw measurements、效应量和置信区间；
- 完整 workload matrix，不能只展示获胜 cell；
- compressed size、memory、correctness、compatibility 和跨架构结果；
- 主要 ancestry、inspiration edges、archive retention 和 edit taxonomy；
- 参数调整、局部优化、结构修改和 corpus 特化的分类；
- model requests、tokens、cash、device-hours 和 wall time；
- failed/invalid candidate 分类和 protocol deviations；
- 一篇英文 case-study report，以及可供推广文章引用的中文事实摘要；
- strong、modest-positive、negative 或 invalid 的最终结论和 claim boundary。

证据包需要让读者浏览候选 diff、谱系、评测协议、性能分布和资源账本。无需制作面向普通用户的一键 Demo。

## Go/no-go 条件

满足以下条件后才能启动正式大规模搜索：

- evaluator 在目标 host 上可以稳定区分预定的最小实际改进；
- root 与 no-op candidate 不产生系统性假提升；
- correctness binary 与 performance binary 不互相污染；
- hidden validation 和 sealed holdout 的访问边界已经实现并测试；
- 三个搜索 arm 的预算和 seed policy 已冻结；
- 对多 lane CPU 与内存带宽干扰已有测量；
- 按 pilot 的有效 job 率和时长，正式预算可以完成并保留验证余量。

遇到以下情况应暂停并修改协议，不要用结果后修补：

- 1% 阈值在目标 host 上需要的重复次数使搜索吞吐不可接受；
- 不同 build 配置仍产生无法解释的性能差异；
- corpus license、泄漏隔离或 sealed holdout 无法审计；
- `make check` 与计划的兼容性测试无法拦截已知错误 fixture；
- 多 lane 排序与单 lane 排序不一致；
- pilot 显示主要“改进”来自 benchmark 特化或改变压缩率边界。

## 仍由接收 agent 决定的事项

以下内容没有在本次选型中决定：

- 正式 upstream commit 是否沿用前期测量 revision；
- 目标 host、主架构和第二验证架构；
- corpus 的具体来源、许可和三层分配；
- compression levels、API mode、文件大小和 dictionary/streaming 范围；
- primary endpoint、Pareto 聚合、回退阈值和 memory limit；
- Gate 1 到 Gate 4 的重复数与晋级比例；
- model、prompt、manual seed policy 和每个 arm 的 job 数；
- full test、sanitizer、fuzzer 和 legacy compatibility 的具体命令；
- pilot、最低可发表预算和扩展预算。

接收 agent 应用预实验数据决定这些事项，并把选择理由和未采用方案写入协议。

## 相关材料

- 总体推广与论文计划：[2026-08-03-loreley-promotion-plan.md](2026-08-03-loreley-promotion-plan.md)
- 第一个案例：[2026-08-02-markdown-it-py-deepseek-case-study.md](2026-08-02-markdown-it-py-deepseek-case-study.md)
- 第二个案例：[2026-08-03-pathspec-deepseek-case-study.md](2026-08-03-pathspec-deepseek-case-study.md)
- Zstandard 官方 benchmark mode：[programs/zstd.1.md](https://github.com/facebook/zstd/blob/82d322c4973d9e2968d94047a40892bc6d9a9bdf/programs/zstd.1.md)
- Zstandard 官方 build-to-build benchmark：[tests/automated_benchmarking.py](https://github.com/facebook/zstd/blob/82d322c4973d9e2968d94047a40892bc6d9a9bdf/tests/automated_benchmarking.py)
- Zstandard 官方测试说明：[TESTING.md](https://github.com/facebook/zstd/blob/82d322c4973d9e2968d94047a40892bc6d9a9bdf/TESTING.md)

## 可直接转交的任务摘要

> 我们已经决定用 Zstandard 作为 Loreley 的第三个案例。请先根据本 handoff 设计并冻结正式协议，再实现 evaluator、完成噪声校准和小规模 pilot，最后运行 Loreley quality-diversity、champion-sequential 和 root-independent 三个同预算 arm。实验必须区分 training、hidden validation 和 sealed holdout，固定 build provenance，保护 benchmark 与 corpus，并同时报告压缩速度、解压速度、compressed size、memory、正确性和兼容性。前期 M4 Pro 数据显示，Zstandard 的单-level 1% evaluator 约 53 到 58 秒，levels 1 到 5 约 3.6 到 4.0 分钟，但这些参数必须在目标 host 上重新校准。最终交付完整 commits、diff、谱系、raw measurements、统计结果、资源账本和 claim boundary，不需要制作公开 Demo。
