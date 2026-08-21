# Loreley design-partner brief

Loreley 正在寻找有真实代码库、自动 evaluator 和明确改进价值的设计合作方。合作目标是在双方预先冻结的协议下运行一次仓库级搜索，交付可审计的候选 diff、评测结果、谱系和资源账本。

合作以无人值守的 evaluator 为起点。Evaluator 把 correctness gates 和目标指标转成结构化结果，双方据此冻结实验协议和预算。

方法、受控实验和三个 capability cases 见 [arXiv:2608.19703](https://arxiv.org/abs/2608.19703)。论文在 Zstandard 上用 1,008 个 physical candidate jobs 比较 Loreley QD、Sequential Champion 和 Independent Root：archive retention 和后续采样实际发生，但 48-job endpoint 没有建立 QD 相对两种 baseline 的优势。合作协议不会预设 QD 必然胜出。

## 适合的场景

一个目标通常需要满足：

- 构建、测试和性能评测可以由脚本或服务自动运行；
- evaluator 能输出 pass/fail 和至少一个可比较的数值指标；
- 一次有效改善有可估算的工程或商业价值；
- 仓库存在多种实现路径，允许 agent 修改不止一个常数或参数；
- 可以提供隔离的代码镜像、构建环境和与目标相称的算力。

当前优先场景包括压缩与存储、数据库与查询执行、编译器和 EDA、SAT/SMT、推理 serving、调度、企业 Java/C++ 性能热点，以及有仿真或回放 evaluator 的内部算法。

目标项目不限语言。Loreley 的 evaluator 入口使用 Python，但可以调用任意构建系统、容器、硬件测试台或远程评测服务。

## 合作前需要的最小信息

请先提供不含机密内容的以下信息：

| 信息 | 需要回答的问题 |
| --- | --- |
| 目标 | 要改善什么指标，方向是什么，1% 改善值多少钱或多少工程时间 |
| 正确性 | 哪些测试、协议、形式验证或业务约束必须通过 |
| 时延 | 一次 training evaluation 的 P50 和 P95，完整 validation 需要多久 |
| 噪声 | 同一 binary 或部署重复测量时，典型 CV、配对差异或置信区间是多少 |
| 并发 | evaluator 可以安全运行多少 lanes，是否共享 CPU、GPU、I/O 或外部配额 |
| 范围 | agent 可以修改哪些文件，哪些目录、接口和依赖必须保护 |
| 数据 | 哪些数据可用于 training，哪些可以保留为 validation 和 sealed holdout |
| 身份 | 什么变化才需要重新评测：Git tree、binary、容器镜像、trace 还是部署版本 |
| 预算 | 可用的 model requests、tokens、candidate evaluations、device-hours 和日历时间 |

如果目前不知道 evaluator 噪声，第一阶段先做 calibration，再根据结果确定搜索规模。

## 建议合作流程

### 1. Evaluator calibration

双方先固定 root revision、构建环境、工作负载和指标定义。对 root/root 运行交错或配对重复，测量 lane bias、噪声、P50/P95 和失败率。1% 改进是否可测，由这些结果决定。

产出：evaluator contract、baseline noise report、candidate identity 规则、推荐并发和停止条件。

### 2. Frozen pilot

在模型调用前封存：

- protected scope 和 correctness gates；
- training、validation 和 sealed holdout；
- seed policy；
- archive objectives；
- finalist 数量与排序规则；
- job、unique-identity、token、评测时间和日历时间上限。

一个可讨论的起始范围是 128–256 个 physical jobs，并验证至少 training Top 10。它不是固定套餐：evaluation 时延、模型时延和候选重复率决定实际预算。若一次 evaluator 需要数小时，pilot 应使用 staged gates、早停和更小的 finalist 集合。

产出：全部 terminal outcomes、通过 gate 的候选、主要谱系、独立 validation、一个预登记 holdout 结论、失败分类和资源账本。

### 3. 结果复盘与扩展决定

Pilot 后共同决定：扩大 unique-identity endpoint、调整 evaluator、运行同预算搜索基线，或停止。停止本身是有效结果；如果 signal-to-noise、候选成功率或经济价值不足，报告会保留原因，不用继续消费预算。

如果合作目标包含 target-specific 的策略比较，还需像已发表的 Zstandard 实验一样运行 Quality-Diversity、Independent Root 和 Sequential Champion 三个同预算 arms，并加入独立搜索重复。跨机器复现是否必要取决于目标指标和部署环境。双方会在 pilot 后单独确定这部分预算。

## 双方投入

合作方提供：

- 可授权使用的仓库镜像和目标 revision；
- 可重复构建环境；
- correctness 和性能 evaluator；
- domain owner 与 evaluator owner；
- 约定的计算资源和数据边界；
- 对候选可维护性与业务价值的最终判断。

Loreley 项目方提供：

- evaluator contract 和实验协议设计；
- Loreley target adapter、scope gates、运行编排和审计记录；
- candidate identity、cache 和失败处理方案；
- 搜索运行、独立验证、谱系分析和资源核算；
- 可由合作方审核的候选 diff 与证据报告。

## 数据与访问边界

- 不要在公开 GitHub issue 中提交私有代码、数据、凭据、内部主机名或未披露指标。
- 公开 issue 只用于提交非机密摘要；私有材料需要另行约定传输、保留和删除方式。
- 在使用外部 coding model 前，双方必须确认代码和上下文是否允许发送给该 provider。需要本地或指定 provider 时，应在协议中固定。
- Evaluator 可以留在合作方环境中，但需要提供可审计的输入、输出、版本和失败语义。
- 是否公开 winner、diff、数据、成本和合作方名称逐项约定。默认不把私有仓库内容写入公开案例。

## 提交场景

请使用 [design-partner intake](https://github.com/NeapolitanIcecream/loreley/issues/new?template=design-partner.yml) 提交非机密摘要。至少包括仓库类型、目标指标、evaluation P50/P95、可用并发、已知噪声和预算范围。

若这些信息还不齐，可以先提交 evaluator calibration 需求。Calibration 会先测量噪声、时延和可用并发，再确定是否进入搜索。

## 现有证据

- [论文：方法、1,008-job matched policy experiment 与三个 capability cases](https://arxiv.org/abs/2608.19703)
- Matched Zstandard experiment：7 个配对 block，每个 policy/block 48 jobs；QD 相对 Sequential Champion 为 -0.135%（95% BCa 区间 -0.556% 至 +0.161%），相对 Independent Root 为 +0.320%（-0.082% 至 +0.686%）；两项比较均未建立 QD 优势
- Archive engagement：4/7 个最终 QD winners 的 primary-parent ancestry 包含 retained non-incumbent；计入 inspiration context 后为 6/7，但后一个计数不证明 context 造成了 edit
- [三案例统一证据报告](../research/2026-08-07-loreley-case-study-evidence-report.md)
- [`markdown-it-py`：前瞻性 6.75% throughput gain](../research/2026-08-02-markdown-it-py-deepseek-case-study.md)
- [`python-pathspec`：reference workloads +25.14%，候选为 post-hoc selection](../research/2026-08-03-pathspec-deepseek-case-study.md)
- [Zstandard：预登记结果与 Top 10 holdout 补测](../research/2026-08-07-zstandard-gpt-v19-case-study-report.md)

后三项 capability campaigns 共 348 jobs，与 1,008-job matched experiment 分开报告。新仓库能够取得的收益取决于 evaluator 质量、可搜索空间和运行预算；现有结果不提供跨仓库成功率或平均收益估计。
