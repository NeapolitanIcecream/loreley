# Proposal: Embedding Cache 跨实验复用的最小安全版本

## 背景

Loreley 的 repo-state embedding 流程会为仓库中的 eligible files 计算文件级 embedding，再聚合成 commit/repo-state embedding。大型仓库的 root bootstrap 可能需要处理大量文件，这会带来明显的 API 成本和启动等待时间。

现有实现已经有两层缓存：

- 文件级 cache：`map_elites_file_embedding_cache`，以 `blob_sha` 为 key，存储文件级 embedding vector。
- commit 级 aggregate：`map_elites_repo_state_aggregates`，以 `commit_hash` 为 key，存储 `sum_vector` 和 `file_count`。

这些缓存目前绑定在单个 Loreley experiment database 内。因为 smoke、canary、long run 通常使用不同 DB，即使它们指向同一个仓库、同一个 root commit、同一套 embedding 配置，后续实验也无法直接复用 smoke/canary 已经计算出的文件 embedding。

## 问题

最常见的问题是：smoke 阶段为了验证链路已经计算过 root repo-state embedding，但正式 long run 使用新 DB 后仍然会重新为相同文件内容支付 embedding 成本。

这个问题不应通过引入一个全新的 embedding 系统解决。它应该作为现有 repo-state 文件级 cache 的增量能力来处理：让一个 experiment DB 可以显式、安全地导入另一个兼容 experiment DB 中的文件级 embedding cache。

## 目标

最小安全版本只解决一个问题：

> 在兼容性可以被明确证明或由 operator 显式声明时，将一个 DB 中的 repo-state 文件级 embedding cache 导入另一个 DB，避免重复调用 embedding API。

具体目标：

- 复用 `map_elites_file_embedding_cache` 中的 `blob_sha -> vector` rows。
- 复用前校验 embedding 语义 fingerprint。
- 不覆盖目标 DB 中已有 cache rows。
- 不改变 repo-state embedding 的现有执行流程。
- 不让 `CodeEmbedder` 承担 cache 语义。
- 为不能复用的情况输出明确原因。
- 支持旧版本 cache 的显式 attestation，但不自动信任旧 cache。

## 非目标

为了保持简单和安全，MVP 明确不做以下内容：

- 不自动发现 smoke/canary/long run 的继承关系。
- 不按 phase 名称、run 目录或 symlink 自动找上游 DB。
- 不导入 `map_elites_repo_state_aggregates`。
- 不复用 commit/repo 级 aggregate。
- 不引入全局 cache DB、cache service、read-through cache、后台同步、TTL 或 eviction。
- 不改造所有 embedding 路径。
- 不让 `CodeEmbedder` 查询 DB cache。
- 不支持没有 manifest 的旧 DB 自动参与复用。
- 不提供 per-file 级别的 miss 原因诊断。
- 不新增 UI、dashboard 或指标系统。

## 设计原则

### 只复用文件级 cache

MVP 只跨 DB 复用文件级 embedding cache：

```text
blob_sha -> file embedding vector
```

目标 DB 仍然自己执行 root bootstrap，重新枚举 eligible files，并根据文件级 vectors 聚合 root commit aggregate。

这样可以省掉绝大多数 embedding API 成本，同时避免复用 commit aggregate 带来的 eligibility、ignore rule、file count 和聚合语义风险。

### 不复用 commit aggregate

commit aggregate 的语义比文件级 vector 更强。它依赖：

- root commit 或 candidate commit 的 eligible file set。
- ignore rule。
- preprocess 配置。
- max file size。
- 文件级 vector。
- aggregate 算法。
- repo root/subpath 解释。

如果跨 DB 直接导入 `commit_hash -> sum_vector + file_count`，后续增量更新可能缺少旧 blob 的文件级 vector，尤其是修改文件时需要从 parent aggregate 中减掉 old blob vector，再加上 new blob vector。

因此 MVP 不导入 commit aggregate。目标 DB 在导入文件级 cache 后，自行 bootstrap aggregate，并在后续 incremental ingestion 中继续使用目标 DB 内的文件级 cache 和新生成的 aggregate。

### Cache 判断发生在 CodeEmbedder 上游

现有 repo-state 流程是：

1. 枚举 commit 下的 eligible files。
2. 提取每个文件的 `blob_sha`。
3. 查询 `map_elites_file_embedding_cache`。
4. 只对 cache misses 执行 preprocess、chunk 和 embedding。
5. 将新 vector 写回文件级 cache。
6. 用 cached vectors 和新 vectors 聚合 commit embedding。

跨 DB 导入只影响第 3 步的命中率。`CodeEmbedder` 仍然只负责对传入的 chunked text 调用 embedding API，不参与 DB cache 查询。

## Embedding 语义 Fingerprint

文件级 cache 不能只用 `blob_sha`、model、dimensions 判断兼容性。相同 blob 内容在不同 preprocess/chunk 配置下可能产生不同文件级 vector。

MVP 引入 `embedding_semantics_fingerprint`，用于判断 source DB 和 target DB 的文件级 cache 是否可以复用。

fingerprint 至少包含：

- embedding model。
- embedding dimensions。
- embedding provider identity 的非敏感形式：
  - provider kind/name。
  - normalized base URL origin/path，不包含用户名、密码、query 或 token。
  - resolved model id，如果 provider 会把请求 model alias 映射到实际模型。
- preprocess 配置：
  - allowed extensions。
  - allowed filenames。
  - excluded globs。
  - max file size。
  - strip comments。
  - strip block comments。
  - max blank lines。
  - tab width。
- chunk 配置：
  - target lines。
  - min lines。
  - overlap lines。
  - max chunks per file。
  - boundary keywords。
- repo-state file embedding algorithm version。
- file aggregation algorithm version。

fingerprint 不需要包含当前 experiment id。跨实验复用正是它要允许的场景。

fingerprint 应由 canonical JSON 计算，例如：

```json
{
  "kind": "repo_state_file_embedding",
  "schema_version": 1,
  "embedding_provider": {
    "provider": "openai",
    "base_url": "https://api.openai.com/v1",
    "requested_model": "text-embedding-3-small",
    "resolved_model": "text-embedding-3-small"
  },
  "embedding_dimensions": 1536,
  "preprocess": {
    "allowed_extensions": [".py", ".ts"],
    "allowed_filenames": ["Dockerfile", "Makefile"],
    "excluded_globs": ["tests/**", "node_modules/**"],
    "max_file_size_kb": 512,
    "strip_comments": true,
    "strip_block_comments": true,
    "max_blank_lines": 2,
    "tab_width": 4
  },
  "chunk": {
    "target_lines": 80,
    "min_lines": 20,
    "overlap_lines": 8,
    "max_chunks_per_file": 64,
    "boundary_keywords": ["def ", "class ", "function "]
  },
  "algorithm": {
    "repo_state_file_embedding": "v1",
    "file_chunk_aggregation": "weighted_average_v1"
  }
}
```

具体列表应跟随当前实现收敛，但原则是：任何会改变文件级 vector 的配置或算法都必须进入 fingerprint。

`allowed_extensions`、`allowed_filenames`、`excluded_globs` 这类 eligibility scope 严格来说主要决定哪些 blob 会参与 repo-state embedding，不一定改变单个 blob 的文件级 vector。MVP 仍然把它们纳入 fingerprint，是为了保持边界简单且保守。代价是某些理论上安全的跨 scope 复用会被拒绝。

## 数据模型

新增一张小表用于记录 cache manifest：

```text
embedding_cache_manifests
- id uuid primary key
- cache_kind varchar(64) not null unique
- fingerprint varchar(64) not null
- payload jsonb not null
- source varchar(64) not null
  -- generated | operator_attested
- created_at timestamptz not null
- updated_at timestamptz not null
```

MVP 可以只支持一个 active manifest：

```text
cache_kind = 'repo_state_file_embedding'
```

`cache_kind` 必须唯一。MVP 不引入 `active` flag；如果未来需要保留历史 manifest，再通过单独的 history 表或显式 `active` 语义扩展，避免当前 import/startup 读取多条 manifest 时出现不确定行为。

fresh DB 或 scheduler/bootstrap 初始化时，manifest 缺失时的处理必须 fail-closed：

- manifest 不存在且 `map_elites_file_embedding_cache` 为空：可以用当前 settings 生成并写入 `source='generated'`。
- manifest 不存在但 `map_elites_file_embedding_cache` 非空：不能自动生成。必须失败，或者要求 operator 先执行 attestation。

如果 manifest 已存在但 fingerprint 和当前 settings 计算值不一致，启动或 import 应失败，并提示 DB 与当前 embedding 语义不兼容。

## 导入命令

新增显式导入命令：

```bash
uv run loreley embedding-cache import \
  --source-dsn "$SMOKE_DATABASE_URL"
```

命令使用当前 `DATABASE_URL` 作为 target DB。

流程：

1. 连接 source DB 和 target DB。
2. 读取 source manifest。
3. 读取或按 fail-closed 规则生成 target manifest。
4. 比较 `cache_kind` 和 `fingerprint`。
5. 若 fingerprint 不一致，拒绝导入。
6. 从 source `map_elites_file_embedding_cache` 读取 rows。
7. 对每个 source row 做 row-level validation。
8. 向 target insert rows，冲突时 `DO NOTHING`。
9. 输出导入摘要。

row-level validation 至少检查：

- `embedding_model` 与 manifest payload 中的 expected stored model 一致，通常是 `embedding_provider.requested_model`；旧 manifest 可以使用顶层 `embedding_model` 兼容字段。
- `dimensions` 与 manifest payload 中的 `embedding_dimensions` 一致。
- `vector` 非空。
- `vector` 长度等于 `dimensions`。

任何 source row 校验失败时，import 应失败，不把该批坏数据写入 target。不能只比较 manifest 后直接 `INSERT ... DO NOTHING`。

输出示例：

```text
Embedding cache import complete
source_rows=125842
inserted_rows=118902
already_present_rows=6940
skipped_rows=0
fingerprint=4d3b...
source_manifest=generated
target_manifest=generated
```

## 旧 Cache 接入

旧版本 DB 可能没有 manifest。它通常只有：

- `blob_sha`
- `embedding_model`
- `dimensions`
- `vector`

这些信息不足以自动证明 cache 兼容，因为缺少 preprocess、chunk 和算法版本信息。

因此 MVP 的默认行为是：

- source DB 没有 manifest 时，拒绝 import。
- target DB 没有 manifest 且 cache 为空时，可以用当前 settings 生成 manifest。
- target DB 没有 manifest 但 cache 非空时，拒绝 import。
- 旧 source DB 必须经过 operator attestation 才能作为 import source。

### Attestation 命令

提供显式命令：

```bash
uv run loreley embedding-cache attest \
  --database-url "$OLD_DATABASE_URL" \
  --from-current-settings
```

该命令表示 operator 声明：旧 DB 中的文件级 cache 与当前 settings 计算出的 embedding semantics 兼容。

命令执行：

1. 检查 DB 中是否已有 manifest。
2. 检查 `map_elites_file_embedding_cache` 中是否只有一个 model/dimensions 组合。
3. 用当前 settings 计算 fingerprint。
4. 写入 manifest，`source='operator_attested'`。
5. 输出明确警告：这是 operator 声明，不是系统自动证明。

也可以支持更显式的形式：

```bash
uv run loreley embedding-cache attest \
  --database-url "$OLD_DATABASE_URL" \
  --fingerprint "$EXPECTED_FINGERPRINT"
```

MVP 不根据旧 Loreley 版本号自动推导 fingerprint。

## 日志和可观测性

MVP 只要求 CLI 和启动日志清楚，不新增 UI 或指标系统。

导入时必须输出：

- source DB。必须 sanitize DSN，不能输出用户名、密码、token 或 query secret。
- target DB。必须 sanitize DSN，不能输出用户名、密码、token 或 query secret。
- source manifest source。
- target manifest source。
- fingerprint。
- source rows。
- inserted rows。
- already-present rows。
- skipped rows。
- 拒绝导入时的具体原因。

root bootstrap 日志建议补充 cache 摘要：

```text
Repo-state root bootstrap cache summary commit=<sha> eligible_files=<n> unique_blobs=<n> hits=<n> misses=<n> embedded=<n> fingerprint=<hash>
```

这样 operator 可以快速判断正式实验是否复用了 smoke/canary 的 embedding cache。

成功判据：导入后，目标 DB 的 root bootstrap 日志中 `misses` 应显著低于未导入时的水平，理想情况下对相同 root/scope 接近 0。如果仍然长时间看到大量 embedding 调用，说明可能是 fingerprint/import 未命中，或瓶颈来自不受 repo-state 文件级 cache 覆盖的其他 embedding 路径。

## 上游空响应恢复性

大型仓库 root bootstrap 可能触发大量 embedding batch。某些 OpenAI-compatible provider 在偶发情况下会返回 HTTP 200，但 SDK 解析后没有 embedding data。这个问题应在 embedding 调用边界做有限恢复，避免一个瞬时空响应直接终止 scheduler startup。

MVP 处理方式：

- 只识别明确的空 embedding data 场景：
  - SDK 抛出 `ValueError("No embedding data received")`。
  - response 的 `data` 缺失或为空。
  - 单个 response item 的 `embedding` 缺失或为空。
- 将这些场景转换成本地 retryable exception。
- 复用 `MAPELITES_CODE_EMBEDDING_MAX_RETRIES` 和 `MAPELITES_CODE_EMBEDDING_RETRY_BACKOFF_SECONDS`。
- 不重试普通 `ValueError`。
- usage event 只在 response validation 完成后记录；无效响应不记录 usage。
- retry 耗尽后保留明确错误信息，例如 `Embedding API returned no embedding data.`。

这不是 repo-state bootstrap checkpoint。MVP 不改变当前事务边界，也不把已经成功的 batch 分批提交。若 provider 持续返回空响应，bootstrap 仍会失败；后续是否引入分批 cache checkpoint 应单独评估。

## 正确性约束

### Insert-only

导入时不覆盖目标 DB 中已有 cache rows。目标 DB 的已有 rows 保持优先。

如果 source 和 target fingerprint 一致，但同一个 `blob_sha` 已存在于 target，导入忽略该 row。

### Fingerprint mismatch fail-closed

fingerprint 不一致时必须拒绝导入。不能降级成 warning 后继续。

### No manifest fail-closed

source DB 没有 manifest 时必须拒绝导入，除非 operator 先执行 attestation。

target DB 没有 manifest 且文件级 cache 非空时也必须拒绝导入，除非 operator 先执行 attestation。不能自动生成 manifest 来“洗白”未知来源的旧 rows。

### Source row validation fail-closed

导入前必须校验 source rows 的 model、dimensions 和 vector 形状。任一 row 校验失败时，import 失败，避免把损坏或不兼容的 cache row 带入 target。

### 不改变 runtime embedding 语义

导入只是提前填充目标 DB 的文件级 cache。repo-state bootstrap 和 incremental ingestion 仍然按现有逻辑决定 cache hit/miss、执行 missing file embedding、聚合 commit vector。

## 增量更新安全性

这个设计不会破坏后续实验中的 embedding 增量更新。

原因是目标 DB 不导入 commit aggregate，而是使用导入后的文件级 cache 自己生成 root aggregate。后续 candidate commit 的 incremental ingestion 仍然有完整的 parent aggregate，并且可以从目标 DB 的文件级 cache 读取 diff 中 old blob 和 new blob 的 vector。

如果某个 new blob 不在导入 cache 中，现有逻辑会把它作为 miss，进入 `CodeEmbedder`，然后写回目标 DB。

因此导入文件级 cache 只会减少 API 调用，不会绕过增量更新所需的数据生成过程。

## 测试计划

MVP 测试覆盖：

- target DB 无 manifest 且 cache 为空时，import 前自动生成 target manifest。
- target DB 无 manifest 且 cache 非空时，拒绝 import。
- source/target fingerprint 一致时，导入 source 中不存在于 target 的 cache rows。
- 已存在 rows 不覆盖。
- source/target fingerprint 不一致时拒绝导入。
- source DB 无 manifest 时拒绝导入。
- source row 的 model、dimensions 或 vector 长度不合法时拒绝导入。
- attestation 为旧 source DB 写入 `operator_attested` manifest。
- attestation 拒绝 model/dimensions 混杂的旧 cache。
- manifest 表对 `cache_kind` 有唯一性约束。
- root bootstrap 在导入后能从目标 DB 命中文件级 cache，并减少 `cache_misses`。
- 不导入 `map_elites_repo_state_aggregates`。
- embedding SDK 首次返回 `No embedding data received`、第二次成功时，batch retry 后返回 embedding。
- embedding response 连续缺失 data 时，按配置次数 retry 后失败，且不记录 usage event。
- 普通 `ValueError` 不被 embedding retry 机制重试。

## 推荐实施顺序

1. 增加 fingerprint 计算函数和单元测试。
2. 增加 `embedding_cache_manifests` 表和 migration。
3. 在 DB 初始化或 repo-state bootstrap 前确保 target manifest 存在且兼容当前 settings。
4. 增加 `embedding-cache attest` 命令。
5. 增加 `embedding-cache import` 命令。
6. 补充 root bootstrap cache summary 日志。
7. 增加跨 DB 导入的集成测试。

## 结论

这个方案是现有 repo-state 文件级 embedding cache 的增量增强，不是独立 embedding 子系统。

MVP 只做显式、可审计、fail-closed 的跨 DB 文件级 cache 导入。它避免重复 embedding API 成本，同时不复用 commit aggregate、不改变 `CodeEmbedder`、不自动信任旧 cache，从而把实现复杂度和正确性风险控制在较小范围内。
