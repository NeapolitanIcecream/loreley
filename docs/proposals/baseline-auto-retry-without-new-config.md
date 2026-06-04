# Proposal: Baseline auto-retry without new user configuration

## Status

Accepted / Implemented

## Date

2026-06-04

## Context

Loreley 在 campaign 开始调度前，会先为 root commit 建立一条
`campaign_baselines` 记录。这个 baseline 是后续比较改进幅度的基准。

当前默认策略是：

- `BASELINE_BOOTSTRAP_POLICY=required`：baseline 不可用时，scheduler 不 dispatch、不 seed、不 schedule。
- `BASELINE_BOOTSTRAP_POLICY=warn`：baseline 不可用时继续调度，但记录 degraded 状态，baseline delta 不可用。

这个 gate 本身是必要的。它防止系统在没有可靠基准的情况下消耗 worker 预算，也防止后续结果变得不可比较。

问题出在失败恢复上。

`BaselineBootstrapService.ensure_or_load_baseline()` 会按 `baseline_key_hash` 查找已有 baseline。如果已有行状态是 `valid`、`failed` 或 `degraded`，并且调用方没有传 `force_rerun=True`，它会直接复用这条记录。

Scheduler 调用这个方法时不会传 `force_rerun=True`。因此，在默认 `required` 策略下，如果某次 baseline evaluator 因环境问题失败，失败行会被持久化。之后每个 scheduler tick 都只会读到这条失败行，然后直接被 baseline gate 挡住。Evaluator 不会再被调用，环境恢复后也不会自动自愈。

用户看到的症状是反复出现类似日志：

```text
Scheduler tick blocked by campaign baseline status=failed
```

这不是 evaluator 专属问题。Evaluator 第一次失败可能来自 Docker、服务健康检查、权限、超时或外部依赖。但失败行永久挡住 scheduler，是 Loreley scheduler/baseline 复用逻辑的问题。

## Problem

当前系统把非 valid baseline 当成了长期结论：

- `valid` 行复用是正确的。
- `failed` / `degraded` 行永久复用会让 transient failure 变成 permanent stuck。
- operator 可以手动 force rerun，但这要求用户知道内部状态并主动介入。

直接新增配置可以解决，例如：

```text
BASELINE_BOOTSTRAP_RETRY_FAILED=true
BASELINE_BOOTSTRAP_RETRY_COOLDOWN_SECONDS=300
```

但这会增加用户更新成本。用户需要学习新配置、决定开关和冷却时间，还要在不同部署里同步 `.env`。对于一个应该默认自愈的调度器，这个配置面太重。

## Goals

- 不新增用户可见配置。
- 不改变 `BASELINE_BOOTSTRAP_POLICY` 的含义。
- 保留 baseline gate，不让无有效基准的 campaign 在 `required` 下继续调度。
- 让 transient baseline failure 在环境恢复后自动自愈。
- 避免每个 scheduler tick 都重跑 evaluator。
- 保留 operator 手动 `force_rerun=True` 重跑非 valid baseline 的行为。
- 保持 valid baseline 的稳定复用。

## Non-Goals

- 不移除 baseline-first gate。
- 不把 evaluator 内部重试和 scheduler 级恢复混在一起。
- 不要求用户迁移数据库或修改 `.env`。
- 不保证所有配置错误都能自愈。比如 primary metric 配错时，自动重试不会让配置变正确。

## Proposed Design

把 failed/degraded baseline 从“永久结论”改成“带内置冷却期的可重试状态”。

不新增环境变量。代码内部使用一个保守常量：

```python
_BASELINE_RETRY_COOLDOWN_SECONDS = 300
```

行为规则：

1. `valid` baseline 永远复用。
2. `failed` / `degraded` baseline 在冷却期内复用。
3. 冷却期到了，如果失败类型可重试，则重新运行 baseline evaluator。
4. 重试成功后，更新同一条 `campaign_baselines` 行为 `valid`。
5. 重试失败后，更新同一条行的失败信息和完成时间，下一次继续等待冷却期。
6. `force_rerun=True` 继续作为 operator 对非 valid baseline 的手动覆盖入口。
7. `valid` baseline 即使传入 `force_rerun=True` 也继续复用，避免破坏已经建立的 campaign 基准。

冷却时间基准使用已有字段，不引入 schema 变更：

1. 优先 `finished_at`
2. 其次 `updated_at`
3. 最后 `created_at`

这些字段来自数据库或测试替身时可能是 timezone-aware，也可能是 naive。
实现必须先把时间归一化为 UTC 再比较；naive datetime 按 UTC 解释，避免在
SQLite、PostgreSQL 和测试 fake 之间出现 offset-aware/offset-naive 比较错误。

并发边界保持简单：scheduler 本身已有 experiment advisory lock，但 operator
baseline ensure 可能和 scheduler 自动重试同时触发。实现不在 evaluator 运行期间
长时间持有数据库行锁；如果发生并发触发，最多可能重复运行一次 evaluator，最后仍由
`baseline_key_hash` 唯一约束和 `_persist_baseline_attempt()` 覆盖同一条 row。

## Retry Classification

不要让明显的配置错误无限重试。Baseline service 可以内置一个小的分类函数。

建议默认重试这些失败：

- `baseline_evaluation_failed`
- `evaluator_error`
- `infrastructure_error`
- `timeout`
- `worker_timeout`
- `service_unavailable`
- `evaluation_missing_result`

建议默认不重试这些失败：

- `primary_metric_not_configured`
- `primary_metric_missing`
- `primary_metric_non_finite`
- `primary_metric_direction_conflict`

直觉上可以这样理解：

- evaluator 没跑成，或者 evaluator 报运行错误：可以等环境恢复后自动再试。
- evaluator 跑成了，但结果不符合 campaign contract：需要修配置或修 evaluator，自动重试大概率没用。

对未知失败类型，建议保守处理：

- 如果 `failure_kind` 以 `_error` 结尾，可以按可重试处理。
- 其他未知失败先不自动重试，保留 operator force rerun。

## Implementation Plan

主要修改 `loreley/scheduler/baselines.py`。

新增内部 helper：

```python
_BASELINE_RETRY_COOLDOWN_SECONDS = 300

def _baseline_retry_reference_time(row: CampaignBaseline) -> datetime | None:
    ...

def _baseline_retry_cooldown_elapsed(row: CampaignBaseline, *, now: datetime) -> bool:
    ...

def _baseline_timestamp_as_utc(value: datetime | None) -> datetime | None:
    ...

def _baseline_failure_is_retryable(row: CampaignBaseline) -> bool:
    ...

def _should_retry_existing_baseline(row: CampaignBaseline, *, now: datetime) -> bool:
    ...
```

然后把 `ensure_or_load_baseline()` 里的复用逻辑从：

```python
if existing is not None and existing.status in _RECORDED_BASELINE_STATUSES:
    return self._result_from_row(existing, key_hash=key.hash, policy=policy)
```

调整为：

```python
if existing is not None and existing.status == BASELINE_STATUS_VALID:
    return self._result_from_row(existing, key_hash=key.hash, policy=policy)

if (
    existing is not None
    and existing.status in {BASELINE_STATUS_FAILED, BASELINE_STATUS_DEGRADED}
    and not force_rerun
    and not _should_retry_existing_baseline(existing, now=datetime.now(timezone.utc))
):
    return self._result_from_row(existing, key_hash=key.hash, policy=policy)

# otherwise evaluate and persist into the same row
```

如果准备自动重试，写一条清晰日志。日志复用现有 `scheduler.baselines`
logger，字段要包含 key、prior status、failure kind 和 cooldown：

```text
Campaign baseline retrying key=<hash> status=failed failure_kind=evaluator_error cooldown_seconds=300
```

重试结果仍走现有 `_persist_baseline_attempt()`，它已经会按 key 找同一条 row 并覆盖状态、metric、failure summary 和时间戳。

## User-Facing Behavior

用户不需要改任何配置。

`BASELINE_BOOTSTRAP_POLICY=required` 的含义保持简单：

- baseline valid：可以调度。
- baseline invalid：暂时不能调度。

新增的默认恢复逻辑只改变一件事：如果 invalid 是可重试失败，scheduler 会隔一段时间自己再试，而不是永远卡住。

`BASELINE_BOOTSTRAP_POLICY=warn` 也会受益。Degraded baseline 期间可以继续调度，但 service 会在冷却期后尝试恢复成 valid，让 baseline delta 重新可用。

## Acceptance Criteria

- failed same-key baseline 不会永久挡住 campaign。
- scheduler 不会每个 tick 都重跑 baseline evaluator。
- valid baseline 仍然不重跑。
- cooldown 内的 failed/degraded baseline 仍然复用旧行。
- cooldown 后的 retry 成功时，同一条 row 更新为 `valid`。
- cooldown 后的 retry 失败时，同一条 row 更新失败信息和 `finished_at`。
- manual `force_rerun=True` 仍然能立即重跑非 valid baseline。
- `required` 下 baseline 未 valid 前仍然阻塞调度。
- `warn` 下 degraded baseline 仍然允许调度。

## Test Plan

Focused tests:

```bash
uv run pytest tests/scheduler/test_baseline_bootstrap.py tests/scheduler/test_baseline_scheduler_gate.py
```

新增或更新测试：

- failed row 在 cooldown 内不会重跑 evaluator。
- failed row 超过 cooldown 后会重跑 evaluator。
- retry 成功会把同一条 row 更新为 `valid`。
- retry 失败会更新同一条 row 的 failure summary 和 finished time。
- valid row 即使超过 cooldown 也不会重跑。
- non-retryable failure kind 不会自动重跑。
- `force_rerun=True` 仍然立即重跑非 valid row。
- scheduler 在 retry 成功后不再返回 `baseline_blocked=1`。

Related tests after implementation:

```bash
uv run pytest tests/scheduler tests/api/test_operator_routes.py tests/api/test_app.py
```

## Risks

The main risk is retrying a failure that is not actually transient. The cooldown keeps this from becoming a busy loop, and the retry classifier avoids obvious campaign contract errors.

Another risk is hiding a persistent environment problem behind periodic retries. The fix should therefore log each retry attempt clearly, including key, prior status, failure kind, and cooldown.

A small concurrency risk remains: scheduler and operator can both decide to retry the same
non-valid row before either one persists the result. This is acceptable because it is bounded
to rare operator overlap, avoids holding database locks across external evaluator work, and
the unique `baseline_key_hash` row remains the durable source of truth.

## Alternatives Considered

### Add retry configuration

Rejected for now.

It is flexible, but it expands the user-facing configuration surface for behavior that should be a safe default. Most users should not need to know that baseline retry exists.

### Put retry logic in scheduler main loop

Rejected.

Scheduler should ask whether the baseline is ready. The decision to reuse or re-evaluate a baseline belongs inside `BaselineBootstrapService`, next to baseline keying and persistence.

### Retry every failed/degraded baseline unconditionally

Rejected.

This can waste evaluator budget on configuration problems such as missing primary metric. A small internal retry classifier gives better default behavior without adding user configuration.
