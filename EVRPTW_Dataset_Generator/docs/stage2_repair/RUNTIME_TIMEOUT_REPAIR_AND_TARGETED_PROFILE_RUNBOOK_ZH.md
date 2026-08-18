# Stage-2 超时修复与定点性能诊断运行手册

> 状态：运行时修复已实现并通过测试；Chicago、Dallas、LA 定点运行尚未执行。
> 本文只申请定点诊断，不申请新的 140-family pilot，更不申请 7,500-family full run。

## 1. 旧 pilot 的处置

`us_10city_trainval_pilot_v6` 已判定失败并终止。73 个已完成 family 仅保留为诊断证据，
不得 resume、复制或拼入后续 pilot。人工终止报告位于：

```text
EVRPTW_Dataset/Instances_v2/us_10city_trainval_pilot_v6/
  reports/stage2_repair/manual_termination_report.json
```

已确认的慢 family：

| 用途 | 城市 | family ID | 旧观测 |
|---|---|---|---:|
| 首个 7,200 s STOP trigger | Chicago | `mf_75f0c9cfc1b6c358fc60c916` | 9,139.19 s |
| 终止时仍在运行 | Dallas | `mf_68a04e303dc43462b506b7c0` | 约 10 小时仍未完成 |
| 最大 charger-roster smoke | Los Angeles | `mf_3392e476dea1c527fccb9cc5` | 需重新绑定新代码测量 |

旧 run 的已完成、rejection 和 timing 都不能作为新 pilot 的 acceptance 输入。

## 2. 冻结运行时合同

```text
runtime_contract_id   = family_process_timeout_and_abort_v2
FAMILY_WALL_TIMEOUT_S = 7200
TERMINATION_GRACE_S   = 60
RUNNER_EXIT_SLACK_S   = 30
STOP_POLICY           = abort_all_inflight_after_grace
```

每个 family attempt 都是 supervisor 用 `start_new_session=True` 启动的独立 POSIX
session/process group。supervisor 保存 PID/PGID，并用 monotonic clock 单独计时。

首次 hard STOP 后的状态流固定为：

```text
timeout/hard gate
  -> 立即停止提交 queued family
  -> timeout offender 立即 SIGTERM
  -> 其他 in-flight family 最多保留 60 s 完成原子发布
  -> grace 到期后 SIGTERM 仍在途 process groups
  -> timeout+grace+30 s 前 SIGKILL/reap 所有幸存 descendants
  -> runner 以 passed=false 退出
```

timeout 不走普通 rejection retry：

```text
outcome             = timed_out
reason_code         = family_wall_timeout
retryable           = false
attempt_consumed    = true
retry_stopped_early = true
```

同一 output root 中若已有该 family 的 timeout ledger，resume 直接标为 unresolved，
不会启动新 attempt。

## 3. staging 与原子发布

每个 attempt 的工作目录为：

```text
.inflight/<family_id>/<attempt_id>/
```

完整流程为：

```text
staging materialization
  -> staging family verifier
  -> verifier passed
  -> atomic rename 到 materialized/families/<family_id>
```

正式 reader 只读取 `materialized/families`。timeout/abort 后 `.inflight` 可保留供诊断，
但 supervisor 会把其中状态为 complete 的 `family_manifest.json` 改名为
`family_manifest.timeout_partial.json`，防止 partial artifact 表示为 completed。

## 4. timeout ledger 与 run report

supervisor ledger：

```text
timeouts/<family_id>.json
```

它记录 run/family/attempt、city/track/day/scale、seed、PID/PGID、UTC start/deadline/
termination 时间、monotonic deadline、elapsed、latest heartbeat、SIGTERM/SIGKILL、
stdout/stderr、partial path、global trigger 和 queue/in-flight/cancelled counts。

进程级诊断目录：

```text
.runtime/<run_id>/<family_id>/<attempt_id>/
  envelope.json
  heartbeat.json
  stdout.log
  stderr.log
  result.json
```

hard STOP 的顶层 `stage2_run_report.json` 必须满足：

```text
passed = false
hard_stop_triggered = true
unresolved_count >= 1
runtime_contract.remaining_process_group_count = 0
```

## 5. 性能采样字段

heartbeat 保存全部阶段；成功 family manifest 保存 selection/matrix profile，run result
另存 staging verification profile。采样阶段包括：

- customer preflight；
- customer spatial activation；
- charger preflight；
- full eligible charger-roster batched Dijkstra；
- energy closure 与 charger selection；
- terminal index construction；
- terminal selection 总计；
- selected-terminal matrix construction；
- staging verification。

每个阶段至少保存 wall time、process CPU time、peak RSS。routing workload 额外保存：

- roster/terminal count；
- terminal access-option count；
- physical node、directed edge、turn-transition count；
- distance Dijkstra unique source/target 与 batch count；
- running-time Dijkstra unique source/target 与 batch count。

算法仍使用完整 eligible charger roster、exact directed physical-distance closure 和 exact
zero-turn edge-state running-time closure；profiling 没有加入 Haversine、prefix、lossy prefilter
或静默 skip。

## 6. 测试证据

`tests/test_runtime_supervisor.py` 使用真实 subprocess/process group 和 2 秒级 timeout，覆盖：

1. 超时按时终止；
2. parent 与 grandchild 整组清除；
3. SIGKILL 后 ledger 完整；
4. staging complete manifest 被隔离且正式目录无 completed artifact；
5. queued family 不启动；
6. peer 在 grace 后终止；
7. runner 满足 timeout+grace+slack 上界；
8. resume 不 retry timeout family；
9. SIGINT cleanup；
10. SIGTERM cleanup；
11. worker 退出后残留 grandchild 被判 hard failure 并清除；
12. contract ID 与 stop policy 冻结。

当前验证结果：12/12 runtime tests passed；完整 generator suite 173/173 passed。

## 7. 定点运行前纪律

必须先满足：

```text
branch = stage2-repair-candidate
working tree = clean
candidate commit = pushed
target output root = 不存在的新目录
```

只使用 Git commit provenance 绑定代码；按用户要求不生成或校验额外文件 SHA/哈希清单。
三个 target 各用独立 root，不能复用 v6 的 73 个 materialized families。

## 8. 单 family 定点命令模板

对每个 target，先在新 root 重建绑定当前 commit 的 plan；只允许复用批准的 v4 frozen
customer splits。以下 `<target-root>` 必须替换为一个从未使用的新目录。

```bash
cd /data/Maojie/ICLR/EVRPTW-DB

PYTHON_BIN=/home/npg/miniconda3/envs/maojie/bin/python \
INSTANCE_MODE=non_release_pilot \
FROZEN_SPLIT_ROOT=/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Dataset/Instances_v2/us_10city_trainval_pilot_v4 \
WORKERS=1 \
FAMILIES_PER_WORKER_TASK=1 \
PILOT_FAMILIES_PER_CITY=7 \
INSTANCE_OUTPUT_ROOT=<target-root> \
./generate_instances.sh \
  --stages preflight splits plan \
  --cities new-york los-angeles chicago houston phoenix philadelphia \
           san-antonio san-diego dallas fort-worth \
  --tracks train validation
```

Chicago 或 Dallas：

```bash
PYTHON_BIN=/home/npg/miniconda3/envs/maojie/bin/python \
INSTANCE_MODE=non_release_pilot \
RUN_DISCIPLINE=targeted_profile \
WORKERS=1 \
FAMILIES_PER_WORKER_TASK=1 \
PILOT_FAMILIES_PER_CITY=7 \
FAMILY_WALL_TIMEOUT_S=7200 \
TERMINATION_GRACE_S=60 \
RUNNER_EXIT_SLACK_S=30 \
INSTANCE_OUTPUT_ROOT=<target-root> \
./generate_instances.sh \
  --stages preflight materialize verify \
  --cities new-york los-angeles chicago houston phoenix philadelphia \
           san-antonio san-diego dallas fort-worth \
  --tracks train validation \
  --family-ids <exact-family-id>
```

LA 使用同一命令，但设 `RUN_DISCIPLINE=la_smoke`，family ID 固定为
`mf_3392e476dea1c527fccb9cc5`。

## 9. 定点结果判定与后续优化边界

三个 target 都必须满足：

```text
family wall time < 7200 s
no timeout/abort/unresolved
existing family verifier passed
remaining process group count = 0
performance profile fields complete
```

如果任一 target 超时，立即读取 timeout ledger 的 latest heartbeat 定位阶段；不能换 seed
或直接重跑。只允许基于证据采用 exact 优化：batched many-to-many Dijkstra、source/target
去重、city×depot×day exact cache、turn/depot-star cache、严格 lower-bound exact pruning、
避免重复构建相同 line graph、只读图共享或 mmap。

只有 Chicago < 7,200 s、Dallas < 7,200 s、LA smoke passed、verifier passed、无 orphan、
代码再次 clean commit 并 push 后，才可以把新 140-family pilot 提交 reviewer。新 pilot 必须
使用全新 root、`WORKERS=12`、`FAMILIES_PER_WORKER_TASK=1`，并从 0 开始。
