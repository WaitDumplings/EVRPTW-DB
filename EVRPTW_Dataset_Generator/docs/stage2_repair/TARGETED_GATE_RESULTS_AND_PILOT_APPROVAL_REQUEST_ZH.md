# Stage-2 三个定向 Gate 结果与 140-family Pilot 放行申请

日期：2026-08-18（America/Los_Angeles）  
候选分支：`stage2-repair-candidate`  
候选 Git commit：`a52e8fffd00e0c9c96e0fdbe205e6a6576346fbd`

本文只使用 Git commit provenance，不生成或校验 SHA256/文件哈希清单。

## 1. 申请结论

Chicago、Dallas 和 Los Angeles 最大 charger-roster smoke 已全部在独立、从未使用的新 root
中完成。三个 family 均满足：

```text
family wall time < 7200 s
materialized = true
existing verifier passed = true
rejected = 0
timed out = 0
unresolved = 0
remaining process group count = 0
```

LA smoke 的冻结 stop rule 返回 `GREEN`、`pilot_allowed=true`、
`exact_performance_optimization_required=false`。因此申请 reviewer 批准下一步：在另一个全新
root 中，以 `WORKERS=12`、`FAMILIES_PER_WORKER_TASK=1` 从 0 开始运行 140-family
non-release pilot。

本申请不授权直接生成 release artifact、压缩包或 restore bundle。

## 2. 代码与测试冻结状态

本次 candidate 包含：

- region growth 的 exact community×decile capacity cache；
- exact incremental frontier 与 crossing-min cache；
- global assignment 的 exact maximum-flow feasibility gate；
- region competition deterministic sequence 的惰性 prefix cache；
- feasibility 的指数搜索与二分搜索；
- 在 first feasible round 上运行原 `nx.min_cost_flow`。

没有改变 customer/charger roster、quota、seed、tie-break、D-1～D-6、5:2 day type、CS 目标、
charging power、zero-turn 或 schema v2。test-only reference 实现仍保留；新增多轮 first-feasible
与完全不可行边界 differential tests。完整 generator suite：`207/207 passed`。

运行前状态：candidate commit 已 push，working tree clean。三个 plan 的 code provenance 都绑定
上述同一 commit，并复用批准的 v4 frozen customer splits。

## 3. 三个 fresh-root 结果

| Target | Family | Run wall | Materialize | Terminal selection | Parent matrix | Verifier | Runtime contract |
|---|---|---:|---:|---:|---:|---|---|
| Chicago | `mf_75f0c9cfc1b6c358fc60c916` | 399.011 s | 388.366 s | 323.651 s | 48.588 s | passed | clean |
| Dallas | `mf_68a04e303dc43462b506b7c0` | 806.776 s | 798.130 s | 716.521 s | 67.298 s | passed | clean |
| Los Angeles | `mf_3392e476dea1c527fccb9cc5` | 538.989 s | 529.178 s | 441.499 s | 64.247 s | passed | clean |

`clean` 表示 rejected/timed-out/unresolved 全部为 0，且
`remaining_process_group_count=0`。

输出 roots：

```text
EVRPTW_Dataset/Instances_v2/stage2_targeted_chicago_monotone_search_v7_a52e8ff
EVRPTW_Dataset/Instances_v2/stage2_targeted_dallas_monotone_search_v7_a52e8ff
EVRPTW_Dataset/Instances_v2/stage2_targeted_la_monotone_smoke_v7_a52e8ff
```

每个 root 的正式证据为 `stage2_run_report.json` 和已发布 family 下的
`family_manifest.json`；没有复用旧 diagnostic partial artifact。

## 4. Spatial activation 证据

| Target | Growth steps / wall | First feasible competition round | Expansions | Global assignment wall | Peak family RSS |
|---|---:|---:|---:|---:|---:|
| Chicago | 1004 / 0.164 s | 123 | 1722 | 210.760 s | 3,954,421,760 B |
| Dallas | 1845 / 0.259 s | 468 | 11700 | 606.727 s | 4,065,583,104 B |
| Los Angeles | 1198 / 0.177 s | 0 | 0 | 29.676 s | 5,494,276,096 B |

Chicago 的 search 精确收敛为 round 122 infeasible、123 feasible；最终 min-cost-flow 为
66.697 秒。Dallas 收敛为 round 467 infeasible、468 feasible；最终 161,929-node、
1,524,900-edge min-cost-flow 为 258.423 秒。LA 初始候选集合已可行，不发生 competition
expansion。

Dallas 证明旧线性逐轮逻辑确实会浪费：若逐轮检查到 468，必须重建和求解数百次大图；新
实现只检查指数 probes 和二分 probes，同时返回同一个 first feasible round 与同一最终
min-cost objective。

LA 的 peak RSS 略高于名义 5 GiB/worker 模型，但 12 workers 即使同时达到该峰值也约为
65.9 GB，显著低于本机 270.1 GB physical memory；`WORKERS=12` 仍满足保守内存边界。

## 5. LA 最大 charger-roster smoke

LA charger 输入为 1,999；冻结 connectivity gate 排除 6 个不可达 charger，完整
connectivity-eligible roster 为 1,993。此处没有为了性能做 prefix、Haversine prefilter 或
截断。

```text
roster terminals                         2994
distance Dijkstra unique sources         3246
distance Dijkstra batches                 136
running-time Dijkstra unique sources     3705
running-time Dijkstra batches             309
exact roster Dijkstra wall             324.164 s
selected chargers                          50
final parent terminals                    1051
unreachable full-CS return count             0
```

LA stop rule：

```text
status                                  GREEN
terminal_selection_s                  441.499
green_terminal_selection_limit_s     3600
family_total_s                         529.178
red_family_total_limit_s             7200
pilot_allowed                            true
```

## 6. 申请批准后的唯一下一步

获批后：

1. 保持已验证的 executable code 不变；允许 evidence-only 文档提交，并再次确认最终 HEAD
   clean/pushed、相对 `a52e8ff` 没有 executable diff；
2. 使用不存在的全新 140-family root；
3. 从 v4 frozen splits 重建 preflight/splits/plan；
4. `WORKERS=12`、`FAMILIES_PER_WORKER_TASK=1`；
5. 从 0 materialize 全部 140 families，不复制三个 target family；
6. timeout family 不换 seed、不重试；任一 hard STOP 立即停止整批；
7. 完成 existing verifier、Phase-1 aggregate 和 release-gate 报告；
8. pilot 全通过后，才讨论压缩/restore bundle。

继续遵守用户指令：不增加 SHA256 或额外文件哈希校验。
