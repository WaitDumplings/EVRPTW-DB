# Stage-2 Spatial Growth 精确缓存优化审核申请

> 性质：只提交诊断证据和待批准方案，不申请立即修改方法，不申请 140-family pilot，
> 不申请 7,500-family full run。

## 1. 当前结论与 STOP

Chicago 和 Dallas 两个已知慢 family 都在同一个函数中表现出稳定的单核 CPU 热点：

```text
activate_spatial_customers -> _grow_regions
```

两次运行均未进入 global customer min-cost-flow、charger preflight、charger selection 或
terminal matrix。因此当前慢点不是车辆参数、charger roster、terminal matrix 或 batched
Dijkstra。新的 140-family pilot 继续保持禁止状态。

这两次都是人工早停的 diagnostic-only run：`reason_code=runner_signal`，不是 timeout、
不是 rejection、不是 acceptance pass。两次均满足：

```text
materialized families = 0
remaining process group count = 0
SIGKILL required = false
retry = false
```

## 2. 代码与运行绑定

诊断代码：branch `stage2-repair-candidate`，commit `01ff67a`。该提交已推送，完整 generator
suite 174/174 通过，其中 runtime supervisor integration tests 12/12 通过。

输入纪律：

- `INSTANCE_MODE=non_release_pilot`；
- `RUN_DISCIPLINE=targeted_profile`；
- `WORKERS=1`；
- `FAMILIES_PER_WORKER_TASK=1`；
- `FAMILY_WALL_TIMEOUT_S=7200`；
- `TERMINATION_GRACE_S=60`；
- `RUNNER_EXIT_SLACK_S=30`；
- customer split 只复用批准的 v4 frozen split；
- 每个城市使用独立的新 output root；
- 未生成或校验额外 SHA/文件哈希清单。

## 3. 实测证据

| 指标 | Chicago | Dallas |
|---|---:|---:|
| family ID | `mf_75f0c9cfc1b6c358fc60c916` | `mf_68a04e303dc43462b506b7c0` |
| track/day/scale | train/weekend/cus1000 | validation/weekend/cus1000 |
| diagnostic elapsed | 170.017 s | 118.012 s |
| customer preflight | 45.373 s | 28.137 s |
| customer preflight CPU | 45.356 s | 28.135 s |
| eligible territory rows | 308,379 | 163,647 |
| quota matrix | 0.055 s | 0.099 s |
| quota cells / regions | 51 / 14 | 89 / 25 |
| community graph | 0.047 s | 0.021 s |
| graph nodes / edges | 2,643 / 11,744 | 1,189 / 4,179 |
| region seed selection | 1.419 s | 1.397 s |
| region seed CPU | 1.407 s | 1.387 s |
| observed peak RSS | 4,043,087,872 B | 3,036,614,656 B |
| last observed growth progress | step 300, pass 40 | step 400, pass 47 |
| frozen growth upper bound | 37,002 | 29,725 |
| measured increment | 100→300 in 75.671 s | 200→400 in 37.319 s |
| increment throughput | about 2.64 step/s | about 5.36 step/s |
| worker CPU | about 100% of one core | about 100% of one core |

Chicago root：

```text
EVRPTW_Dataset/Instances_v2/
  stage2_targeted_chicago_spatial_probe_v3_01ff67a
```

Dallas root：

```text
EVRPTW_Dataset/Instances_v2/
  stage2_targeted_dallas_spatial_probe_v3_01ff67a
```

吞吐只用于定位复杂度，不能用 `upper_bound / throughput` 当作准确完成时间；region 可能在
达到上界前满足 quota。旧 Chicago 同一 family 的 terminal selection 实测 9,066.872 秒，
与这里发现的重复全表扫描一致。

## 4. 根因

当前 `_grow_regions` 的选择规则本身没有问题，问题是每一步用 DataFrame 全表操作重新计算
本来可以精确维护的充分统计量。

### 4.1 unmet quota 每次重复扫描 customer territory

`_region_unmet_cells` 每次调用都执行：

```python
customers["community_id"].isin(communities)
groupby("radial_decile").size()
```

Chicago 的扫描对象是 308,379 行，Dallas 是 163,647 行。该函数在每个 pass 的每个 region
调用一次，失败报告阶段还会再次调用。

### 4.2 每个 frontier candidate 再次做 equality 全表扫描

`_neighbor_score` 对每个 candidate 执行：

```python
customers["community_id"].eq(candidate)
```

随后才判断 candidate 是否覆盖 unmet decile。一个 growth step 可能评价多个 candidate，
因此总扫描次数高于 growth step 数。

### 4.3 frontier 和 crossing minimum 每步从 region 全量重建

每一步还会重新遍历 region 内所有 community，重建 successors/predecessors union；随后每个
candidate 又遍历 region members，查询双向 edge 并计算最小 crossing weight。

复杂度的主项因此接近“growth 迭代次数 × territory 行数”，而不是“territory 一次预聚合 +
局部 community graph 更新”。

## 5. 待批准的 exact 优化

申请允许一个仅在单 family、单次 `activate_spatial_customers` 内存活的精确缓存。它不跨
family，不改变任何输入或随机过程。

### 5.1 一次性 customer capacity 表

进入 `_grow_regions` 前只做一次 groupby，建立：

```text
community_decile_capacity[community_id][0..9] = exact integer count
community_decile_support[community_id]        = exact nonzero decile set
region_quota[region_id][0..9]                 = exact integer target
```

每个 region 初始化 `region_capacity` 为 seed community 的 capacity；加入 chosen community
时做 10 维整数加法。unmet 判断仍是逐 decile 的 `capacity < quota`，结果逐项相同。

### 5.2 增量 exact frontier

为每个 region 维护当前 boundary community set：

```text
frontier = union(successors(member), predecessors(member)) - region_members
```

加入 chosen 后，只把 chosen 的 successors/predecessors 合入并移除所有 region members。
这与每一步从头计算上述集合在集合意义上完全相同。

### 5.3 增量 exact crossing minimum

预先把每条 incident directed edge 转成：

```text
incident_min[member][candidate] = min(
    weight(member -> candidate),
    weight(candidate -> member)
)
```

region 加入一个 member 时，以 `min(old, incident_min[new_member][candidate])` 更新 frontier
candidate 的 crossing minimum。它等价于原实现对全部 region members 收集双向 edge weight
再取 `min`。

### 5.4 完全冻结的选择键

以下 tuple 必须原样保留：

```python
(
    0 if candidate_deciles intersects unmet else 1,
    exact_min_crossing_weight,
    stable_u64(seed, "region_growth", region, candidate),
    candidate,
)
```

`order`、`maximum_steps`、pass 顺序、一次只扩展一个 community、error code、diagnostics 和
growth step 计数全部不变。

## 6. 明确不改变的内容

该方案不允许：

- Haversine selector 或任何 lossy spatial prefilter；
- 减少 customer/charger eligible roster；
- charger prefix；
- 修改 D-1～D-6、5:2 day type、CS 目标或充电功率链；
- 修改 community graph、region seed、quota、redraw cap 或 stable tie-break；
- 近似距离、近似 capacity、提前接受 unmet quota；
- 静默跳过慢 family；
- timeout 后换 seed 重试。

## 7. 等价性验证要求

若 reviewer 批准，实施时必须同时保留一个 test-only reference 路径，并完成：

1. 随机小图 differential tests：多组 customer 分布、directed graph、quota、seed；
2. reference 与 optimized 的 regions、growth_steps 和成功/失败 error code 逐项相同；
3. 对每一步 chosen community 做 trace 比较；
4. disconnected、zero-customer transit community、单向 edge、tie weight、empty frontier；
5. callback 开关前后 customers、assignment、radial baseline 与非计时 metadata 相同；
6. 完整 generator suite 通过；
7. clean commit push 后，从全新 root 完整运行 Chicago 和 Dallas；
8. 两个 family 都必须 `<7200 s` 且 existing verifier passed；
9. 再运行 LA 最大 charger-roster smoke；
10. 三个 target 全通过后，才重新申请 140-family pilot。

## 8. 请求 reviewer 明确签字的问题

请只回答以下放行项：

```text
ALLOW exact family-local community×decile capacity cache: YES / NO
ALLOW exact incremental region frontier cache:           YES / NO
ALLOW exact incremental crossing-min cache:               YES / NO
REQUIRE per-step differential trace equality:             YES / NO
ALLOW implementation before another full target run:      YES / NO
```

在得到明确 YES 前，code agent 不修改 `_grow_regions` 的选择实现，也不启动新的 140-family
pilot。
