# Stage-2 Spatial Growth 精确缓存优化审核申请

> 状态：三项 exact cache 已获用户执行授权并实现；205/205 generator tests 通过。
> Chicago、Dallas、LA 的优化后完整定点运行尚未执行。本文件不申请 140-family pilot，
> 不申请 7,500-family full run。

## 1. 当前结论与 STOP

Chicago、Dallas 两个已知慢 family，以及 Los Angeles 最大 charger-roster smoke，都在
同一个函数中表现出稳定的单核 CPU 热点：

```text
activate_spatial_customers -> _grow_regions
```

三次运行均未进入 global customer min-cost-flow、charger preflight、charger selection 或
terminal matrix。因此当前慢点不是车辆参数、charger roster、terminal matrix 或 batched
Dijkstra。LA smoke 被上游热点遮挡，不能记为 smoke passed。新的 140-family pilot 继续
保持禁止状态。

这三次都是人工早停的 diagnostic-only run：`reason_code=runner_signal`，不是 timeout、
不是 rejection、不是 acceptance pass。三次均满足：

```text
materialized families = 0
remaining process group count = 0
SIGKILL required = false
retry = false
```

## 2. 代码与运行绑定

诊断代码：branch `stage2-repair-candidate`。Chicago/Dallas 绑定已推送 commit `01ff67a`；
LA 绑定只增加审核文档、执行源码相同的已推送 commit `6e291fa`。完整 generator suite
优化前基线 174/174 通过，其中 runtime supervisor integration tests 12/12 通过。

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

| 指标 | Chicago | Dallas | Los Angeles |
|---|---:|---:|---:|
| family ID | `mf_75f0c9cfc1b6c358fc60c916` | `mf_68a04e303dc43462b506b7c0` | `mf_3392e476dea1c527fccb9cc5` |
| track/day/scale | train/weekend/cus1000 | validation/weekend/cus1000 | train/weekday/cus1000 |
| diagnostic elapsed | 170.017 s | 118.012 s | 196.020 s |
| customer preflight | 45.373 s | 28.137 s | 81.182 s |
| customer preflight CPU | 45.356 s | 28.135 s | 81.167 s |
| eligible territory rows | 308,379 | 163,647 | 381,604 |
| quota matrix | 0.055 s | 0.099 s | 0.073 s |
| quota cells / regions | 51 / 14 | 89 / 25 | 68 / 17 |
| community graph | 0.047 s | 0.021 s | 0.056 s |
| graph nodes / edges | 2,643 / 11,744 | 1,189 / 4,179 | 3,112 / 13,282 |
| region seed selection | 1.419 s | 1.397 s | 2.151 s |
| region seed CPU | 1.407 s | 1.387 s | 2.135 s |
| observed peak RSS | 4,043,087,872 B | 3,036,614,656 B | 5,505,241,088 B |
| last observed growth progress | step 300, pass 40 | step 400, pass 47 | step 300, pass 27 |
| frozen growth upper bound | 37,002 | 29,725 | 52,904 |
| measured increment | 100→300 in 75.671 s | 200→400 in 37.319 s | 100→300 in 72.847 s |
| increment throughput | about 2.64 step/s | about 5.36 step/s | about 2.75 step/s |
| worker CPU | about 100% of one core | about 100% of one core | about 100% of one core |

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

Los Angeles root：

```text
EVRPTW_Dataset/Instances_v2/
  stage2_targeted_la_smoke_v3_6e291fa
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

Chicago 的扫描对象是 308,379 行，Dallas 是 163,647 行，LA 是 381,604 行。该函数在
每个 pass 的每个 region 调用一次，失败报告阶段还会再次调用。

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

## 8. 执行授权与当前实现

用户在收到本审核申请后以“开始”授权按本文方案实施。code agent 按最严格口径执行：

```text
exact family-local community×decile capacity cache: implemented
exact incremental region frontier cache:           implemented
exact incremental crossing-min cache:               implemented
per-step differential trace equality:               required and passed
new 140-family pilot:                               still prohibited
```

原始 DataFrame-scan 实现保留为 test-only reference。新增 30 组随机 directed graph differential
tests，以及 zero-customer transit、单向 edge、tie crossing weight、empty frontier 和失败
diagnostics 测试。optimized/reference 的每一步 chosen community、selection tuple、regions、
growth_steps、progress events 和 error diagnostics 逐项相同。

growth cache 完成时的完整 generator suite 为 205/205 passed。后续 Chicago 定向运行确认
1004 个 growth steps 已缩短至 0.166 秒，并暴露下一个热点为 global unique-customer
assignment；该热点的处理和新增验证见下一节。

## 9. Global assignment 精确搜索补充

### 9.1 新证据

Chicago 初始 assignment graph 为 63,189 nodes、154,568 edges，其中 91,381 条为
cell-customer candidate edges。初始 maximum flow 为 997/1000；逐轮旧逻辑到第 44 轮仍为
997/1000，累计 616 次 competition expansion。每轮都重建大图并重新验证，单轮 maximum
flow 从约 1.6 秒增加到约 4.6 秒，因此继续线性逐轮探测没有意义。

### 9.2 不变的数学对象

每个 region 的 competition expansion 顺序只依赖：

- 本 region 当前 community members；
- frozen target deciles；
- directed community graph 的 exact incident crossing minimum；
- frozen `stable_u64` tie-break。

它不依赖其他 region 的扩展，也不依赖上一轮 flow 的具体取值。每一轮只增加 community，
因此只会增加 assignment candidate edges；可行性关于 round index 单调不减。

### 9.3 已实施的 exact 搜索

实现保留 test-only 逐轮 reference，并执行：

1. 对每个 region 惰性缓存 deterministic expansion sequence；
2. 只生成下一次 probe 所需的 sequence prefix，不预展开整个连通分量；
3. round 0 先做 exact maximum-flow feasibility；
4. 以 1、2、4、8……指数探测找到首个可行上界，或证明所有 frontier exhausted；
5. 在最后不可行轮与可行上界之间二分；
6. 回放到与 reference 完全相同的 first feasible round；
7. 仅在该轮运行原 `nx.min_cost_flow`，objective、capacity、唯一性和 tie weight 均不变。

搜索不近似 flow、不删候选、不改变 quota、region、seed 或评分键。最终 regions、expansion
count、逐步 decision trace、assignment 和失败 error code 必须与 reference 相同。

### 9.4 验证状态

新增测试强制首个可行点位于第 5 轮且图在第 8 轮后仍有 tail，从而实际覆盖指数探测与二分
回退；optimized/reference 的 assignment records、10 次 expansion、每步 selection tuple 和
最终 region members 逐项相同。原一轮可行与完全不可行边界测试继续通过。

当前完整 generator suite 为 207/207 passed。下一放行点是 clean commit/push 后，从全新
root 完整跑完 Chicago、Dallas 和 LA；只有三者 `<7200 s`、verifier passed、零 orphan，
才能重新申请 140-family pilot。
