# EVRPTW-DB V2 连通性修复与生成 Pipeline（唯一权威 Runbook）

> 状态：**R-1～R-6 已签字；只允许执行到 140-family pilot**  
> 日期：2026-08-17  
> 分支：`stage2-repair-candidate`  
> 规范基线：`STAGE2_REPAIR_DIRECTIVE_V2.1_FINAL.md`  
> 当前动作边界：先测试、commit/push `stage2-repair-candidate` 并确认 clean tree；
> 然后可连续执行 CLE_v2、C1、C2、LA smoke、140-family pilot。full 7,500、
> profile promotion、archive/restore、merge/tag 均未获批准。

## 0. 签字版 R-1～R-6 放行纪律

- **R-1**：固定 11 城 CLE 唯一根目录为
  `EVRPTW_Dataset/CLE_v2/us_11city`。`CLE_v1/us_11city` 只读保留，reader
  必须拒绝，不能 resume 或升级。
- **R-2 / C1-Q1**：固定规则
  `connectivity_quarantine_precedes_customer_split_v1`。quarantine rate 按
  `city × terminal_kind` 和 unique ID 计算；customer 分母是通过非连通性
  source/geometry/road-anchor 条件、在 Stage-1/Stage-2 connectivity filtering 前的
  全城市 pre-split universe，不依赖 train/heldout；分子是 Stage-1 directional、
  Stage-2 node/turn 的 unique union。quarantined customer 保留完整 ledger，但
  `split_pool=null`、`split_assignment_status=excluded_pre_split_connectivity`、
  `generation_eligible=false`。customer 上限 0.1%，charger 上限 1.0%；超限是
  stop-and-review，不得通过 split assignment 或静默删除降低 rate。该阈值只作
  engineering bug detector。
- **R-3**：LA smoke：terminal selection/total 分别满足 `<=3600/<=7200` 为 GREEN；
  `3600<terminal<=7200` 且 total `<=7200` 为 AMBER；其余 RED。GREEN/AMBER
  可进 pilot，AMBER 必须登记 exact-performance optimization。pilot 任一 family
  任一 stage `>7200s` 停止新提交并 drain 在途；4 小时按
  `elapsed×planned/completed` 外推，0 completed 或外推 `>36h` 停止，此后每小时复查；
  同一 `family_id+roster fingerprint+reason_code` non-retryable 第二次出现即停止。
- **R-4**：Phase B 前必须 tests pass → candidate commit → push → clean tree。
  CLE index、plan、run 和 family manifest 绑定 exact commit。artifact 相关代码改变后，
  必须新 commit、新 output root，并从最早受影响阶段重跑。
- **R-5**：smoke 前必须依次通过 C1 connectivity/PF-1 与 C2 Amazon H3、PF-2、
  METRIC-HOLDOUT、四项 leakage、5:2 slot ledger；任一 primary support cell 为 0 停止。
- **R-6**：pilot 后先生成完整 pilot report；只有导师显式签字才可冻结 acceptance
  config、写入 report ID/hash，并生成新的 `release_calibrated`、
  `official_generation_eligible=true` profile。该变更还须形成并 push clean acceptance
  commit。禁止把 candidate-profile 产物事后升格。

## 1. 给审核人的一页结论

前一轮 10 城 pilot 的失败不是车辆电池、能耗或充电功率造成的，而是 Stage 1 对
有向边端点投影的 round-trip 标签过于宽松，加上 Stage 2 在大候选池上缺少与最终
canonical zero-turn 路由一致的连通性预检。

修复后的三层防线是：

1. Stage 1 按实际有向 edge access 语义分别标记 inbound/outbound eligibility；
2. Stage 2 在 territory 和全终端 closure 之前，对 customer/charger 做 node-level
   和 canonical turn-line-graph 双向预检，并把坏点隔离到确定性 ledger；
3. 如果过滤后仍不足，或最终 selected-terminal closure 仍出现不可达，错误是
   non-retryable，同一个固定坏 roster 不再白跑四次。

旧 CLE 没有新合同 `directed_projection_roundtrip_v2`，当前 reader 会明确拒绝，
所以不能误用旧 CLE 续跑。修复没有改变导师已冻结的 D-1～D-6、5:2 日型、Amazon
cohort、CS 目标、充电功率链、zero-turn、schema v2 输出根目录或 Q90 gate。

审核稿历史测试结果为 **134 passed**；它不是最终 candidate 证据。最终通过数、耗时和
peak RSS 必须在 commit 前重新测量并记录。这些测试也不等于真实 11 城 CLE 和 pilot
已通过。

## 2. 本次明确没有做的事情

- 没有运行 `generate_cle.sh`；现有 CLE 是旧 connectivity contract，不能用于新 pilot。
- 没有运行新的 `generate_instances.sh`。
- 没有读取或生成 Test2 / Jacksonville instance。
- 没有开始 full 7,500-family generation。
- 没有创建、传输或恢复 slim archive。
- 没有删除旧失败 pilot；它仅作为修复前证据，不能作为新 run 的 resume root。
- full 7,500、archive/restore、merge/tag 仍禁止。

旧 pilot 根目录是：

```text
EVRPTW_Dataset/Instances_v2/us_10city_trainval_pilot
```

批准后的新 pilot 必须使用新目录，例如：

```text
EVRPTW_Dataset/Instances_v2/us_10city_trainval_pilot_v2
```

## 3. 修复前证据与根因

旧 pilot 在约 7 小时后停止：计划 140 个 family，已 materialize 67 个，记录 202
个 rejected attempts。已有 family 的 `terminal_selection` 时间分布为：

| 样本数 | min | P50 | P90 | max |
|---:|---:|---:|---:|---:|
| 67 | 53.001 s | 350.187 s | 2,089.929 s | 19,103.490 s |

202 个 rejection 中 152 个是 `ValueError`，50 个是 `SpatialActivationError`。这批
结果来自修复前 schema `cle_evrptw_family_terminal_selection_v2`，不能用于证明本次
修复后的性能或 acceptance。

### 3.1 精确坏点审计

| 城市 | 审计对象 | node round-trip bad | canonical turn bad |
|---|---:|---:|---:|
| Houston | 341,511 个 train customer 候选 | 1 | 未单独审计 |
| Phoenix | 309,217 个 train customer 候选 | 3 | 未单独审计 |
| Los Angeles | 2,000 个 charger 候选 | 1 | 7 |

关键坏点如下：

- Houston customer
  `msft_nsi_msft_usbf_houston_004823097`：one-way `secondary_link`，physical edge
  `0eb9bf8e70bc1cf08760`，OSM way `15311389`，fraction `1.0`；depot 无法到达。
- Phoenix customers
  `...phoenix_000722556`、`...phoenix_001951371`：physical edge
  `8385227de5fba5b19bb0`，OSM way `396101715`，fraction `0.0`；无法返回 depot。
- Phoenix customer `...phoenix_001738061`：physical edge
  `a03351e513704e05245b`，OSM way `262969271`，fraction `0.0`；无法返回 depot。
- Los Angeles charger `afdc_113090`：physical edge
  `49d5b8b8196dac5710ba`，OSM way `48226731`，fraction `0.0`；node 和 turn 层均
  无法返回。
- Los Angeles chargers `afdc_159025`、`afdc_160987`、`afdc_176457`、
  `afdc_176458`、`afdc_176459`、`afdc_179517`：都在 physical edge
  `0d7838d9782599888a0c`、OSM way `48144161`；node 层可达，但 canonical
  turn topology 无法返回。

完整审计证据见 `TERMINAL_CONNECTIVITY_AUDIT_PILOT.md`。

### 3.2 原代码为什么会误判

一个 terminal 可以投影到有向边 `u -> v` 的内部或端点。Stage 2 的实际语义是：

```text
进入 projection：reference graph -> u -> projection
离开 projection：projection -> v -> reference graph
```

旧 Stage 1 的 endpoint SCC label 只看 projection 所在端点。fraction=1 时，即使 `v`
在主 SCC，`u` 仍可能无法从主 SCC 到达；fraction=0 时存在对偶问题。旧
`protected_roundtrip_eligible` 只是 `anchor_in_reference_scc`，没有分别验证上述
两个方向，也没有验证 zero-turn line graph 的 immediate-reversal 禁止规则。

旧 Stage 2 又把一个固定坏点升级成整个 family failure：customer 阶段把 depot 与
完整 split roster 做 star；charger 阶段把全部 charger 候选放入 closure。retry 只
换随机 seed，不会改变固定坏点集合，所以四次 retry 没有意义。

## 4. Stage 1 修复：directional projection contract

实现位置：

- `src/evrptw_cle/protected_connectivity.py`
- `src/evrptw_cle/customer_access.py`
- `src/evrptw_cle/cle.py`

### 4.1 新判定

对一个 terminal 的全部 `directed_projection_offsets`：

```text
protected_inbound_access_eligible
    = 存在至少一个 directed ref，其 u 属于 reference SCC

protected_outbound_access_eligible
    = 存在至少一个 directed ref，其 v 属于 reference SCC

protected_roundtrip_eligible
    = protected_inbound_access_eligible
      AND protected_outbound_access_eligible
```

inbound 和 outbound 可以由不同的 reciprocal/directed refs 提供；这与 Stage 2
“从某个 ref 进入、从某个 ref 离开”的多 access-option 语义一致。审核人需要明确
确认这一点是否接受；当前实现没有强制两方向必须来自同一 ref。

新状态值：

```text
passed_reference_scc_directional_access
quarantine_no_reference_scc_directional_access
quarantine_no_reference_scc_inbound_access
quarantine_no_reference_scc_outbound_access
```

新字段会传播到 service node、road projection node、connector、customer、depot 和
charger 表，并进入 CLE verifier 的 required columns。

### 4.2 CLE manifest hard contract

新 CLE manifest 必须包含：

```json
{
  "connectivity_contract": {
    "id": "directed_projection_roundtrip_v2",
    "inbound_semantics": "reference_scc_to_directed_edge_u_then_projection",
    "outbound_semantics": "projection_then_directed_edge_v_to_reference_scc",
    "stage2_canonical_turn_topology_preflight_required": true
  }
}
```

CLE verifier 会拒绝缺失或错误的 ID；Stage 2 reader 也会在 split、plan 或 routing
以前拒绝旧 CLE，错误信息要求先 rebuild Stage 1。这是刻意的 hard stop，不提供
兼容 fallback。

## 5. Stage 2 修复：精确 preflight 与 quarantine

实现位置：

- `src/evrptw_stage2/routing.py`
- `src/evrptw_stage2/selection.py`
- `src/evrptw_stage2/reader.py`

### 5.1 两层连通性

`route_depot_star()` 现在返回
`cle_evrptw_depot_terminal_star_v2`，为每个候选分别给出：

```text
node_outbound_reachable
node_return_reachable
turn_outbound_reachable
turn_return_reachable
connectivity_eligible = 四者 AND
```

node 层使用与 runtime 相同的 directed physical graph、edge projection 和 connector
cost。turn 层使用 canonical directed line graph；turn 时间仍是 D-4 冻结的 0，
但 virtual access split node 的 immediate edge reversal 在 topology 上禁止。因此
Los Angeles 六个“node 可达、turn 不可返回”的 charger 会被识别。

### 5.2 执行顺序

每个 family 的 terminal selection 顺序现在是：

```text
选择 depot
  -> 对完整 split customer roster 做 depot-star node + turn preflight
  -> 隔离 bad customers
  -> 在剩余 customers 上做 source-specific T_env 与 energy territory
  -> spatial activation 得到 N customers
  -> 对完整 candidate-eligible charger roster 做 depot-star node + turn preflight
  -> 隔离 bad chargers
  -> 对剩余 charger roster 做 exact full-terminal/energy closure 与 relevance fill
  -> 对 depot + selected customers + selected chargers 做最终四矩阵 closure
```

坏点不再使整个 family 立即失败。只有过滤后 customer 少于 N、charger 少于 K，或
最终 exact closure 仍不可达时，family 才失败。

### 5.3 确定性 quarantine ledger

family selection metadata schema 从
`cle_evrptw_family_terminal_selection_v2` 升为
`cle_evrptw_family_terminal_selection_v3`，新增：

```text
terminal_connectivity.schema
  = cle_evrptw_terminal_connectivity_quarantine_v1
terminal_connectivity.policy
  = depot_bidirectional_node_and_canonical_turn_topology_v1
customer/charger input_count
customer/charger eligible_count
customer/charger quarantined_count
customer/charger quarantine_ledger
customer/charger depot_star report
applied_before_territory_and_full_terminal_closure = true
```

每个 ledger row 保存：

```text
terminal_kind, source_id, reason_codes, physical_edge_id,
anchor_scc_id, directed_edge_ref_count, directed_projection_offsets
```

允许的 reason codes：

```text
node_unreachable_from_depot
node_cannot_return_to_depot
turn_unreachable_from_depot
turn_cannot_return_to_depot
```

本次策略是**隔离，不自动 remap**。自动把坏点换投影道路可能改变真实 access
语义，需要单独的方法批准，当前没有擅自加入。

### 5.4 最终不变量

preflight 不是最终矩阵的替代品。`route_terminals()` 仍对最终 depot + selected
customers + selected chargers 计算完整 directed all-pair closure；任一矩阵出现
`inf` 都抛出 `TerminalConnectivityError`。四张矩阵仍必须 finite、nonnegative、
float32、zero diagonal，并由 family verifier 检查。

## 6. Retry 修复

实现位置：

- `src/evrptw_stage2/parallel.py`
- `scripts/build_stage2_instances.py`

`TerminalConnectivityError.retryable = false`。rejection ledger 新增：

```text
retryable
retry_stopped_early
next_attempt_seed = null  # non-retryable 时
```

fixed-roster connectivity failure 首次出现后立即停止该 family，不再重复四次。其它
例如 structure source、territory 或 order matching 可能随 seed/source 改变的失败，
仍默认 retryable。

同时修复了 resume lifetime cap：`MAX_ATTEMPTS_PER_FAMILY=4` 是 family 跨重启的
累计上限，不是每次启动再获得四次。已记录的 attempt number 会从剩余序号继续；
用尽后直接 unresolved。

## 7. 冻结方法：本次没有改变

以下以导师签字版 V2.1 为准，不因连通性修复重新讨论：

| 合同 | 冻结内容 |
|---|---|
| D-1 | `METRIC-HOLDOUT` 为整 station；其余 station-days 按 usable mass 分 GEN-TRAIN / GEN-EVAL |
| D-2 | 每个 structure source 使用自己的 P99 `T_env` 和 decile edges |
| D-3 | deterministic region-first nested partition |
| D-4 | canonical turn times 全为 0；split-node immediate reversal 拓扑禁止 |
| D-5 | primary strata 的每个 M2/M3 component 必须满足 generated-to-holdout Q90 <= real-to-real Q90 |
| D-6 | `p_battery = 0.90 * min(p_station_or_imputed, p_vehicle_cap)` |
| B-3 | direct depot-customer-depot energy sufficient screen 使 CS energy core 为空；K 个 CS 是 relevance fill |
| 日型 | weekday:weekend = 5:2 |

Amazon cohort：

- `METRIC-HOLDOUT = DCH2 + DLA9 + DSE2`；不得用于生成。
- train 只用 `GEN-TRAIN`；validation/Test1/Test2/Test3/scalability 用 `GEN-EVAL`。
- primary Cus100/500/1000 必须是 `SINGLE_STRUCTURE_DAY` +
  `SINGLE_ORDER_DAY`。
- composite 只允许 Cus2000 report-only。
- station-day、template ID、route ID 的 pool leakage assertions 都是 hard error。

## 8. 冻结 V2 参数

权威配置：

```text
configs/cle_evrptw_stage2_v2.json
configs/us_reference_instance_profile_v2.json
configs/amazon_cohort_split_v1.json
configs/us_national_charging_power_medians_v1.json
```

### 8.1 基础、城市和 split

| 参数 | 值 |
|---|---|
| dataset/schema | `CLE_EVRPTW_v2` / `cle_evrptw_stage2_config_v2` |
| master seed | `20260810` |
| horizon | 28,800–86,400 s，即 08:00–24:00 |
| train cities | New York, Los Angeles, Chicago, Houston, Phoenix, Philadelphia, San Antonio, San Diego, Dallas, Fort Worth |
| held-out city | Jacksonville，仅 full Test3 |
| community | Census Block Group × `anchor_scc_id` |
| customer split | complete-community 80/20 |
| pilot tracks | train + validation only |

### 8.2 Vehicle、energy、charging

| 参数 | 值 |
|---|---:|
| vehicle | Rivian Commercial Van Delivery 700 reference v1 |
| battery | 100 kWh |
| nominal range | 257 km |
| energy coefficient | 0.38910505836575876 kWh/km |
| cargo capacity | 18,500,000 cm3 |
| initial SOC | 1.0 |
| vehicle DC cap | 100 kW |
| vehicle AC L2 cap | 11 kW |
| charging | full、linear、derating 0.90 |
| missing AC L2 power | frozen national median 6.5 kW |
| missing CCS DC power | frozen national median 200 kW |
| ports/fleet | infinite ports、unlimited fleet |

缺失 charging power 如果连 frozen national mode median 都没有，则 hard error；禁止
退回 vehicle cap。energy 不含速度项或 auxiliary load。

### 8.3 Road、turn、matrix

- family 直接选择 CLE weekday/weekend reference-speed column。
- road residual 对各 day/type 都为 mean=min=max=1、std=0。
- minimum speed 5 km/h。
- connector speed 是同 city/day、operating mode U edge 的 length-weighted median。
- canonical right/left/U-turn 时间均为 0 s。
- 3/8/20 s geometry turn adapter 是 noncanonical test-only。
- 每个 parent 存四张 square `float32` matrix：

```text
distance_matrix_km.npy
distance_path_travel_time_s.npy
running_time_shortest_matrix_s.npy
running_time_path_distance_km.npy
```

energy matrix 不存，按 path distance × energy coefficient 派生。terminal order 固定为
depot、customers、charging stations。

### 8.4 Family、view、CS 数量

full plan 为 7,500 parent families：

| cohort | families | parent |
|---|---:|---|
| core train | 5,000 | Cus1000 |
| validation | 500 | Cus1000 |
| Test1 new seed | 500 | Cus1000 |
| Test2 held-out locations | 500 | Cus1000 |
| Test3 Jacksonville | 500 | Cus1000 |
| same-city unseen scalability | 500 | Cus2000 |

当前代码计划 173,000 views：Cus50 101,000；Cus100 52,000；Cus500 12,000；
Cus1000 7,500；Cus2000 500。固定 CS 数为 Cus50=10、Cus100=20、
Cus500/1000/2000=50。

train Cus1000 parent 拆成 20×Cus50、10×Cus100、2×Cus500、1×Cus1000；
evaluation 使用一条 strict nested chain；Cus2000 带一个 Cus1000 control subset。

### 8.5 空间与订单

- structure source 先于 territory。
- territory：split pool + source P99 `T_env` + direct round-trip battery sufficient
  condition，`pool_floor=1.0`。
- quota：deterministic min-cost controlled matrix rounding；route row、decile column、
  total N 必须 exact。
- seed：quota-descending network-time max-min。
- growth：directed OSM cross-block-group adjacency。
- region redraw cap：3。
- customer sampling weight：`max(1, residential_units)`。
- nested partition：`deterministic_region_first_v1`。
- order template 必须 distinct，并同时满足 volume、TW、service、return horizon；
  order resample 不能改 source mode。

## 9. 验收标准

### 9.1 CLE hard gate

每个新 CLE 必须：

- manifest contract ID 为 `directed_projection_roundtrip_v2`；
- customer/depot/charger 和相关 access 表有 inbound/outbound/roundtrip 字段；
- technical verification passed；
- portable package verification passed；
- cohort index：`status=complete`、`verified_cle_count=11`、`failures=[]`。

还应输出每城 directional quarantine counts，并确认 Houston/Phoenix 已知 endpoint
坏点不再是 `protected_roundtrip_eligible=true`。Stage 1 不负责 turn-line-graph gate，
所以 LA 六个 turn-only charger 可以通过 Stage 1，但必须在 Stage 2 preflight 被隔离。

### 9.2 Family correctness hard gate

- parent customer count 恰好 N，ID 全局唯一，split pool 明确；
- route row margins、decile column margins、total N exact；
- nested child size exact、pairwise disjoint、union exact；
- selected terminal 四矩阵全 finite、shape/dtype/zero diagonal 正确；
- energy derivation、charging cache、order provenance、feasibility certificate 正确；
- quarantine ledger counts 与 masks 一致；
- final exact closure 不出现 `TerminalConnectivityError`。

### 9.3 D-5 realism gate

primary scales 为 Cus100/Cus500/Cus1000；Cus50 是单独 compatibility gate；Cus2000
和 composite strata report-only。对每个可评价的：

```text
day_type × scale × source_mode × M2/M3 component
```

必须满足：

```text
Q0.90(D_generated_to_holdout) <= Q0.90(D_real_to_real)
```

primary stratum 缺 support 也失败。pair subsampling 禁止。station-block bootstrap
confidence intervals 只报告，不改变 frozen gate。Jacksonville/Test2 不参与校准阈值。
M1、M4、M5 仍按 V2.1 角色报告，不得事后改变 Q90 门槛来让结果通过。

### 9.4 Reliability / anti-waste gate

pilot handoff 至少报告：

- 计划、materialized、verified、unresolved 数；
- first-attempt 和 conditional-attempt success；
- retryable/non-retryable rejection 数和原因；
- 每城 customer/charger quarantine 数、ID、reason codes；
- known Houston/Phoenix/LA bad IDs 是否按预期被隔离；
- `terminal_selection`、matrix、order、verification 的 P50/P90/max；
- worker peak RSS、总 wall time；
- M1–M5、Q90、H3/PF support、四类 leakage assertions；
- arbitrary-child-index slim restore exact hash 测试。

任何固定 roster connectivity error 不应出现重复 attempt。若同一 non-retryable
reason 被重复记录，视为 retry control regression。

## 10. 已冻结的工程纪律与仍待实测的性能

### 10.1 Charger full-roster 性能

本次修复保证坏 charger 会在 exact preflight 后隔离，但过滤后的完整 charger roster
仍用于后续 all-terminal/energy closure。Los Angeles 可有约 2,000 个 charger，旧
pilot 已出现单个 `terminal_selection` 19,103 s。因此：

- correctness defect 已修；
- performance defect **没有被证明已修**；
- 不能直接从 134 个单元测试推断 140-family pilot 会很快。

签字版已经允许在 C1/C2 通过后先跑 1-city/1-family LA smoke，再条件执行 140-family
pilot。若要改变 full-roster closure 算法，必须保持 D-6、
B-3、bidirectional energy communicating set 和 exact selected-terminal closure，不可用
haversine/prefix selector 回退。

### 10.2 Worker task chunk

launcher 的一般默认值仍为 25，但本次 pilot 已冻结为
`FAMILIES_PER_WORKER_TASK=1`，配合 `WORKERS=12` 和 lifetime attempts=4。它只改变
调度，不改变 seeds 或数据内容。

### 10.3 Timeout / stop rule

R-3 stop rule 已按第 0 节冻结。这些上限只触发停止新提交、drain 在途并送审，不是
scientific acceptance threshold；禁止把超时 family 删除后继续形成有 selection bias
的 corpus。

### 10.4 Ledger 体积

当前每个 family manifest 保存完整 customer/charger quarantine ledger，审计最完整，
但同一 depot/roster 的重复 ledger 可能增大 slim artifact。reviewer 可接受当前方案，
或另行批准 city/depot-level dedup contract；在批准前不改为只存 count。

## 11. 已批准边界内的 pipeline

执行环境：

```text
/home/npg/miniconda3/envs/evrptw-cle/bin/python
```

所有阶段用 `/usr/bin/time -v` 分开计时并保留 stdout/stderr、exit code、Git revision。
CLE 与 instance 不能并发运行。

### Phase A：测试、固定 candidate revision 与 source preflight

```bash
cd /data/Maojie/ICLR/EVRPTW-DB
cd EVRPTW_Dataset_Generator
PATH=/home/npg/miniconda3/envs/evrptw-cle/bin:/usr/local/bin:/usr/bin:/bin \
  /usr/bin/time -f 'elapsed=%e peak_rss_kb=%M' \
  /home/npg/miniconda3/envs/evrptw-cle/bin/python -m pytest -q
cd ..
git add -A
git commit -m 'Implement reviewed Stage-2 connectivity and release gates'
git push origin stage2-repair-candidate
git status --short
git rev-parse HEAD
free -h
df -h .

cd EVRPTW_Dataset_Generator
PATH=/home/npg/miniconda3/envs/evrptw-cle/bin:/usr/local/bin:/usr/bin:/bin \
  PYTHONPATH=src \
  /home/npg/miniconda3/envs/evrptw-cle/bin/python \
  scripts/prepare_us11_sources.py --check-only
```

如果 source 缺失，先单独下载并计时；如果要严格统计“不含下载”的 runtime，只有
在 `--check-only` 已证明全部 required inputs 存在后，才能用
`PREPARE_CLE_SOURCES=0` 计 CLE computation。首次 NSI cache 不完整时，NSI 网络时间
仍会混入 CLE；应先补齐 cache 或在报告里明确标注。

### Phase B：重建全部 11 城 CLE

新 connectivity 字段来自 CLE build，所以不能 patch 旧 parquet 后继续。

```bash
cd /data/Maojie/ICLR/EVRPTW-DB
mkdir -p logs

/usr/bin/time -v -o logs/generate_cle_v2_connectivity.time \
  env PREPARE_CLE_SOURCES=0 \
      PYTHON_BIN=/home/npg/miniconda3/envs/evrptw-cle/bin/python \
      NSI_WORKERS=4 \
      KEEP_CLE_WORK=0 \
      ./generate_cle.sh \
  > logs/generate_cle_v2_connectivity.log 2>&1
```

CLE 生成顺序仍是：

```text
preflight -> roads -> buildings -> depots -> cles -> package -> index
```

验收：

```bash
jq '{status, verified_cle_count, failures}' \
  EVRPTW_Dataset/CLE_v2/us_11city/cle_index.json

rg -n 'directed_projection_roundtrip_v2' \
  EVRPTW_Dataset/CLE_v2/us_11city/cities/*/manifest.json
```

期望 11 个 contract hit，`status=complete`、`verified_cle_count=11`、`failures=[]`。
随后先做 terminal connectivity audit，不直接启动 pilot。

### Phase C0：只生成固定 140-family split/plan（不 materialize）

```bash
PILOT_ROOT=/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Dataset/Instances_v2/us_10city_trainval_pilot_v4
INSTANCE_MODE=non_release_pilot WORKERS=1 FAMILIES_PER_WORKER_TASK=1 \
PILOT_FAMILIES_PER_CITY=7 \
INSTANCE_OUTPUT_ROOT="$PILOT_ROOT" \
./generate_instances.sh \
  --stages preflight splits plan \
  --cities new-york los-angeles chicago houston phoenix philadelphia \
           san-antonio san-diego dallas fort-worth \
  --tracks train validation
```

### Phase C1：新 CLE 的 10 城 terminal connectivity/PF-1 audit

审计范围：十个 train cities 的城市级 pre-split customer universe 和完整
candidate charger roster；至少对 Houston、Phoenix、Los Angeles 输出 known bad IDs
的新 eligibility、node/turn masks、edge refs。执行顺序固定为：非连通性
source/geometry eligibility → Stage-1/Stage-2 connectivity audit → quarantine →
仅对 connectivity-eligible customers 建 complete-community 80/20 split。gate：

- Stage 1 endpoint traps 被 directional flag 隔离；
- Stage 2 turn-only traps 被 canonical turn preflight 隔离；
- 每个 quarantined customer 在 city ledger 恰好一行，包括 community 缺失者；
- quarantined customer 的 split pool 必须为 null，且不得出现在 split/family/view；
- 每个 generation-eligible customer 必须恰好属于 train/heldout 之一；
- R2 customer denominator 是 connectivity filtering 前的城市级 unique universe；
- depot 本身四个 reachability mask 都为 true；
- 不存在 ledger/mask count mismatch。

这一阶段只读新 CLE，不 materialize family：

```bash
cd EVRPTW_Dataset_Generator
PYTHONPATH=src /home/npg/miniconda3/envs/evrptw-cle/bin/python \
  scripts/audit_stage2_terminal_connectivity.py \
  --cle-root ../EVRPTW_Dataset/CLE_v2/us_11city \
  --profile configs/us_reference_instance_profile_v2.json \
  --plan-root "$PILOT_ROOT/generation_plan" \
  --split-root "$PILOT_ROOT/customer_splits" \
  --block-group-preset configs/us_census_block_groups_v1.json \
  --block-group-source-dir data/sources/census_block_groups_2025 \
  --output "$PILOT_ROOT/reports/stage2_repair/connectivity_audit.json"
```

输出 schema 固定为 `cle_evrptw_phase_c1_terminal_connectivity_audit_v2`，
必须逐城报告 Stage-1、Stage-2 node、Stage-2 turn 和 union unique-ID rate、pre-split
denominator、split/ledger assertions、Houston/Phoenix/LA known IDs，且 PF-1 每个审计
depot 的严格 lower-bound eligible CS 不少于 50。C0 的 eligible split membership 必须
与上一批准基线逐 ID 相同；不得 resume `pilot_v3`。

### Phase C2：Amazon H3/PF/leakage/5:2 preflight

```bash
PYTHONPATH=src /home/npg/miniconda3/envs/evrptw-cle/bin/python \
  scripts/run_stage2_release_preflight.py \
  --amazon-artifact-root ../EVRPTW_Dataset/Calibration_v2/amazon_stage2_v3 \
  --cohort-split configs/amazon_cohort_split_v1.json \
  --connectivity-audit "$PILOT_ROOT/reports/stage2_repair/connectivity_audit.json" \
  --plan-root "$PILOT_ROOT/generation_plan" \
  --output "$PILOT_ROOT/reports/stage2_repair/release_preflight.json"
```

C2 必须证明 H3 为冻结三站（或记录冻结确定性替代搜索）、所有 primary Cus100/500/1000
weekday/weekend 的 SINGLE_STRUCTURE_DAY 与 SINGLE_ORDER_DAY support 非零、PF-2 与
METRIC-HOLDOUT primary support 非零、四项 leakage assertion 全真，并且每个
city×track 已预分配精确 5 weekday + 2 weekend。

### Phase D：Los Angeles 单 family timing smoke

为了先验证真实图上的运行时间，建议新临时 output root、1 city、train、1 family、
1 worker、1 family/task。具体 city 建议包含 Los Angeles，因为 charger roster 最大且
存在 turn-only 证据；命令只在 reviewer 批准后执行：

```bash
INSTANCE_MODE=non_release_pilot \
RUN_DISCIPLINE=la_smoke \
WORKERS=1 \
FAMILIES_PER_WORKER_TASK=1 \
PILOT_FAMILIES_PER_CITY=1 \
MAX_ATTEMPTS_PER_FAMILY=4 \
PYTHON_BIN=/home/npg/miniconda3/envs/evrptw-cle/bin/python \
INSTANCE_OUTPUT_ROOT=/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Dataset/Instances_v2/la_connectivity_timing_smoke_v2 \
./generate_instances.sh \
  --stages preflight splits plan materialize verify \
  --cities los-angeles \
  --tracks train
```

必须先检查 quarantine ledger、final matrix finite 和 stage timing；RED 停止。AMBER
可进入 pilot，但必须在报告登记 exact-performance optimization。

### Phase E：10 城 train/validation calibration pilot

批准后计划为：10 cities × 2 tracks × 7 families = 140 families。每个
city×track 精确 5 weekday + 2 weekend，所以每 track 共 50 weekday + 20 weekend。
Test2、Test3/Jacksonville、Test1、Cus2000 都不进入此 pilot。

```bash
cd /data/Maojie/ICLR/EVRPTW-DB

/usr/bin/time -v -o logs/generate_instances_pilot_v4.time \
  env PATH=/home/npg/miniconda3/envs/evrptw-cle/bin:/usr/local/bin:/usr/bin:/bin \
      PYTHON_BIN=/home/npg/miniconda3/envs/evrptw-cle/bin/python \
      INSTANCE_MODE=non_release_pilot \
      RUN_DISCIPLINE=pilot \
      WORKERS=12 \
      FAMILIES_PER_WORKER_TASK=1 \
      PILOT_FAMILIES_PER_CITY=7 \
      MAX_ATTEMPTS_PER_FAMILY=4 \
      INSTANCE_OUTPUT_ROOT="$PILOT_ROOT" \
      ./generate_instances.sh \
        --stages materialize verify metrics \
        --cities new-york los-angeles chicago houston phoenix philadelphia \
                 san-antonio san-diego dallas fort-worth \
        --tracks train validation \
  > logs/generate_instances_pilot_v4.log 2>&1
```

`WORKERS=12`、`FAMILIES_PER_WORKER_TASK=1`、`MAX_ATTEMPTS_PER_FAMILY=4` 是本次
pilot 的冻结值。stop rule 触发后不得删除失败 family 或自行改 selector/threshold。

### Phase F：pilot report 与等待最终签字

先生成冻结的 realism/Q90 pairing ledger 和 0.85/0.90/0.95 charging sensitivity：

```bash
cd EVRPTW_Dataset_Generator
PYTHONPATH=src /home/npg/miniconda3/envs/evrptw-cle/bin/python \
  scripts/evaluate_stage2_realism.py \
  --instance-root "$PILOT_ROOT" \
  --amazon-artifact-root ../EVRPTW_Dataset/Calibration_v2/amazon_stage2_v3 \
  --cohort-split configs/amazon_cohort_split_v1.json \
  --output-dir "$PILOT_ROOT/reports/stage2_repair"

PYTHONPATH=src /home/npg/miniconda3/envs/evrptw-cle/bin/python \
  scripts/evaluate_charging_sensitivity.py \
  --instance-root "$PILOT_ROOT" \
  --profile configs/us_reference_instance_profile_v2.json \
  --output-dir "$PILOT_ROOT/reports/stage2_repair/charging_sensitivity"
```

然后用 `scripts/build_stage2_pilot_report.py` 固结 C1、C2、LA smoke、140-family
run、Phase-1、Q90 和 sensitivity evidence。到这里必须停止并交导师审核；本轮不执行
profile promotion。

```bash
cd EVRPTW_Dataset_Generator
PYTHONPATH=src /home/npg/miniconda3/envs/evrptw-cle/bin/python \
  scripts/build_stage2_pilot_report.py \
  --run-report "$PILOT_ROOT/stage2_run_report.json" \
  --phase1-report "$PILOT_ROOT/reports/phase1/summary.json" \
  --q90-report "$PILOT_ROOT/reports/stage2_repair/q90_gate.json" \
  --connectivity-audit "$PILOT_ROOT/reports/stage2_repair/connectivity_audit.json" \
  --release-preflight "$PILOT_ROOT/reports/stage2_repair/release_preflight.json" \
  --la-smoke-report ../EVRPTW_Dataset/Instances_v2/la_connectivity_timing_smoke_v2/stage2_run_report.json \
  --charging-sensitivity "$PILOT_ROOT/reports/stage2_repair/charging_sensitivity/charging_sensitivity_summary.json" \
  --output "$PILOT_ROOT/reports/stage2_repair/pilot_acceptance_report.json"
```

### Phase G：profile promotion 与 full 7,500（**NOT APPROVED**）

只有新签字后才可运行 `scripts/promote_stage2_profile.py`，提交并 push acceptance
commit，确认 clean tree，再以 `INSTANCE_MODE=official`、新正式 output root 和
`--full-run-approved` 启动 full。`research` full 被 runner 明确禁止。本轮不得执行。

```bash
INSTANCE_MODE=official \
WORKERS=12 \
FAMILIES_PER_WORKER_TASK=<REVIEWER_APPROVED_VALUE> \
MAX_ATTEMPTS_PER_FAMILY=4 \
PYTHON_BIN=/home/npg/miniconda3/envs/evrptw-cle/bin/python \
INSTANCE_OUTPUT_ROOT=/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Dataset/Instances_v2/us_11city \
./generate_instances.sh --full-run-approved
```

当前 profile 仍是 `candidate_calibration` 且 `official_generation_eligible=false`，
因此 full 命令现在必然被拒绝。

full gate：`passed=true`、selected=verified=7,500、unresolved=0、Phase-1 hard gates
全通过、D-5 `release_calibrated=true`，并完成 stratified human review。

## 12. Archive、云盘传输与 restore（**NOT APPROVED**；只作未来合同说明）

### 12.1 创建 matrix-free archive

```bash
cd /data/Maojie/ICLR/EVRPTW-DB
PYTHON_BIN=/home/npg/miniconda3/envs/evrptw-cle/bin/python \
./auto.sh archive create \
  --archive /data/EVRPTW_Dataset_us11city_v2_slim.tar.zst \
  --compression-threads 12
```

creator 会拒绝未通过 CLE、Stage-2、Phase-1、V2.1 Q90 gate 的输入。archive 包含
portable CLE 和 matrix-free instance tree；每张被省略矩阵的 path、shape、dtype、
SHA-256 写入 reconstruction contract。zstd 默认 level 9，并生成 `.sha256` sidecar。

### 12.2 另一服务器 restore

```bash
git clone <repository> /path/to/EVRPTW-DB
cd /path/to/EVRPTW-DB

./auto.sh archive start \
  --archive /cloud-transfer/EVRPTW_Dataset_us11city_v2_slim.tar.zst \
  --destination /data \
  --workers 12 \
  --families-per-worker-task <REVIEWER_APPROVED_VALUE>

./auto.sh archive status --destination /data
./auto.sh archive logs --destination /data --follow
./auto.sh archive wait --destination /data
```

restore 先验证 archive SHA、tar member safety、required code commit ancestry、依赖和
磁盘，再在 private staging 解包并原子发布。每张矩阵按 CLE + profile + terminal
projection 重建，必须与 export contract SHA 完全一致；中断后 exact-hash complete
families 可以复用。

正式 archive 前后都应做 arbitrary child view -> parent family mapping 测试；最终还
需要在独立 destination 覆盖全部 7,500 families 的 full restore，phase 必须
`succeeded`。

## 13. 测试证据

执行命令：

```bash
cd /data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Dataset_Generator
PATH=/home/npg/miniconda3/envs/evrptw-cle/bin:/usr/local/bin:/usr/bin:/bin \
/usr/bin/time -f 'elapsed=%e peak_rss_kb=%M' \
/home/npg/miniconda3/envs/evrptw-cle/bin/python -m pytest -q
```

最终 candidate 实测：`147 passed`（pytest quiet progress 无失败），
`elapsed=8.72 s`，`peak_rss_kb=606820`。另用 `pytest --collect-only` 确认共 147 tests。

覆盖本次修复的关键 regression tests：

- fraction=1 的 one-way inbound trap；
- fraction=0 的 one-way return trap；
- reciprocal refs 分别提供 inbound/outbound；
- node 可达但 immediate reversal 禁止的 canonical turn trap；
- stale CLE connectivity contract 被 reader 拒绝；
- `TerminalConnectivityError` non-retryable；
- 普通 `ValueError` 保持 retryable；
- restart 后 attempts 是 lifetime cap；
- slim/reconstruction fixture 需要新 CLE contract。

此外，当前旧 Houston/Phoenix/LA CLE 的只读 load 都按预期失败并提示 rebuild；这是
防止误续跑的 hard gate。

## 14. 代码与 schema 变更摘要

本次连通性修复直接涉及：

```text
src/evrptw_cle/protected_connectivity.py
src/evrptw_cle/customer_access.py
src/evrptw_cle/cle.py
src/evrptw_stage2/reader.py
src/evrptw_stage2/routing.py
src/evrptw_stage2/selection.py
src/evrptw_stage2/parallel.py
scripts/build_stage2_instances.py
tests/test_protected_connectivity.py
tests/test_cle.py
tests/test_stage2.py
tests/test_reconstruction.py
```

schema/contract 表：

| 对象 | 旧 | 新 |
|---|---|---|
| V2 output root | `Calibration_v2` / `Instances_v2` | 不变 |
| CLE connectivity contract | 缺失/anchor-only | `directed_projection_roundtrip_v2` |
| depot star report | v1 | `cle_evrptw_depot_terminal_star_v2` |
| family terminal selection | v2 | `cle_evrptw_family_terminal_selection_v3` |
| quarantine ledger | 无 | `cle_evrptw_terminal_connectivity_quarantine_v1` |
| Stage-2 run report | `cle_evrptw_stage2_run_report_v2` | 不变 |
| canonical matrix policy | zero-turn v3 | 不变 |

仓库中还包含 V2.1 签字修订产生的 Amazon、spatial activation、charging、metrics、
archive/reconstruction 等改动；本稿不把它们误称为“这次连通性 bug 的最小 diff”。
最终 commit 前应以完整 `git diff` 再生成 changed-file summary。

## 15. Reviewer 必须逐项回答

- [ ] 接受 Stage 1 inbound/outbound 可由不同 directed refs 提供。
- [ ] 接受 Stage 1 做 directional SCC gate、Stage 2 再做 canonical turn gate 的分层。
- [ ] 接受坏点 quarantine，不做自动 road remap。
- [ ] 接受完整 quarantine ledger 写入每个 family manifest，或提出版本化 dedup 方案。
- [ ] 接受 fixed-roster connectivity error non-retryable。
- [ ] 接受 attempts=4 是跨 resume 的 lifetime cap。
- [ ] 确认 D-1～D-6、B-3、5:2、Q90 和 schema v2 输出根目录均未改变。
- [x] 允许通过 C1/C2 后跑 Los Angeles 1-family timing smoke。
- [x] pilot 固定 `FAMILIES_PER_WORKER_TASK=1`、workers=12、attempts=4。
- [x] stop rule 按 R-3，无空白分支且不得静默丢弃。
- [ ] 审核 full charger-roster closure 的性能风险；未批准优化前不得宣称已解决。
- [ ] 批准后先重建 11 CLE，再做只读 terminal audit；不得复用旧 CLE。
- [ ] terminal audit 通过后，单独批准 140-family train/validation pilot。
- [ ] pilot evidence 通过后，才可批准 `--full-run-approved`。
- [x] candidate commit/push 必须在 CLE_v2 之前完成。
- [ ] pilot 报告经新签字后，才允许 profile promotion/full。

## 16. 建议签字格式

```text
方法审核：APPROVE / REJECT / CHANGES REQUIRED
连通性修复：APPROVE / REJECT / CHANGES REQUIRED
Los Angeles timing smoke：ALLOW / DO NOT ALLOW
10-city 140-family pilot：ALLOW / DO NOT ALLOW
FAMILIES_PER_WORKER_TASK：____
工程 stop rule：____
full 7,500-family run：当前保持 NOT APPROVED
archive/restore：当前保持 NOT APPROVED
candidate commit/push：ALLOW AND REQUIRED
审核人：____
日期：____
备注：____
```
