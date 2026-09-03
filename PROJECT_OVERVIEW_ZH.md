# EVRPTW-DB 项目背景、数据流程与 Benchmark 总结

> 文档日期：2026-08-25  
> 项目根目录：`/data/Maojie/ICLR/EVRPTW-DB`  
> 当前正式 Stage-2 数据根：`EVRPTW_Dataset/Instances_v2/us_11city_full_clean_v7_bbde5db_20260823`  
> 数据生成代码基线：`bbde5db48dc3f939906fbafdfb18b5b973ae04f1`

本文面向项目汇报、论文写作、代码交接和实验复现，统一说明 EVRPTW-DB 的研究背景、问题定义、数据来源、两阶段生成方法、数据划分、文件组织、求解器 benchmark、评估协议和当前进度。

需要特别注意：仓库中保留了若干历史审核稿和修复 runbook，其中部分内容已经被后续 V2/V3/V7 合同取代。本文优先依据当前根目录 `README.md`、实际 release 配置、当前代码、正式 Stage-2 报告和 benchmark 脚本；历史文档只用于解释设计演化。

---

## 1. 一句话概括

EVRPTW-DB 是一个面向实际城市配送场景的 **Electric Vehicle Routing Problem with Time Windows（EVRPTW）数据集与统一求解 benchmark**。

项目将真实城市道路、潜在配送位置、depot 和充电设施等静态环境，与每天变化的客户激活、包裹需求、服务时间、时间窗和车辆运行条件分开建模：

```text
公开真实地理与基础设施数据
            ↓
Stage 1: City Logistics Environment (CLE)
            ↓
Amazon 运营模板 + 冻结的生成规则
            ↓
Stage 2: CLE-EVRPTW operating-day instances
            ↓
Gurobi / ALNS / VNS-TS / RL benchmark
            ↓
统一 route replay、anytime checkpoints 与跨方法比较
```

数据集最准确的定位是：

> **Infrastructure-grounded, real-geography, semi-synthetic EVRPTW benchmark.**

即：道路、城市边界、建筑/住宅位置、充电设施和 depot 候选由真实公开数据提供；具体某一天哪些客户被激活、需求多少、服务多久、时间窗是什么，则由经 Amazon Last Mile 2021 运营数据校准的生成模型产生。

它不是对某条真实 Amazon 配送路线的复原，也不能把所有 OSM warehouse、AFDC charger 或生成订单解释为经过人工逐站验证的真实运营记录。

---

## 2. 为什么要做这个项目

传统 EVRPTW benchmark 通常存在以下限制：

1. 使用欧式平面距离或人工随机坐标，无法反映真实单行道、道路等级和城市拓扑；
2. 充电站、depot 和客户位置缺少现实基础设施依据；
3. 不同规模的实例彼此独立，难以判断算法在同一物理环境下的规模泛化能力；
4. train、validation 和 test 容易发生空间或模板泄漏；
5. Exact、metaheuristic 和 RL 使用不同的距离、能耗、充电或可行性定义，使算法结果不能公平比较；
6. 大规模实例如果重复保存完整路网和矩阵，会造成非常高的存储和传输成本。

EVRPTW-DB 的目标是建立一套同时满足以下要求的研究基础设施：

- 真实有向道路网络和城市边界；
- 有数据来源和版本记录的客户、depot、charging station 候选；
- 一致的距离、运行时间、能量和充电合同；
- 空间上严格区分 train、validation 和多个 test 条件；
- 支持 Cus50 到 Cus2000 的层次化规模实验；
- 支持 Exact、metaheuristic 和 learning-based 方法公平比较；
- 数据可压缩、传输、确定性恢复并验证；
- 每条正式结果路线都可以被独立 replay，而不是只相信求解器自行报告的 objective。

---

## 3. EVRPTW 问题定义

每个 operating-day instance 包含：

- 一个 depot；
- $N$ 个需要被服务的客户；
- 一组可访问的 charging stations；
- 一支同质电动车队；
- directed terminal-to-terminal distance、travel-time 和 energy cost；
- 每个客户的需求、服务时间和时间窗；
- 每个 charging station 的可用充电功率；
- 车辆容量、电池容量、工作时间范围和充电策略。

求解器需要输出若干条从 depot 出发并返回 depot 的车辆路线，使得：

1. 每个客户恰好服务一次；
2. 车辆载货量不超过容量；
3. 到达客户时满足时间窗，允许提前到达后等待；
4. 行驶过程中电池不能低于零；
5. 必要时可以访问 charging station；
6. 所有路线在 operating horizon 结束前返回 depot；
7. 最小化所有车辆的总 directed travel distance。

当前统一 objective 为：

\[
\min \sum_{(i,j)\in\text{used arcs}} d_{ij}
\]

其中 $d_{ij}$ 来自真实有向路网的 `distance_matrix_km`，并非节点间欧式距离。

### 3.1 当前冻结的车辆与运行参数

| 项目 | 当前合同 |
|---|---|
| 车辆参考 | Rivian Commercial Van Delivery 700 proxy |
| Cargo capacity | 18,500,000 cm³ |
| Battery capacity | 100 kWh |
| Nominal range | 257 km |
| Specific energy | 0.389105 kWh/km |
| Energy model | 与path distance线性相关；不含speed dependence与auxiliary load |
| AC L2 vehicle cap | 11 kW |
| DC vehicle cap | 100 kW |
| Charging derating | 0.90 |
| Charging behavior | 每次访问充电站后full recharge，线性充电 |
| Fleet size | unlimited；实际使用车辆数由解决定 |
| Charging ports | infinite-port modeling assumption |
| Operating horizon | 08:00–24:00，即28,800–86,400秒 |
| Day-type ratio | weekday : weekend = 5 : 2 |

充电站 $q$ 的有效功率为：

\[
P_q^{effective}=0.90\min(P_q^{station},P^{vehicle\ cap})
\]

如果到站时剩余电量为 $b_q$，充满所需时间为：

\[
t_q^{charge}=\frac{B-b_q}{P_q^{effective}}\times3600
\]

---

## 4. 两阶段数据设计

## 4.1 Stage 1：City Logistics Environment（CLE）

CLE 是可被许多 operating-day instances 复用的静态城市环境。它冻结：

- land-only city service boundary；
- 真实 directed OSM road graph；
- edge-level legal/reference speeds；
- 潜在住宅服务位置；
- depot candidates；
- public charging-site candidates；
- 设施到道路的projection和access connectors；
- directed connectivity、SCC和canonical turn-topology证据；
- 所有输入来源、版本、hash、QA和release状态。

CLE **不包含**：

- 当天active customer；
- package count或volume demand；
- realized service time；
- realized time window；
- 某一天具体的配送路线。

这种设计使一个大城市路网只需构建一次，后续可以在同一物理环境上生成大量不同运营日和不同规模实例。

## 4.2 Stage 2：CLE-EVRPTW instances

Stage 2 在CLE上采样一天的EV配送问题。它负责：

- 选择depot；
- 从指定spatial split pool中激活恰好 $N$ 个客户；
- 为每个客户分配互不重复的Amazon order template；
- 选择相关charging stations；
- 选择weekday或weekend road state；
- 建立terminal closure与四张parent matrices；
- 构造Cus50/Cus100/Cus500/Cus1000/Cus2000 views；
- 验证每个view的基础可行性和数据合同。

Stage 1与Stage 2分离还有一个重要作用：防止把未来某一天的active demand泄漏进城市静态环境或solver input。

---

## 5. 公开数据来源及各自作用

| 数据源 | 在项目中的作用 |
|---|---|
| Census TIGER/Line Place + Hydrography | city-proper边界与water removal |
| OpenStreetMap / Geofabrik | directed road graph、road tags、depot/设施证据 |
| Microsoft US Building Footprints | 建筑polygon和geometry support |
| USACE National Structure Inventory | 住宅occupancy、unit和structure evidence |
| NREL Alternative Fuels Data Center | public charging station位置、端口和功率证据 |
| FHWA HPMS | road functional class与direction-verified legal-speed补充证据 |
| EPA MOVES5 | weekday/weekend、road-class层面的speed-retention profile |
| Census Block Groups | customer communities和spatial train/held-out split |
| Amazon Last Mile Routing Challenge 2021 | depot-day route structure、package volume、service time和time-window模板 |

Amazon的匿名坐标不会被转移到CLE城市。项目迁移的是运营结构和订单属性分布，而不是把Amazon stop伪装成这些城市中的真实地点。

Amazon派生数据受上游CC BY-NC 4.0约束；代码许可证不重新授权这些数据。

---

## 6. 城市设计与泛化测试

### 6.1 十个seen cities

```text
New York City, Los Angeles, Chicago, Houston, Phoenix,
Philadelphia, San Antonio, San Diego, Dallas, Fort Worth
```

这些城市用于train、validation、Test-1、Test-2和same-city unseen-scale测试。

### 6.2 完全held-out city

```text
Jacksonville
```

Jacksonville只用于Test-3，不进入训练、validation、Test-1、Test-2或same-city Cus2000。

边界为2025 Census incorporated place的city-proper land area，不是metro area，也不声称等同于真实承运商service territory。

### 6.3 三类核心测试

| Track | 测试什么 | 城市/客户pool |
|---|---|---|
| Test-1: new seed | 相同城市、相同train空间pool下的新运营实例 | 十个seen cities的train communities |
| Test-2: held-out locations | 相同城市内未见过的完整空间community | 十个seen cities的held-out communities |
| Test-3: held-out city | 跨城市泛化 | Jacksonville |

另外还有：

- `compatibility_cus50`：用于小规模MIP与传统算法直接比较；
- `scalability_cus2000`：相同seen cities上的unseen-scale评估，同时保存同family Cus1000 control。

---

## 7. Stage 1 CLE生成流程

CLE主流程可概括为：

```text
Preflight
  → road extraction and directed graph
  → building/customer evidence
  → depot and charger candidates
  → speed-profile construction
  → terminal-to-road access materialization
  → connectivity audit
  → CLE assembly
  → technical verification
  → portable packaging and strict verification
```

### 7.1 Boundary与道路图

- 使用Census city place boundary；
- 去除water area，得到land-only service boundary；
- 从冻结的Geofabrik PBF提取可驾驶OSM道路；
- 保留有向边和单行道语义；
- 对路网进行reference SCC和连通性审核；
- 每条physical edge与directed edge保持可追溯ID。

### 7.2 潜在服务位置

- 使用NSI住宅结构记录；
- 只保留ordinary residential occupancy families；
- 按structure evidence估计住宅units；
- 与Microsoft building footprints进行geometry匹配；
- 分类为house、manufactured home、small/medium/large apartment；
- 将建筑位置投影到可用road edge并建立双向access connector；
- Stage 1只保存latent opportunity，不激活客户。

### 7.3 Depot与charger

- Depot候选来自OSM logistics/industrial/warehouse等设施证据；
- 同一物理设施的多个access points按严格证据归组；
- AFDC提供public charging-site和power/port证据；
- 原始与resolved coordinates及QA evidence都保留；
- 默认候选必须通过相应的道路access与connectivity规则。

### 7.4 Speed model

每条有向道路保存：

- `legal_speed_kph`；
- `free_flow_speed_proxy_kph`；
- `reference_speed_weekday_kph`；
- `reference_speed_weekend_kph`。

OSM `maxspeed`优先；高置信、方向可验证的HPMS speed可以补充缺失值。MOVES5不是edge-level观测，而是提供national road-class/day-type speed-retention factor：

\[
v^{reference}_{e,d}=v^{legal}_e\rho_{road\ type,d}
\]

这种方法保留每条边自身legal-speed尺度，同时把weekday/weekend运行差异引入静态road state。

### 7.5 Connectivity repair合同

Terminal投影到有向边内部时，进入和离开projection具有不同方向语义：

```text
进入 terminal: reference graph → directed edge u → projection
离开 terminal: projection → directed edge v → reference graph
```

当前CLE必须满足`directed_projection_roundtrip_v2`：

- inbound access存在；
- outbound access存在；
- Stage 2还要做node-level与canonical zero-turn line-graph双向preflight；
- 失败terminal进入deterministic quarantine ledger，而不是静默删除或反复换seed。

---

## 8. Stage 2实例生成流程

正式流程为：

```text
Amazon artifact preparation
  → customer connectivity quarantine
  → community split and adjacency
  → generation plan
  → C3 joint spatial-support gate
  → family-by-family materialization
  → family verification and feasibility gate
  → Phase-1/construct-valid acceptance
  → slim archive
  → independent exact matrix restoration
```

### 8.1 Amazon模板准备

当前紧凑artifact为`amazon_stationday_stage2_v3`，正式报告记录：

- 521个station-days；
- 898,391个order templates；
- overall envelope $T_{env}=3486.4$ 秒。

模板拆分成两类信息：

1. route/stop空间运营结构；
2. customer-level package volume、service time和time window。

Primary Cus100/500/1000只使用`SINGLE_STRUCTURE_DAY + SINGLE_ORDER_DAY`；同station多日composite只允许Cus2000 report-only。

### 8.2 防止Amazon模板泄漏

- `METRIC-HOLDOUT`整个station隔离：DCH2、DLA9、DSE2；
- generation pool拆为GEN-TRAIN和GEN-EVAL；
- train只使用GEN-TRAIN；
- validation、Test-1/2/3和unseen-scale使用互斥的GEN-EVAL track ledgers；
- station-day、template ID和route ID均有hard leakage assertions。

### 8.3 Customer community与空间split

```text
community_id = Census Block Group × directed-road SCC
```

整个community被分到train或held-out，不允许把同一community中的building随机拆开。

当前held-out fraction为20%。这种设计避免train和Test-2只隔一栋建筑造成过度相似。

### 8.4 Family terminal selection

每个family按以下顺序构建：

1. 选择physical depot group及access point；
2. 对完整customer roster做node和canonical-turn depot-star preflight；
3. 隔离不连通customers；
4. 使用source-specific Amazon $T_{env}$和direct round-trip battery sufficient condition定义territory；
5. 采用region-first spatial activation选择恰好 $N$ 个customers；
6. 对charging roster做同样的node/turn preflight；
7. 根据已激活区域选择相关charging stations；
8. 对最终depot、customers和chargers做完整all-pair terminal closure；
9. 为customers做distinct Amazon order-template assignment；
10. 输出parent family、nested views、可行性certificate和Phase-1证据。

### 8.5 Retry与失败语义

- 每个family跨resume累计最多4次attempt；
- 随seed/source变化的失败可以retry；
- 固定roster connectivity failure为non-retryable；
- 每次拒绝保存reason、seed、stage与ledger；
- 同一个坏terminal不会通过换seed被掩盖。

---

## 9. Parent Family与Nested View设计

为了避免为每个规模重复计算和保存巨大矩阵，数据采用family/view分层结构。

### 9.1 Parent families

- Core parent为Cus1000；
- 每个Cus1000 parent保存一次terminal order和四张dense matrices；
- Cus50、Cus100和Cus500只是parent indices的嵌套子集；
- Cus2000为独立parent，并同时保存同family的Cus1000 control。

### 9.2 Nested partition

一个Cus1000 parent被确定性拆成：

```text
Cus1000
  ├── 2 × Cus500
  ├── 10 × Cus100
  └── 20 × Cus50
```

20个Cus50 leaf彼此互斥，union必须恰好恢复全部1000个parent customers。

这样可以研究：

- 相同物理环境下的scale effect；
- 算法从Cus50到Cus1000的scalability；
- 不同方法是否在相同customer subset上公平比较；
- learning方法对嵌套规模的泛化。

### 9.3 当前正式数量

| Cohort | Parent families | Views |
|---|---:|---:|
| Core train | 5,000 | 65,000 = 50k Cus100 + 10k Cus500 + 5k Cus1000 |
| Core validation | 500 | 1,500 = 每个scale各500 |
| Test-1 new seed | 500 | 1,500 = 每个scale各500 |
| Test-2 held-out locations | 500 | 1,500 = 每个scale各500 |
| Test-3 held-out city | 500 | 1,500 = 每个scale各500 |
| Cus2000 scalability | 500 | 1,000 = 500 Cus1000 controls + 500 Cus2000 |
| Cus50 compatibility train | 共享Core parents | 100,000 |
| Cus50 compatibility val | 共享Core parents | 500 |
| Cus50 compatibility Test-1 | 共享Test-1 parents | 500 |
| **Total physical families** | **7,500** |  |
| **Total logical views** |  | **173,000** |

---

## 10. 每个Family和View存什么

典型目录：

```text
materialized/families/<family_id>/
├── family_manifest.json
├── terminal_index.parquet
├── phase1_metrics.json
├── phase1_observations.parquet
├── phase1_region_pair_metrics.parquet
├── matrices/
│   ├── distance_matrix_km.npy
│   ├── distance_path_travel_time_s.npy
│   ├── running_time_shortest_matrix_s.npy
│   └── running_time_path_distance_km.npy
└── views/<view_id>/
    ├── view_manifest.json
    ├── terminal_parent_indices.npy
    ├── customer_attributes.npz
    └── charging_attributes.npz
```

### 10.1 四张stored parent matrices

| Matrix | 用途 |
|---|---|
| `distance_matrix_km` | distance-shortest path distance；统一objective |
| `distance_path_travel_time_s` | 上述distance-shortest path对应的travel time |
| `running_time_shortest_matrix_s` | fastest running-time path；时间窗传播使用 |
| `running_time_path_distance_km` | fastest-time path对应的distance；用于energy |

Energy矩阵不重复存储，而是由路径distance确定性派生：

\[
E^{distance\ path}_{ij}=D^{distance}_{ij}h
\]

\[
E^{fastest\ path}_{ij}=D^{fastest}_{ij}h
\]

其中 $h=0.389105$ kWh/km。

### 10.2 为什么distance与time使用不同路径

- Objective希望最小化真实directed distance；
- 时间窗传播必须使用fastest-running-time path；
- 电池消耗应与实际选择的fastest-time path distance一致；
- 因此不能用一张矩阵同时代替distance、time和energy。

### 10.3 View attributes

每个view保存：

- parent terminal indices；
- customer volume demand和package count；
- service time；
- 一个客户时间窗；
- charging-station power；
- vehicle和charging contract；
- compact single-customer feasibility certificate；
- family/view/provenance IDs。

---

## 11. 数据质量、可行性与Acceptance

项目把“内部正确”“可移植”“构造有效”和“完全真实”分开描述。

### 11.1 Hard correctness gates

至少包括：

- schema和manifest一致；
- ID唯一、行数和terminal order正确；
- 四张矩阵均为finite、nonnegative、float32、zero diagonal；
- depot/customer/charger在node和canonical turn topology上双向可达；
- customer demand、service time和time window合法；
- charging power与车辆cap合同一致；
- family/view嵌套索引一致；
- customer-level feasibility certificate可验证；
- Amazon train/eval/metric-holdout不存在泄漏；
- Phase-1必需文件完整；
- corpus不存在unresolved family。

### 11.2 Construct-valid acceptance

当前`stage2_construct_valid_v3`将以下两类证据区分：

- Amazon运营迁移有效性：作为主要hard gate；
- 生成城市与Amazon之间的空间统计差异：作为report-only diagnosis，而不是强迫真实城市模仿Amazon匿名空间。

### 11.3 不能过度声称的内容

当前Stage-2正式生成报告为`release_eligible=true`，表示它通过了冻结的technical candidate和construct-valid合同。但底层CLE manifest仍保留人工科学审核blockers，例如：

- Microsoft/NSI geometry review；
- customer road-access review；
- depot release review；
- charger coordinate/release review；
- delivery-community interpretation。

因此论文和汇报应使用：

> **Infrastructure-grounded semi-synthetic benchmark based on technically verified candidate pools.**

不应使用：

> “所有客户、depot和charging stations都经过人工现场验证”或“完全真实运营数据集”。

---

## 12. 数据划分与训练含义

### 12.1 Core train

- 10个seen cities；
- train communities；
- GEN-TRAIN Amazon pool；
- Cus100/Cus500/Cus1000；
- 每个scale约5,000,000 customer exposures。

### 12.2 Validation

- 相同seen cities和train spatial pool；
- 使用GEN-EVAL中的独立validation ledger；
- 用于模型选择、超参数和early stopping；
- 不等同于最终Test-1。

### 12.3 Test-1

- seen cities；
- train communities；
- 新family/view seeds；
- 独立GEN-EVAL template ledger；
- 测试相同分布条件下的新实例泛化。

### 12.4 Test-2

- seen cities；
- 完整held-out communities；
- 独立GEN-EVAL ledger；
- 测试同城市未见空间位置的泛化。

### 12.5 Test-3

- 完全held-out Jacksonville；
- 独立GEN-EVAL ledger；
- 测试跨城市泛化。

### 12.6 Cus2000

- seen cities的train community pool；
- 训练中未见的更大规模；
- 允许same-station composite Amazon source；
- report-only scalability track，并带Cus1000 control。

---

## 13. Benchmark方法

当前统一benchmark包括：

```text
Exact/Gurobi_Solver
MetaHeuristics/ALNS_Solver
MetaHeuristics/VNS_TS_Solver
Reinforcement_Learning/TERRAN
```

## 13.1 Gurobi Exact MILP

用途：小规模Exact/MIP baseline，主要用于Cus50。

主要合同：

- binary routing arcs；
- 每个客户exactly once；
- unlimited homogeneous fleet；
- capacity、time-window和battery propagation；
- charging station以固定数量dummy copies展开，当前`cs_copies=2`；
- 每次charging visit按station-specific effective power充满；
- MIP objective为directed distance；
- `MIPGap=0`，但只有实际证明optimal才标记`COMPLETED_OPTIMAL`；
- time limit结束时有有效incumbent则标记`COMPLETED_WITH_INCUMBENT`。

所有Gurobi incumbent都由独立route replay重新验证。

## 13.2 ALNS

Adaptive Large Neighborhood Search baseline：

- 使用Stage-2 feasibility certificate构造可行warm start；
- 多种destroy/repair operators和自适应选择；
- 全程受统一wall-clock budget控制；
- objective、time、energy、charging与Gurobi合同一致；
- 每个checkpoint solution独立replay。

## 13.3 VNS-TS

Variable Neighborhood Search + Tabu Search baseline：

- 首先发布完整certificate对应的safe feasible incumbent；
- 进行deadline-bounded route consolidation；
- 再执行VNS/Tabu distance improvement；
- scalable `fast` profile对不同scale使用确定性bounded neighborhoods；
- 不允许输出partial-customer solution。

## 13.4 TERRAN / RL

Reinforcement-learning方向通过统一EVRPTW environment消费相同instance语义：

- action包括depot、customers和charging stations；
- action mask表示当前可行动作；
- state跟踪位置、时间、容量、电池和已服务客户；
- route export使用与benchmark一致的节点序列；
- 最终仍应通过EVRPTW Core replay，不能只依赖environment reward。

当前Exact和metaheuristic的Stage-2 runner与结果合同更成熟；RL部分应在正式比较前再次核对charging mode与当前`full_charge_linear_derated_v2`完全一致。

---

## 14. 公平比较与评估协议

### 14.1 相同输入合同

所有正式solver必须使用：

- objective：`distance_matrix_km`；
- travel time：`running_time_shortest_matrix_s`；
- battery：`running_time_path_energy_kwh`；
- demand/capacity：cm³；
- time/service/TW：秒；
- station-specific full linear charging；
- 相同vehicle、battery、charger和horizon配置。

### 14.2 Anytime checkpoints

当前Cus50正式合同：

```text
5 min / 30 min / 60 min / 120 min
= 300 / 1800 / 3600 / 7200 seconds
```

每个checkpoint保存当时已经发现且通过replay的最好route与objective。

严格因果规则：

- checkpoint之后才发现的解不能回填到更早checkpoint；
- 如果算法自然提前结束，最终可行解只能forward-fill到更晚checkpoint；
- 若截止时没有incumbent，保持空值，不能人为写入惩罚objective。

### 14.3 建议报告指标

| 指标 | 含义 |
|---|---|
| Feasible coverage | 在时间预算内找到合法解的instance比例 |
| Objective distance | 经replay后的总directed distance |
| Gap to Gurobi incumbent/optimum | Metaheuristic/RL相对Exact参考差距 |
| Gurobi MIP gap | incumbent与best bound之间的证明差距 |
| Runtime to first feasible | 首次得到可行解的速度 |
| Anytime curve | 5/30/60/120分钟objective变化 |
| Vehicle count | 辅助分析；当前不是primary objective |
| Replay pass rate | 求解器输出经过独立验证的比例 |
| Completion status | optimal、time-limited incumbent、no incumbent、invalid等 |

不能把“Gurobi在2小时内最好的incumbent”自动称为“optimal”；只有Gurobi证明optimality后才成立。

### 14.4 统一状态

- `COMPLETED_OPTIMAL`
- `COMPLETED_WITH_INCUMBENT`
- `UNFINISHED_NO_INCUMBENT`
- `INVALID_INCUMBENT`
- `NO_FEASIBLE_SOLUTION`

---

## 15. Route Replay为什么重要

每个solver可能在内部使用不同的数据结构、时间单位或charging expansion。为了防止“objective看起来更好但实际违反约束”，所有published routes都由独立validator逐arc重放，检查：

- customer coverage和重复访问；
- directed distance；
- running time；
- waiting；
- service time；
- time windows；
- cargo capacity；
- battery consumption；
- station-specific charging time；
- operating-horizon return。

只有replay通过的incumbent才能进入summary、time trace和checkpoint solution。

---

## 16. 数据压缩、传输与恢复

四张parent matrices约占157.37 GB，是确定性cache而不是不可替代的原始数据。

Slim archive保留：

- 11-city CLE；
- family/view参数；
- terminal selection；
- order、charging和vehicle attributes；
- reconstruction contract；
- matrix hashes/identity metadata；
- 但省略每个family的`matrices/`目录。

在目标服务器上：

```text
Slim archive
  → extract to private staging
  → validate CLE and instance contracts
  → reconstruct four matrices for each parent family
  → validate all 7,500 families
  → atomic promotion
```

求解器不区分matrix来自直接生成还是确定性恢复；只要schema和portable identity一致，就进入相同adapter和validator。

当前正式archive：

```text
/data/Maojie/ICLR/EVRPTW-DB_Releases/
EVRPTW_Dataset_us11city_full_clean_v7_bbde5db.tar.zst
```

独立恢复验证报告确认：

```text
passed = true
selected_family_count = 7500
```

当前流程为了性能明确记录`file_hash_validation_performed=false`；它验证结构、合同和确定性矩阵恢复，但没有对所有大型文件重新执行完整SHA扫描。汇报时应如实说明。

---

## 17. 当前正式数据状态

截至2026-08-25：

| 项目 | 状态 |
|---|---|
| Clean Stage-2 output | Passed |
| Families | 7,500 |
| Logical views | 173,000 |
| Unresolved family IDs | 0 |
| Stage-2 report | `passed=true` |
| Stage-2 run manifest | `release_eligible=true` |
| C3 joint spatial support | `passed_full_plan` |
| Construct-valid V3 | 已执行并通过clean pipeline |
| Slim archive creation/inspection | Passed |
| Independent matrix restoration | 7,500/7,500 passed |
| Generation workers | 30 |
| Recorded Stage-2 run wall time | 58,924.5 s，约16.37小时 |
| Clean end-to-end pipeline elapsed | 89,759.9 s，约24.93小时 |
| Generator regression test log | 247 tests completed |

当前benchmark状态快照：

- Cus50 Test-1 Gurobi正在运行；
- 输入为500个Cus50 views；
- 30个独立workers，每个Gurobi process 1 thread；
- 每个instance最多7,200秒；
- checkpoints为300/1800/3600/7200秒；
- `cs_copies=2`，`MIPGap=0`；
- 当前log只能证明任务已启动，尚不能作为最终Gurobi结果表。

当前运行日志：

```text
logs/benchmarks/gurobi_cus50_test1_e660b96/run.log
```

---

## 18. 正式Benchmark执行流程

### 18.1 数据前置检查

在任意solver benchmark之前必须确认：

1. 使用同一Git commit；
2. 使用同一restored release ID；
3. `matrix_restore_report.json`通过且family count为7,500；
4. solver环境和许可证可用；
5. 输出目录不与另一个active runner共享；
6. 时间预算、checkpoint、seed和solver profile冻结。

### 18.2 Cus50 Test-1

```bash
bash EVRPTW_Benchmark/test_scripts/run_gurobi_cus50_test.sh
bash EVRPTW_Benchmark/test_scripts/run_alns_cus50_test.sh
bash EVRPTW_Benchmark/test_scripts/run_vnsts_cus50_test.sh
```

顺序运行三种方法：

```bash
bash EVRPTW_Benchmark/test_scripts/run_all_cus50_tests.sh
```

### 18.3 Cus500 Core Test-1/2/3

```bash
bash EVRPTW_Benchmark/test_scripts/run_gurobi_cus500_tests.sh
bash EVRPTW_Benchmark/test_scripts/run_alns_cus500_tests.sh
bash EVRPTW_Benchmark/test_scripts/run_vnsts_cus500_tests.sh
```

每个solver覆盖3个track × 500 views = 1,500 instances。若全部实例用满2小时，在单服务器上代价很高，应使用multi-server shard。

### 18.4 Multi-server shard

例如10台服务器中的server 3：

```bash
EVRPTW_SHARD_COUNT=10 EVRPTW_SHARD_INDEX=3 \
  bash EVRPTW_Benchmark/test_scripts/run_all_cus500_tests.sh
```

所有服务器必须使用相同shard count，且shard index恰好覆盖`0..count-1`一次。

### 18.5 Resume

Exact和metaheuristic launcher都支持`--skip_completed`。是否跳过由完整run-contract fingerprint决定，包含：

- algorithm/profile；
- budget/checkpoints；
- base与per-view seed；
- search参数；
- replay policy；
- portable dataset identity。

worker count、shard/range和output path不影响同一view的随机seed或问题身份。

---

## 19. Benchmark输出

### Gurobi

```text
gurobi_summary.csv
gurobi_time_trace.csv
solutions/*.pkl
solutions/checkpoints/*.pkl
```

### ALNS

```text
alns_summary.csv
alns_time_trace.csv
solutions/<run-contract-fingerprint>/*.pkl
solutions/checkpoints/<run-contract-fingerprint>/*.pkl
```

### VNS-TS

```text
vns_ts_summary.csv
vns_ts_time_trace.csv
solutions/<run-contract-fingerprint>/*.pkl
solutions/checkpoints/<run-contract-fingerprint>/*.pkl
```

每个summary应至少保留：

- instance/view/family ID；
- solver和profile；
- seed与run-contract fingerprint；
- benchmark status；
- objective distance；
- runtime；
- vehicle count；
- replay结果；
- Gurobi bound/gap或metaheuristic search metadata；
- charging/matrix/schema provenance。

跨solver比较使用：

```bash
python EVRPTW_Benchmark/compare_solver_summaries.py \
  --reference_summary <gurobi_summary.csv> \
  --candidate_summary <alns_or_vnsts_summary.csv> \
  --reference_name gurobi \
  --candidate_name candidate \
  --save_path <comparison.csv>
```

---

## 20. 代码结构

```text
EVRPTW-DB/
├── EVRPTW_Core/
│   └── evrptw_core/
│       ├── schema.py          # 统一instance/solution schema
│       ├── io.py              # 数据和solution I/O
│       └── validation.py      # 独立route replay/validation
├── EVRPTW_Dataset_Generator/
│   ├── configs/               # 冻结profile与acceptance合同
│   ├── src/evrptw_cle/        # Stage-1 CLE构建
│   ├── src/evrptw_stage2/     # Stage-2 planning/materialization/routing
│   ├── scripts/               # source preparation、generation、restore、QA
│   └── docs/                  # 数据模型、schema、性能与历史审核记录
├── EVRPTW_Dataset/
│   ├── CLE_v2/                # portable static city environments
│   ├── Calibration_v2/        # compact Amazon-derived artifacts
│   └── Instances_v2/          # Stage-2 families/views和历史runs
├── EVRPTW_Benchmark/
│   ├── Exact/Gurobi_Solver/
│   ├── MetaHeuristics/ALNS_Solver/
│   ├── MetaHeuristics/VNS_TS_Solver/
│   ├── Reinforcement_Learning/TERRAN/
│   ├── test_scripts/          # 当前标准Cus50/Cus500 launchers
│   └── results/
├── generate_cle.sh
├── generate_instances.sh
├── run_clean_full_pipeline.sh
├── create_dataset_archive.sh
├── restore_dataset_archive.sh
└── auto.sh
```

---

## 21. 关键入口与复现方式

### 21.1 环境

Generator推荐环境：

```bash
cd EVRPTW_Dataset_Generator
conda env create -f environment.yml
conda activate evrptw-cle
```

Benchmark shell默认环境为`maojie`，可用`EVRPTW_CONDA_ENV`覆盖。

### 21.2 生成CLE

```bash
./generate_cle.sh
```

### 21.3 生成Stage 2

正式完整生成必须使用release profile、fresh output root和显式full-run approval。当前已完成结果不需要重复生成；复现时优先参考：

```text
run_clean_full_pipeline.sh
EVRPTW_Dataset_Generator/configs/us_reference_instance_profile_v2_release.json
EVRPTW_Dataset_Generator/configs/stage2_acceptance_v3_full_7500.json
```

### 21.4 Restore archive

```bash
./restore_dataset_archive.sh start \
  --archive /path/to/EVRPTW_Dataset_us11city_full_clean_v7_bbde5db.tar.zst \
  --destination /path/to/restore_root \
  --workers 30

./restore_dataset_archive.sh status --destination /path/to/restore_root
./restore_dataset_archive.sh logs --destination /path/to/restore_root --follow
./restore_dataset_archive.sh wait --destination /path/to/restore_root
```

---

## 22. 推荐的论文/汇报叙事

可以按下面的逻辑介绍项目：

1. 传统EVRPTW benchmark缺少真实城市拓扑、基础设施和严格泛化split；
2. 我们提出两阶段数据设计，将可复用城市环境与每日运营实例分开；
3. CLE整合OSM、Census、Microsoft、NSI、AFDC、HPMS和MOVES5；
4. Stage 2只迁移Amazon运营模板，不迁移匿名坐标；
5. family/view结构在同一物理环境下提供Cus50–Cus2000的嵌套规模；
6. Test-1、Test-2、Test-3分别测试new-seed、unseen-location和unseen-city泛化；
7. Exact、ALNS、VNS-TS和RL共享同一个resource/charging/objective合同；
8. 每条route都由独立validator replay；
9. matrices可从slim release确定性恢复，降低发布与多服务器部署成本；
10. 当前正式Stage-2包含7,500 families和173,000 views，并通过完整生成与独立恢复验证。

---

## 23. 常见误解

### 误解1：这是完全真实的Amazon配送数据

不是。真实地理和基础设施来自公开数据；operating-day demand由Amazon模板校准后生成。

### 误解2：Amazon匿名坐标被映射到了11个城市

没有。项目只迁移运营结构和订单属性，不迁移匿名坐标。

### 误解3：Distance、time和energy都来自同一最短路径

不是。Distance objective与fastest-time resource propagation使用不同canonical paths，因此保存四张矩阵。

### 误解4：一个family就是一个instance

不完全是。Family拥有parent terminals和matrices；多个不同scale view共享同一family。

### 误解5：Gurobi在2小时后给出的解就是optimal

不是。只有证明optimality才标为`COMPLETED_OPTIMAL`；否则只是time-limited incumbent。

### 误解6：Solver报告feasible就足够

不够。正式结果必须通过独立route replay。

### 误解7：`release_eligible=true`等于每个物理站点人工验证完成

不是。它表示Stage-2通过冻结的技术与construct-valid合同；底层CLE仍显式保留人工科学审核状态。

---

## 24. 建议优先阅读的权威文件

1. `README.md`：项目定位与当前总体流程；
2. `EVRPTW_Dataset_Generator/docs/STAGE2_INSTANCE_MODEL.md`：Stage-2英文权威模型；
3. `EVRPTW_Dataset_Generator/docs/OUTPUT_SCHEMA.md`：CLE/family/view schema；
4. `EVRPTW_Dataset_Generator/docs/PIPELINE.md`：Stage-1 CLE流程；
5. `EVRPTW_Dataset_Generator/docs/STAGE2_PERFORMANCE.md`：并行生成、resume和资源模型；
6. `EVRPTW_Benchmark/README.md`：统一solver合同；
7. `EVRPTW_Benchmark/test_scripts/README.md`：当前Cus50/Cus500标准运行方式；
8. `EVRPTW_Benchmark/Exact/Gurobi_Solver/README.md`；
9. `EVRPTW_Benchmark/MetaHeuristics/ALNS_Solver/README.md`；
10. `EVRPTW_Benchmark/MetaHeuristics/VNS_TS_Solver/README.md`；
11. 当前正式数据的`stage2_run_report.json`和`run_manifest.json`。

历史repair文档适合了解为什么引入connectivity quarantine、canonical-turn preflight和construct-valid V3，但不应覆盖当前配置和实际生成报告。

---

## 25. 最终总结

EVRPTW-DB的核心贡献不是简单“生成很多VRP文件”，而是建立了一条从公开城市数据到可验证算法实验的完整链路：

```text
真实静态环境
  +
防泄漏的半合成运营实例
  +
嵌套多规模数据组织
  +
统一Exact/heuristic/RL求解合同
  +
独立可行性replay
  +
可传输、可恢复、可审计的发布流程
```

当前7,500-family / 173,000-view正式Stage-2数据已经完成生成、构造有效性评估、archive和7,500-family独立matrix restoration。下一阶段工作的重点是完成Cus50和Cus500上的统一solver benchmark，形成coverage、objective、gap、runtime和anytime curve结果，再扩展和校准RL方法。
