# EVRPTW-DB 11 城 Instance 生成全流程（代码实现审阅版）

> 文档性质：本说明以仓库当前实现为准，面向方法审核、运行复现和后续 release 审计。
>
> 对应代码版本：以每次 fresh run 的 `stage2_run_report.json` 和 release manifest 中的 `code_commit` 为准。
>
> 正式数据必须写入不存在的 fresh root；不得 resume 或复制旧 run 的 family。
>
> 重要边界：CLE 的构建方式与此前介绍完全相同，本次没有修改 CLE 生成方法。Stage 2 使用已经通过技术验证和 directed-connectivity quarantine 的 CLE candidate pools；所有 CLE manual-release blockers 永久保留。因此数据集应描述为 **infrastructure-grounded semi-synthetic**，不能描述为 fully real，也不能宣称所有 customer/depot/charger 已逐点人工核验。

## 1. 一句话方法概述

CLE 决定 11 个目标城市的真实基础设施和有向道路地理；Amazon Last Mile 2021 数据决定配送日的运营结构与订单属性。Stage 2 先冻结 7,500 个 matrix families 及 173,000 个 nested views，再为每个 family 做 depot × Amazon structure 的联合可行性规划（C3），随后生成真实有向路网矩阵、选择真实 AFDC charging stations、匹配不重复的 Amazon order templates，逐 family 验证后原子发布，最后聚合 Phase-1 指标。

核心证据边界是：

```text
Amazon controls operational templates.
CLE controls target-city geography.
```

Amazon 不向目标城市搬运匿名坐标，也不规定纽约、Los Angeles、Phoenix 等城市的局部道路形态。

## 2. 代码入口与权威配置

### 2.1 入口

- 总入口：`generate_instances.sh`
- 主 runner：`EVRPTW_Dataset_Generator/scripts/build_stage2_instances.py`
- C3 并行 supervisor：`EVRPTW_Dataset_Generator/scripts/apply_stage2_joint_support_gate_parallel.py`
- C3 单 task worker：`EVRPTW_Dataset_Generator/scripts/apply_stage2_joint_support_gate.py`
- 单 family materializer：`EVRPTW_Dataset_Generator/src/evrptw_stage2/materialize.py::materialize_family`
- family 进程 supervisor：`EVRPTW_Dataset_Generator/src/evrptw_stage2/subprocess_parallel.py::run_supervised_materialization`
- family verifier：`EVRPTW_Dataset_Generator/src/evrptw_stage2/artifacts.py::verify_materialized_family`
- Phase-1 聚合：`EVRPTW_Dataset_Generator/src/evrptw_stage2/metrics.py::aggregate_phase1_metrics`

### 2.2 配置

- 数据规模、split、view tree：`EVRPTW_Dataset_Generator/configs/cle_evrptw_stage2_v2.json`
- 车辆、能耗、充电、路况、CLE 使用边界：`EVRPTW_Dataset_Generator/configs/us_reference_instance_profile_v2_release.json`
- Amazon train/evaluation cohort：`EVRPTW_Dataset_Generator/configs/amazon_cohort_split_v1.json`
- Amazon operational acceptance：`EVRPTW_Dataset_Generator/configs/amazon_operational_transfer_acceptance_v2.json`

JSON 配置是 executable source of truth。方法说明可参考：

- `EVRPTW_Dataset_Generator/docs/STAGE2_INSTANCE_MODEL.md`
- `EVRPTW_Dataset_Generator/docs/STAGE2_INSTANCE_MODEL_CN.md`
- `EVRPTW_Dataset_Generator/docs/stage2_repair/D5_CONSTRUCT_VALIDITY_REVISION_V2_ZH.md`

## 3. 输入数据与 claim boundary

### 3.1 CLE 输入（方法未改）

Stage 2 从以下母版读取 11 城 CLE：

```text
EVRPTW_Dataset/CLE_v2/us_11city/cities/<city>/
```

`reader.py::load_portable_cle` 强制检查：

1. CLE schema 为 `evrptw_city_logistics_environment_v1`；
2. connectivity contract 为 `directed_projection_roundtrip_v2`；
3. portable package 已验证；
4. speed manifest 为 `evrptw_directed_speed_profiles_v6`；
5. customer/depot/charger 均有 protected inbound、outbound、roundtrip 字段；
6. directed speed 表含 weekday/weekend reference speeds；
7. 规模下限至少为 2,000 customers、1 depot、50 chargers。

正式 benchmark 使用的 CLE contract 是：

```text
frozen_technical_candidate_v1
```

因此实际读取字段是：

- customer：`cle_default_instance_eligible`
- depot：`depot_candidate_eligible`
- charger：`charger_candidate_eligible`

这项授权不清除 CLE manifest 中的 manual blockers。当前 provenance 中保留的 blocker 类别包括 Microsoft NSI geometry、customer road-access review、depot release、charging coordinate/release 和 delivery communities 等。

### 3.2 Amazon 输入

原始输入来自 Amazon Last Mile Routing Research Challenge 2021 的：

```text
model_build_inputs/route_data.json
model_build_inputs/package_data.json
model_build_inputs/travel_times.json
```

入口脚本在 compact artifact 不存在时才构建：

```text
EVRPTW_Dataset/Calibration_v2/amazon_stage2_v3/
```

Amazon 数据承担两个彼此分开的角色：

1. structure source：station-day、route、stop count、depot-to-stop time decile 和 route 内 travel-time 统计；
2. order source：package count、总体积、planned service time、time window。

Primary family 只允许 `SINGLE_STRUCTURE_DAY + SINGLE_ORDER_DAY`。只有 Cus2000 report-only scalability family 允许同 station、同 day type 的 composite；不允许跨 station composite，也不通过复制 template 补齐数量。

## 4. 正式 corpus 的精确规模

当前 generation plan 的实际 registry 为：

```text
parent families = 7,500
views           = 173,000
matrix dtype    = float32
parent matrices = 4 / family
```

### 4.1 Family cohort

| Cohort | Families | 城市分配 | Customer pool | Parent |
| --- | ---: | --- | --- | --- |
| core/train | 5,000 | 10 个训练城市各 500 | train | Cus1000 + 50 CS |
| core/val | 500 | 10 个训练城市各 50 | train | Cus1000 + 50 CS |
| test1_new_seed | 500 | 10 个训练城市各 50 | train | Cus1000 + 50 CS |
| test2_heldout_locations | 500 | 10 个训练城市各 50 | heldout | Cus1000 + 50 CS |
| test3_heldout_city | 500 | Jacksonville 500 | all_release_eligible | Cus1000 + 50 CS |
| unseen_scale_same_cities | 500 | 10 个训练城市各 50 | train | Cus2000 + 50 CS |

因此每个训练城市有 700 个 parent families，Jacksonville 有 500 个，总计 7,500。

### 4.2 View 数量

| Scale | View 数 | Customer | CS |
| --- | ---: | ---: | ---: |
| Cus50 | 101,000 | 50 | 10 |
| Cus100 | 52,000 | 100 | 20 |
| Cus500 | 12,000 | 500 | 50 |
| Cus1000 | 7,500 | 1,000 | 50 |
| Cus2000 | 500 | 2,000 | 50 |

合计 173,000 views。

### 4.3 Nested view tree

对 train Cus1000 parent：

```text
1 × Cus1000
├── 2 × Cus500
├── 10 × Cus100
└── 20 × Cus50
```

代码先构造 region-first 的 20 个互斥 Cus50 leaves。Cus100、Cus500 是固定 leaf group，20 个 Cus50 的 union 必须精确等于 parent，siblings 必须互斥。

evaluation family 不保存所有分支，而是由 `evaluation_chain_leaf` 确定一条严格嵌套链：

```text
Cus50 leaf → 对应 Cus100 → 对应 Cus500 → Cus1000
```

Test2/Test3 不含 Cus50 compatibility view。Cus2000 family 保存一个 Cus2000 view 和其前 1,000 customers 的 deterministic Cus1000 control view。

Child view 不复制 parent matrix；它只保存 `terminal_parent_indices.npy`，加载时从 parent matrix 切片。较小 scale 所需 CS 不是 parent charger list 的简单前缀，而是从 parent 的 50 个 chargers 中按该 child customers 的 road-time replacement delta 重新选择。

### 4.4 Weekday/weekend

`planning.py::build_generation_plan` 在每个 `city × cohort` 内按 5:2 权重做 largest-remainder integer allocation，然后 seeded shuffle 到 family slots。该 slot 的 day type 在 retry 时不改变。

注意：当 cohort count 不能被 7 整除时，代码保证的是 largest-remainder 后的最接近整数分配，不是字面上的整数 5:2。例如每城 50 个 family 会分成 36 weekday + 14 weekend；500 个会分成 357 + 143。

## 5. 从零开始的真实执行顺序

正式 `INSTANCE_MODE=official` 的代码路径是：

```text
candidate revision / clean-tree check
→ Amazon compact artifacts（若尚不存在）
→ preflight
→ customer splits
→ generation plan
→ C3 joint depot × structure support gate
→ family materialization
→ staging family verification
→ corpus-level family verification
→ Phase-1 aggregation
→ terminal run report
```

`generate_instances.sh` 将它拆成三次调用：

```text
build_stage2_instances.py --stages preflight splits plan
apply_stage2_joint_support_gate_parallel.py
build_stage2_instances.py --stages materialize verify metrics
```

### 5.1 C0：preflight、split、plan

#### Repository/profile preflight

`check_candidate_revision.py` 和 provenance loader 要求：

- branch 为 `stage2-repair-candidate`；
- worktree clean；
- generation plan 绑定当前 code commit；
- official profile 为 `release_calibrated`；
- profile 具有 acceptance v3、EV activity audit 和 advisor signoff；
- `hash_validation_performed=false`。

Official C3 并不重新运行早期 C2 pilot，它读取已经 promotion 到 release profile 的 C2/acceptance evidence。换言之，C2 是正式运行的已冻结前置证据，C3 是本次正式 corpus 的逐-family capacity gate。

#### Customer split

实现：`community.py::build_customer_split`。

```text
community_id = city × Census Block Group GEOID × anchor directed SCC
```

完整 community 被分到 `train` 或 `heldout`，不允许把同一 community 内的 individual customers 随机拆开。目标 heldout fraction 是 0.20，assignment 同时平衡 location type、residential-unit band 和 customer mass，并记录 split restart。

`community_adjacency.parquet` 来自真实 directed OSM inter-community edges。无 customer 的 transit-only road communities 仍被保留，因为它们可以连接两个住宅 communities。

每城输出：

```text
customer_splits/<city>/
├── customer_split_manifest.parquet
├── community_manifest.parquet
├── community_adjacency.parquet
└── customer_split_report.json
```

这些 frozen split artifacts 不允许静默覆盖；目录只有一部分存在时 runner 会报 stale/incomplete，而不是继续拼接。

#### Plan

实现：`planning.py::build_generation_plan`。

Plan 在任何昂贵选择之前固定：family ID、cohort、city、track、pool、day type、parent size、CS count、family seed 及 depot/customer/charger/road-state/view seeds。

Plan 输出分 cohort 的 `family_index.parquet`、`view_index.parquet` 和 `split_registry.json`。代码断言 family/view IDs 唯一、view 必须指向 plan 内 family、一个 matrix family 只能属于一个 cohort。

代码内部使用 BLAKE2 生成 deterministic seed/ID/fingerprint；这只是确定性命名与绑定，不是对 dataset artifact 做 SHA256/内容完整性校验。按当前纪律，evidence inventory 与跨版本比对使用 path/size/mtime 或直接解析后的表/数组比较，不执行 SHA256/file hash validation。

### 5.2 C1：directed connectivity contract

C1 的基础 contract 已在 CLE 中冻结为 `directed_projection_roundtrip_v2`。Stage 2 仍会在每个 depot 下执行 runtime connectivity audit：

1. directed node graph 上 depot→terminal；
2. directed node graph 上 terminal→depot；
3. canonical zero-turn line graph 上 depot→terminal；
4. canonical zero-turn line graph 上 terminal→depot。

Customer/charger 必须四项都可达。坏点被写入 deterministic quarantine ledger 并过滤，不再因为 candidate roster 中存在一个固定孤立点而让整个 family 无限换 seed 重试。

若 quarantine 后固定 roster 少于所需 N/K，错误为 non-retryable connectivity failure；改变 seed 不能绕过。

### 5.3 C3：joint depot × Amazon structure support

实现：

- `selection.py::assess_joint_spatial_support_pair`
- `apply_stage2_joint_support_gate.py`
- `apply_stage2_joint_support_gate_parallel.py`

C3 对每个 family 做以下操作：

1. 按 `depot_seed` 得到 deterministic depot candidate order；
2. 按 track、day type、parent scale 和 `customer_superset_seed` 得到 deterministic Amazon structure source order；
3. 按 depot rank × source rank 顺序尝试 pair；
4. 对 pair 建立 directed road state 和 depot-star；
5. 过滤 split pool、双向 node/turn connectivity、Amazon source P99 time envelope、direct roundtrip battery sufficient condition；
6. 检查 route × radial-decile aggregate capacity；
7. 实际执行 controlled rounding、region seed、community growth 和 global assignment；
8. 第一个 aggregate gate 与 exact activation 都成功的 pair 被冻结。

C3 不是“只看总容量”的粗 precheck。它实际生成 exact selected customers，因此可以提前消除过去那种 materialization 重复做昂贵空间选择、反复 rejection 的浪费。

#### Territory

对已选 depot/source，customer 必须：

- 属于 family 声明的 train/heldout/all-release-eligible pool；
- 通过 directed node + canonical turn 双向连通；
- `depot_running_time_s <= source_t_env_s`，其中 `T_env` 是该 Amazon structure source 的 P99；
- `(outbound_distance + return_distance) × (100/257) <= 100 kWh`。

最后一项是单 customer direct-roundtrip 的充分条件，不代表 multi-customer route 永远不需要充电。

#### Spatial activation

Amazon source route × depot-time-decile count 先在仅允许 downscale 的前提下做 deterministic controlled matrix rounding。row margins、column margins 和总数 N 必须同时精确。

每个保留的 source route 对应一个目标 delivery region。region seeds 采用 quota-aware network-time max-min；随后沿真实 `community_adjacency` 做 round-robin growth，最终通过 global min-cost flow 一次性分配 customer IDs。一个 latent customer 在 parent 中最多出现一次。

如果 candidate community competition 导致 flow 不可行，代码扩展受影响 region 并重试；最多 `region_redraw_cap=3`。accepted sample 不会被事后手改。

同时保存一个 size-matched radial baseline；它只匹配 depot-time target，不要求 contiguous community growth，且不作为 benchmark instance，只用于 M1/M5 等对照诊断。

#### C3 selection capsule

优化后的 C3 把 exact activation 保存为 versioned handoff：

```text
reports/stage2_repair/c3_selection_capsules/<task-id>.metadata.json
reports/stage2_repair/c3_selection_capsules/<task-id>.selected_customers.parquet
reports/stage2_repair/c3_selection_capsules/<task-id>.radial_baseline.parquet
```

Capsule 对每个 family 绑定：

- family ID、city、day type、parent customer count；
- customer/road-state seeds；
- selected depot；
- selected structure source IDs；
- joint-support contract；
- capacity-contract fingerprint。

读取时上述字段逐项比较；路径必须是 output-root 内的相对路径；selected customer 与 radial baseline 行数必须等于 parent N。缺失、越界、schema 错误、binding 不一致均为 non-retryable `SelectionCapsuleError`。

Selected customer capsule 固定保存 17 个 selection 字段，包括 latent location ID、经纬度、physical edge/projection/connector、SCC/community、route/decile、住宅属性和 depot running time。

#### C3 并行策略

当前正式参数：

```text
C3_WORKERS=30
C3_FAMILIES_PER_TASK=25
```

每城内部按 family ID 排序后每 25 个组成一个 task；tasks 在城市间 round-robin 排队，使首轮 30 个 workers 覆盖全部城市，减少单城市长尾。当前 7,500-family plan 共 300 tasks。

C3 timeout 的实际语义是：某 task 连续 7,200 秒没有完成新的 family 才 timeout；不是整个 25-family task 总 wall time只能有 7,200 秒。失败时 supervisor 对全部 running process groups 先 SIGTERM，等待 60 秒后仍存活则 SIGKILL，并回收进程。

进度文件：

```text
<instance-root>/stage2_c3_progress.json
```

它包含 planned/completed、每城完成数、active task IDs、pending/completed task count。

### 5.4 Family materialization

实现：`materialize.py::materialize_family`。

Materialization 必须看到完整 C3 registry：

```text
status = passed_full_plan
covered_family_count = 7500
```

每个 family 的步骤如下。

#### 1. Road state

从 CLE directed speed table 选择与 family day type 对应的 weekday 或 weekend reference-speed column。canonical profile 不再叠加随机 edge multiplier。

`d96dd03` 的 topology cache 只跨 family 复用静态拓扑；每个 family 通过 `with_road_state` 重建本 family 的 weighted state。family 内 adjacency/depot-star cache 不跨 road state 使用。

#### 2. Replay C3 selection

Materializer 重新加载同一 depot 和 structure source，并严格验证 C3 capsule binding。验证通过后直接读取 exact selected customers 和 radial baseline，不再重新执行 territory Dijkstra、community growth 和 global assignment。

这是本轮最主要的性能优化：C3 的昂贵 customer selection 成果不再在 materialization 中重复计算。

#### 3. Charger preflight 与选择

Customer geography 固定后才选择 charger。输入是 CLE 中所有 compatible、candidate-eligible 的真实 AFDC sites；不插 synthetic CS，也不把大 roster 有损裁成一个小 roster。

流程为：

1. 全 charger roster 对 depot 做 node/turn 双向 connectivity quarantine；
2. 对 `depot + selected customers + eligible chargers` 计算 directed roster closure；
3. 在 depot/CS infrastructure graph 上按 battery-feasible arcs 找出既能从 depot 到达、也能回 depot 的 communicating set；
4. 对每个 customer/charger 计算 directed replacement-time delta；
5. deterministic greedy 选择固定 K 个 chargers。

Greedy objective 是：

```text
mean(nearest delta)
+ 0.25 × P90(nearest delta)
+ 0.10 × max(nearest delta)
```

Parent Cus1000/Cus2000 固定 50 CS；child Cus50/Cus100 分别从 parent 50 CS 中重新选择 10/20；Cus500/Cus1000 使用 50。

#### 4. Charging power

代码先存储：

```text
charging_power_kw = min(AFDC reported power 或 national mode median,
                        vehicle mode cap)
```

缺失 station power 时使用冻结的全国 mode median：

- AC Level 2：6.5 kW；
- DC fast：200 kW。

Vehicle cap 为 AC 11 kW、DC 100 kW。没有相应 national median 时直接失败，不允许退回 vehicle cap。

需要特别精确地描述实现：`terminal_index.parquet` 和 `charging_attributes.npz` 中保存的是上述 **cap 后、derating 前** 的 power；实际 certificate/charging-time 计算再乘 `charging_power_derating_factor=0.90`。因此电池侧计算功率是：

```text
p_battery = 0.90 × stored_charging_power_kw
```

0.90 是 benchmark derating factor，不表述为充电效率。

#### 5. Terminal index

Parent terminal 顺序固定为：

```text
index 0                  : depot
index 1 ... N            : customers
index N+1 ... N+K        : charging stations
```

`terminal_index.parquet` 保存 terminal source ID、坐标、directed access projection、customer community/region/decile/住宅属性、charger mode/power/provenance；order matching 完成后再写入 customer 的 order template ID、station-day ID 和 source mode。

#### 6. Four parent matrices

`routing.py::PhysicalRoadNetwork.route_terminals` 计算并保存四张 `float32` matrices：

| 文件字段 | 含义 |
| --- | --- |
| `distance_matrix_km` | directed shortest physical-distance path 的距离 |
| `distance_path_travel_time_s` | 上述 shortest-distance path 在本 family road state 下的时间 |
| `running_time_shortest_matrix_s` | canonical zero-turn directed fastest travel time |
| `running_time_path_distance_km` | fastest-time path 对应的 physical distance |

Canonical turn penalties 为 right/left/U-turn 全 0；virtual access-connector split node 禁止 immediate directed-edge reversal。3/8/20 秒 geometry-turn adapter 只用于 test，不生成 canonical release matrices。

Selected terminal exact all-pair closure 中只要存在一个 `inf` 就抛出 non-retryable connectivity error。矩阵允许非对称；非对称来自单行道、不同方向 legal/reference speed 和不同路径。

能耗矩阵不重复存储：

```text
specific energy = 100 / 257 = 0.389105058... kWh/km
energy          = path distance × specific energy
```

#### 7. Amazon order-template matching

矩阵完成后，`orders.py::match_amazon_order_templates` 才把 Amazon order templates 附着到 target-city customers。

先从 directed matrix 构造每个 customer 的 single-customer feasibility certificate，允许 depot/CS full-battery states 和多次 CS hops。然后建立 customer-template bipartite graph；一条 edge 只有在以下条件全部成立时存在：

- demand ≤ 18,500,000 cm3；
- 从 08:00 出发的 earliest service start 不晚于 TW end；
- service 完成并按 certificate 返回 depot 不晚于 24:00；
- customer energy path certificate finite。

SciPy maximum bipartite matching 必须覆盖全部 customers。一个 template 在 family 内不能复用；TW 不移动、不扩宽；若一个 source 发生 Hall failure，则记录该 source attempt 并尝试下一个 admissible source。

Primary family 不允许 composite。Cus2000 可以使用同 station、同 day type composite，但其 release role 是 report-only。

#### 8. View attributes 与 certificate

每个 view 保存：

- `terminal_parent_indices.npy`；
- `customer_attributes.npz`：package count、demand、service、TW、arrival/return、charging flags/visit count、energy margin 等；
- `charging_attributes.npz`：charging power 和 full-CS-to-depot time cache；
- `view_manifest.json`。

每个 child view 继承 parent 已匹配的 templates，但使用 child 自己的 customer/charger matrix slice 重新计算 feasibility certificate。若 inherited template 在 child 中违反 TW、horizon、energy 或 capacity，整个 family attempt 失败。

Full-CS-to-depot cache 允许经其它 CS 多跳，包含 travel 和 intermediate full-charge time。它是静态 instance data，不是 runtime action mask；代码明确要求 `runtime_mask_stored=false`。

#### 9. Phase-1 family metrics

每个 family 同时写：

```text
phase1_metrics.json
phase1_observations.parquet
phase1_region_pair_metrics.parquet
```

Hard construction gates 包括 exact N、customer ID 唯一、split pool 存在、route/decile row/column margins、global uniqueness、view union/disjoint/child sizes。

M1–M5 当前角色：

- M1 depot radial fidelity：Amazon operational transfer；
- M4 route/region structure：by-construction hard audit；
- M2 nearest-neighbour road time：report-only；
- M3 within-region pairwise road time：report-only；
- M5 community concentration：report-only。

历史 `station_block_q90_m2_m3_v1` 的 0/24 FAIL 必须永久保留，但不再决定 release；不能删结果，也不能把阈值从 1 调到 6.89。

#### 10. Staging verification 与 atomic publish

每个 attempt 在独立 process group 和独立 staging root 下运行：

```text
.inflight/<family-id>/<attempt-id>/materialized/families/<family-id>/
```

Family 文件全部写完后，worker 先对 staging family 运行 verifier。只有 verifier PASS 才通过 `os.replace` 原子移动到：

```text
materialized/families/<family-id>/
```

已发布 family 不允许覆盖。Resume 时若 final family 已存在，代码先重新 verify；通过则记为 `reused_verified`，失败则停止。

### 5.5 Materialization 并行、retry 与 STOP

当前正式参数：

```text
WORKERS=30
FAMILIES_PER_WORKER_TASK=1
MAX_ATTEMPTS_PER_FAMILY=4
FAMILY_WALL_TIMEOUT_S=7200
TERMINATION_GRACE_S=60
RUNNER_EXIT_SLACK_S=30
STOP_POLICY=abort_all_inflight_after_grace
```

每一个 family attempt 都由 `family_process_worker` 在独立 session/process group 中运行，因此可以单独终止，不会留下无法控制的 child process。

如果一个 attempt 是允许 retry 的随机 rejection，runner 使用 deterministic attempt seed 进入下一次，最多 4 次。C3 已冻结的 depot、structure source、customer-selection seed、road-state seed 和 capacity contract 不会被 retry 偷偷更换；downstream order/charger randomness 可以随 attempt namespace 改变。

Programming/runtime fault、OSError、MemoryError、SelectionCapsuleError 和显式 non-retryable connectivity/contract failure不会被当作普通 seed rejection。Timeout/hard stop 后不能靠同一 run 自动重新提交 timeout family。

Materialization 进度：

```text
<instance-root>/stage2_progress.json
<instance-root>/stage2_progress_events.jsonl
```

Snapshot 原子替换，events 为 append-only + fsync。字段包括 planned/completed/materialized/verified/rejected/timed_out/aborted/unresolved、active IDs、not-started 和 last-completed family。

### 5.6 Corpus verifier

Staging PASS 后，runner 仍会对全部 7,500 个 published families 再运行一次 corpus-level verifier；`WORKERS=30` 时使用 multiprocessing `spawn`，in-flight verification queue 上限为 `workers × 4`。

当前 verifier 实际检查：

- terminal row count 与 source ID uniqueness；
- exactly four matrix names；
- 每张 matrix shape、`float32`、finite、nonnegative、zero diagonal；
- view count、view/family ID、customer/CS shape；
- parent-index matrix slice shape；
- 两张 derived energy matrices 与 distance × scalar 一致；
- package/demand/TW array shape；
- stored certificate 与重新计算 certificate 一致；
- TW、24:00 horizon、vehicle volume、positive charger power；
- full-CS return cache 与重算一致；
- runtime mask 未存储。

任一 family FAIL 会令 run report 失败，不进入正常 PASS 发布语义。

### 5.7 Phase-1 corpus aggregation

`aggregate_phase1_metrics` 读取全部 family metrics 和 rejection ledgers，生成：

```text
reports/phase1/
├── family_metrics.parquet
├── stratified_metrics.csv
├── corpus_metrics.csv
├── rejected_attempts.parquet             # 有 rejection 时
├── amazon_source_family_ledger.parquet
├── amazon_template_usage.parquet
├── source_usage_summary.json
├── matching_bias_audit.parquet
├── fragmentation_audit.parquet
├── charger_selection_audit.parquet
└── summary.json
```

Aggregation 报告 first-attempt success、conditional attempt success、rejection reason、region redraw/fallback、M1–M5、Amazon source/template usage、within-family template reuse、matching bias、charger selection 和 child fragmentation。

`generate_instances.sh` 自动执行的是 Phase-1 aggregation；`evaluate_amazon_operational_transfer_v2.py`、cross-city diagnostic、construct-valid acceptance 和最终 release/packaging 属于独立的后续 acceptance 流程，不能把 `reports/phase1/summary.json` 单独等同于最终 dataset release 签字。

### 5.8 Run report 与 BrokenPipe 语义

最终顺序是：

```text
verification/metrics completed
→ atomic persist run_manifest.json
→ atomic persist terminal stage2_run_report.json
→ terminal_report_committed=true
→ best-effort concise stdout summary
```

完整大 JSON 默认只写文件；stdout 只打印摘要和 report path。PASS report 已提交后发生 `BrokenPipeError` 只能进入 observability warning ledger，不能把 generation outcome 改回 failed。

## 6. Family 输出结构

```text
materialized/families/<family-id>/
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
└── views/<view-id>/
    ├── view_manifest.json
    ├── terminal_parent_indices.npy
    ├── customer_attributes.npz
    └── charging_attributes.npz
```

Cus1000 parent terminal count 为 1,051；Cus2000 parent 为 2,051。Plan registry 估计四张 parent matrices 合计 157,368,120,000 bytes（约 157.37 GB 十进制，约 146.56 GiB），尚未包括 Parquet、NPZ、JSON 和 filesystem overhead。后续 archive/restore 设计必须意识到 parent matrices 才是主要体积来源。

## 7. 当前正式运行命令

当前 run 对应以下参数：

```bash
INSTANCE_MODE=official \
WORKERS=30 \
C3_WORKERS=30 \
C3_FAMILIES_PER_TASK=25 \
FAMILIES_PER_WORKER_TASK=1 \
MAX_ATTEMPTS_PER_FAMILY=4 \
FAMILY_WALL_TIMEOUT_S=7200 \
TERMINATION_GRACE_S=60 \
INSTANCE_OUTPUT_ROOT=/data/.../EVRPTW_Dataset/Instances_v2/<fresh-root> \
./generate_instances.sh --full-run-approved
```

入口脚本还把 OMP/OpenBLAS/MKL/NumExpr thread 数默认固定为 1，避免 30 个 processes 每个再内部过度并行。

## 8. 已完成的 150-family full-path toy 证据

在启动 7,500 正式数据前，代码用 75 × 2 frozen toy templates 覆盖 11 城、六个 tracks、weekday/weekend 和 Cus1000/Cus2000 两类 parent。优化版结果：

```text
planned      = 150
materialized = 150
verified     = 150
unresolved   = 0
hard stop    = false
Phase-1      = completed
```

优化前 root：

```text
EVRPTW_Dataset/Instances_v2/us_11city_full_path_toy_75x2_629bbea
```

优化后 root：

```text
EVRPTW_Dataset/Instances_v2/us_11city_full_path_toy_75x2_d96dd03
```

30 workers 下 materialize + staging verify + corpus verify + metrics 的 wall time：

| 指标 | 优化前 | 优化后 | 变化 |
| --- | ---: | ---: | ---: |
| runner wall | 2,283.86 s | 1,471.00 s | -35.59% |
| summed family materialization work | 50,225.58 s | 35,658.55 s | -29.00% |
| summed terminal selection | 34,387.30 s | 20,328.30 s | -40.88% |
| max worker peak RSS | 7.04 GB | 5.38 GB | -23.62% |

优化后 family wall distribution 约为 P50 160.66 s、P90 487.39 s、max 871.89 s。两版 150 families 的 customer terminal tables、四张 parent matrices、1,970 views、view NPY/NPZ 和 Phase-1 tables 做过直接解析后的逐项比较，优化后与正确 baseline 等价；未使用 SHA256/file hash。

C3 本身仍需完成 exact activation，因此 selection capsule 主要消除了 materialization 的重复选择，而不是把 C3 变成近零成本。150 toy 的端到端总 wall 从约 58.5 分钟降到约 45 分钟，约快 23%。

## 9. Full 7,500 的时间估计

基于 150-family full-path toy 和当前 30-worker 正式 C3 的实际吞吐，当前工程估计为：

| 阶段 | 估计 wall time |
| --- | ---: |
| preflight + split + plan | 分钟级；当前已经完成 |
| C3 joint support | 约 7–9 小时 |
| materialization + staging verification | 约 18–23 小时 |
| corpus verification + Phase-1 aggregate | 约 2–3 小时 |
| 从零总计 | 约 27–34 小时 |

这是 wall-time 工程估计，不是固定 SLA。主要不确定性来自大城市 charger-roster all-pair Dijkstra、Cus2000 families、family runtime long tail、filesystem throughput 和 30 个 worker 的实际 RSS/CPU contention。

不能用 `150 toy wall × 50` 做简单线性结论而忽略 task mix 和并行尾部；也不能把 C3 完成误认为 matrices 已生成。C3 只冻结 selection capsules；真正四张 parent matrices 在 materialization 才生成。

## 10. 实时状态与 post-generation watcher

C3 进度位于 `<instance-root>/stage2_c3_progress.json`；materialization、corpus verification 与 metrics 的统一进度位于 `<instance-root>/stage2_progress.json`。

`EVRPTW_Dataset_Generator/scripts/watch_full_corpus_feasibility.py` 在 Stage-2 terminal report 提交前只读等待；结束后核对 7,500 个 planned/materialized/verified/published ID 集合、全部 family feasibility summary、Phase-1 hard gates、timeout/rejection/unresolved 和残余 process groups。PASS 时写出：

```text
<instance-root>/reports/post_generation/full_corpus_feasibility_gate_v1.json
<instance-root>/reports/post_generation/READY_FOR_CLEANUP_AND_CLEAN_RERUN.json
```

总控入口 `run_clean_full_pipeline.sh` 只接受不存在的 output/archive/restore 路径，依次执行 generator tests、fresh Stage-2 C0/C3/materialization/verification/metrics、watcher、construct-valid v3、slim archive、完整 restore rehearsal，最后才允许 push。该链不执行 SHA256 或 file-content hash validation。

## 11. Release 前必须显式区分的通过条件

### 11.1 Generation correctness

正式 generation 至少应满足：

```text
planned       = 7500
materialized  = 7500
verified      = 7500
timed_out     = 0
unresolved    = 0
aborted       = 0
remaining process groups = 0
all Phase-1 hard gates = PASS
```

### 11.2 Amazon operational transfer

Amazon hard gate 应只检查实际迁移的变量：M1 radial、M4 route/region size、package、volume、service、TW、day type、source provenance、template equality 和 matching bias。

### 11.3 Cross-city spatial diagnostics

M2/M3/M5 必须完整报告，但不能改变 operational acceptance 的 `passed`。它们描述的是 target-city road morphology、spatial concentration 及 proposed/radial baseline 差异。

### 11.4 Packaging

即使 7,500/7,500/7,500 与 Phase-1 PASS，也仍需完成 full-corpus acceptance review、archive inventory、restore rehearsal 和从零全流程复跑证据后，才能把代码仓库和 dataset archive 作为正式 release artifact。生成结束不自动等于 packaging/release 完成。

## 12. 代码审阅中发现、release 前应处理的实现注意项

本节不是方法修改，而是按当前代码逐行核对后需要 reviewer 知道的事实。

### 12.1 Verifier contract 已统一并有 regression gate

`materialize.py` 与 `artifacts.py` 现在共享 `evrptw_stage2.contracts.STAGE2_GENERATION_CONTRACT`，当前值为 `stage2_construct_valid_v3`。因此 current-contract parent-level order provenance、三份 Phase-1 文件、schema/family ID/hard-gate 和 observation row-count 检查都会执行；测试同时防止 writer/verifier 常量再次漂移。

### 12.2 Stored charging power 的命名容易引起误解

字段名是 `effective_charging_power_kw`，但代码实际存的是 station/median 经 vehicle cap 后的 power，0.90 derating 在 certificate/charging time 计算时才乘。论文和 data card 应明确“stored pre-derating power + profile derating”，避免读者再次乘或漏乘。

### 12.3 Metrics stage 与最终 acceptance 是两件事

主入口的 `metrics` 只运行 Phase-1 aggregate；Amazon operational transfer v2 和最终 acceptance 是独立脚本。最终 run 报告里有 Phase-1 summary 不代表所有 release gates 已自动执行。

### 12.4 不执行 SHA256/hash evidence validation

当前 profile、C3 reports、capsules 和 CLE reference 都明确记录 `hash_validation_performed=false` 或 `content_digest_validation_performed=false`。Deterministic internal IDs/fingerprints 仍用于 seed 与 binding consistency，但不能对外表述为 artifact hash validation。

## 13. 给 Nanpeng 的简短版本

CLE 部分与此前介绍一致，没有修改。我们在 11 城 CLE 的真实有向 OSM 路网、住宅候选点、depot 和 AFDC charging stations 上生成 EVRPTW instances；Amazon 数据只用于校准真实配送日的 route/stop 结构、package volume、service time 和 time windows，不把 Amazon 匿名坐标搬到这些城市。

正式数据先冻结 7,500 个 parent families 和 173,000 个 nested scale views。每个 family 先做 depot × Amazon day structure 的联合可行性检查，在有向路网上过滤不可达点，并按 Amazon route × radial-time decile 精确选择 customer；再从完整真实 charger roster 中选取与 customer geography 相关且能量连通的 CS，计算四张 directed parent matrices，最后通过 maximum bipartite matching 给 customers 分配互不重复且时间窗/容量/能量可行的 Amazon order templates。每个 family 先在 staging 中验证，PASS 后原子发布；全体完成后再做一次 7,500-family verification 和 Phase-1 aggregate。

当前用 30 个 C3 workers 和 30 个 materialization workers。150-family 全路径 toy 已经 150/150 生成并验证通过；优化后 toy 端到端约 45 分钟，比优化前约 58.5 分钟快约 23%。完整 7,500-family 从零工程估计约 27–34 小时，其中 C3 约 7–9 小时，matrix materialization、两层 verification 和 metrics 约 20–26 小时。
