# EVRPTW-DB：从 CLE 到 11 城 Instance 的完整生成说明

> 文档类型：代码实现、方法审核与复现运行说明。
>
> 更新日期：2026-08-27。
>
> 最终数据 root：EVRPTW_Dataset/Instances_v2/us_11city_full_clean_v7_bbde5db_20260823。
>
> 数据生成 commit：bbde5db48dc3f939906fbafdfb18b5b973ae04f1。
>
> 当前仓库 HEAD：6ac73b5e03179d2c095a83acf12997ea1f58e55a。数据生成后的改动主要是 matrix restore 兼容性修复和 benchmark solver/launcher；7,500 个 family 的 generation provenance 仍绑定 bbde5db。

## 1. 方法结论与 claim boundary

EVRPTW-DB 的 instance 不是在欧氏平面上随机撒点，也不是把 Amazon 匿名城市坐标搬到目标城市。代码中的数据分工是：

```text
CLE 决定目标城市的真实地理、道路与公共充电基础设施。
Amazon Last Mile 2021 校准配送日结构和订单运营属性。
Stage 2 在可行性约束下把两者组合成 EVRPTW instances。
```

最终 corpus 已完成：

| 项目 | 结果 |
| --- | ---: |
| 城市 | 11 |
| Parent matrix families | 7,500 |
| Nested views / instances | 173,000 |
| Materialized families | 7,500 |
| Verified families | 7,500 |
| Timeout / rejection / unresolved / aborted | 0 / 0 / 0 / 0 |
| Phase-1 hard gates | PASS |
| Full-corpus feasibility watcher | PASS |
| Construct-valid acceptance v3 | PASS |

正确的数据描述是 infrastructure-grounded semi-synthetic EVRPTW benchmark。不能描述为 fully real，也不能声称每个 customer、depot 和 charger 均经过逐点人工 release 审核。

## 2. CLE 母版提供什么

Stage 2 使用：

```text
EVRPTW_Dataset/CLE_v2/us_11city/cities/<city>/
```

11 个城市为 Chicago、Dallas、Fort Worth、Houston、Jacksonville、Los Angeles、New York、Philadelphia、Phoenix、San Antonio 和 San Diego。

CLE 的构建方法与此前介绍一致，本轮没有修改。每个 portable CLE package 提供：

- 有向 OSM 道路拓扑；
- physical edge、terminal projection 和 virtual connector；
- weekday/weekend reference speed；
- Census/NSI residential customer candidates；
- depot candidates；
- AFDC public charging-station candidates；
- road SCC、community 和 protected connectivity 字段；
- source registry、manifest 和 QA evidence。

reader.py::load_portable_cle 强制检查 CLE schema、portable package、directed_projection_roundtrip_v2、speed-profile schema、terminal 双向 connectivity 字段和最低 roster 容量。

正式 benchmark 使用：

```text
official CLE contract = frozen_technical_candidate_v1
customer field         = cle_default_instance_eligible
depot field            = depot_candidate_eligible
charger field          = charger_candidate_eligible
```

这表示允许使用通过技术验证和 directed-connectivity quarantine 的 candidate pools，但所有 CLE manual-release blockers 继续保留。CLE 控制 customer/depot/charger 坐标、道路距离、方向不对称、weekday/weekend running time、community 邻接和 charger infrastructure。

## 3. Amazon 数据提供什么

原始输入为：

```text
model_build_inputs/route_data.json
model_build_inputs/package_data.json
model_build_inputs/travel_times.json
```

代码将其整理到：

```text
EVRPTW_Dataset/Calibration_v2/amazon_stage2_v3/
```

最终 artifact 包含 521 个 station-day structures 和 898,391 个 order templates。Amazon 提供：

- 每日 route 数和每条 route 的 stop 数；
- depot-to-stop time decile 和径向层级；
- package count 与 volume；
- planned service time；
- time-window presence、位置与宽度；
- weekday/weekend 运营差异；
- route 内订单属性的联合结构。

Amazon 不提供目标城市坐标，也不控制目标城市的 block size、nearest-neighbour road time 或 community morphology。

Structure source 和 order source 是两个角色。Primary Cus100/Cus500/Cus1000 family 必须使用 SINGLE_STRUCTURE_DAY + SINGLE_ORDER_DAY。Cus2000 是 report-only unseen-scale cohort，只允许同 station、同 day type composite；不能跨 station composite，也不能复制 template 凑数量。

## 4. 权威代码与配置

### 4.1 Shell 入口

| 文件 | 作用 |
| --- | --- |
| auto.sh | Stage 2 或 archive 总入口 |
| generate_instances.sh | 从现有 CLE 生成 Stage 2 |
| run_clean_full_pipeline.sh | 测试、fresh generation、验收、archive、restore rehearsal |
| create_dataset_archive.sh | 创建省略 dense matrices 的 portable archive |
| restore_dataset_archive.sh | 解压并确定性重建 matrices |

### 4.2 Python 实现

| 文件或函数 | 作用 |
| --- | --- |
| scripts/build_stage2_instances.py | C0、materialization、verify、metrics 主 runner |
| planning.py::build_generation_plan | 冻结 family/view registry |
| community.py::build_customer_split | complete-community split |
| apply_stage2_joint_support_gate_parallel.py | C3 supervisor |
| selection.py::assess_joint_spatial_support_pair | C3 depot × source gate |
| selection.py::select_family_terminals_v2 | replay customer capsule 并选择 chargers |
| materialize.py::materialize_family | 单 family 写盘 |
| routing.py::PhysicalRoadNetwork | 有向 routing 和四矩阵 |
| orders.py::match_amazon_order_templates | order-template matching |
| artifacts.py::verify_materialized_family | family/view verifier |
| metrics.py::aggregate_phase1_metrics | corpus Phase-1 聚合 |
| reconstruction.py::restore_dataset_matrices | portable matrix restore |

### 4.3 配置

- configs/cle_evrptw_stage2_v2.json：城市、规模、split、family/view 数和 acceptance；
- configs/us_reference_instance_profile_v2_release.json：车辆、能耗、充电、road state、CLE contract；
- configs/amazon_cohort_split_v1.json：Amazon generation/evaluation/metric-holdout split；
- configs/stage2_acceptance_v3_full_7500.json：最终 construct-valid acceptance。

这些 JSON 是 executable source of truth。

## 5. Vehicle、energy、charging 与 horizon

Reference vehicle：

```text
vehicle_id       = rivian_commercial_van_delivery_700_reference_v1
battery          = 100 kWh
nominal range    = 257 km
cargo capacity   = 18,500,000 cm3
initial SOC      = 100%
fleet            = unlimited
```

Constant-distance energy：

```text
specific energy = 100 / 257
                = 0.38910505836575876 kWh/km
```

Parent family 不重复存储能耗矩阵。View 加载时派生：

```text
distance_path_energy_kwh
  = distance_matrix_km × specific energy

running_time_path_energy_kwh
  = running_time_path_distance_km × specific energy
```

Charging profile：

```text
DC vehicle cap   = 100 kW
AC L2 cap        = 11 kW
derating factor  = 0.90
charging model   = full-charge linear
ports            = infinite
```

缺失 station power 时使用冻结的 national mode median：AC L2 6.5 kW，DC fast 200 kW；然后应用 vehicle cap。Artifact 中保存 cap 后、derating 前的 power，certificate 再乘 0.90。Operating horizon 为 08:00–24:00，即 28,800–86,400 秒。

## 6. Corpus 与 nested views

10 个 seen cities 为 New York、Los Angeles、Chicago、Houston、Phoenix、Philadelphia、San Antonio、San Diego、Dallas 和 Fort Worth；held-out city 为 Jacksonville。

| Family cohort | Families | Pool | Parent |
| --- | ---: | --- | --- |
| core/train | 5,000 | seen-city train | Cus1000 + 50 CS |
| core/val | 500 | seen-city train | Cus1000 + 50 CS |
| test1_new_seed | 500 | seen-city train | Cus1000 + 50 CS |
| test2_heldout_locations | 500 | seen-city heldout | Cus1000 + 50 CS |
| test3_heldout_city | 500 | Jacksonville all eligible | Cus1000 + 50 CS |
| unseen_scale_same_cities | 500 | seen-city train | Cus2000 + 50 CS |

每个 seen city 有 700 families，Jacksonville 有 500，总计 7,500。

| Scale | Customers | CS | Views | Role |
| --- | ---: | ---: | ---: | --- |
| Cus50 | 50 | 10 | 101,000 | compatibility |
| Cus100 | 100 | 20 | 52,000 | primary |
| Cus500 | 500 | 50 | 12,000 | primary |
| Cus1000 | 1,000 | 50 | 7,500 | primary |
| Cus2000 | 2,000 | 50 | 500 | report-only scalability |

Train Cus1000 family 的 tree 为 1×Cus1000、2×Cus500、10×Cus100、20×Cus50，共 33 views。20 个 Cus50 leaves 两两互斥，union 精确等于 parent。Evaluation family 保存一条严格嵌套 chain；Test2/Test3 没有 Cus50。Cus2000 family 还保存前 1,000 customers 的 deterministic Cus1000 control view。

Child view 不复制矩阵，只保存 terminal_parent_indices.npy 并从 parent matrices 切片。

Weekday/weekend 在每个 city × cohort 内按 5:2 largest-remainder 分配，再 deterministic shuffle。例如 50 families 分为 36 weekday + 14 weekend；retry 不改变 day type。

## 7. 从 CLE 开始的阶段顺序

```text
frozen CLE + release profile
→ code/branch/clean-tree preflight
→ Amazon compact artifact（若不存在）
→ C0 customer split
→ C0 generation plan
→ C1/C2 frozen evidence binding
→ C3 depot × structure joint-support
→ family materialization
→ staging verification
→ atomic publish
→ 7,500-family corpus verification
→ Phase-1 aggregation
→ terminal Stage-2 report
→ full-corpus feasibility watcher
→ construct-valid acceptance v3
→ slim archive
→ full matrix restore rehearsal
```

generate_instances.sh 将核心生成拆成三步：

```text
build_stage2_instances.py --stages preflight splits plan
apply_stage2_joint_support_gate_parallel.py
build_stage2_instances.py --stages materialize verify metrics
```

## 8. C0：customer split 与 generation plan

Community ID 是 city × Census Block Group GEOID × anchor directed SCC。完整 community 被分配到 train 或 heldout，不允许随机拆分同一 community 内的 individual customers。目标 heldout fraction 为 0.20，并平衡 location type、residential-unit band 和 customer mass。

每城输出：

```text
customer_splits/<city>/
├── customer_split_manifest.parquet
├── community_manifest.parquet
├── community_adjacency.parquet
└── customer_split_report.json
```

Generation plan 在昂贵计算前冻结 family ID、cohort、city、track、pool、day type、parent size、CS count、family/depot/customer/charger/road-state/view seeds 和所有 view rows。

split_registry.json 最终断言：

```text
family_count = 7,500
view_count   = 173,000
four stored matrices per parent family
```

内部 deterministic ID、seed 和 fingerprint 用于可重复命名与 contract binding，不代表执行 SHA256 file validation。

## 9. C1/C2：连通性与冻结放行证据

CLE 已冻结 directed_projection_roundtrip_v2。Stage 2 对 depot 与完整 terminal roster 检查 node graph 和 canonical zero-turn topology 的 depot→terminal、terminal→depot，共四个 reachability masks。四项都为真才是 connectivity eligible。

固定不可达 customer/charger 被 deterministic quarantine，而不是让 family 反复换 seed。Quarantine 后 roster 不足属于 non-retryable failure。

C2 历史审核覆盖 Amazon generation/evaluation/metric-holdout leakage、single-day support、H3/PF capacity、C0 membership/5:2、connectivity acceptance 和 profile promotion。正式 v7 不重复人工视觉审核；它读取 promotion 到 release_calibrated profile 的冻结证据，并重新检查与 fresh C0 的 binding。

## 10. C3：depot × Amazon structure 联合支持

C3 在生成大矩阵前，为每个 family 搜索第一个可行的 depot candidate × Amazon structure source：

1. 按 seed 固定 depot 和 source 排序；
2. 建 family road state 和 depot-star；
3. 过滤 split pool 与 node/turn 双向 connectivity；
4. 应用 Amazon source P99 territory-time envelope；
5. 应用 direct depot-customer-depot battery sufficient condition；
6. 检查 route × radial-decile aggregate capacity；
7. 执行 controlled matrix rounding；
8. 执行 region seeds、community growth 和 global min-cost-flow assignment；
9. 接受 aggregate 与 exact activation 都通过的第一个 pair。

Quota rounding 必须同时精确满足 row margins、column margins 和总 customer 数 N。每个 latent_customer_id 在 parent 内最多出现一次；competition 失败时只扩展受影响 region，最多 region_redraw_cap=3。

C3 保存：

```text
reports/stage2_repair/c3_selection_capsules/<task-id>.metadata.json
reports/stage2_repair/c3_selection_capsules/<task-id>.selected_customers.parquet
reports/stage2_repair/c3_selection_capsules/<task-id>.radial_baseline.parquet
```

Capsule 绑定 family、city、day type、N、depot、structure source、seeds 和 capacity contract。Materializer 逐项复核后 replay exact customers，不重复 territory Dijkstra、community growth 和 global assignment。

最终参数为 C3_WORKERS=30、C3_FAMILIES_PER_TASK=25，共 300 tasks。进度写入 stage2_c3_progress.json；最终为 7,500/7,500 和 300/300。

## 11. Family materialization

Materialization 必须先看到 C3 status=passed_full_plan 和 covered_family_count=7500。

### 11.1 Road state

按 family day type 选择 CLE weekday/weekend directed speed。Canonical release 不叠加随机 residual multiplier；right/left/U-turn penalty 均为 0，virtual connector immediate reversal 在拓扑上禁止。静态 topology 可复用，但 weighted road state 不跨 family 混用。

### 11.2 Charger selection

Customer 固定后：

1. 对全部 compatible、candidate-eligible AFDC roster 做双向 node/turn quarantine；
2. 建 depot + customers + eligible chargers directed closure；
3. 建 battery-feasible depot/CS communicating set；
4. 计算 customer/charger road-time replacement delta；
5. 以 mean + 0.25×P90 + 0.10×max nearest delta 做 deterministic greedy。

不插 synthetic CS，也不对完整 roster 做有损 prefix 截断。Child Cus50/Cus100 根据自己的 customers 从 parent eligible roster 重新选择 10/20 CS，不使用简单前缀。

### 11.3 Parent terminal 与四矩阵

Terminal 顺序固定为 depot、N customers、K chargers。每个 family 保存四张 float32 directed matrices：

| 文件 | 定义 |
| --- | --- |
| distance_matrix_km.npy | shortest physical-distance path 的距离 |
| distance_path_travel_time_s.npy | shortest-distance path 的时间 |
| running_time_shortest_matrix_s.npy | canonical fastest running time |
| running_time_path_distance_km.npy | fastest-time path 对应距离 |

最终 terminal closure 只要出现 inf 就 non-retryable fail。矩阵允许方向不对称。

Cus1000+50CS terminal 数为 1,051，四矩阵 17,674,128 bytes；Cus2000+50CS 为 2,051，四矩阵 67,306,128 bytes。全部 parent matrix plan estimate 为 157,368,120,000 bytes。

### 11.4 Order-template matching

矩阵完成后，代码先计算允许 depot/CS full-battery states 和多 CS hops 的 single-customer certificate，再建立 customer-template bipartite graph。Edge 要求 demand≤18,500,000 cm3、TW 可服务、service 后 24:00 前可返 depot、energy path finite。

Maximum bipartite matching 必须覆盖全部 customers；family 内 template 不复用，TW 不平移、不扩宽。Primary family 禁止 composite。

### 11.5 View artifacts

每个 view 保存 view_manifest.json、terminal_parent_indices.npy、customer_attributes.npz 和 charging_attributes.npz。属性包括 package、volume demand、service、TW、arrival/return、charging visits、energy margin、charging power 和 full-CS-to-depot cache。

Child 继承 parent order templates，但在自己的 customer/charger matrix slice 上重算 certificate。runtime_mask_stored=false。

### 11.6 Atomic publish、retry 与 timeout

每个 attempt 在独立 process group 和 .inflight staging root 写完，staging verifier PASS 后才通过 os.replace 发布。已发布 family 不覆盖；resume 时重新 verify 后记为 reused_verified。

```text
WORKERS                  = 30
FAMILIES_PER_WORKER_TASK = 1
MAX_ATTEMPTS_PER_FAMILY  = 4
FAMILY_WALL_TIMEOUT_S    = 7200
TERMINATION_GRACE_S      = 60
RUNNER_EXIT_SLACK_S      = 30
STOP_POLICY              = abort_all_inflight_after_grace
start method             = spawn
```

Programming fault、OSError、MemoryError、SelectionCapsuleError、fixed connectivity/contract failure 不作为普通 seed rejection。进度写入 stage2_progress.json 和 append-only stage2_progress_events.jsonl。

## 12. Verification、Phase-1 与 acceptance

Family verifier 检查 terminal IDs、四矩阵 names/shape/float32/finite/nonnegative/zero diagonal、view shape/index slice、派生能耗、package/demand/service/TW、order provenance、certificate 重算、capacity/horizon/energy、charger power、full-CS cache、runtime mask 和 Phase-1 files/hard gates。

Verification 有两层：staging PASS 才发布；全部完成后再次验证 7,500 个 published families。Materializer、verifier 和 reconstruction 共享 contracts.py 中同一个 STAGE2_GENERATION_CONTRACT。

Phase-1 输出 family_metrics、stratified/corpus metrics、Amazon source/template usage、matching bias、fragmentation 和 charger audit。最终 7,500 families 全部 first-attempt success，rejection=0，all_hard_gates_passed=true。

M1 radial 和 M4 route/region structure 属于 operational transfer；M2 nearest-neighbour、M3 within-region pairwise 和 M5 community concentration 是 report-only spatial diagnostics。历史 Q90 v1 的 0/24 FAIL 保留，但不参与 construct-valid v3 passed。

最终 construct-valid v3 审计 7,500 families、173,000 views 和 24,750,000 inherited customer observations；family_artifacts_modified=false，hash_validation_performed=false，status=passed。

## 13. Family 输出结构

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

## 14. 正式生成命令与进度

前置条件是 stage2-repair-candidate、clean working tree、release_calibrated profile、已有 CLE/Amazon artifact、fresh output root 和足够资源。

```bash
cd /path/to/EVRPTW-DB
conda activate maojie

SKIP_CLE_BUILD=1 \
INSTANCE_MODE=official \
WORKERS=30 \
C3_WORKERS=30 \
C3_FAMILIES_PER_TASK=25 \
FAMILIES_PER_WORKER_TASK=1 \
MAX_ATTEMPTS_PER_FAMILY=4 \
FAMILY_WALL_TIMEOUT_S=7200 \
TERMINATION_GRACE_S=60 \
INSTANCE_OUTPUT_ROOT="$PWD/EVRPTW_Dataset/Instances_v2/us_11city_new_clean_run" \
./auto.sh stage2 --full-run-approved
```

查看 C3：

```bash
python -m json.tool EVRPTW_Dataset/Instances_v2/us_11city_new_clean_run/stage2_c3_progress.json
```

查看 materialization/verification：

```bash
python -m json.tool EVRPTW_Dataset/Instances_v2/us_11city_new_clean_run/stage2_progress.json
```

最终必须检查 stage2_run_report.json、reports/phase1/summary.json、reports/post_generation/watcher_progress.json 和 reports/stage2_repair/stage2_acceptance_v3_construct_valid.json。

## 15. v7 实测时间和内存

| 阶段 | 实测 wall time |
| --- | ---: |
| C0+C3+materialize+verify+Phase-1 | 88,154 s = 24 h 29 m 14 s |
| 其中 materialize+verify+metrics runner | 58,924.5 s = 16 h 22 m 5 s |
| Construct-valid v3 | 936 s = 15 m 36 s |
| Slim archive create | 561 s = 9 m 21 s |
| Archive inspect | 68 s = 1 m 8 s |
| 30-worker restore rehearsal（7,499 restored + 1 reused） | 19,119.7 s = 5 h 18 m 40 s |

Generation server physical memory 为 270,076,866,560 bytes；最大单 generation worker RSS 为 5,559,250,944 bytes，runner peak RSS 为 14,969,749,504 bytes。

Worker 数不能跨服务器照抄。Raynor 的 restore worker 曾以约 7.8 GiB RSS 被 OOM killer 杀死；30 个 restore processes 峰值可能超过 230 GiB，尚未包含系统和其它任务。Archive status 的 available_free_bytes 是磁盘，不是 RAM。共享服务器应先查看 free -h 和当前大内存进程；不清楚时从 4 workers 开始，内存充分且无其它大任务时再用 8–12。不要同时运行大规模 restore、solver 和 PyTorch job。

## 16. Slim archive 与 matrix restore

四张 dense matrices 可由 frozen CLE、family/view parameters、profile 和 road-state seed 重建，因此 portable archive 省略 matrices，保留 CLE、lightweight artifacts、terminal/order attributes、plan 和 acceptance evidence。

最终 archive：

```text
EVRPTW_Dataset_us11city_full_clean_v7_bbde5db.tar.zst
compressed bytes = 7,331,778,640
logical bytes    = 16,294,525,419
members          = 1,002,201
families/views   = 7,500 / 173,000
```

创建：

```bash
./auto.sh archive create \
  --archive /path/to/EVRPTW_Dataset_us11city_full_clean_v7_bbde5db.tar.zst \
  --cle-root "$PWD/EVRPTW_Dataset/CLE_v2/us_11city" \
  --instance-root "$PWD/EVRPTW_Dataset/Instances_v2/us_11city_full_clean_v7_bbde5db_20260823" \
  --profile "$PWD/EVRPTW_Dataset_Generator/configs/us_reference_instance_profile_v2_release.json" \
  --compression-threads 12
```

另一台服务器 restore：

```bash
./auto.sh archive start \
  --archive "$PWD/EVRPTW_Dataset/EVRPTW_Dataset_us11city_full_clean_v7_bbde5db.tar.zst" \
  --destination /writable/path/evrptw_runtime \
  --workers 4
```

查看：

```bash
./restore_dataset_archive.sh status --destination /writable/path/evrptw_runtime
./restore_dataset_archive.sh logs --destination /writable/path/evrptw_runtime --follow
```

最终必须满足 phase=succeeded、selected_family_count=7500、restored_count+reused_count=7500、matrix_restore_report.passed=true。在此之前禁止运行 Gurobi、ALNS 或 VNSTS；目录存在不代表 matrices 已恢复完成。

Restore 对已有四张完整且结构有效的 matrices 记为 reused_existing_cache；缺失时先写 .matrices-rebuild-*，四张都通过后才原子替换为 matrices/。已存在但不完整的正式 cache 会被拒绝。相同 archive+destination 在正常中断后可以复用完整 families。

当前不做 SHA256：file_hash_validation_performed=false。Archive 做 member/layout/provenance 检查；restore 做 shape、dtype、finite、nonnegative 和 full-family verifier。内部 ID/fingerprint 只用于 deterministic binding。

## 17. Reviewer 验收清单

- [ ] fresh root、clean tree、正确 branch/profile/CLE contract；
- [ ] C0 family/view、membership、leakage、5:2 精确；
- [ ] C3 covered 7,500；
- [ ] planned/materialized/verified = 7,500/7,500/7,500；
- [ ] views = 173,000；
- [ ] timeout/rejection/unresolved/abort = 0；
- [ ] remaining process groups = 0；
- [ ] directed matrices、nested views、order inheritance 和 certificate PASS；
- [ ] Phase-1 hard gates、watcher、construct-valid v3 PASS；
- [ ] M2/M3/M5 仅作为 spatial diagnostics；
- [ ] CLE manual blockers 和 semi-synthetic claim boundary 保留；
- [ ] archive inspect PASS；
- [ ] full restore report 覆盖 7,500 且 passed；
- [ ] benchmark 只在 restore succeeded 后启动。

## 18. 给 Nanpeng 的简短版本

CLE 部分与此前介绍一致，没有修改。我们在 11 城 CLE 的真实有向 OSM 路网、住宅候选点、depot 和 AFDC charging stations 上生成 EVRPTW instances；Amazon 只校准 route/stop 结构、package volume、service time 和 time windows，不把匿名坐标搬到目标城市。

代码先冻结 7,500 个 parent families 和 173,000 个 nested views。每个 family 先做 depot × Amazon day structure 联合可行性检查，在有向路网上 quarantine 不可达点，并按 route × radial-time decile 精确选择 customers；随后从完整真实 charger roster 中选能量连通且对 customer geography 有帮助的 CS，计算四张 directed parent matrices，再通过 maximum bipartite matching 分配不重复且 TW/容量/energy 可行的 Amazon order templates。

每个 family 在 staging 验证后原子发布，全部完成后再次验证 7,500 families、聚合 Phase-1、执行 full-corpus feasibility 和 construct-valid v3。最终从 CLE 到完整 Stage 2 实测约 24.5 小时；portable archive 约 7.33 GB，目标服务器还需约 5.3 小时重建完整 matrices（30 workers 大内存服务器实测，小服务器必须降低 worker）。
