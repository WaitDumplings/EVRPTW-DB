# EVRPTW-DB：CLE 与 Instance 完整生成流程（审核稿）

> **已被新的 V2 连通性修复审核稿取代。** 本文保留为 2026-08-16 的历史流程
> 快照，其中部分 V1 路径、turn penalty 和 Phase-1 gate 描述已经过时。当前审核请
> 使用
> `EVRPTW_Dataset_Generator/docs/stage2_repair/STAGE2_CONNECTIVITY_REPAIR_AND_PIPELINE_REVIEW_ZH.md`。

> 状态：生成前审核稿，2026-08-16。
>
> 本文只描述当前 EVRPTW-DB 代码准备如何生成、验证、压缩和恢复数据。
> 不复用 EVRPTW-DB_Legacy 的 CLE 或 instances。审核通过并得到明确授权前，
> 不继续生产运行、不制作正式压缩包、不提交当前改动。

## 1. 一页结论

完整流程：

    固定公开源
      → 11 城 CLE + portable verification
      → Amazon 三个 JSON 转为无坐标 calibration artifact
      → complete-community train/held-out split
      → 7,500 parent families / 173,000 nested views
      → 12 个 spawn workers 生成四张 parent matrices
      → 每 family verification + Phase-1 metrics
      → corpus 验收
      → CLE + slim instances 压缩
      → 独立目录 full exact restore
      → 全部成功后再提交代码

审核人首先需要知道：

1. 当前 operations profile 是 development_calibration，且
   official_generation_eligible=false。当前完整结果只能标为 research，不能称为
   official scientific release。
2. 当前代码实际计算出 7,500 families、173,000 views；Legacy 的 172,500 views
   不适用于当前版本。
3. 当前只冻结 correctness hard gates。M1 是候选比较指标，M2/M3/M5 是
   report-only；技术 passed 不自动等于 realism 已获得论文级认可。
4. PBF preparer 当前使用 --skip-sha256。派生产物、矩阵 contract 和最终 archive
   仍做 SHA-256，但正式发布是否补做 raw-source hash freeze 需要审核决定。
5. Amazon 派生数据受 CC BY-NC 4.0 约束。

## 2. 权威入口与配置

入口：

- generate_cle.sh：准备公共源并生成 11 城 CLE。
- generate_instances.sh：生成 Amazon calibration 和 Stage-2 instances。
- auto.sh archive create：生成 CLE + matrix-free slim instances 压缩包。
- auto.sh archive start/status/logs/wait：在另一服务器恢复全部矩阵。

权威配置：

- EVRPTW_Dataset_Generator/configs/us_11city_cle_v1.json
- EVRPTW_Dataset_Generator/configs/us_11city_population_v1.json
- EVRPTW_Dataset_Generator/configs/cle_evrptw_stage2_v1.json
- EVRPTW_Dataset_Generator/configs/us_reference_instance_profile_v1.json
- EVRPTW_Dataset_Generator/configs/us_moves5_speed_profile_v1.json
- EVRPTW_Dataset_Generator/configs/us_census_block_groups_v1.json

权威实现/规范：

- scripts/build_cle_cohort.py
- scripts/build_stage2_instances.py
- src/evrptw_stage2/planning.py
- src/evrptw_stage2/materialize.py
- src/evrptw_stage2/artifacts.py
- src/evrptw_stage2/metrics.py
- docs/PIPELINE.md
- docs/STAGE2_INSTANCE_MODEL_CN.md
- docs/OUTPUT_SCHEMA.md

README、Legacy manifest 与 planning 代码数量不一致时，以当前配置加载器和
build_generation_plan() 的计算结果为准。

## 3. 环境、CPU、内存和磁盘

创建专用环境：

    cd /data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Dataset_Generator
    conda env create -f environment.yml
    conda activate evrptw-cle
    python --version
    osmium --version
    python -m pytest -q tests

当前已建立：

    /home/npg/miniconda3/envs/evrptw-cle

环境是 Python 3.11，包含 GeoPandas、NetworkX、NumPy、OSMnx、PyArrow、Pyogrio、
SciPy、Shapely、AWS CLI 和 osmium-tool，并以 pip -e . 安装项目。

资源：

- Stage-2 使用 WORKERS=12。
- OMP/OpenBLAS/MKL/NumExpr 每进程限制为 1 线程。
- 内存保护模型为每 worker 5 GiB + parent/OS 4 GiB，12 workers 至少约 64 GiB。
- 当前服务器约 251 GiB RAM。
- matrix 数值本体计划值：157,368,120,000 bytes。
- 30,000 个 NPY header 约 3,840,000 bytes。
- matrix 文件合计约 157,371,960,000 bytes，即 157.37 GB / 146.56 GiB。
- 应至少预留 300 GiB；若 full output、slim staging、独立 full restore 同时存在，
  建议预留 500 GiB 以上。

历史文档中的 154.79 GiB 等口径与当前代码 byte-level 估计不完全一致，发布前应
统一 GB/GiB。

## 4. 城市与 held-out 设计

十个 train cities：

    new-york, los-angeles, chicago, houston, phoenix,
    philadelphia, san-antonio, san-diego, dallas, fort-worth

完全 held-out city：

    jacksonville

边界为 2025 Census incorporated place 的 city proper land-only boundary，不是
metro area，也不声称是真实承运商 service territory。

## 5. Phase 0：公共源准备

scripts/prepare_us11_sources.py 检查 36 个固定文件：

| 类型 | 数量 | 用途 |
| --- | ---: | --- |
| Geofabrik PBF | 7 | directed OSM roads、depot/charging evidence |
| Microsoft state building GeoJSON | 7 | building polygon 和面积 |
| FHWA HPMS city windows | 11 | road class 和方向验证后的缺失限速证据 |
| AFDC/OSM/Census charging evidence | 4 | raw AFDC、address anchor、OSM POI、resolved AFDC |
| 2025 Census Block Group ZIP | 7 | Stage-2 community/split |

NSI 不包含在 36-file contract 中；它在 CLE customer stage 按 deterministic tiles
下载并缓存。已有 NSI_CACHE_ROOT/<city>/raw_tiles 时会预先复制到 resumable work。

下载/复用顺序：

    PBF → Microsoft buildings → HPMS windows → AFDC snapshot
      → OSM charging POI → Census address evidence → resolved AFDC
      → Census Block Groups

只检查：

    cd EVRPTW_Dataset_Generator
    PYTHONPATH=src python scripts/prepare_us11_sources.py --check-only

准备源：

    export NLR_API_KEY='操作者私下设置；不写入仓库或日志'
    PYTHONPATH=src python scripts/prepare_us11_sources.py
    unset NLR_API_KEY

AFDC filter：

    Fuel Type Code = ELEC
    Status Code = E
    Access Code = public
    Country = US

NLR_API_KEY 只在全国 AFDC snapshot 缺失时需要，兼容旧名 NREL_API_KEY。Census
只提供 address anchor QA，不当作精确 EVSE 坐标。resolved geometry 优先级为人工
override、OSM exact-address、raw AFDC，原始坐标始终保留。

当前 PBF 命令含 --skip-sha256，以减少大型 PBF 的第二次顺序读取。审核可选择：

- research 接受该策略，最终 archive/派生 contract 仍做 SHA；
- final release 前补算全部 raw-source hash 并冻结 manifest。

## 6. Phase 1：11 城 CLE

### 6.1 计划命令

先完成公共源，再把下载与 CLE 计算尽量分开：

    cd /data/Maojie/ICLR/EVRPTW-DB
    mkdir -p logs

    /usr/bin/time -v -o logs/generate_cle.time \
      env PREPARE_CLE_SOURCES=0 \
          PYTHON_BIN=/home/npg/miniconda3/envs/evrptw-cle/bin/python \
          NSI_WORKERS=4 \
          KEEP_CLE_WORK=0 \
          ./generate_cle.sh \
      > logs/generate_cle.log 2>&1

第一次无 NSI cache 时，NSI 网络时间仍在 CLE 阶段内。若要求严格排除所有下载，
需先增加/运行独立 NSI source-only acquisition；当前入口无法在第一次运行中完全
拆开 NSI 下载与 CLE computation。

| 环境变量 | 默认值 |
| --- | --- |
| CLE_PROFILE | configs/us_11city_cle_v1.json |
| CLE_WORK_ROOT | work/us-11city-v1 |
| CLE_RELEASE_ROOT | EVRPTW_Dataset/CLE_v2/us_11city |
| NSI_CACHE_ROOT | data/sources/nsi-us-11city |
| NSI_WORKERS | 4 |
| PREPARE_CLE_SOURCES | 1 |
| KEEP_CLE_WORK | 0 |

无参数完整成功且 KEEP_CLE_WORK 不为 1 时，仅删除 generator 自己管理的 work tree；
失败或 partial stage 保留 work 以便续跑。

### 6.2 七个 CLE stages

    preflight → roads → buildings → depots → cles → package → index

#### preflight

检查 profile、boundary、PBF、building、AFDC、HPMS、MOVES 和路径合同。失败停止。

#### roads

从 frozen PBF 建 directed drive graph：

| 参数 | 值 |
| --- | ---: |
| component policy | all |
| query buffer | 5,000 m |
| query simplify | 100 m |
| routing buffer ladder | 0, 1, 2, 5, 10, 20 km |
| retained node coverage | ≥0.99 |
| retained physical-road-length coverage | ≥0.995 |
| residual isolated-component threshold | 严格少于 100 nodes |
| synthetic intercomponent edges | 禁止 |

保留 one-way、parallel edges、OSM ID/tags/geometry。边界外连接道路标为
transit_only，不能承载 customer/depot/charger；不把 one-way 改成 two-way。

#### buildings

从 7 个 Microsoft state files 按城市提取 polygon。已完成 manifest 可续跑复用。

#### depots

- Tier A：明确 carrier/dispatch/logistics facility evidence。
- Tier B：warehouse/logistics proxy，可进入 research pool。
- Tier C：ambiguous/generic/retail parcel shop，默认排除。
- 1,000 m² 只是 sensitivity flag，不是 hard filter。

#### cles（逐城市）

每城执行：

1. 验证 operational graph；
2. HPMS–OSM physical corridor conflation；
3. Microsoft extraction/reuse；
4. NSI 下载、classification、structure grouping；
5. NSI point–building G1/G2 matching；
6. customer/depot/charger road projection 和 connectors；
7. AFDC/OSM/Census facility evidence；
8. OSM/HPMS legal speed + MOVES5 weekday/weekend reference speed；
9. SCC/protected round-trip eligibility 和 technical QA。

住宅使用普通 RES1/RES2/RES3；institutional residence 不进入普通 customer pool。
G1 为 point covered by polygon；G2 为距离 ≤10 m 且面积差 ≤4 倍。customer access
大于 200 m 只是 QA flag。

HPMS 参数：

| 参数 | 值 |
| --- | ---: |
| candidate radius | 75 m |
| overlap buffer | 25 m |
| minimum overlap | 0.20 |
| max orientation delta | 30° |
| high-confidence distance | 25 m |
| high-confidence overlap | 0.50 |
| high-confidence orientation | 15° |
| ambiguity distance margin | 10 m |
| ambiguity overlap margin | 0.20 |

HPMS F_SYSTEM 只来自 high-confidence corridor match；SPEED_LIMIT 只有在
high-confidence 且 OSM physical segment 有唯一可验证方向时才能补 directed edge。

MOVES5 使用 sourceTypeID 32 和 08:00–24:00 window：

    restricted:   weekday retention 0.758791, weekend 0.841456
    unrestricted: weekday retention 0.545059, weekend 0.564078

    reference_speed = legal_speed × day/type retention

Stage-2 只接受 evrptw_directed_speed_profiles_v6 和 versioned profile ID，拒绝
Legacy 单列 reference_speed_kph。

#### package/index

把 technical CLE 复制到 private staging，加入 operational GraphML，改为
package-relative paths，移除 machine-local runtime paths，重算 hash，运行 strict
portable verifier，原子发布到：

    EVRPTW_Dataset/CLE_v2/us_11city/cities/<city>/

最后生成 cle_index.json/csv。

### 6.3 CLE acceptance

    jq '{status, verified_cle_count, failures}' \
      EVRPTW_Dataset/CLE_v2/us_11city/cle_index.json

必须为：

    status = complete
    verified_cle_count = 11
    failures = []

每城还需 technical_verification_passed=true 和 portable_package_verified=true。
release_eligible 是独立 scientific status，不能用 portability 代替。

## 7. Amazon calibration

只读三个 training JSON：

    route_data.json
    package_data.json
    travel_times.json

计划复用已有路径，不重新下载：

    /data/Maojie/Maojie_Github/dataset/almrrc2021/
      almrrc2021-data-training/model_build_inputs

artifact 输出：

    EVRPTW_Dataset/Calibration_v1/amazon_stage2_v2

Amazon 坐标不迁移到目标城市。只保留 station-day route/stop structure、
depot-time decile、pairwise-time statistics、package volume/count、planned service、
TW template 和 provenance。

单日不足 N 时，只允许同一 station、同一 day type 多日 composite；禁止跨 station
合并和 template duplication。

## 8. Frozen Stage-2 plan

### 8.1 Mode

- research：当前完整计划默认模式。
- official：当前 profile official_generation_eligible=false，会拒绝。
- non_release_pilot：必须设置 --pilot-families-per-city。

生产拟用 INSTANCE_MODE=research。

### 8.2 Split/cohort

    community_id = Census Block Group × directed-road SCC

完整 community 80/20 分到 train/held-out，不按 building 随机切分。adjacency 来自
真实 OSM 跨 community directed edges。

| Cohort | Families | Customer pool | Parent |
| --- | ---: | --- | --- |
| core/train | 5,000 | 10 城 train | Cus1000 |
| core/val | 500 | 10 城 train | Cus1000 |
| test1 new seed | 500 | 10 城 train | Cus1000 |
| test2 held-out locations | 500 | 10 城 held-out communities | Cus1000 |
| test3 held-out city | 500 | Jacksonville | Cus1000 |
| scalability | 500 | 10 城 train | Cus2000 |
| **总计** | **7,500** |  |  |

每 city×cohort 用 largest-remainder 精确分配 weekday/weekend 5:2，再 frozen-seed
shuffle；retry 不改变 slot day type。

### 8.3 Views

当前代码实际值：

| Scale | Customers | CS | Views |
| --- | ---: | ---: | ---: |
| Cus50 | 50 | 10 | 101,000 |
| Cus100 | 100 | 20 | 52,000 |
| Cus500 | 500 | 50 | 12,000 |
| Cus1000 | 1,000 | 50 | 7,500 |
| Cus2000 | 2,000 | 50 | 500 |
| **总计** |  |  | **173,000** |

训练 Cus1000 parent 嵌套为 20×Cus50、10×Cus100、2×Cus500、1×Cus1000。
Cus50 leaves 互斥且 union 等于 parent；Cus100/Cus500 由固定 leaves 合成。
validation/test 只保留一个 strict nested chain。Cus2000 family 有 Cus2000 view 和
Cus1000 control-subset view。

Master seed 为 20260810。family/view ID 和各类 RNG seed 用稳定 BLAKE2b namespace
派生。

## 9. 每个 family 的 materialization

1. 等概率选 physical facility group；组内优先 Tier A，否则 Tier B depot。
2. 在 day-type CLE reference speed 上从 depot 路由。
3. Territory customer 必须属于指定 pool、connector 合法、depot time ≤ Amazon
   T_env，且 depot→customer→depot 最快路径能耗 ≤100 kWh。
4. Controlled matrix rounding 将 Amazon route×time-decile counts 缩放到 N，同时
   精确保留 route row、decile column 和总数。
5. Region seed 使用 quota-descending/network-time max-min。
6. 沿 directed community adjacency 进行 road-contiguous growth。
7. Global min-cost flow 同时满足所有 region-decile cells；customer ID 全局唯一。
8. 生成 size-matched radial baseline，仅用于 M1/M5，不是 instance。
9. Customer geography 固定后，从真实 AFDC 选择 CS；不创建 synthetic CS。
10. 计算四张 parent matrices。
11. 用 bipartite matching 分配不重复 Amazon order templates；volume、TW、service、
    return horizon 必须可行。
12. 写 nested views、full-CS-to-depot multi-hop cache、feasibility certificate。
13. 计算 Phase-1 metrics 并运行 family verifier。

MAX_ATTEMPTS_PER_FAMILY=4。失败 attempt 的 seed、error、time、reason 保存在
rejections。下一 attempt 使用确定性 replacement seed；四次均失败则进入
unresolved_family_ids，完整运行失败。

## 10. Operations/profile 参数

### 10.1 Vehicle/charging

| 参数 | 值 |
| --- | ---: |
| horizon | 28,800–86,400 s（08:00–24:00） |
| day weights | weekday:weekend = 5:2 |
| battery/range | 100 kWh / 257 km |
| energy | 0.3891050584 kWh/km |
| cargo | 18,500,000 cm³ |
| initial SOC | 1.0 |
| DC/AC caps | 100/11 kW |
| charging | full, linear, efficiency 1.0 |
| capacity | infinite ports, unlimited fleet |

Research missing-power policy：先用同城同 mode median；仍缺失时用 vehicle mode cap。

### 10.2 Road/turn

Road residual 对所有 day/type 都是 mean=min=max=1、std=0，不叠加随机 traffic
multiplier。Turn penalties：right 3 s、left 8 s、U-turn 20 s；straight ≤30°，
U-turn ≥150°。Signal delay 未建模。

### 10.3 Package/service/TW

| 参数 | 值 |
| --- | ---: |
| max packages/location | 72 |
| extra-package mean/dispersion | 0.62194 / 0.327 |
| package median/sigma | 7,000 cm³ / 1.0 |
| max package | 300,000 cm³ |
| service base | 28.1626 s |
| service beta/package | 46.9063 s |
| service beta/volume | 0.000358429 s/cm³ |
| service noise sigma | 0.75 |
| service clamp | 5–8,007 s |
| order resample cap | 64 |

TW presence：weekday Beta(6.1,60.7)，weekend Beta(5.9,58.5)；strain share
0.80/0.76。TW 只 sample-then-validate，不做 feasibility clipping。

### 10.4 Four matrices

每 family 恰好四张 float32 square matrix：

    distance_matrix_km.npy
    distance_path_travel_time_s.npy
    running_time_shortest_matrix_s.npy
    running_time_path_distance_km.npy

Cus1000/Cus2000 terminal counts 是 1,051/2,051。Energy matrix 不存储，按 path
distance ×0.3891050584 派生。Terminal order 固定为 depot、customers、CS。

## 11. 12-worker full command

    cd /data/Maojie/ICLR/EVRPTW-DB
    mkdir -p logs

    /usr/bin/time -v -o logs/generate_instances.time \
      env INSTANCE_MODE=research \
          WORKERS=12 \
          FAMILIES_PER_WORKER_TASK=25 \
          MAX_ATTEMPTS_PER_FAMILY=4 \
          PYTHON_BIN=/home/npg/miniconda3/envs/evrptw-cle/bin/python \
          AMAZON_MODEL_BUILD_INPUTS=/data/Maojie/Maojie_Github/dataset/almrrc2021/almrrc2021-data-training/model_build_inputs \
          ./generate_instances.sh \
      > logs/generate_instances.log 2>&1

输出：

    CLE_ROOT=EVRPTW_Dataset/CLE_v2/us_11city
    AMAZON_ARTIFACT_ROOT=EVRPTW_Dataset/Calibration_v1/amazon_stage2_v2
    INSTANCE_OUTPUT_ROOT=EVRPTW_Dataset/Instances_v1/us_11city

并行细节：

- start method：spawn。
- 每 task 处理同一城市 25 families，复用 routing topology。
- materialization 最多 workers×2=24 个 in-flight tasks。
- verification 最多 workers×4=48 个 in-flight tasks。
- 12 workers 的并行单位是 family chunks，不是城市或单个矩阵分片。

续跑：

- source、Amazon artifact、完整 split/plan 可复用；
- 已有 family 必须先通过 verify_materialized_family；
- partial/conflicting matrix directory 不静默覆盖；
- rejection ledger 和 deterministic retry state 保留；
- 中断后在同一 output root 重跑同一命令。

## 12. Family verifier

逐 family 检查：

- family/view schema、ID、terminal count/source-ID uniqueness；
- order template/station-day/source-mode 完整；
- 三份 Phase-1 family files 和 hard gates；
- 四 matrices 恰好齐全；
- shape、float32、finite、nonnegative、zero diagonal；
- view parent indices 和 customer/CS/matrix shapes；
- energy 与 path distance 线性一致；
- package/demand/TW shapes；
- feasibility certificate；
- full-CS-to-depot cache；
- runtime action mask 不得存储；
- asymmetry 等 matrix statistics。

profile 不是 release_calibrated 时会产生 warning。

## 13. Phase-1 evaluation

Hard gates（任一失败则 family 不接受）：

1. customer count 恰好 N；
2. parent customer IDs 唯一；
3. split pool 明确；
4. route row margins exact；
5. decile column margins exact；
6. global customer uniqueness；
7. view union exact；
8. child views pairwise disjoint；
9. child sizes exact。

Diagnostics：

| 指标 | 内容 | 当前状态 |
| --- | --- | --- |
| M1 | proposal depot-time 对 Amazon target 的 normalized W1，并比较 radial baseline | candidate，无冻结阈值 |
| M2 | network nearest-neighbor time 对 Amazon | report-only |
| M3 | within-region pairwise P50/P90 对 Amazon routes | report-only |
| M4 | region count/size 和 controlled-rounding audit | construction audit |
| M5 | community count/largest share/HHI 对 radial baseline | diagnostic-only |

Reliability 还报告 territory reserve、energy-screen removal、seed fallback、region
redraw、community growth、assignment competition、first-attempt/conditional success 和
rejection reasons。

审核必须在看到 full result 前决定 M1–M5 是只报告、冻结阈值，还是专家分层签字；
不应生成后挑“刚好通过”的 threshold。

## 14. Corpus acceptance

CLE：

    status=complete
    verified_cle_count=11
    failures=[]

Stage-2：

    jq '{
      passed,
      mode,
      workers: .execution.workers,
      selected: .execution.selected_family_count,
      unresolved: (.unresolved_family_ids|length),
      verified: (.verified|length),
      wall_seconds: .performance.run_wall_seconds
    }' EVRPTW_Dataset/Instances_v1/us_11city/stage2_run_report.json

最低要求：

    passed=true
    mode=research
    workers=12
    selected=7500
    unresolved=0
    verified=7500

Phase-1 summary 最低要求：

    all_hard_gates_passed=true
    successful_parent_family_count=7500

还必须审查 M1 share、first/conditional attempt success、rejection reason、region/fallback
distributions，以及 stratified_metrics.csv 的 city×day_type×scale 分布。

## 15. Timing

分别计时：

1. public source download；
2. CLE computation；
3. Amazon artifact；
4. full Stage-2；
5. slim export/compression；
6. full exact restore。

/usr/bin/time -v 保存 wall time、CPU、RSS、I/O；Stage-2 report 还保存
run_wall_seconds 和 worker peak RSS。

历史性能文档仅称 2 workers 约 22–30 小时。12 workers 不保证线性加速，会受 graph
size、memory bandwidth、CPU contention 和 filesystem throughput 限制。建议审核后
先跑 exact-seed pilot，再决定 full run。不能用 Legacy 的“2 分钟”推断当前完整流程。

## 16. Outputs

    EVRPTW_Dataset/
      CLE_v2/us_11city/
      Calibration_v1/amazon_stage2_v2/
      Instances_v1/us_11city/
        customer_splits/
        generation_plan/
        materialized/families/
        rejections/
        reports/phase1/
        stage2_run_report.json

每 family：

    family_manifest.json
    terminal_index.parquet
    phase1_metrics.json
    phase1_observations.parquet
    phase1_region_pair_metrics.parquet
    matrices/*.npy
    views/<view_id>/
      view_manifest.json
      terminal_parent_indices.npy
      customer_attributes.npz
      charging_attributes.npz

split 目录只存 indices，不复制矩阵。

## 17. Slim archive 和 restore

三层验收后：

    ./auto.sh archive create \
      --archive /data/EVRPTW_Dataset_us11city_research_slim_v1.tar.zst \
      --compression-threads 12

创建器会强制检查 CLE 11/11、Stage-2 0 unresolved、7,500 verified、Phase-1 hard
gates；复制 portable CLE；导出 matrix-free family/view tree；记录每张原矩阵的
SHA/shape/dtype；拒绝 link/special file/同名输出；zstd level 9、12 threads；流式
重检 archive 后才发布 .sha256。

另一服务器：

    git clone <repository>
    cd EVRPTW-DB
    ./auto.sh archive start \
      --archive /path/release.tar.zst \
      --destination /data \
      --workers 12 \
      --families-per-worker-task 25
    ./auto.sh archive status --destination /data
    ./auto.sh archive wait --destination /data

Restore 验证 archive SHA、全部 tar members、required code commit ancestry 和磁盘；
private staging 解包后原子发布；重建每张矩阵并要求与 export contract SHA 完全一致。
中断可续跑，exact-hash 完成的 family 可复用。正式 archive 后还必须在独立
destination 做一次覆盖全部 7,500 families 的 full restore，phase 必须 succeeded。

## 18. 当前暂停状态

截至 2026-08-16：

- 无活动 CLE/instance 生产进程。
- EVRPTW_Dataset 仍为空骨架，无 CLE/instances。
- 已缓存 7/7 PBF、7/7 Microsoft buildings、11/11 HPMS windows。
- 36-file contract 中 25 个已存在；AFDC/charging evidence 和 7 个 Block Group ZIP
  未完成。
- source cache 约 19 GB。
- AFDC 仍需操作者私下提供 NLR key。
- Generator tests 此前 115/115 通过。
- archive-create 改动和 tiny exact restore test 尚未提交；按约定正式 archive 完成
  前不提交。

## 19. 审核 checklist

- [ ] 接受 11 城 cohort 和 Jacksonville held-out design。
- [ ] 接受 research 标签，或先把 profile 校准为 official eligible。
- [ ] 确认当前权威数量为 7,500 / 173,000，而非 Legacy 172,500。
- [ ] 接受 complete-community 80/20 split。
- [ ] 接受 nested views 和 Cus2000/Cus1000 control。
- [ ] 接受 Amazon template transfer 和不迁移坐标。
- [ ] 接受 constant-distance energy、linear charging、unlimited fleet/ports。
- [ ] 接受 static weekday/weekend road state。
- [ ] 接受四 matrices，不存 energy matrix/runtime mask。
- [ ] 预先决定 M1–M5 验收策略。
- [ ] 决定 --skip-sha256 和 final raw-source hash policy。
- [ ] 审核 Amazon CC BY-NC 及所有 attribution。
- [ ] 接受 4 attempts 和 selection-bias/rejection reporting。
- [ ] 接受 12 workers、25 families/task 和 300–500 GiB 磁盘规划。
- [ ] 批准先 pilot，再 full run。
- [ ] 审核 archive create/restore 后再批准生产和 commit。

## 20. 建议审批顺序

1. 模型/参数审核。
2. 完成并冻结 sources。
3. 只生成/验收 11 CLE。
4. San Diego deterministic pilot：1 worker、每 cohort 1 family，检查 schema、M1–M5
   和 exact replay。
5. 批准 12-worker full 7,500-family run。
6. 审核 corpus/stratified quality。
7. 批准 slim archive + full exact restore。
8. 最后批准 Git commit。

每一步都有独立 gate；任一 gate 未通过，不继续扩大计算与存储成本。
