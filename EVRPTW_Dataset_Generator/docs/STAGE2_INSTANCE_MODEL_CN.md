# CLE-EVRPTW Stage 2 实例生成模型

本文档是英文规范
[STAGE2_INSTANCE_MODEL.md](STAGE2_INSTANCE_MODEL.md) 的中文审阅版，用于逐项
检查从 City Logistics Environment（CLE）到经典静态 EVRPTW 实例的模型、
公式、参数来源和假设。代码字段名与最终公开引用以英文版为准。

## 1. 研究范围与真实性边界

美国参考实现应被准确描述为：

> 基于真实地理、整合公开数据、运营层半合成的 benchmark。

- CLE 中的城市边界、定向道路拓扑、建筑位置、设施候选和充电站证据来自
  公开数据；
- 每日激活顾客、包裹、需求、服务时间、时间窗和道路状态由版本化模型生成；
- 生成的实例不是一条真实发生的 Amazon 路线；
- Amazon ARCD 的匿名坐标不用于给十个 CLE 城市放置顾客。

V1 保持经典 EVRPTW 合同：

- 每个 matrix family 选择一个 depot；
- 同质、无限 EV fleet，车辆初始满电；
- demand 与 capacity 均使用体积 `cm3`；
- 每个激活 service location 最多一个 time window；
- charging station 具有无限 port；
- 采用线性、必须充满的 full-charging policy；
- 一个 instance 内道路耗时静态，不考虑 8 点与 14 点的差异；
- 优化目标为最小物理行驶距离；
- time 和 energy 只用于可行性；
- 不保存运行时 mask。

当前 horizon 为 `08:00-24:00`。它是可配置 benchmark 设置，不表示所有
真实运营都采用同一班次。

## 2. 输入、输出和执行顺序

Stage 2 输入：

1. 可独立加载的 portable CLE；
2. `configs/cle_evrptw_stage2_v1.json` benchmark 合同；
3. `configs/us_reference_instance_profile_v1.json` 美国运营参数适配器；
4. Census block-group polygon；
5. `scripts/analyze_amazon_arcd_statistics.py` 生成的 Amazon ARCD 聚合统计。

执行顺序：

1. 先冻结完整 community 的 train/held-out location pool；
2. 规划 matrix family 与各 problem scale 的 view；
3. 抽样 weekday/weekend、depot、customer superset 和嵌套 CS 集合；
4. 生成一套静态定向 road state；
5. 计算最短距离路径和带转向惩罚的最快路径；
6. 抽样 package、volume demand、service time 和 TW；
7. 先验收可行性；失败时重新抽样 order，不修改原 TW；
8. 保存单顾客充分可行性 certificate 与 CS 回库 cache；
9. 保存四张 parent matrix，低规模只保存 index view；
10. 对 family 和 view 做结构验证。

## 3. 数据划分和问题规模

### 3.1 完整 community held-out

```text
community_id = Census block group x directed road SCC
```

划分单位是完整 community，不是单个 customer。一个 community 只能进入
train location pool 或 held-out location pool，目标比例为 80/20。

- Test-1：十个训练城市、train pool、新 seed；
- Test-2：十个训练城市、完整 held-out communities；
- Test-3：训练从未使用的 Jacksonville；
- Cus2000：同城 unseen-scale，不属于 exact solver 主比较；
- Cus50：兼容传统小规模与 budgeted MIP 的 track。

### 3.2 训练量和固定 CS 数量

| Scale | Train | CS | 角色 |
| --- | ---: | ---: | --- |
| Cus50 | 100,000 | 10 | compatibility |
| Cus100 | 50,000 | 20 | core |
| Cus500 | 10,000 | 50 | core |
| Cus1000 | 5,000 | 50 | core parent |
| Cus2000 | 0 | 50 | unseen-scale test |

每个训练规模均为 (5 x 10^6) customer exposures。CS 数量按规模固定，避免
显存和 POMO step 的 shape 因 instance 改变。

## 4. Depot、CS 与 customer activation

### 4.1 Depot

每个 matrix family 从 release-eligible Tier-A/B candidate 中抽取一个 depot。
Tier C 不进入实例 depot pool。不同 family 可以使用不同 depot，避免固定单一
起点造成训练泄漏或过拟合。strict candidate 的抽样权重为 2，optional
candidate 为 1。这样既偏向物流证据更强的地点，又不会退化成很小的
strict-only 集合。2:1 是版本化 benchmark 选择，不是实际承运商份额，正式版本
需要提供 sensitivity。

以选中的 depot 为中心，customer catchment 从 40 km 开始，每次增加 10 km，
最大 100 km，直到包含至少 `max(N, ceil(1.5 N))` 个 eligible latent location。
这里的直线距离只做候选池预筛；最终 terminal matrix 仍全部来自定向路网。
这些半径是 development sampling 参数，不代表真实承运商的法定服务半径。

### 4.2 Charging station

以 depot 为中心构造可扩展 catchment。CS 在激活每日 customer 之前，依据完整
community centroid 进行排序；其前 10、20、50 个形成嵌套集合。这样 CS 选择
不会使用当日确切 customer ID。

令 `d_c(q)` 为 community reference point `c` 到候选 charger `q` 的距离，
`D_c` 为它到当前已选 charger 的最近距离。每一步最小化：

```text
sum_c w_c min(D_c, d_c(q))
  + 0.25 * Q90_c[min(D_c, d_c(q))]
  + 0.10 * max_c[min(D_c, d_c(q))].
```

第一项控制加权平均覆盖，后两项避免少数 community 的服务距离过差；`w_c` 是
eligible location 数量，不是当天订单数。0.25/0.10 是公开的 development design
constant，并不是从 AFDC 拟合出的运营参数。

有效充电功率：

```text
p_effective = min(p_station, p_vehicle_cap(mode))
```

AFDC 有功率时使用真实记录；缺失时使用同城市、同 mode 的 median。正式生成
如果整个 city-mode 都没有已知功率则失败；pilot 可以使用车辆 AC/DC cap，
但必须在 manifest 中记录 fallback。

### 4.3 激活 latent service location

CLE 保存潜在物理服务地点，不保存每日订单。实例先选择 community，再在
community 内激活 location：

- house 对应一个 ordering unit；
- apartment location 可以代表多个 residential units；
- `CusN` 表示 `N` 个不同物理 location，不表示 `N` 个包裹或住户。

NSI residential-unit 信息将真实建筑结构连接到每日激活概率和包裹数量。给定
`N`，先从截断 lognormal 抽取 target locations-per-community `T`。当前
center/spread 与 56--205 范围来自 Amazon route stop-count aggregate，但这里只把
`T` 当成空间分散程度的 prior。选中的 community 数量至少为：

```text
max(2, ceil(N/T), ceil(log2(N+1))).
```

community 按 eligible-location 数量、到 depot 的距离衰减和 family-level
lognormal activity multiplier 加权无放回抽样，直到它们合计还包含至少
`ceil(1.08 N)` 个 eligible location。随后，对含 `u_i` 个 residential unit 的
location，基础激活权重为：

```text
a_i(day) = 1 - (1 - p_day)^u_i,
```

再乘以 community activity。算法先保证每个已选 community 至少激活一个
location，再无放回补齐剩余位置。这样 apartment 更容易成为当天服务点，但不会
改变 `CusN` 的定义。

只有 route-size 描述统计直接来自 Amazon。community activity spread、distance
decay、1.08 buffer 和 per-unit probability 都仍是 development cross-data
calibration，不应写成普适 parcel rate。

## 5. 静态道路速度模型

### 5.1 三种不同的速度概念

CLE 的每条 directed edge `e` 保存：

- `v_e_legal`：该方向适用的法定限速；
- `v_e_ref`：商业配送车 reference running speed；
- H/M/U operating mode：
  - H：motorway/trunk transfer；
  - M：城市主要转移道路；
  - U：residential/service/delivery access。

法定限速按以下顺序确定：OSM directional `maxspeed`、OSM generic
`maxspeed`、经过方向验证的 high-confidence HPMS `SPEED_LIMIT`，最后才是
同城分层中位数填补。有可靠的 HPMS physical-corridor conflation 时优先用
`F_SYSTEM` 判定功能等级，否则使用 OSM `highway=*`。只有 corridor match
并不足以使用 HPMS 限速；还必须确认它对应唯一的 OSM 单行方向。H/M/U 是
本项目的 portable crosswalk，不是 NREL 或 MOVES 的原生类别。

H/M/U reference-speed anchor 来自 NREL Fleet DNA 商用车报告，是 mode-level
prior，不是 Rivian 的逐路段 telemetry
（[NREL report](https://doi.org/10.2172/1397153)）。

### 5.2 Instance road factor

EPA MOVES 按 road type、source type 和 day/time 保存平均速度分布。V1 使用
其可迁移的 road-type/day 分层结构：

```text
H   -> urban restricted access
M,U -> urban unrestricted access
```

每个 matrix family 对两类 MOVES road type 各抽一个 weekday/weekend factor：

```text
v_e(instance) = min(v_e(legal),
                    max(v_min, v_e(ref) * alpha(day, MOVES-road-type)))
```

M 与 U 使用同一个 day-level factor，但 `v_e_ref` 不同，所以小区道路仍比
城市主路慢。V1 不额外制造没有数据支撑的 corridor、physical-segment 或
direction 随机扰动。

网络非对称性来自：

- OSM 单行道和 directed topology；
- direction-applicable legal speed；
- A 到 B 与 B 到 A 不同的路径；
- 几何转向代价。

MOVES 支撑分层结构
（[EPA MOVES algorithms](https://www.epa.gov/moves/moves-algorithms)），但 JSON
中当前均值、标准差和上下界仍是 development 参数。正式发布前必须给出冻结
MOVES 版本、提取过程、拟合表和 sensitivity。NPMRDS 可作为 optional observed
adapter，但因授权与 NHS/TMC 覆盖限制，不作为 portable 默认输入
（[FHWA NPMRDS](https://ops.fhwa.dot.gov/publications/fhwahop20028/)）。

新加入的 customer/depot/CS connector 使用 U reference speed，两个方向速度
相同，但不会把原始 OSM 单行道路变成双向。

## 6. 带转向惩罚的最快路径

V1 不加入 traffic-signal delay，只加入 straight/right/left/U-turn 的几何
penalty。

- distance path 最小化物理距离，其 paired time 在相同路径上计算 edge time
  与 turn penalty；
- fastest path 直接在优化中包含 turn penalty，不是选完 path 后再补 penalty。

实现将原 directed graph 转成 edge-state graph。state 表示 incoming edge
`e`；仅当 `head(e)=tail(f)` 时允许从 `e` 转到 `f`：

```text
w(e,f) = travel_time(f) + turn_penalty(e,f)
```

在 edge-state graph 上运行 Dijkstra，再精确加入 terminal connector 与起终点
partial-edge cost。因此，一条略长但少转弯的路径可以成为真正的 fastest path。

## 7. Package、volume、service time 与 TW

主要统计来源是
[Amazon Last Mile Routing Research Challenge 2021](https://doi.org/10.1287/trsc.2022.1173)。
该数据包含 route/stop/package 层记录、package dimensions、planned service
time、TW 和 vehicle volume capacity，但坐标经过匿名扰动，且没有
house/apartment label。

因此我们只迁移聚合统计，不迁移 location；house/apt 条件分布是 NSI/CLE
unit evidence 与 Amazon aggregate 结合的显式半合成模型。

### 7.1 Package count 与 demand

对具有 `u_i` residential units 的激活 location：

1. 按 weekday/weekend 抽样 ordering units；
2. 因 location 已激活，条件化为至少一个 ordering unit；
3. 用 negative-binomial 模型抽取额外包裹；
4. 每个包裹从 truncated lognormal volume 分布抽样并求和。

```text
demand_i = sum(volume_ij)
```

demand 与 vehicle capacity 都使用 `cm3`。

### 7.2 Service time

```text
s_i = clip((beta_0
            + beta_pkg * package_count_i
            + beta_vol * demand_i) * lognormal_noise,
           s_min, s_max)
```

Amazon 的 planned service time 是 package-level；统计脚本先在 stop 内求和，
再拟合 stop-level model。因此一个 apt location 的包裹越多、volume 越大，
service time 一般越长。

### 7.3 一个 location 一个 TW

先按 weekday/weekend Beta 分布抽取该 instance 的 TW presence rate。每个
location 得到完整 horizon，或者一个 `strain`/`loose` interval。中心与宽度
来自 profile。

TW 只会与声明的 `08:00-24:00` 支持范围求交；不会依据 shortest travel time
平移、扩大或缩短。

### 7.4 先抽样、再验证

先生成 package、demand、service time 和 TW，再做充分可行性检查。若 order
draw 不可行，则保留物理 location，重新抽取该 location 的 order 属性，并记录
attempt 与 rejection reason。

如果 location 即使使用最小 service time 也在结构上无法完成单顾客路线，
整个 matrix-family attempt 失败，由外层 deterministic retry 重新选择 family。
代码不会为了让数据可行而修改 TW。

64 次重试只是显式 fail-safe，不是行业统计参数；耗尽后直接报错。

## 8. Reference vehicle 与线性能耗

V1 固定为 Rivian Commercial Van Delivery 700。官方 2025 Reference Guide
提供：

- cargo volume：18.5 m3 = 18,500,000 cm3；
- EPA range：160 mi = 257 km；
- battery：100 kWh LFP；
- AC：11 kW；
- DC：up to 100 kW。

来源：
[Rivian Commercial Van Reference Guide](https://assets.ctfassets.net/2md5qhoeajym/5FQcJgfAOa4vDYu9rWwEYO/2fa75339d6e533532ba08bf395275015/RCV-QuickRef-v17.pdf)。

V1 使用经典 EVRPTW 的常数单位距离能耗：

```text
h = 100 / 257 = 0.3891050584 kWh/km
energy(P) = h * distance(P)
```

速度、等待、转向时间和 auxiliary load 不改变 V1 energy。这是为了与 exact、
heuristic 和 RL 的经典 EVRPTW 模型兼容，不代表真实 EDV 能耗与速度、天气、
payload 和 HVAC 无关。

经典线性建模可参考
[Schneider et al.](https://doi.org/10.1287/trsc.2013.0490) 以及显式使用常数
consumption rate 的
[Operational Research charging-model comparison](https://link.springer.com/article/10.1007/s12351-023-00806-5)。

## 9. Charging 与 CS 回库 cache

在有效功率为 `p_q` 的 CS，从当前能量 `b` 充满所需时间：

```text
charge_time(q,b)
    = (battery_capacity - b) / (eta * p_q) * 3600
```

当前 `eta=1` 表示 AFDC/vehicle-cap 后的 kW 被解释为 battery-side effective
power。这样不需要虚构没有数据支撑的损耗；其它 adapter 可以加入有来源的
efficiency。

每个 view 保存 `full_cs_to_depot_time_s[q]`：从 CS `q` 满电出发，以最快
energy-feasible 方式回 depot，允许经过多个 CS，包含中间充电时间，但不包含
origin CS 或 depot 的充电。该 cache 使用 fastest-path time 与 fastest-path
distance 派生的 energy。

## 10. 只保存四张 matrix

每个 parent family 持久化四张 `float32` matrix：

1. `distance_matrix_km`；
2. `distance_path_travel_time_s`；
3. `running_time_shortest_matrix_s`；
4. `running_time_path_distance_km`。

loader 按需派生：

```text
distance_path_energy_kwh = distance_matrix_km * h
running_time_path_energy_kwh = running_time_path_distance_km * h
```

因此 consumer API 仍提供两张 energy array，但不在磁盘重复保存。之所以不能
只保留两张 matrix，是因为 distance-shortest 与 time-fastest 可能是两条不同
路径，time 与 energy 都必须和实际评估的 path 对齐。

## 11. 当前主要数值参数与证据等级

| 小模型 | 当前数值 | 证据状态 |
| --- | --- | --- |
| Weekday/weekend | 5:2 | benchmark mixture |
| MOVES restricted factor | weekday 0.96/0.035；weekend 0.98/0.030 | 分层有来源，数值仍 development |
| MOVES unrestricted factor | weekday 0.92/0.050；weekend 0.95/0.045 | 分层有来源，数值仍 development |
| Connector speed | 36.403361 km/h | NREL U-mode prior |
| Turn class | straight <=30 degrees，U-turn >=150 degrees | geometry convention |
| Turn penalty | right 3 s，left 8 s，U-turn 20 s | development constants |
| Depot evidence weight | strict:optional = 2:1 | development sampling rule |
| Depot catchment | 40 km 起，10 km 扩展，最大 100 km | development sampling rule |
| Catchment pool buffer | 请求 location 数的 1.5 倍 | development sampling rule |
| Community target | lognormal median 141.6661，sigma 0.1835，范围 56--205 | Amazon route stop-count descriptive prior |
| Community capacity/activity | buffer 1.08；lognormal sigma 0.85；distance decay 30 km | development spatial sampling rule |
| Per-unit order probability | weekday 0.028，weekend 0.022 | cross-data development fit |
| Charger coverage score | weighted mean + 0.25 p90 + 0.10 maximum | development facility-sampling rule |
| Extra package | mean 0.62194，dispersion 0.327 | Amazon aggregate development fit |
| Package volume | median 7,000 cm3，sigma 1.0，cap 300,000 cm3 | Amazon dimensions development fit |
| Service time | 28.1626 + 46.9063/package + 0.000358429/cm3 | Amazon planned-service development fit |
| TW occurrence | weekday Beta(6.1,60.7)，weekend Beta(5.9,58.5) | Amazon aggregate development fit |
| Battery/range/cargo | 100 kWh / 257 km / 18.5 m3 | Rivian official guide |
| AC/DC cap | 11/100 kW | Rivian official guide |
| Charging efficiency | 1.0 | explicit effective-rate assumption |

正式把 profile 改为 `release_calibrated` 之前，需要一起发布：

- `amazon_arcd_training_statistics_v1.json`；
- 原始统计到参数表的 fitting script/notebook；
- MOVES 冻结版本与 query；
- road factor、turn penalty、package/service/TW 的 sensitivity；
- 引用和 source snapshot 说明。

结构测试通过只表示实现合同一致，不等于科学校准已经完成。

## 12. 主要引用

- Amazon ARCD: <https://doi.org/10.1287/trsc.2022.1173>
- Rivian Commercial Van Reference Guide:
  <https://assets.ctfassets.net/2md5qhoeajym/5FQcJgfAOa4vDYu9rWwEYO/2fa75339d6e533532ba08bf395275015/RCV-QuickRef-v17.pdf>
- EPA MOVES: <https://www.epa.gov/moves/moves-algorithms>
- NREL Fleet DNA: <https://www.nrel.gov/transportation/fleettest-fleet-dna.html>
- NREL/TP-5400-65921: <https://doi.org/10.2172/1397153>
- Schneider et al. EVRPTW: <https://doi.org/10.1287/trsc.2013.0490>
- FHWA NPMRDS: <https://ops.fhwa.dot.gov/publications/fhwahop20028/>
