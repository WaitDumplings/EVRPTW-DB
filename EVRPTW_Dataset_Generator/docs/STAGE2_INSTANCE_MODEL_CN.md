# CLE 到 EVRPTW Instance 的生成合同（中文审阅版）

> **D-5 修订（2026-08-19）：**下文 M2/M3 Q90 规则作为历史 v1 合同保留，现由
> [`D5_CONSTRUCT_VALIDITY_REVISION_V2_ZH.md`](stage2_repair/D5_CONSTRUCT_VALIDITY_REVISION_V2_ZH.md)
> 取代：Amazon 只 hard-gate 运营迁移；M2/M3/M5 降为 report-only 空间诊断。

本文对应英文权威规范
[STAGE2_INSTANCE_MODEL.md](STAGE2_INSTANCE_MODEL.md)。代码字段、公开引用和
最终 schema 以英文版为准。本版用于逐步审阅数据逻辑，已移除旧版的 40--100 km
catchment、1.5N 候选池、2:1 depot 权重、lognormal community activity 和
先选 CS 后选 customer 等规则。

## 1. 研究边界

本数据集应描述为“真实地理 + 公开数据整合 + 运营层半合成”。CLE 提供真实城市
边界、OSM 定向路网、潜在住宅服务地点、depot 候选、AFDC 充电站和道路速度
证据。Stage 2 决定某一天激活哪些地点，并给它们匹配真实 Amazon 订单模板。

生成结果不是对某条 Amazon 路线的复原。Amazon 匿名坐标不会被搬到目标城市；
我们只迁移 depot-day 的路线结构、package volume、planned service time 和 TW。

## 2. 从零开始的执行顺序

1. 用三个 Amazon `model_build_inputs` JSON 构造紧凑模板库；
2. 用 Census Block Group × road SCC 冻结 train/held-out customer pool；
3. 预先安排 parent family、weekday/weekend 和子 view；
4. 选择一个真实物理 depot group 及其 access point；
5. 用 Amazon 的 `T_env` 和电池充分条件构造可行 territory；
6. 按 Amazon route × depot-time decile 结构精确激活 N 个 customer；
7. 根据已激活区域选择固定数量的真实 AFDC CS；
8. 选择 CLE weekday 或 weekend 静态路况，计算四张 parent matrix；
9. 用二分图 covering 给每个 customer 匹配不同的 Amazon order template；
10. 构造嵌套 view、充电 cache 和单顾客可行性 certificate；
11. 验证 family，并保存每个 family 和全 corpus 的 Phase-1 指标。

## 3. Amazon 模板

预处理单位是 station × date × day type。空间结构与订单模板分开处理：

- 空间结构保留 source route、route stop count、depot-to-stop time decile 和
  stop-pair travel-time 统计；
- 订单模板把一个 Amazon stop 的 package 数、总体积、总 planned service time
  和一个 TW 合成一条记录。

若单个 station-day 足以支持 N，则标签为 `SINGLE_STRUCTURE_DAY` /
`SINGLE_ORDER_DAY`。不足时，只允许合并同一 station、同一 day type 的多天，
标签为 `SAME_STATION_*_COMPOSITE`。不允许跨 station 合并，也不复制 template。

## 4. Community、split 和 adjacency

```text
community_id = Census Block Group × directed-road SCC
```

Block Group 提供稳定的统计地理单元；road SCC 防止把同一 polygon 内定向上不可
互达的服务点混成一个 routing community。完整 community 进入 train pool 或
held-out pool，不能按单个 building 随机拆分。

`community_adjacency.parquet` 来自真实 OSM 跨 community 的定向 edge，并使用
CLE road time 作 cost。即使某个 transit-only community 没有 customer，也保留
它作为两个住宅 community 之间的路网通道。

## 5. Family 计划和数据划分

每个 city × cohort 使用 largest-remainder 精确分配 5:2 weekday/weekend，再做
seeded shuffle；retry 不允许改变 slot 的 day type。

- Test-1：十城 train pool 的新 seed；
- Test-2：十城完整 held-out communities；
- Test-3：训练未使用的 Jacksonville；
- Cus2000：同城 unseen-scale；
- Cus50：传统小规模 / budgeted MIP compatibility。

训练规模仍为每个 scale 五百万 customer exposures：Cus50 100k、Cus100 50k、
Cus500 10k、Cus1000 5k。CS 数分别固定为 10/20/50/50。

Cus1000 parent 被精确拆成 20 个互斥 Cus50 leaf；固定组合形成 Cus100 和 Cus500。
leaf union 必须等于 parent。Cus2000 family 另存一个同 family 的 Cus1000 control。

## 6. Depot

Stage 1 先把属于同一真实物流设施的 access points 归成 physical facility group：
可依据共同 logistics/industrial land-use containment，或同名/同 operator 且同地址
或几何接触。仅仅距离近不能合并；没有可靠匹配的保持 singleton。

每个 family 先等概率选一个 facility group，再在组内优先用 Tier A，否则用合格
Tier B。Tier C 的 UPS Store/parcel shop 不进入 canonical depot pool。面积保留为
证据，但不再设置无法解释的 `1000 m2` hard gate。

## 7. Territory

选定 depot 后，在当前 day type 的 CLE 参考路况上一次性路由到所有潜在地点。
候选 customer 必须：

- 属于本 family 的 split pool；
- connector 合法；
- depot travel time 不超过 Amazon-derived `T_env`；
- depot→customer→depot 的最快路径能耗不超过一块满电电池。

最后一项是充分条件，不是假设完整多点 route 不需要充电。它只保证每个 customer
单独服务时一定可回库。若 territory 少于 N，直接拒绝本次 family attempt；不再
用拍脑袋的 40/50/.../100 km 半径扩大。

在 `T_env` 和能耗筛选以前，Stage 2 使用与最终 canonical 路由一致的 directed
node graph 与 zero-turn line graph，分别检查 depot→terminal 和 terminal→depot。
不可达 customer 会写入确定性 quarantine ledger 并从 roster 移除，而不是让一个
固定坏点使整个 family 重试。Stage 1 还必须声明
`directed_projection_roundtrip_v2`；旧 CLE 会被 reader 拒绝。

## 8. Step 6：空间激活

### 8.1 精确 quota

从 Amazon structure source 得到 route × depot-time-decile count。若总数大于 N，
先同比例缩小；再用 controlled matrix rounding 同时保持 route row margin、decile
column margin 和总数 N。不能逐 cell 独立 round。

### 8.2 region seed

每个 round 后仍为正数的 Amazon route row 对应一个目标 delivery region。第一个
seed 从合适 decile 选；后续 seed 最大化它到已有 seeds 的最小“对称化路网时间”，
避免所有 region 堆在城市同一角落。目标 decile 无容量时只按“先 b-1、后 b+1”
fallback，并记录次数。

### 8.3 road-adjacent growth 与 global assignment

各 region 轮流沿 `community_adjacency` 扩展。最终不是各 region 独立抽 customer，
而是用 global min-cost flow 同时满足所有 region-decile cell。一个 customer ID 在
整个 parent 里最多出现一次。若多个 region 竞争同一批 customer 导致 flow 不可行，
受影响 region 继续向相邻 community 扩展并重试；扩展次数写入指标。

成功结果必须同时满足：恰好 N 个 customer、全局唯一、route/decile margin 精确、
所有 view union/disjoint/size 精确。失败只能记录并 deterministic retry，不能修改
已经接受的 sample。

### 8.4 radial baseline

同一 territory 另外生成一个 size-matched radial baseline。它匹配 depot-time target，
但不要求 community contiguous growth。它不作为 benchmark instance，只用来检查
复杂的 spatial proposal 是否真的比简单 radial sampling 更合理。

## 9. Charging station

CS 在 customer geography 确定以后选。只能使用 CLE 内真实且兼容的 AFDC 站点，
不插入 synthetic CS。selection 先满足 energy core；由于当前 territory 已使用
direct-roundtrip 充分筛选，core 通常为空。剩余名额覆盖 active communities、
depot-to-region corridor 和覆盖较差的区域。

所有 charger 候选同样先通过 depot 双向 node/turn topology preflight；坏点进入
quarantine ledger。过滤后的 roster 才进入 energy communicating-set 和 relevance
fill。最终选中的 depot、customers、chargers 仍必须通过完整 all-pair closure。

因此 CS 既与配送区域相关，也保留“沿线可补能”的优化意义。固定 CS 数是 benchmark
shape 合同，不代表每个站都会被最优路线访问。

```text
p_battery = 0.90 * min(AFDC reported power 或冻结的全国 connector-mode median,
                       vehicle mode cap)
```

缺失功率使用冻结的全国同 connector-mode median（J1772 L2 6.5 kW、CCS DC
200 kW）；若注册表缺少相应 mode，生成直接报错，禁止退回 vehicle cap。DC cap 为
100 kW，L2/AC cap 为 11 kW。0.90 是 benchmark derating factor，不表述为充电效率。

## 10. 路况、matrix、order 和可行性

Stage 2 直接选择 CLE 的 weekday 或 weekend reference-speed column，不再添加没有
独立依据的 edge 随机 multiplier。非对称性来自单行道、不同方向限速和不同路径。
canonical turn time 为 0；虚拟 access-connector split node 禁止 immediate edge
reversal。3/8/20 秒 geometry adapter 仅测试，不生成 canonical matrix。connector
是双向，但不会修改 OSM physical edge。

parent 保存：最短距离、该路径的 zero-turn 耗时、zero-turn 最快耗时、最快路径距离。
能耗由路径距离乘 `100/257 kWh/km` 得到，不重复存 matrix。

matrix 得到以后才附着 Amazon order。customer-template edge 只有在 volume、TW、
service 和返回 horizon 都可行时存在。maximum bipartite matching 必须覆盖所有
customer；不复用模板，也不移动/放宽 TW。Cus100/500/1000 只能换 single-day source
ID，不能降级 composite；composite 仅用于 report-only Cus2000。每个 customer 保存
template ID、station-day ID 和 source mode。

每个 view 还保存“从每个 CS 满电出发，允许经其它 CS 多跳，最快回 depot”的时间
cache。它是静态 instance data，不是 runtime action mask。

## 11. Phase-1 指标

每个 family 都保存三份文件，全 corpus 另有 aggregation report。

**Hard gates**：N 精确、customer 全局唯一、split pool 明确、route/decile margins
精确、view union/disjoint/child size 精确。任何一个失败都拒绝 family。

**Realism diagnostics**：

- M1：生成 depot-time 与 Amazon target 的 normalized Wasserstein-1，并和 radial
  baseline 比较；
- M2：路网上 nearest-neighbour time 与 Amazon reference 比较；
- M3：region 内 pairwise time P50/P90 与 source route 比较；
- M4：region count/size 和 controlled-rounding audit；
- M5：community count、largest share、HHI，并和 radial baseline 并列。

**Reliability**：territory reserve ratio、energy-screen removal share、decile fallback、
region redraw、community growth、assignment competition expansion、first-attempt success、
conditional attempt success、rejection reason。统计单位是 attempted parent family，失败
attempt 也保留，避免只报告成功 family 的 survivorship bias。

V2.1 已冻结 D-5：primary Cus100/Cus500/Cus1000 strata 的每个 M2/M3 component
必须满足 generated-to-holdout Q90 不大于 real-to-real Q90；缺少 primary support
也失败。Cus50 是单独 compatibility gate，Cus2000/composite report-only。
station-block bootstrap confidence interval 只报告，不改变该 gate。M1、M4、M5
按签字版规定的角色报告，不得在看到结果后修改阈值。
