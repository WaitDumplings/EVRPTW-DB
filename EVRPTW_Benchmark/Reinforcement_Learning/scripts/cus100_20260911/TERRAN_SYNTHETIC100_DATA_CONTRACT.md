# TERRAN-derived synthetic Cus100 数据合同

日期：2026-09-11。适用范围只有本轮五方法（AM-EVRPTW、DRL-TS、TERRAN、RRNCO、EVRPTW-RL）的 Euclid-100 训练与合成 validation。Road-100 仍使用冻结 EVRPTW-D。两种训练数据不是配对母实例；本轮没有生成或启动 Cus500/1000 数据或任务。

## 冻结来源、正式数据与状态

- 上游：`git@github.com:NanpengYu/TERRAN.git`，固定 commit `926665870d08e24b1acea33a57f1412c96376758`。
- 未修改的本机克隆：`EVRPTW_Benchmark/results/terran_generator_audit_20260911/`。
- 原配置：上游 `configs/config_100c.json`。直接以绝对配置路径实例化 `Solomon_EVRPTW_Generation` 并顺序调用 `_generate_instances()`；不使用仍硬编码 `./config.json` 的 `prepare_and_save_dataset()`，不调用用于转换 Solomon 文本的默认 CLI。
- canonical adapter：`common/terran_synthetic.py`，版本 `terran_synthetic_canonical_xy_v1`。
- 生成器修复模式：`canonical_candidate_feasibility_v4`。这是带公开必要可行性修复的 **TERRAN-derived** 数据，不是未改动的 TERRAN 原样生成分布或原论文复现。
- 正式数据根：`EVRPTW_Dataset/TERRAN_synthetic100_feasible4_20260911/`。训练目标 50,000 实例，validation 500 实例；每例 100 客户、20 充电站、1 仓库，共 121 节点。
- 根目录 `corpus_manifest.json` 是完成状态与所有最终 hash 的权威记录。只有 `complete=true` 且 `formal_training_authorized=true`、train=50,000、val=500 时才允许正式启动。
- `train_smoke/` 仅包含正式顺序流的前 128 例，用于显存和接口调试；其 index sidecar 明确 `complete=false, usage=smoke_only`。不能把该子集当成正式训练支持集。
- 早期失败、provisional、原始坏字典均保留在 `results/terran_*` 中；它们不属于正式数据，也不允许部署器拿来代替正式支持集。

## 原生执行行为与必要补丁

保留的上游实际行为：

| 项目 | 配置声明 | 本轮实际执行 |
| --- | --- | --- |
| R / C / RC 抽样权重 | 0.3 / 0.3 / 0.4 | 保留原权重与 Python random 类型选择 |
| RC 随机客户比例 | `mix_random_ratio=0.3` | 代码读取另一字段 `rs_random_ratio`，实际缺省 0.5；本轮保留 0.5 |
| TW 比例 | 候选比例 [0.25,0.5,0.75,1]，权重 [1,1,1,4] | 原函数返回索引 0..3，并在对象构造时只抽一次；本轮 train/val 对象均得到 3，概率比较因此不会触发去除 TW |
| 初始位置范围 | 均匀候选 [0,100]² | 聚簇 Gaussian 客户可落到该方框外；不裁剪，不伪造经纬度 |
| 容量、电池、充电、服务、horizon | 随 R1/R2/C1/C2/RC1/RC2 改变 | 保留原值，按下列单位合同转换 |

实际执行未改动上游时发现的额外错误：完整 val500 中 **42 例（8.4%）** 有 ready > due（共 92 客户）；训练前 1000 例有 87 例（8.7%，180 客户）。例如原 val index 30 / RC1，horizon=240，却生成 TW=[276.20415657468163,240]。原始报告与坏字典保存在 `results/terran_generator_actual_audit_20260911/raw_audit.json` 及同目录 `*_bad_*.pkl`。

只添加 ready <= due 检查仍不够：单独 `twguard1` provisional 中 train1000 有 92 例、val500 有 41 例至少一个客户即使忽略充电也无法等待服务后按时回仓。证据：`results/terran_synthetic_provisional_twguard1_20260911/reachability_audit.json`。上游的回仓检查使用 `2*travel+service`，没有包含实际 TW 等待。

最终补丁仅应用到复制进正式 corpus/provenance 的生成文件，不改用户的其他 TERRAN 仓库，也不修改 canonical 环境或五种模型架构：

1. 在两个聚簇候选 TW 生成点加入物理可行性 guard，保留已有原生接受条件。guard 同时检查 TW、容量以外的电池和旅行、等待服务、充电、horizon 内返仓。
2. 见证类型是 `depot → [CS a] → customer → [CS b] → depot`。a/b 可省略，但同时存在时必须不同，符合 canonical 每车每站至多一次的路线规则。没有放宽环境去允许上游使用的前后重复同一站。
3. non-RS-based 聚簇函数先对 centroid 做同样的宽 TW 可达性预检，避免为拓扑上不可用的固定中心反复抽客户。
4. 该函数每个未完成 cluster 最多抽 10,000 个候选；若超限，回滚仅这个 cluster 已追加的客户/TW，然后重新抽中心。保留此前 cluster、实例类型与充电站。公开记录 candidate rejection、centroid rejection/restart 计数，避免原候选循环无界卡死。
5. 每例转换为 float32 canonical 字段后，重新构造其全部 100 条独立单客户见证，由独立 route validator 全部重放。任何失败立即停止整次生成并写失败原始字典；**不丢弃、重采样整实例或根据学习模型成功率筛选数据**。

完整 unified diff 为正式 corpus 中 `provenance/canonical_candidate_feasibility_v4.patch`。原配置和有效配置逐字相同；所有差异仅在该补丁与 adapter 中公开。

## 单位与字段映射

这是明确选定的 synthetic 物理解释；原 Solomon 单位本身不是经过测量的公里、分钟、体积或 kWh。换算保持上游的时间、负载、电池及充电相对可行性；生成后只使用下表的 canonical 数据。

| 量 | canonical 映射 |
| --- | --- |
| 位置 | 原归一化 XY × `pos_scale=100` × 1 km；明确 `coordinate_system=synthetic_cartesian_xy_km` |
| 距离 D | 上述平面 XY 的 L2 距离，单位 km；不调用 Haversine、路网 snap 或旧 `common/euclidean.py` |
| 时间 T | 原距离 / 原 `velocity_base` × 60 秒；本配置 velocity=1，等效速度 60 km/h |
| TW / service / horizon | 原归一化字段先乘原 max_time 恢复，再乘 60 秒；horizon 直接原 max_time × 60 秒 |
| 能量 E | 原旅行时间 × 原 `energy_consumption` × (100/257) kWh；本配置恰为 D × 0.38910505836575876 kWh/km |
| 电池 | 原 battery_capacity × (100/257) kWh；不是强行套用固定 100 kWh 电池 |
| cargo | canonical 等效载荷容量统一为 18,500,000 cm³ |
| 客户需求 | 原 normalized demand × 18,500,000 cm³；每类型原 demand_capacity 同时保留，对应 cm³/原单位为 18,500,000 / demand_capacity，负载占比完全不变 |
| 充电 | 上游输出的 charging_rate 已经是 inverse_charging_rate 的倒数；P=charging_rate × (100/257) × 3600/60 kW，所有站同实例同速率，derating=1 |
| 满充时长 | 原 battery_capacity / 原 charging_rate × 60 秒；到站补满，实际时间按缺失电量/P 计算 |
| package_counts | synthetic schema 占位每客户为 1；不声称来自真实包裹数量，容量约束以需求体积为准 |

原节点顺序为 `[depot, 20 CS, 100 customers]`；canonical 是 `[depot, 100 customers, 20 CS]`。保存两个方向的节点映射，并一致重排坐标、TW、需求及全部 D/T/E 矩阵。

实际原类型时间单位保留：service 为 600 或 5400 秒，horizon 在 13,800–203,400 秒之间；这可能跨越真实工作日，属于选择的 synthetic 时间尺度，不伪称 EVRPTW-D 实际班次。电池范围约 24.179–106.276 kWh。`region_id`/`city_slug` 使用明确 sentinel `terran_synthetic`；真实 city、operational day 字段为 null/显式 synthetic 标识。

D、T、E、charging power 一次写入 canonical。环境转移、奖励、图关系输入、独立 verifier 和缓存读取同一套数组。正式 USD 目标不继承上游 TERRAN 的 reward 或旧费用：

`C = 0.151750972762646 * D_km + 413.6331536717643 * vehicles_started`

其来源是 `configs/rivian_energy_vehicle_cost_v2.json`。reward scale 对整个 C 做正比例缩放；不能通过分别改距离/车辆系数弥补 synthetic 的较大车辆数。E 的训练参考尺度由单独 train-only 有界校准确定，不读 validation/test 拟合。

## RNG、ID、存储与跨机复现

- 训练 seed 为 1234。数据 seed 派生字符串：`terran_synthetic_canonical_xy_v1:base_seed=1234:split={train|val}`；SHA256 前四字节按 little-endian 解析为 uint32。
- train 数据 seed=2311891958，val=3307617993。每个 split 在对象构造之前调用完整 Python random、NumPy 和 Torch seed；每个 split 只有一个 generator 对象，连续顺序抽取，不由五个训练进程各自在线生成。
- 保存每 split 的完整 CPU RNG 前/后状态，源软件版本、原始字典与转换版本。生成以 `CUDA_VISIBLE_DEVICES=''` 执行，不占训练 GPU。
- ID 为 `terran100-feasible4-seed1234-{split}-{six_digit_index}`。ID 与原始/转换后字节 hash 固定；索引和所有 shard 路径相对 corpus 根存储，可原样复制三机。
- `train/view_index.parquet` 的 split/track 均是 train；`val/view_index.parquet` 的 split=val、track=validation，兼容统一 validation filter。
- raw 和 canonical 使用每 1000 例一个顺序 pickle shard；index 保存每条 record 的相对路径、offset、length、SHA256。loader 在读记录时重新校验 hash，在读 index 时校验 manifest 中的 index hash。
- root manifest 保存全部源文件、config、patch、adapter/helper、index 和数据 shard hash；train/val ID 和 raw 内容 hash 无交集。五模型使用共同冻结 ID stream，按各自 batch 仅改变分组，不改变相同 ID 指代的数据。

可重生成命令（新输出目录必须为空，不会覆盖既有语料）：

```bash
cd /data/Maojie/ICLR/EVRPTW-DB
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/npg/miniconda3/envs/maojie/bin/python \
  -m EVRPTW_Benchmark.Reinforcement_Learning.scripts.generate_terran_synthetic100 generate \
  --upstream-root EVRPTW_Benchmark/results/terran_generator_audit_20260911 \
  --output EVRPTW_Dataset/TERRAN_synthetic100_feasible4_20260911 \
  --train-count 50000 --val-count 500 \
  --distribution-mode canonical_candidate_feasibility_v4
```

固定原 commit 后，上述命令使用新的空目录可重生成相同 raw/canonical 记录。运行时间和绝对输出路径等运行元数据不属于数据内容相同的判断；应核对 per-record / per-shard / index hash。

## 验证证据

- 未修复原始来源非法 TW 比例及字典：`results/terran_generator_actual_audit_20260911/`。
- 仅倒置修复仍等待后无解的反例：`results/terran_synthetic_provisional_twguard1_20260911/reachability_audit.json`。
- 最终 v4 preliminary train1000+val500：全部结构及 150,000 客户 singleton 独立重放通过，无整实例筛选；train1000 9.60 秒，val500 4.94 秒。
- 正式数据 train 前128 + val 前128：完整 RNG 独立重生成 raw hash 全一致；D/T/E、TW/service、需求比例、双向节点顺序全通过往返核验。
- R1/R2/C1/C2/RC1/RC2 **各3例，共18例、1800条完整单客户路线**：canonical fast env 逐步 action mask 可执行，全部正常完成；环境与两个独立 verifier 的 D/K/C 一致。
- 三项回归测试（非法 TW、等待后回仓、仅重复 CS 的伪见证）通过：`common/tests/test_terran_synthetic_feasibility.py`，`3 passed`。
- 上述增强审计：`results/terran_synthetic_feasible4_unit_rng_replay_18cases_20260911.json`，脚本为 `scripts/audit_terran_synthetic100.py`。
- 正式生成逐例校验所有 100 客户的可行性见证；最终数量、类型计数、候选拒绝/中心重试统计与文件 hash 以正式 root manifest 的完整快照为准。

<!-- FINAL_CORPUS_SNAPSHOT -->
## 最终完成快照

正式生成已完成：train=50,000，val=500；50,500 个 raw 内容 hash 唯一，train/val 的 ID 与 raw 内容交集均为 0。全部 5,050,000 客户单路线见证在生成时独立重放通过。占用 8.939 GiB。

| split | R1 | R2 | C1 | C2 | RC1 | RC2 | candidate拒绝 | centroid拒绝 | centroid重试 | 整实例过滤/重抽 | 生成秒数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 7627 | 7641 | 7376 | 7621 | 9747 | 9988 | 40256 | 651 | 0 | 0 | 505.94 |
| val | 70 | 79 | 76 | 64 | 103 | 108 | 404 | 8 | 0 | 0 | 4.96 |

候选/中心重试属于公开的生成器逻辑修复，不能被解读成未改原分布；整实例过滤/重抽数单独为零。

- corpus manifest SHA256：`445150e84a3d81516f66ef3551cb3ce7a550ff258e3f92a924850970a5856d22`
- train view index SHA256：`3b539bb493925deaff04460bdc1ef6e541892edcd12bc23872812d7d02d955ab`
- val view index SHA256：`183600f24998469520876bf29a8875b05c2324517273cf959e69fb45412fe75b`
- provenance `canonical_candidate_feasibility_v4.patch` SHA256：`40805fdc11a6f2bca4d772bd7bd2138705108d4fc7d7c35a9a51d1017839eb59`
- provenance `effective_config_100c.json` SHA256：`47d5751eeb0dc8ff9ad7cae6ebb9d69628484e736b033daec521b3b0c92b0e54`
- provenance `instance_generator.py` SHA256：`9e86effc75d43e4a3e189fc541d1dcd218cb7dbdb84c3aaf20c77ab45e5039fd`
- provenance `terran_synthetic_adapter.py` SHA256：`9bd0198bdb717d05398f4b38a264953efde06bba5fcc109af80bce1adfbec597`
- provenance `terran_synthetic_feasibility.py` SHA256：`c3cd946b2aacefb0de08c89de9da50fa34dc016360305695ca83030aee4ce075`
- provenance `upstream_config_100c.json` SHA256：`47d5751eeb0dc8ff9ad7cae6ebb9d69628484e736b033daec521b3b0c92b0e54`

完成审计：`results/terran_synthetic_feasible4_completion_audit_20260911.json`。源码 adapter/helper 与冻结 provenance hash 仍一致；最后一个实例的累计候选计数与 manifest 完全一致。
