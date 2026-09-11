# Cus100 本轮范围与启动预检

日期：2026-09-11。本文描述冻结的实验范围与检查规则；运行状态以各服务器生成的 JSON 为准。实际显存、smoke 证据和预计耗时见本轮 `CUS100_SMOKE_REPORT.md` 及对应测量结果。

## 范围与逐卡分配

仅训练五个适配 DRL 模型的 Road Cus100 和 TERRAN-derived synthetic Euclidean Cus100，共十项，训练 seed 均为 1234。旧结果不续训、不改名复用。本启动器不启动 Cus50、Cus500、Cus1000、非学习正式实验或机制消融，也不自动启动测试。最终 checkpoint 冻结后的测试编号保留在注册表，目标统一为 Road Cus100/T1。

| 服务器角色 | 物理 GPU | 训练编号 | 方法 | 训练源 / validation 源 | physical = effective batch |
|---|---:|---|---|---|---:|
| 2080ti_4_1 | 0 | TR02 | AM-EVRPTW | Road / Road | 108 |
| 2080ti_4_1 | 1 | TR01 | AM-EVRPTW | synthetic E / synthetic E | 108 |
| 2080ti_4_1 | 2 | TR06 | TERRAN | Road / Road | 384 |
| 2080ti_4_1 | 3 | TR05 | TERRAN | synthetic E / synthetic E | 384 |
| 2080ti_4_2 | 0 | TR04 | DRL-TS | Road / Road | 24 |
| 2080ti_4_2 | 1 | TR03 | DRL-TS | synthetic E / synthetic E | 24 |
| 2080ti_4_2 | 2 | TR18 | EVRPTW-RL | Road / Road | 200 |
| 2080ti_4_2 | 3 | TR17 | EVRPTW-RL | synthetic E / synthetic E | 200 |
| 2080ti_3_1 | 0 | TR10 | RRNCO | Road / Road | 50 |
| 2080ti_3_1 | 1 | TR09 | RRNCO | synthetic E / synthetic E | 50 |
| 2080ti_3_1 | 2 | — | 预留 | 无正式任务 | — |

最终可执行参数以 `cus100_seed1234_jobs.jsonl` 为准；manifest 中 `enabled` 和 `calibration_status` 控制是否通过正式启动检查。上表配置本身不代表训练已启动。

## 已核查的本机身份

仅本机角色 `2080ti_4_1` 已实测。真实 hostname 为 `npg-To-Be-Filled-By-O-E-M`，四张卡均为 NVIDIA GeForce RTX 2080 Ti：

| 物理 GPU | 实际 UUID |
|---:|---|
| 0 | GPU-03349f36-2776-81f9-0166-75926041cbe3 |
| 1 | GPU-2b152219-8d5f-2a1c-c230-df45655d433a |
| 2 | GPU-1bd5dcf4-92cf-fc56-28f8-5b10c52af9e5 |
| 3 | GPU-90987749-43d4-9e85-7ac5-1ab6e4586554 |

`2080ti_4_2` 和 `2080ti_3_1` 是部署角色名称，其 hostname、UUID、驱动和当前占用未被远程核查。用户在对应机器手动启动时，preflight 会读取真实设备信息并保存。每项训练只绑定一张实际 GPU 的 UUID；不会通过 DDP 自动占用其他卡。每项任务使用 2 个 CPU 线程，设置 OMP/MKL/OPENBLAS/NUMBA 为 2。

显存目标为单进程实测峰值 9.5–10.3 GiB；显存口径及完整测量见 smoke 报告。相同卡型号不等于相同主机 CPU、I/O 或驱动环境，运行时会另存实际 Python、torch、numpy、pandas、numba、pyarrow 和驱动版本。

## 数据与目标

- Road root：`EVRPTW_Dataset/Instances_v2/us_11city_full_clean_v7_bbde5db_20260823/`。训练为 `generation_plan/core/train/view_index.parquet` 中 50,000 个 Cus100 views、5,000 families；validation 为 `generation_plan/core/val/view_index.parquet` 中 500 个 Cus100 views、500 families。训练/验证 view 与 family 的交集均为零。
- Synthetic root：`EVRPTW_Dataset/TERRAN_synthetic100_feasible4_20260911/`。独立 `train/view_index.parquet` 50,000 个实例与 `val/view_index.parquet` 500 个实例，五个 E 模型共享同一冻结语料。生成器、修复、单位和完整校验见 `TERRAN_SYNTHETIC100_DATA_CONTRACT.md`。
- 本阶段不使用 Road test 实例调参、校准、排错或选模。E 用独立 synthetic E validation 选模，G 用冻结 Road validation 选模。

共同正式目标为：

```text
C_USD = 0.151750972762646 × D_km + 413.6331536717643 × K
```

其中 D 包含实际行驶的全部路段、充电绕行与最后返仓；K 为有效 depot→非 depot 出发次数。完整目标配置为 `configs/rivian_energy_vehicle_cost_v2.json`。E/G 不改变目标系数，只用各自训练域的单一正标量缩放整个 C：

| 训练域 | 共享 reward contract | S：完整 C 的归一标量 | 终止失败 base | 未服务比例系数 |
|---|---|---:|---:|---:|
| Road G | `configs/drl_reward_contract_energy_vehicle_v3.json` | 432.7147451224062 | 4.913082471598237 | 1.0 |
| Synthetic E | `results/cus100_20260911/artifacts/reward_synthetic_feasible4/reward_contract.json` | 6530.963114223516 | 3.371478213719905 | 1.0 |

上表简写的 `configs/` 位于 `EVRPTW_Benchmark/Reinforcement_Learning/`，`results/` 位于 `EVRPTW_Benchmark/`。五个 E 模型完全共享 E 合同，五个 G 模型共享原 G 合同。E 标定由完整 50,000 实例训练池按独立 SHA256 命名空间固定选择 500 个实例，运行确定性 singleton-best-fit 构造并经两套独立 verifier 检查；不运行随机 ALNS 搜索，不读取 validation/test。S 为 reference C 的中位数，失败 base 为 `q99_linear(C/S)+1`；这是训练参考标定，不声称它构成每个极端实例的通用可行性保证。

## 训练预算与暴露量口径

训练与 validation 的候选数均为 30；训练 horizon 为 240，validation horizon 为 360。每 100 logical epochs 在完整 500 个 validation 实例上选模。最低预算 5,000，最高 10,000；完成第 5,000 epoch 后才开始累计连续五次 validation 无改善的提前停止计数。因此正常提前停止最早为第 5,500 epoch；未满足条件则继续至预算上限。错误退出不等于正常提前停止。

physical batch 等于 effective batch，因此不存在为达到上表 effective batch 而额外累积多个物理批次。TERRAN 保留原生 PPO 的 3 次 update epochs × 4 个 minibatches，每个 logical epoch 执行 12 次 optimizer updates；另外四个方法每个 logical epoch 执行 1 次 optimizer update。TERRAN 的 12 次 PPO 更新复用同一批 rollout，不产生 12 倍的新样本 ID。

| 方法（每个 E/G 配置分别计） | 主 stream 每 epoch 实例 | 5,000 epochs 主 stream 实例 / 客户暴露 | 10,000 epochs 主 stream 实例 / 客户暴露 | 10,000 epochs optimizer updates |
|---|---:|---:|---:|---:|
| AM-EVRPTW | 108 | 540,000 / 54,000,000 | 1,080,000 / 108,000,000 | 10,000 |
| TERRAN | 384 | 1,920,000 / 192,000,000 | 3,840,000 / 384,000,000 | 120,000 |
| DRL-TS | 24 | 120,000 / 12,000,000 | 240,000 / 24,000,000 | 10,000 |
| RRNCO | 50 | 250,000 / 25,000,000 | 500,000 / 50,000,000 | 10,000 |
| EVRPTW-RL | 200 | 1,000,000 / 100,000,000 | 2,000,000 / 200,000,000 | 10,000 |

表中是主训练 ID stream 暴露；每个实例另采样 30 条训练轨迹。五个方法共享同一 source/seed 的完整池打乱循环，但按不同预算消费不同长度前缀。相同长度复用同一物化文件，不同长度逐 ID 与前缀摘要验证一致。因此本配置匹配的是 logical epoch 上限与轨迹数，不是完全相同的总样本暴露量、optimizer updates 或 GPU 时间。

原生 greedy baseline 的额外前向与 probe 数据读取单独计量，不计入上述主 stream 预算。公共训练器的固定/替换 probe 池只来自训练集；`baseline_history.jsonl` 记录 `probe_instances`、`optimizer_step`、是否更新、阶段和调度来源，`training_result.json` / checkpoint 记录 `baseline_eval_count`、`baseline_update_count`。每次配对 probe 包含 actor 与 baseline 的两组前向；替换 probe 池也属于额外训练域访问。RRNCO 使用 leave-one-out 时不建立此独立 probe 池。不能把主 stream 计数描述为包括所有 baseline 访问的总读取量。DRL-TS 保留每 250 optimizer updates 的 native-adapter baseline 比较，并区分 soft/hard 阶段。

## 启动、来源与运行证据

三机都使用本目录对应角色的 `full.sh`，不调用旧全量队列。预检要求校准通过、输出目录为空、目标 GPU 无其他计算进程、语料完整、数据/配置/stream hash 一致。E 还要求 `corpus_manifest.json` 中 `complete=true` 且 `formal_training_authorized=true`，并将 reward 标定绑定到准确的 corpus 与 train-index 摘要。

每次启动保存当前 tracked 文件实际内容和新增 untracked 源码的完整快照，以内容及 Git 风格执行位计算 `source_version`；Git HEAD 仅作参考。组写权限或 umask 变化不改变源码身份。各进程的 `launch_record.json` 记录 source version、manifest hash、真实 hostname/GPU UUID 和命令，实际进度及结束状态见 `CUS100_RUN_REGISTRY.md` 指向的 JSON。launcher 启动成功不代表模型训练完成；完成检查还要求有效 `training_result.json`、预算范围和已保存的 best checkpoint。

正式训练启动后，这些源码和静态交付文档保持冻结；后续状态只写运行产物。
