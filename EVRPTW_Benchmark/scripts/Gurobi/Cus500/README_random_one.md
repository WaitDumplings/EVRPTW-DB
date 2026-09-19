# 单个随机 Road Cus500：Gurobi 默认参数与实时日志

从指定 Road test cohort 的 500 个 Cus500 实例中等概率抽取 **一个**，默认 T1。
只创建一个模型，直接调用 Gurobi；不启动多实例进程池，不设置单线程限制。

在仓库根目录运行：

```bash
conda activate maojie
bash EVRPTW_Benchmark/scripts/Gurobi/Cus500/random_one.sh
```

这是前台运行。默认不设置 `Threads`、`ThreadLimit`、`MIPGap`、`MIPGapAbs`
或 `TimeLimit`，由 Gurobi 使用其默认/环境参数。本机 Gurobi 13.0.2 验证为
`Threads=0`（自动）、`MIPGap=0.0001`（0.01%）、`TimeLimit=Infinity`。
自动线程数由 Gurobi 决定，并不承诺占满全部逻辑 CPU。完整有效参数记录在
`run_config.json`，不会将 `OPTIMAL` 状态对应的非零残余 gap 改成零。
参见 [Gurobi 参数文档](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html)。

## 可选参数

```bash
# 仅选样、检查真实数据并输出配置，不构建或优化模型。
bash EVRPTW_Benchmark/scripts/Gurobi/Cus500/random_one.sh --dry_run

# 固定抽样，仍使用默认 Gurobi 参数。
bash EVRPTW_Benchmark/scripts/Gurobi/Cus500/random_one.sh --selection_seed 20260919

# 如需限制优化阶段时间，显式追加此参数；默认没有这个限制。
bash EVRPTW_Benchmark/scripts/Gurobi/Cus500/random_one.sh --time_limit_s 1800

# 指定其它发布目录、track 和全新输出目录。
bash EVRPTW_Benchmark/scripts/Gurobi/Cus500/random_one.sh \
  --dataset_root /path/to/us_11city \
  --track T2 --selection_seed 1234 --output_dir /data/gurobi_cus500_one_run
```

`--selection_seed` 只控制抽到哪个实例，不改 Gurobi Seed。未指定时随机产生种子并保存。
`--output_dir` 必须是新目录，避免覆盖原实验。`--cs_copies` 默认 2：每个物理充电站
在全解中最多使用两个副本，这是模型限制。解释器可由 `PYTHON_BIN` 指定。
数据目录可由 `--dataset_root`、`EVRPTW_DATASET_ROOT`、`CUS100_ROAD_ROOT` 指定；
无覆盖时使用仓库内 `us_11city_full_clean_v7_bbde5db_20260823`。

## 目标与输出

新实验优化：

```text
C_USD = (0.39 × 100/257) × D_time_km + 413.6331536717643 × K
```

成本距离为 `running_time_path_distance_km`，时间为
`running_time_shortest_matrix_s`，电量为 `running_time_path_energy_kwh`。
解经独立重放后才标记 `verified_feasible`。原生 MIP objective 与 verified cost
分列记录，不能把没有可行解的 objective 或 gap 填成零。

启动时打印实际选中的实例 ID 和输出目录。默认保存到仓库下：

```text
EVRPTW_Benchmark/results/gurobi_single_Cus500_dtime_<UTC时间>_<PID>/
```

| 文件 | 内容 |
|---|---|
| `gurobi.log` | Gurobi 原生优化日志 |
| `progress.csv` / `progress.jsonl` | incumbent objective、bound、gap、Runtime 与 wall time 的完整精度数值 |
| `latest_progress.json` | 最近一次进度，原子替换，可在运行中查看 |
| `run_config.json` | 实例/随机种子、参数、成本来源、源文件哈希、CPU 信息、运行状态 |
| `summary.csv` / `summary.json` | 最终 objective、verified cost、gap、时间、车辆数与距离 |
| `solution.pkl` / `solution.json` | 路线内容和独立重放诊断 |
| `error.json` | 异常时保留错误及 traceback |

`mip_gap` 是比例，`mip_gap_percent = 100 × mip_gap`。无 incumbent 时留空。
`candidate_objective_usd` 单独记录 MIPSOL 候选，不把更差候选误当成当前最好解。

`gurobi_runtime_s` 是 Gurobi 优化器 Runtime；`wall_runtime_s` 从模型构建前开始，
最终包含构模、优化及解后重放。二者均不含实例读取。可选 TimeLimit 只限制优化器
时间，不等于包含构模的总 wall-clock 预算。

**每次回调观测到 objective、bound 或 MIPGap 改变都立即写入 CSV 并 flush，
即使同一秒内更新多次也保留。** `--log_interval_s` 仅限制数值不变时的心跳行，
不限制变化记录；每次 MIPSOL 和最终状态也会写入。记录的是 Gurobi 回调可观测的
变化，无法承诺捕获两次回调之间未对外暴露的内部变化。

`progress.csv` 的核心列如下（其他列用于溯源）：

| 列 | 含义 |
|---|---|
| `gurobi_runtime_s` | 到这次更新为止，优化器已运行的秒数 |
| `wall_runtime_s` | 到这次更新为止，从开始构模计算的总秒数 |
| `mip_gap` | 当前相对 gap 比例，如 0.15 表示 15% |
| `mip_gap_percent` | 当前 gap 百分数，如 15.0 |
| `objective_value_usd` | 当前最好 incumbent 的美元目标值 |
| `best_bound_usd` | 当前目标下界 |

可以显式指定新的输出目录，便于直接找到文件：

```bash
bash EVRPTW_Benchmark/scripts/Gurobi/Cus500/random_one.sh \
  --output_dir /data/gurobi_cus500_one
# 另一个终端实时查看；该文件从启动后即持续写入。
tail -f /data/gurobi_cus500_one/progress.csv
```

这是事件驱动记录，不保证每秒都收到 Gurobi 回调；构模期间会先留下
`model_build_start`，模型完成后写 `model_ready`。Ctrl-C/SIGTERM 会请求 Gurobi
停止并在正常返回后保存已找到的 incumbent；强制 SIGKILL 无法生成最终汇总。

## 验证范围

已通过真实 Gurobi 小实例求解和回调日志测试；真实 Cus500 完成随机选择与数据
检查（dry-run）。准备脚本时没有启动完整 Cus500 优化。历史 DRL/搜索实验数据
没有被改写，本脚本输出属于新的 D_time 口径。
