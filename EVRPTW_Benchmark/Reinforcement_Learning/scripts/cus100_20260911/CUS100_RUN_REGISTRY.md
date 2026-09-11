# Cus100 本轮运行注册表

日期：2026-09-11。这是静态路径注册表，不是一份实时训练状态快照。十项任务是否开始、失败、提前停止或完成，以各服务器生成的 `launchers/<server>/status.json` 和每项 `launch_record.json` 为准；不存在这些文件时不得推断任务已经启动。ETA 和测量结果见本轮 smoke/耗时报告。

下表所有输出路径都相对于统一输出根目录：

```text
EVRPTW_Benchmark/results/cus100_20260911/
```

默认根目录可由 `CUS100_OUTPUT_ROOT` 覆盖；实际绝对路径会写入 `launch_record.json`。每项均为新训练，seed 1234。`best.ckpt` 是 validation 选出的 checkpoint；如训练器另保存 `best_overall.ckpt`，其选择与 epoch 以 `validation_summary.json` 为准。路径登记不代表文件已经生成。

| TR | 冻结后 Road Cus100/T1 的 EV | 服务器 / GPU | 方法 / 训练域 | 输出目录 | stdout / stderr | best 路径 | 实际状态来源 |
|---|---|---|---|---|---|---|---|
| TR02 | EV14 | 2080ti_4_1 / 0 | AM-EVRPTW / Road | `runs/TR02/` | `runs/TR02/stdout.log` / `stderr.log` | `runs/TR02/best.ckpt` | `runs/TR02/launch_record.json` |
| TR01 | EV10 | 2080ti_4_1 / 1 | AM-EVRPTW / synthetic E | `runs/TR01/` | `runs/TR01/stdout.log` / `stderr.log` | `runs/TR01/best.ckpt` | `runs/TR01/launch_record.json` |
| TR06 | EV30 | 2080ti_4_1 / 2 | TERRAN / Road | `runs/TR06/` | `runs/TR06/stdout.log` / `stderr.log` | `runs/TR06/best.ckpt` | `runs/TR06/launch_record.json` |
| TR05 | EV26 | 2080ti_4_1 / 3 | TERRAN / synthetic E | `runs/TR05/` | `runs/TR05/stdout.log` / `stderr.log` | `runs/TR05/best.ckpt` | `runs/TR05/launch_record.json` |
| TR04 | EV22 | 2080ti_4_2 / 0 | DRL-TS / Road | `runs/TR04/` | `runs/TR04/stdout.log` / `stderr.log` | `runs/TR04/best.ckpt` | `runs/TR04/launch_record.json` |
| TR03 | EV18 | 2080ti_4_2 / 1 | DRL-TS / synthetic E | `runs/TR03/` | `runs/TR03/stdout.log` / `stderr.log` | `runs/TR03/best.ckpt` | `runs/TR03/launch_record.json` |
| TR18 | EV78 | 2080ti_4_2 / 2 | EVRPTW-RL / Road | `runs/TR18/` | `runs/TR18/stdout.log` / `stderr.log` | `runs/TR18/best.ckpt` | `runs/TR18/launch_record.json` |
| TR17 | EV74 | 2080ti_4_2 / 3 | EVRPTW-RL / synthetic E | `runs/TR17/` | `runs/TR17/stdout.log` / `stderr.log` | `runs/TR17/best.ckpt` | `runs/TR17/launch_record.json` |
| TR10 | EV46 | 2080ti_3_1 / 0 | RRNCO / Road | `runs/TR10/` | `runs/TR10/stdout.log` / `stderr.log` | `runs/TR10/best.ckpt` | `runs/TR10/launch_record.json` |
| TR09 | EV42 | 2080ti_3_1 / 1 | RRNCO / synthetic E | `runs/TR09/` | `runs/TR09/stdout.log` / `stderr.log` | `runs/TR09/best.ckpt` | `runs/TR09/launch_record.json` |

表中的 `stderr.log` 与同一行 `stdout.log` 位于相同 `runs/TRxx/` 目录。所有测试 EV 编号仅为后续结果归属映射；本启动器不自动运行这十组 T1 测试，也不会在训练阶段读取 T1。

## 服务器运行状态入口

| 服务器角色 | 任务集 | 运行状态 JSON | 启动和设备预检记录 |
|---|---|---|---|
| 2080ti_4_1 | TR02、TR01、TR06、TR05 | `launchers/2080ti_4_1/status.json` | `launchers/2080ti_4_1/preflight.json`、`started.json` |
| 2080ti_4_2 | TR04、TR03、TR18、TR17 | `launchers/2080ti_4_2/status.json` | `launchers/2080ti_4_2/preflight.json`、`started.json` |
| 2080ti_3_1 | TR10、TR09；GPU 2 预留 | `launchers/2080ti_3_1/status.json` | `launchers/2080ti_3_1/preflight.json`、`started.json` |

本机 `2080ti_4_1` 的真实 hostname 和四张卡 UUID 已核查，详见 `CUS100_SCOPE_AND_PREFLIGHT.md`。另外两台仅配置角色与 GPU 索引；真实 hostname/UUID 由用户在对应服务器启动时自动采集，不将角色名当成已经验证的 SSH hostname。

AM-EVRPTW、DRL-TS、RRNCO、EVRPTW-RL 的训练趋势查看同目录 `logical_epoch_history.jsonl` 和 `train_history.jsonl`；TERRAN 的训练趋势查看 `logs/train_log.csv`。五种方法的 validation 趋势均查看 `validation_history.jsonl`、`validation_summary.json`；最终预算和退出证据以 `training_result.json` 与 `launch_record.json` 交叉核对。baseline 的额外训练域访问及更新事件查看 `baseline_history.jsonl` 和最终 baseline 计数，不能合并为主训练 ID stream 的样本预算。

冻结数据、stream 和目标证据位于 `artifacts/`；其中 `shared_stream_preparation.json` 保存跨方法前缀一致性检查，`reward_synthetic_feasible4/` 保存仅用 E 训练池生成的 500 例标定 cohort、reference routes、统计和 reward contract。源码快照位于 `provenance/`；每次运行的 `source_version` 指向实际源码内容，而不是仅用旧 Git HEAD 代替。

本注册表在正式启动后保持静态，后续训练状态由上述运行产物持续更新。
