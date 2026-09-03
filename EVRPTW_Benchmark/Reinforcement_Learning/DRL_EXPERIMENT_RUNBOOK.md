# DRL 实验运行手册（protocol v1）

本目录只调度四个已冻结 baseline：AM-EVRPTW、EVRPTW-RL、DRL-TS、TERRAN。统一目标是 independent verifier 通过后的 directed-road total distance。Edge-DIRECT-H、LEHD、`E -> R` 和 `R -> Inject -> R` 不会生成 GPU job。

## 共同准备

四台服务器都在仓库根目录执行，并设置相对各自机器的路径：

```bash
export EVRPTW_REPO_ROOT="$PWD"
export EVRPTW_DATASET_ROOT="$PWD/EVRPTW_Dataset/Instances_v2/us_11city"
export EVRPTW_OUTPUT_ROOT="$PWD/EVRPTW_Benchmark/results/DRL_protocol_v1"
export EVRPTW_CONDA_ENV=maojie
conda activate "$EVRPTW_CONDA_ENV"
```

launcher 会检查 GPU 数量与型号、branch/commit、真实数据索引、输出目录和磁盘空间。代码不包含服务器绝对路径，也不做 corpus SHA256 扫描。

## 四台服务器

推荐直接使用已冻结的四个 checkpoint server bundle。每个目录都包含
独立 `jobs.jsonl`、任务摘要以及 nohup launcher：

```text
scripts/2080ti_4_1  # 第一台 4×2080Ti，13 个 full training jobs
scripts/2080ti_4_2  # 第二台 4×2080Ti，13 个 full training jobs
scripts/2080ti_3_1  # 3×2080Ti，10 个 full training jobs
scripts/a6000_2_1   # 2×A6000，12 个 Cus1000 full training jobs
```

分别在对应服务器执行：

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/2080ti_4_1/pilot.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/2080ti_4_2/pilot.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/2080ti_3_1/pilot.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/a6000_2_1/pilot.sh
```

这些入口默认 nohup/setsid 离线运行；同目录的 `status.sh` 和
`logs.sh` 用于查看进度。通过 pilot gate 后使用 `full.sh`，中断恢复
使用 `resume.sh`。四个目录不包含 evaluation job。

如果四台机器不共享 `EVRPTW_OUTPUT_ROOT`，构建 pilot gate 前必须汇总
四台 pilot 产物。正式训练结束后，把全部 `checkpoint_selected.pt`、
`training_result.json`、`validation_summary.json`、atomic training state 和
provenance 汇总到中央测试服务器，再单独冻结测试服务器的 evaluation 分配。

以下旧入口继续兼容。

两台 4×2080Ti 分别使用不同的 `SERVER_INDEX`：

```bash
SERVER_INDEX=0 bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/run_drl_4x2080ti.sh pilot
SERVER_INDEX=1 bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/run_drl_4x2080ti.sh pilot
```

3×2080Ti：

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/run_drl_3x2080ti.sh pilot
```

2×A6000：

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/run_drl_2xa6000.sh pilot
```

每张 GPU 内串行、GPU 之间并行。一个 job 失败只停止所属 GPU 队列；其余 GPU 完成在途队列。`SIGINT/SIGTERM` 会传给所有子进程组。

训练 rollout 使用所有方法共享的按规模预算：Cus50=80、Cus100=140、
Cus500=600、Cus1000=1200。该预算只限制 sampling actor 和训练 baseline，
validation、T1/T2/T3 与 scale-transfer evaluation 继续使用环境的完整动态
horizon，避免把训练预算耗尽误报为最终求解失败。

每个 REINFORCE data pass 以及每个 TERRAN epoch 都记录：

- 实际 trajectory steps 的 mean/P50/P90/P99/max；
- `rollout_budget_exhausted_count`；
- `rollout_budget_exhausted_rate`。

这些字段用于审核预算是否过紧；修改预算后必须重新跑 batch/memory pilot，旧的
RTX 2080 Ti 吞吐与 wall-time 外推不再视为当前配置证据。

## 固定训练预算

正式训练不遍历全部 train views，也不使用 full-data pass 计量。epoch 是跨模型
一致的 logical epoch，而不是某个模型恰好能装入显存的 physical batch：

```text
logical environments = logical epochs × environments per logical epoch
customer exposures = logical environments × customer count
```

同一 scale 的四个模型使用相同 epoch 数、每 epoch 环境数和 exposure。每个
logical epoch 从按 `seed` 确定性打乱的 train pool 中无放回取样；如果 logical
batch 超过某模型的安全 physical batch，就拆成可整除的 micro-batches、累计梯度，
最后只做一次 optimizer update。

| Scale | Logical epochs | Environments / epoch | Total environments | Customer exposures |
|---|---:|---:|---:|---:|
| Cus50 | 100 | 100 | 10,000 | 500,000 |
| Cus100 | 200 | 25 | 5,000 | 500,000 |
| Cus500 | 500 | 2 | 1,000 | 500,000 |
| Cus1000 | 1,000 | 1 | 1,000 | 1,000,000 |

Cus50/Cus100/Cus500 都使用约10%的 train pool。Cus1000 的1,000 epochs 对应
1,000个训练环境，即20%的 train pool；这是显式的 scale-specific 预算，不应描述
为四个 scale 完全相同的 exposure。

正式 job 结束时执行一次500-view validation并选择 checkpoint；T1/T2/T3 和
Cus2000 不参与训练或 checkpoint selection。中断前若 fixed-budget job 尚未原子
提交完成状态，该 job 从同一 seed 的确定性抽样序列开头重跑。

## Pilot 与 full 边界

收集四台服务器的 pilot 输出及 evidence 后构建放行报告：

```bash
python EVRPTW_Benchmark/Reinforcement_Learning/scripts/run_drl_pilot_checks.py \
  --output-root "$EVRPTW_OUTPUT_ROOT" \
  --dataset-root "$EVRPTW_DATASET_ROOT"

python EVRPTW_Benchmark/Reinforcement_Learning/scripts/build_drl_pilot_gate.py \
  --output-root "$EVRPTW_OUTPUT_ROOT" \
  --evidence-root "$EVRPTW_OUTPUT_ROOT/pilot_evidence"
```

协议中的 `pilot.full_runtime_budget_approved=false` 仍会使 gate 保持 STOP。
统一 logical-epoch 预算已经替代候选 100 full-data passes，但仍须
审核逐方法 wall-time、rollout-cap telemetry 和 RTX A6000 Cus1000 pilot，生成
新的 clean commit，并显式批准 runtime budget；不得按 method/scale/seed 单独
增加 exposure 预算。

只有 `$EVRPTW_OUTPUT_ROOT/pilot_gate_report.json` 的 `passed=true` 时，
`full` 才会启动。否则 runner 硬停止。放行后把对应命令中的 `pilot`
替换为 `full`。中断后的同一完整训练队列用 `resume`；已具有通过
result、selected checkpoint 和 verifier summary 的 job 不会重跑，从未启动的
后续 job 会按 fresh job 启动。

查看状态时使用对应目录的 `status.sh`。仅检查命令与队列、不用 GPU
时，可用前台入口 `run.sh pilot --dry-run --skip-gpu-preflight`。

全局 manifests 仍保留冻结的 evaluation 定义，但不会被这四个
checkpoint-only server bundle 调度。中央测试服务器的 GPU 拓扑确定后，再
从全局 manifests 生成独立 test bundle。

## 汇总

```bash
python EVRPTW_Benchmark/Reinforcement_Learning/scripts/summarize_drl_experiments.py \
  --output-root "$EVRPTW_OUTPUT_ROOT" \
  --destination "$EVRPTW_OUTPUT_ROOT/summary"
```

输出保留 per-instance 行、feasibility 分母、conditional verified distance、vehicle count、推理/训练时间、GPU memory、T2−T1/T3−T1，以及 paired Cus1000/Cus2000 transfer 单元。validation 只用于 checkpoint selection；T1/T2/T3 和 Cus2000 永不用于训练或选 checkpoint。
