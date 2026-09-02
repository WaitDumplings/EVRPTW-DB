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

## Pilot 与 full 边界

收集四台服务器的 pilot 输出及 evidence 后构建放行报告：

```bash
python EVRPTW_Benchmark/Reinforcement_Learning/scripts/build_drl_pilot_gate.py \
  --output-root "$EVRPTW_OUTPUT_ROOT" \
  --evidence-root "$EVRPTW_OUTPUT_ROOT/pilot_evidence"
```

只有 `$EVRPTW_OUTPUT_ROOT/pilot_gate_report.json` 的 `passed=true` 时，`full` 才会启动。否则 runner 硬停止。放行后把对应命令中的 `pilot` 替换为 `full`。中断后的同一完整训练队列用 `resume`；已具有通过 result、selected checkpoint 和 verifier summary 的 job 不会重跑。

评估在训练全部完成后运行：

```bash
# 三台 2080Ti 服务器：greedy
SERVER_INDEX=0 bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/run_drl_4x2080ti.sh evaluate
SERVER_INDEX=1 bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/run_drl_4x2080ti.sh evaluate
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/run_drl_3x2080ti.sh evaluate

# A6000：best-of-50 与 Cus1000/Cus2000 paired transfer
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/run_drl_2xa6000.sh evaluate
```

查看状态时使用相同 launcher 的 `status` 模式。仅检查命令与队列、不用 GPU 时追加 `--dry-run --skip-gpu-preflight`。

## 汇总

```bash
python EVRPTW_Benchmark/Reinforcement_Learning/scripts/summarize_drl_experiments.py \
  --output-root "$EVRPTW_OUTPUT_ROOT" \
  --destination "$EVRPTW_OUTPUT_ROOT/summary"
```

输出保留 per-instance 行、feasibility 分母、conditional verified distance、vehicle count、推理/训练时间、GPU memory、T2−T1/T3−T1，以及 paired Cus1000/Cus2000 transfer 单元。validation 只用于 checkpoint selection；T1/T2/T3 和 Cus2000 永不用于训练或选 checkpoint。
