# DRL 实验运行手册（protocol v1）

> [!IMPORTANT]
> **历史文档提示（2026-09-04）：** 本文后续出现的 1,000-epoch 预算、pilot
> 流程及相关旧数值仅保留作历史记录，不再用于当前正式实验。当前配置以
> [`scripts/rq_v1/README.md`](scripts/rq_v1/README.md) 为准：预算版本为
> `v11`，minimum/max epochs 为 5,000/6,000，已取消 pilot gate；Cus1000 的
> validation/test 使用 `sample` 解码且 `n_traj=100`。

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
scripts/rq_v1/2080ti_4_1  # 第一台 4×2080Ti，24 个 full training jobs
scripts/rq_v1/2080ti_4_2  # 第二台 4×2080Ti，21 个 full training jobs
scripts/rq_v1/2080ti_3_1  # 3×2080Ti，15 个 full training jobs
scripts/rq_v1/a6000_2_1   # 2×A6000，12 个 Cus1000 full training jobs
```

分别在对应服务器执行：

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/2080ti_4_1/pilot.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/2080ti_4_2/pilot.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/2080ti_3_1/pilot.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/a6000_2_1/pilot.sh
```

这些入口默认 nohup/setsid 离线运行；同目录的 `status.sh` 和
`logs.sh` 用于查看进度。当前服务器自己的 pilot 全部通过后使用
`full.sh`，中断恢复使用 `resume.sh`。四个目录不包含 evaluation job。

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

当前活动候选为 `drl_rq_runtime_budget_v4_cus1000_b2_val100`，完整定义见
`reports/LOGICAL_EPOCH_TRAINING_BUDGET_V5_CUS1000_B2.md`。

正式训练不遍历全部 train views，也不使用 full-data pass 计量。四个规模均运行
1,000 个 logical epochs；同一 scale 的四个模型与三个 seed 使用一致的每 epoch
base-instance 数、确定性训练流和 effective batch：

```text
logical environments = 1000 × environments per logical epoch
customer exposures = logical environments × customer count
```

训练流在每个 `city × day_type` stratum 内按冻结 seed 有放回采样，再确定性打乱。
这让训练预算不受有限 train index 大小限制，同时保证同一 scale/condition/seed 的
四个方法消费完全相同的 instance ID 序列。

| Scale | Logical epochs | Instances / epoch | Effective batch | Total instances | Customer exposures |
|---|---:|---:|---:|---:|---:|
| Cus50 | 1,000 | 200 | 200 | 200,000 | 10,000,000 |
| Cus100 | 1,000 | 50 | 50 | 50,000 | 5,000,000 |
| Cus500 | 1,000 | 4 | 4 | 4,000 | 2,000,000 |
| Cus1000 | 1,000 | 2 | 2 | 2,000 | 2,000,000 |

显存相关的 physical batch 可按模型更小；四个方法都通过顺序梯度累积恢复上述
effective batch。Cus1000 固定为 `physical=1 / effective=2`，所以 GPU 同时只驻留
一个大 instance，但每个 logical epoch 在两个 rollout 的梯度累积完成后才更新。
`1000 epochs × 2 instances` 的单 job 规划值约为48小时；这是待 pilot 校准的线性
估计，不是硬超时。

每 50 epochs 固定保存并验证一次，共 20 个 checkpoint。Cus50/100/500 每次使用
固定500个 validation views；Cus1000 每次使用固定100个 views，以避免周期验证量
超过训练量。所有选择验证均执行完整 horizon 和独立 route verifier。Model
selection 只用 val：
先最大化 verifier feasibility rate；若相同，再最小化通过样本的平均 directed-road
distance。最终同时保存 `best.ckpt` 和兼容旧 evaluation 脚本的
`checkpoint_selected.pt`，并在 `validation_history.jsonl` 保留全部20次结果。
Cus1000 训练结束后对已经选定的 `best.ckpt` 再执行一次完整500-view val audit，
写入 `validation_final_audit.json`；该审计不重新选择 checkpoint。T1/T2/T3 和
Cus2000 不参与 checkpoint selection。

三个 REINFORCE trainer 在每个 50-epoch validation 边界同步更新
`checkpoint_latest.pt` 与原子状态，可从最近边界恢复。TERRAN 保留每个边界的
`checkpoint_epoch_*.pt` 和在线 `best.ckpt`；当前固定 epoch 队列不承诺中途自动 resume，
避免把尚未恢复训练流 cursor 的行为误写成正式保证。

## Pilot 与 full 边界

每台服务器独立放行自己的 full 队列。`full.sh` 只检查同一
`jobs.jsonl` 中属于当前服务器的 pilot 是否在同一 executable commit 下全部
产生 PASS `job_result.json`、`checkpoint_selected.pt` 和
`validation_summary.json`。因此四台机器不需要共享输出目录，也不需要等待其他
服务器的 pilot；切换 executable commit 后必须在新 commit 下重跑本机 pilot。

以下命令仍可在所有机器的结果汇总到中央服务器后生成论文级全局 pilot evidence，
但它不再作为单台服务器启动 full 的运行时依赖：

```bash
python EVRPTW_Benchmark/Reinforcement_Learning/scripts/run_drl_pilot_checks.py \
  --output-root "$EVRPTW_OUTPUT_ROOT" \
  --dataset-root "$EVRPTW_DATASET_ROOT"

python EVRPTW_Benchmark/Reinforcement_Learning/scripts/build_drl_pilot_gate.py \
  --output-root "$EVRPTW_OUTPUT_ROOT" \
  --evidence-root "$EVRPTW_OUTPUT_ROOT/pilot_evidence"
```

全局报告继续用于论文审核、跨服务器配置一致性和最终 acceptance，不控制某台
服务器的本地队列。不得按 method/scale/seed 单独增加 exposure 预算。

本机 pilot 全部通过后，把对应命令中的 `pilot` 替换为 `full`。若本机任一
pilot 缺失或失败，runner 会硬停止并列出具体 job ID。中断后的同一完整训练队列
用 `resume`；已具有通过
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
