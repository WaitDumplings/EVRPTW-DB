# Ablation / final 实验入口

分支 `ablation`；新成本实验统一使用最快时间路径距离 D_time，时间 T_time，电量 κD_time。
此目录是新的 Road 训练入口，不调用旧队列、恢复旧 checkpoint 或启动 test evaluation。
非学习入口仍在 `EVRPTW_Benchmark/scripts/`，见其 `ABLATION_NONLEARNING_AUDIT.md`。

## 本轮 Road Cus100 参数诊断

五个模型从头训练，各 300 **logical epochs**，每 100 epoch 在相同完整 500-instance
Road Cus100 validation cohort 上验证（100、200、300）。固定 seed=1234，train/val
各实例 30 trajectories；验证保留完成解中通过独立 verifier 的最低 D_time 美元成本。
没有 early stopping，没有 warm start，没有额外局部搜索或 test 选参。

| 模型 | Cus100 每卡 instances | Cus500 每卡 instances | Cus1000 每卡 instances |
|---|---:|---:|---:|
| AM-EVRPTW | 108 | 4 | 1 |
| EVRPTW-RL | 200 | 24 | 12 |
| DRL-TS | 24 | 2 | 1 |
| TERRAN | 384 | 16 | 4 |
| RRNCO | 50 | 22 | 4 |

Cus100 使用一张卡训练一个模型，可并行运行多个模型；四卡机器先跑四个，第五个接
最先空出的卡。Cus500 每模型同步双卡；Cus1000 根据**所选可见 GPU**，有四卡用
四卡，否则三卡，少于三卡报错。大规模物理 batch 是已测配置或保守初值，尚未全部
完成新 D_time、多卡显存测量，不能把上表当作显存利用率保证。全局 batch = 每卡
instances × GPU 数，不包含 trajectories；不要求不同模型拥有相同 sample exposure。

Cus100 rollout cap 240/360（train/val）；Cus500 通常1700/2550，EVRPTW-RL保留其
明确的600/700配置；Cus1000为1250/1875。大规模脚本配置不会在这轮自动开跑。
`--batch-size` 可覆盖每卡 batch。TERRAN 大规模 PPO time chunk 分别16/8。

学习方法及参数选择：

- EVRPTW-RL 使用修正后的 **mean aggregation**，LR=0.001、EMA warmup=1000。
  这轮300 epoch尚未转到greedy-rollout baseline，不能视为完整baseline schedule效果。
- DRL-TS LR=0.0001，保留完整训练的soft-stage边界2500；短预算用
  `min(2500, epochs)`登记合法边界，因此这轮全部300 epoch在soft阶段，验证始终hard。
  不把短试跑改为150 soft +150 hard；未来较长fresh实验自然使用2500边界。
- TERRAN使用一致的legacy PPO+PBRS配置，不跨规模切换stable-cost critic架构。
  LR=0.0001、PPO3轮/4 minibatches、vf_coef=0.1、Smooth-L1 critic、backbone梯度0.1。
  新多卡入口真实同步模型梯度，rank0独立验证和保存；只支持fresh fixed-epoch运行。
- AM LR=0.0001；原生 `steps_per_epoch=2500 × baseline_warmup_epochs=1` 对应
  前2500次optimizer updates使用EMA baseline。这轮300 logical epochs各一次更新，
  因此也全部处于EMA阶段，尚不能评价切换greedy-rollout baseline后的表现。
- RRNCO full关系模式、stable AFT、nearest关系选择、LOO baseline、LR=0.0001。
- AdamW、weight decay=0.01；各模型其它原生参数保存到各自checkpoint/训练signature。

`configs/reward_dtime.json` 将旧v3训练集标定的数值尺度作为本轮**固定超参**保留，
并显式声明新的 D_time objective。它不是对 D_time 的重新标定，也没有声称旧统计
是在新矩阵下计算。五模型共享这些数值；原始USD评价没有这些归一化或auxiliary。

## 启动

```bash
conda activate maojie
cd /data/Maojie/ICLR/EVRPTW-DB
# 当前工作目录应已检出 ablation 分支。
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/ablation_final/cus100_300.sh
```

Shell 为前台入口，长作业可在 tmux 中执行。可指定新目录：

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/ablation_final/cus100_300.sh \
  --output-root /data/ablation_road100_pilot
```

默认自动找发布目录；外部数据用 `ABLATION_ROAD_ROOT` 或 `--road-root` 指定。
默认使用当前 Python，可通过 `ABLATION_PYTHON` 指定环境。代码包含全部配置，
stream按训练索引和seed在启动时确定性生成，不要求复制历史results/artifacts。
数据集仍需先在目标机器部署；不会重新生成实例。

后续大规模 fresh 实验（本轮尚未执行）示例：

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/ablation_final/train.sh \
  --scale 500 --models rrnco --gpus 0,1 --epochs 10000
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/ablation_final/train.sh \
  --scale 1000 --models evrptw_rl --gpus auto --epochs 10000
```

`--dry-run` 只准备stream与命令/manifest，不启动训练。GPU已被计算任务占用时拒绝启动，
不终止外部任务；桌面GNOME服务可共存。训练过程中SIGTERM/Ctrl-C只停止本调度器
自己的进程组。输出根必须不存在；当前统一入口不提供resume以防科学配置混用。

## 结果

输出根包含 `status.json`（实际来源文件哈希、启动命令、GPU UUID、进程状态）和
持续更新的 `validation_summary.csv`。单个模型目录
`runs/<method>_road_cus100_seed1234/` 包含：

- `validation_history.jsonl`：100/200/300完整验证结果及best选择标志。
- `training_result.json`、checkpoint：训练预算与最终完成状态。
- `stdout.log` / `stderr.log`：运行日志。
- REINFORCE的`logical_epoch_history.jsonl`；TERRAN的`logs/train_log.csv`。

只有正常退出、训练达到声明epoch，且每个声明验证点均完整评测指定实例数，才记为
completed。参数诊断应联合看FR、可行实例cost与训练稳定性；300epoch不是收敛证明。
不同batch意味着不同样本暴露，不能据此单独证明架构优劣。


用只读报告脚本汇总或持续观察：

```bash
python EVRPTW_Benchmark/Reinforcement_Learning/scripts/ablation_final/report.py \
  --output-root /data/ablation_final_road_cus100_300_20260919 --watch
```

默认每30秒刷新 `analysis/summary_by_epoch.csv`、`summary_at_300.csv`、
`RESULTS_SUMMARY.md`、cost及FR曲线，所有任务终止后退出。报告保留缺失、失败及
排队状态；1 epoch profile不会成为300 epoch结果。CSV保留原始浮点精度。
Cost使用各模型各次验证自身的可行实例，**不是五模型共同可行交集**；FR使用完整
验证cohort。AM的2500-update EMA warmup、EVRPTW-RL的1000-update EMA warmup、
DRL-TS的soft阶段分别标注，不能把这轮短试跑当作完整训练流程的最终排名。
