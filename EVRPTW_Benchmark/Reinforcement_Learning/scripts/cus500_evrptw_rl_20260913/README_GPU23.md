# EVRPTW-RL Road Cus500：GPU2/3 双卡重训

在已准备好 Road 数据的空闲 2080 Ti 服务器上执行：

```bash
cd /data/Maojie/ICLR/EVRPTW-DB
git fetch origin
git switch cus100-evrptw-stability-4-2-20260914
git pull --ff-only
conda activate maojie
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus500_evrptw_rl_20260913/evrptw_rl_gpu23.sh
```

GPU2 和 GPU3 通过 torchrun/NCCL 共同训练一个 Road Cus500 模型，每张卡处理不同实例，梯度同步后进行同一次全局更新。入口固定这两张物理卡，检查空闲后直接后台训练，无 watcher、无启动前校准。

| 参数 | 值 |
|---|---|
| 模型 | EVRPTW-RL，`--graph-aggregation mean` 数值稳定性适配 |
| 每卡 / 全局 batch | 24 / 48，梯度累积 1 |
| seed | 1234 |
| train / validation n-traj | 30 / 30 |
| train / validation rollout 上限 | 600 / 700，按本次要求显式设置 |
| 数据 | 10,000 Road Cus500 train / 固定 500 val |
| epoch | min 5000 / max 10000；每 100 epoch 验证 |
| 早停 | 5000 epoch 后连续 5 次验证未改善 |
| 优化器 / baseline | AdamW 1e-3；前 1000 更新 EMA，之后 greedy rollout |
| 缓存 | 每 rank 的 train/val pool 各最多 256 例 |

batch 24 来自旧 sum 模型、rollout 1700/2550 的双卡实测配置；新版 mean 与 600/700 上限未重新测量目标服务器显存，不能把旧测量当作新版已达到 9.5–10.3 GiB 的证据。较短 rollout 通常会减少显存使用；此入口直接沿用 batch 24，不进行启动前调参。保留 activation checkpoint stride 1。原 objective、reward 和充电站辅助惩罚配置保持一致。mean 是明确记录的模型适配，不能作为原 sum 训练的继续训练。

配置由 `config_gpu23_mean.json` 保存。默认每 epoch 暴露 48×500=24,000 个客户，最大训练流 480,000 个实例条目。启动时自动从已有 Road train index 生成确定性训练流，不生成或重划分原始数据，不读取 test。

数据路径自动查找当前仓库的完整 release 或 `us_11city`，也支持同级 `EVRPTW-DB` 仓库的数据。自定义位置可用 `--road-root /path/to/release`，或 `CUS500_ROAD_ROOT` / `CUS100_ROAD_ROOT`。

默认独立输出：
`EVRPTW_Benchmark/results/cus500_evrptw_rl_mean_gpu23_20260914`。
模型目录：`runs/evrptw_rl_road_cus500_mean_seed1234_2gpu`。
旧 Cus100 或旧 sum 结果不在此目录中。

查看进度：

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus500_evrptw_rl_20260913/evrptw_rl_gpu23.sh --mode status
```

自定义输出可用 `--output-root /path/to/new/run` 或 `CUS500_EVRPTW_GPU23_OUTPUT_ROOT`，查看进度时使用同一位置。若要开始另一轮从头训练，可指定带时间的输出目录：

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus500_evrptw_rl_20260913/evrptw_rl_gpu23.sh \
  --output-root "EVRPTW_Benchmark/results/cus500_evrptw_rl_mean_$(date +%Y%m%d_%H%M%S)"
```

每卡 batch、累积和缓存可显式使用 `--batch-size`、`--accumulation-steps`、`--instance-cache-size`；该入口清理旧任务遗留的通用 batch/cache 环境覆盖变量，默认稳定使用此配置。改变 batch 会改变训练流和样本预算，应单独记录。
