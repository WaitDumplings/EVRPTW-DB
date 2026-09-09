# TERRAN stable_cost_v1 实验

这是一个显式启用的新训练方案：`training.algorithm: stable_cost_v1` 和 `model.critic_mode: stable_cost_v1` 必须同时选择。它继续使用 TERRAN 构造式策略、Stage-2 实例和独立路线验证器，不启动 ALNS，也不替换其他方法的默认训练流程。

## 目标与训练口径

环境保留真实经济成本：`C = electricity_price × consumption_per_km × distance_km + vehicle_fixed_cost × vehicles_started`。每步奖励是累计成本的负增量，单位 USD；`gamma=1`，不使用按 Cus 规模标定的 `S_N`，不使用旧 PBRS、成功奖励或旧 reward contract。车辆、电价和能耗参数仍由同一个 `rivian_energy_vehicle_cost_v2.json` 指定。

成本 critic 学习 USD cost-to-go；失败 critic 学习最终失败概率。critic 使用独立参数、优化器和梯度裁剪，读取 detached encoder 特征。共享 PopArt 更新目标统计并保持更新瞬间的原始 USD 预测，不改变环境目标。actor 使用真实未服务客户集合和剩余预算上下文。

初始阶段使用同实例多轨迹的 leave-one-out 可行性比较；达到持续成功率阈值后启用成本优化。失败约束通过单独的 USD 乘子调整，`failure_target=0.01` 是训练目标，不是可行性保证。`dual_initial` 和 `dual_max` 以车辆固定费用为单位指定；具体 USD 值由引擎记录。所有最终报告仍须使用独立 verifier 检查的完整解。

一个逻辑 batch 的所有 rollout 在同一组冻结参数下收集，存放在 CPU；PPO 依次处理物理 microbatch 和时间 chunk，最后才更新参数。actor 对每条轨迹的实际动作求和，再除以逻辑轨迹数和共同的固定 rollout budget；critic 对每条轨迹的有效状态取平均，再对轨迹取平均。不会给长轨迹额外增加实例权重，也不会把每个 microbatch 当成独立更新。

Actor 更新后还会在完整逻辑 batch 上检查经验 KL；超过阈值时恢复更新前的 actor 和 Adam 状态，学习率减半，最多重试三次。该保护只能约束这批已采样状态上的经验变化，不是全状态空间的 KL 或训练稳定性证明。

## 数据和默认 profiles

已检查的当前 `generation_plan/core/train/view_index.parquet` 和 `core/val/view_index.parquet` 实际只包含下列三个规模。旧 server 脚本的 Cus50 枚举不能作为该数据存在的依据。

| Profile | 训练 views | 验证 views | 充电站 | 物理 batch | 累积次数 | 有效 batch | 每实例轨迹 | PPO chunk | Actor LR | Train/val 步数上限 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cus100.yaml` | 50,000 | 500 | 20 | 64 | 2 | 128 | 16 | 32 | 1e-4 | 200 |
| `cus500.yaml` | 10,000 | 500 | 50 | 64 | 2 | 128 | 16 | 16 | 5e-5 | 950 |
| `cus1000.yaml` | 5,000 | 500 | 50 | 32 | 4 | 128 | 16 | 16 | 3e-5 | 1900 |

Profiles 位于 `../configs/stable_cost_v1/`。共用参数包括 critic LR `1e-4`、每逻辑 batch 两轮 PPO、10000 epochs、每个 epoch 原子保存最新 checkpoint、每 100 epochs 验证 500 个实例，每实例采样 100 个候选。不同规模的 batch、chunk、学习率和时间预算是数值计算设置；经济目标系数相同。显存上限仍须在目标机器上实测。

CLI 接受任意 `Cus正整数`。有同名 profile 时优先使用；否则按 `N≤100`、`100<N≤500`、`N>500` 选择三个基础 profile，并把默认 train/val 步数上限设为 `ceil(1.9N)`。已有 profile 的默认预算也至少达到该下限，显式 `--rollout-steps` 可用于覆盖训练预算。CLI 会核对所选数据集真实包含该规模；因此当前数据仍只能训练表中的三个规模，将来提供相应数据后可直接使用例如 `--scale Cus50`。

新实验使用 Stage-2 的确定性 shuffle-cycle sampler，不复用旧 registered stream。每一轮通过 seed 和 pass number 生成不放回排列，完成一轮后生成下一轮，因此不会在旧 Cus1000 的 40000 条有限 stream 处耗尽。`sampled_view_ids.jsonl` 保存采样记录，checkpoint 保存样本游标。索引文件 hash 记录于新实验中；这不等于对所有 materialized family 文件重新做完整数据认证。

## 准备、运行与检查

在仓库根目录运行。此机器已验证的 Python 为 `/home/exx/anaconda3/envs/maojie/bin/python`。CLI 优先使用 `--dataset-root`，其次 `EVRPTW_DATASET_ROOT`，再查找仓库数据目录和当前机器的 `/data/evrptw_runtime` 数据目录。

先准备一个新的 Cus500 实验；不传 warm-start 参数即从头初始化：

```bash
/home/exx/anaconda3/envs/maojie/bin/python -m EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.stable_cli prepare \
  --scale Cus500 \
  --dataset-root /data/evrptw_runtime/EVRPTW_Dataset/Instances_v2/us_11city \
  --output-dir /path/to/new/Cus500-run \
  --seed 1234 \
  --warm-start-checkpoint /path/to/actor-checkpoint.pt
```

`--warm-start-checkpoint` 只加载兼容的 actor/backbone 权重；新 critic、PopArt、优化器、乘子和数据游标重新初始化。新增 actor 上下文参数由新模型初始化。初始化来源写入训练契约。

检查 `resolved_config.yaml` 后，使用该配置后台启动：

```bash
/home/exx/anaconda3/envs/maojie/bin/python -m EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.stable_cli launch \
  --resolved-config /path/to/new/Cus500-run/resolved_config.yaml --gpu 0
```

Cus1000 使用 `--scale Cus1000`、另一个输出目录和目标 GPU。也可以直接 `launch --scale ... --output-dir ... --gpu ...`，一次完成准备与启动。`run` 子命令用于前台运行。`prepare` 不占用 GPU；launcher 不停止已有任务。此次工作中用户已授权验证后替换旧 Cus500/Cus1000 训练，但应由负责调度的进程核对旧 PID 并保留原日志/checkpoint，不能通过该工具隐式终止任意 GPU 进程。

后台 launcher 在未设置时将 `OMP_NUM_THREADS` 和 `MKL_NUM_THREADS` 默认为 `1`，避免多个训练进程各自创建过多 CPU 线程；显式设置的环境变量会保留。

```bash
/home/exx/anaconda3/envs/maojie/bin/python -m EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.stable_cli status \
  --output-dir /path/to/new/Cus500-run
```

`status` 返回进程是否匹配该 run、当前进度、最新训练和验证指标。`process_matches_run=false` 还可能表示当前进程命名空间无法看到宿主进程，需要结合宿主机状态判断。

输出目录必须是新的，已有运行不能被悄悄覆盖或重复启动。准备好的 `--resolved-config` 不能再混入训练参数覆盖。新配置可通过 `--epochs`、`--physical-batch-size`、`--effective-batch-size`、`--n-traj`、`--rollout-steps`、`--ppo-step-chunk-size` 调整；有效 batch 必须能被物理 batch 整除，`n_traj` 至少为 2。完整配置和来源写入 `provenance.json`，引擎实际配置与签名写入 `resolved_config.yaml` 和 `training_contract.json`。

恢复只接受新的 stable-cost checkpoint，并使用新目录：

```bash
/home/exx/anaconda3/envs/maojie/bin/python -m EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.stable_cli launch \
  --scale Cus500 --output-dir /path/to/new/resumed-run --gpu 0 \
  --resume /path/to/previous-run/checkpoint_latest.pt
```

Resume 恢复 actor、critic、两套优化器、PopArt、乘子、训练阶段和样本游标。`--resume` 与 `--warm-start-checkpoint` 互斥。当前恢复严格核对训练签名和 seed，包括总 epochs、batch、chunk、时间预算及训练数据索引；如原实验使用自定义参数，恢复时必须保留。修改训练设置应创建新实验，不能标成原实验的严格恢复。

## 工程参考与比较范围

参考仓库 `/data/Maojie/AAAI/CaliRoute` 的 TERRAN 使用静态 encoder 缓存、PPO 时间分块和独立实例 batch；`offline2online/trainer.py` 提供同实例多轨迹比较和 reference advantage 的实现经验。本方案复用这些工程思想，保留当前 EVRPTW-DB 已有的紧凑 CPU observation、静态字段共享和正确的有效数据加权；不引入 CaliRoute 的在线数据生成器、专家 replay 或另一套模型。旧源码中的 gamma `0.99` 和逐时间步均值 loss 也未作为新训练规则照搬。

这组实验改变了 reward、critic、采样和训练预算，结果不能直接混入旧 `drl_rq_protocol_frozen_v1` 的正式表格。比较时同时记录训练实例数、客户曝光量、优化步数、墙钟时间、候选数、horizon 和 verifier 可行率。`metrics.jsonl` 的 `cost_accrued_mean_usd` 包括失败轨迹已经发生的成本，不能当作完整解成本；成本结论应使用 `validation_history.jsonl` 的验证结果。

三个规模使用相同模型参数结构，支持把较小规模 actor 作为较大规模的初始化。这只说明结构兼容，不能替代冻结模型的跨规模评估；真正的尺寸外推需要额外报告未在目标规模继续训练的结果。
