# Road Cus100 → Cus500：五模型双卡 curriculum

五个入口均从 **Road Cus100 第一阶段的 best 权重**开始，在 Road Cus500 上新增
**3000 个 logical epoch**，每100轮用500个验证实例、best-of-30评测一次。
每次启动训练一个模型，两个 GPU 同步更新；只做 Road，不做 Euclidean。

## 准备 checkpoint

每台服务器使用相同的路径规则（每台只需放它要训练的模型）：

```text
/data/cus100_ckpt/
  am.ckpt
  evrptw_rl.ckpt
  drl_ts.ckpt
  terran.ckpt
  rrnco.ckpt
```

将各模型 **G/Cus100 curriculum 第一阶段的 `best_overall.ckpt`** 复制并重命名到这里。
不要放 E 域、旧 D_dist 实验、已在 Cus500 上训练的 checkpoint 或临时 smoke 权重。
脚本只读取指定模型的文件，不需要另外四个都存在；没有文件时明确报错，不回退到旧 archive。
文件名不代表内容可信：启动前在 CPU 上核查实际模型、域、规模、seed、目标、阶段和
架构，严格加载全部权重，记录实际 SHA256 和所选 epoch。

例如，本机已复制第一阶段 AM 的 best epoch1300 到 `/data/cus100_ckpt/am.ckpt`。
该文件与现有双卡 AM 运行的来源权重完全相同；现有训练继续运行，未重复启动。
本机 EVRPTW-RL 第一阶段尚未完成，未将其中间 best 冒充最终文件；其余模型需在对应
服务器第一阶段完成后自行放入。

## 同步代码和启动

在干净的仓库工作区中：

```bash
cd /data/Maojie/ICLR/EVRPTW-DB
git fetch origin
git switch ablation
git pull --ff-only origin ablation
conda activate maojie             # 第三台如使用 caliroute，则激活 caliroute
```

按模型运行一个对应命令，末尾两个数字是 `nvidia-smi` 的物理 GPU 编号：

```bash
./script_curriculum/cus500_curr/am_cus100_to_500.sh 0 1
./script_curriculum/cus500_curr/evrptw_rl_cus100_to_500.sh 0 1
./script_curriculum/cus500_curr/drl_ts_cus100_to_500.sh 1 2
./script_curriculum/cus500_curr/terran_cus100_to_500.sh 0 1
./script_curriculum/cus500_curr/rrnco_cus100_to_500.sh 0 1
```

这些是可选的独立入口，不要把上面五行当成共享 GPU 的启动队列。
脚本默认后台运行，打印 launcher PID、日志路径，可以断开 SSH；GPU 已有计算任务时会拒绝启动。
默认沿用当前激活的 conda 环境；未激活时选择 maojie。
`CURRICULUM_CONDA_ENV=caliroute` 或 `CURRICULUM_PYTHON=/path/to/python` 可明确指定。

查看配置但不占 GPU：

```bash
./script_curriculum/cus500_curr/rrnco_cus100_to_500.sh 0 1 --dry-run
```

末尾加 `--foreground` 在前台运行。普通启动不会自动修改代码；`--pull` 才会显式
执行 `git pull --ff-only origin ablation`。旧 `am.sh` 保留兼容，默认 GPU0/1，但现在也
读取 `/data/cus100_ckpt/am.ckpt`。

## Batch 与已有显存证据

| 方法 | 每卡 batch | 全局 batch | train/val 步数上限 | 证据范围 |
|---|---:|---:|---:|---|
| AM | 12 | 24 | 1700/2550 | 当前来源权重双卡短测约10071/9924 MiB，已启动正式训练 |
| EVRPTW-RL | 24 | 48 | 600/700 | 历史 sum 版双卡约10254 MiB/进程；当前 mean 版沿用，未针对新的来源权重复测 |
| DRL-TS | 2 | 4 | 1700/2550 | 历史双卡约9508 MiB/进程；batch3曾在 baseline encoder OOM |
| TERRAN | 16 | 32 | 1700/2550 | 保留已有保守默认；没有匹配的 Cus500 GPU 显存实测 |
| RRNCO | 22 | 44 | 1700/2550 | 历史双卡约9818 MiB/进程；新来源权重未复测 |

所有方法每实例30条训练轨迹、30条验证轨迹。batch 是实例数，不乘轨迹数。
AM 数值是设备显存（含桌面等开销），其余实测数值是进程显存，不能混为同一口径。
历史测试的权重或目标与本轮不同；不得将其称为所有新来源都通过 GPU 测试。
短测也不能保证未来每一个随机 batch 的峰值。详细来源在 `batch_profiles.json`；
AM 初次实测见 `SMOKE_REPORT.md`。

EVRPTW-RL 保留现有 mean 版的600/700步配置；复杂路线可能达到此上限，训练日志中的
`rollout_budget_exhausted_rate` 和验证可行率会反映这一点。其他模型保留1700/2550步。

## 训练含义与方法差异

- 一个 logical epoch 是一次全局 rollout/update cycle，不是遍历一次语料；TERRAN 在其中执行多次 PPO 更新。
- 优化目标为 `energy_vehicle_cost`：`413.6331536717643*K + 0.39*(100/257)*D_time_km`。
  成本、电量都用最快时间路径距离；独立 replay 复算验证成本。
- 保留 Cus100 策略权重，重置优化器、baseline、epoch/数据流/选优状态，切换到 Cus500 的
  reward normalizer。额外3000轮，无 early stopping；验证先最大化可行率，再最小化可行实例平均成本。
- AM 保留128维/3层/8头；baseline 默认前2500轮 EMA、后500轮 greedy。
- EVRPTW-RL 保留 mean aggregation、128维和3轮 Structure2Vec。
- DRL-TS 从已进入 hard 阶段的来源继续，`soft_stage_end_epoch=0`，不重新执行 soft 阶段。
- RRNCO 保留 full road graph、stable AFT、nearest relation、温度5和LOO baseline。
- TERRAN 保留256维/3层的 legacy PPO+PBRS。双卡入口会实际加载 actor，重置 critic 和 optimizer，
  从新阶段第1轮开始；不会自动换成 `stable_cost_v1` 或修改推理架构。PPO time chunk 为16。

REINFORCE 的验证实例按 rank 分片再合并；TERRAN 在 rank0 验证完整 cohort 后广播结果。
训练梯度按全局分母合并，各 rank 的 BatchNorm forward 使用本地统计，更新后广播 rank0 buffers；
不宣称与同全局 batch 的单卡运行逐位等价。

## 数据与输出

训练使用完整10000条 Road Cus500 train views，验证使用500条独立 val views，并检查实例及
parent family 不重叠；不读取 test。Git 不传输 dataset/checkpoint 二进制文件。
找不到数据时指定：

```bash
export CURRICULUM_ROAD_ROOT=/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Dataset/Instances_v2/us_11city_full_clean_v7_bbde5db_20260823
```

也兼容 `CUS500_ROAD_ROOT`、`CUS100_ROAD_ROOT`、`EVRPTW_DATASET_ROOT`。
checkpoint 根路径可用 `CURRICULUM_CUS100_CKPT_ROOT` 或 `--checkpoint-root` 改写；
`--source-checkpoint` 可直接指定文件，仍需通过相同检查。

输出默认 `/data/curriculum_stage2_cus500/<model>_G_Cus500_stage2_seed1234_<timestamp>_<pid>/`，
包含 `request.json`、`status.json`、`training.log`、checkpoint、训练/验证历史和
`validation_summary.csv`。TERRAN 双卡的逐轮训练诊断写入 `logical_epoch_history.jsonl` 和 `logs/train_log.csv`。
`CURRICULUM_CUS500_OUTPUT_ROOT` 或 `--output-root` 可更改训练目录。
每次创建新目录，启动不覆盖已有结果。

可选参数还包括 `--batch-size`（每卡实例数）、`--epochs`、`--validation-every`、
`--validation-limit`。默认无需修改；改变 batch 会改变实例暴露量，override 不冒充已实测设置。
