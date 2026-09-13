# Road Cus500：RRNCO 与 DRL-TS 双卡部署

每组两张 2080 Ti 同步训练一个模型。默认安排如下：

| 服务器 | 使用 GPU | 本轮任务 | 启动入口 |
|---|---|---|---|
| 2080ti_4_1 | 0、1 | RRNCO Road Cus500，完整道路图 | `2080ti_4_1/full.sh` |
| 2080ti_4_2 | 0、1 | DRL-TS Road Cus500，原生 soft → hard 两阶段 | `2080ti_4_2/full.sh` |
| 2080ti_3_1 | 原 AM 三卡配置 | AM 已安排，沿用原脚本 | `../cus500_am_multigpu_20260912/2080ti_3_1/full.sh` |

TERRAN Cus500 使用用户在另一台服务器已启动的实验。这里仅新增 RRNCO 和 DRL-TS，
不启动 EVRPTW-RL，不修改已有 AM、TERRAN 或 Cus100 任务。

## 在两台服务器准备代码

在 **2080ti_4_1 和 2080ti_4_2 分别执行**：

```bash
cd /data/Maojie/ICLR/EVRPTW-DB
git fetch origin
git worktree add --detach ../EVRPTW-DB-cus500-dual origin/cus500-dual-rrnco-drlts-20260913
cd /data/Maojie/ICLR/EVRPTW-DB-cus500-dual
conda activate maojie

export CUS500_ROAD_ROOT="/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Dataset/Instances_v2/us_11city_full_clean_v7_bbde5db_20260823"
```

独立目录保留原仓库的本地修改和已有实验。若 `EVRPTW-DB-cus500-dual` 已存在，
进入该目录检查实验状态，不要覆盖或重复创建。`caliroute` 环境也可以使用；脚本优先
读取当前 conda 环境的 Python，可用 `CUS500_PYTHON=/绝对路径/bin/python` 显式指定。
启动前检查 CUDA、NCCL 和 Python 依赖，记录实际版本。

无需重新解压或生成 Road 原始数据。新目录通过 `CUS500_ROAD_ROOT` 读取现有 release，
仅在新结果目录生成确定性的 Cus500 训练抽样索引。不会读取 Cus100 deployment manifest。

## 启动与预检

**2080ti_4_1：**

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus500_dual_20260913/2080ti_4_1/full.sh
```

**2080ti_4_2：**

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus500_dual_20260913/2080ti_4_2/full.sh
```

命令完成预检和 CPU 数据准备后，会后台启动并打印 launcher PID、完整命令和 status 路径。
打印成功后可以退出 SSH。加 `--mode preflight` 只检查，不启动；加 `--foreground`
在当前终端运行。两个 shell 使用同一个 `launch.py`，也可直接指定 `--model rrnco`
或 `--model drl_ts`。GPU 编号可改为 `export CUS500_GPUS=1,2`，必须恰好两张不同卡。

启动器不会终止任何已有进程。GPU 0 的桌面进程允许保留，前提是 `/proc/PID/exe`
精确等于 `/usr/libexec/gnome-remote-desktop-daemon`，进程占用以及整卡启动前总占用
均低于 1 GiB，而且保留足够模型显存余量。其他 CUDA compute 进程都会阻止启动。
不同模型、不同结果目录使用相同物理 GPU 时，本套启动器通过 GPU UUID 文件锁互斥。

## 参数和比较口径

模型默认配置分别是 [RRNCO](configs/rrnco.json) 和 [DRL-TS](configs/drl_ts.json)。
显存目标是每卡 NVIDIA-SMI 进程峰值 **9.5–10.3 GiB**，batch 以最终实测配置为准；
硬件、驱动和依赖版本不同会改变实际占用。不要将两张卡的显存相加来估算可装载的 batch。

已完成本机 GPU 0/1 双卡校准（[实测报告](PROFILE_REPORT.md)）：

| 模型 | 每卡 / 全局 batch | 每卡进程峰值 | 预计5500 / 10000更新耗时 |
|---|---:|---:|---|
| RRNCO full graph | 22 / 44 | 9.59 GiB | 3.5–5天 / 6–9天 |
| DRL-TS | 2 / 4 | 9.29 GiB | 17–24小时 / 30–42小时 |

DRL-TS每卡batch3实测OOM，故使用batch2，略低于目标区间。耗时为短测外推，
正式启动100–200epoch后应修正；本次短测不代表500例验证结果或模型收敛。


| 项目 | 设置 |
|---|---|
| 数据 | Road Cus500；10,000 train / 固定 500 val，训练/验证 family 不重叠 |
| 单例规模 | 500 客户 + 50 充电站 + 仓库，共 551 节点 |
| 种子 | train 1234，val 910001234 |
| 训练/验证轨迹数 | 每实例 30 条 |
| 训练/验证 rollout 上限 | 1700 / 2550，与 AM Cus500 一致 |
| 最大 / 最小 epoch | 10000 / 5000 |
| 验证与早停 | 每 100 epoch 验证，5000 后连续 5 次未改善早停 |
| 优化器 | AdamW，学习率 1e-4，weight decay 0.01 |
| checkpoint 选择 | 优先独立验证器完整可行率，再比较成功实例的平均 verified cost |
| CPU 实例缓存 | 每个 rank 的 train/val pool 各最多 256 例，按需加载 |

每张卡处理全局训练 ID stream 的不同位置，完成本地反向传播后同步一次梯度，再统一裁剪
和更新参数。验证分成两个固定分片，保持每个实例的采样 seed，由 rank 0 合并独立验证
结果并写 checkpoint。一次 epoch 是一次全局优化更新，不是完整遍历训练集。

```text
全局 effective batch = 每卡 physical batch × 2 × 梯度累积次数
训练 stream 长度 = 全局 effective batch × 最大 epoch
客户暴露量 = 已完成 epoch × 全局 effective batch × 500
```

模型的 batch 可以不同。比较结果时同时报告 epoch、全局 batch、客户暴露量、GPU-hours
和验证可行率，不能只凭相同 epoch 假定两个实验用过相同训练样本量。

RRNCO 保留 Cus100 正式配方：`graph-mode full`、stable AFT、nearest 距离采样、relation
温度 5、chunk 32、checkpoint bias、每实例 **leave-one-out** REINFORCE baseline。
没有关闭 encoder 或 decoder 的道路关系。为提高 Cus500 的每卡 batch，decoder 使用
activation checkpoint stride 1；重算替代保存中间激活。等价测试覆盖采样 routes、log-likelihood、
全部参数梯度和 RNG 状态，不改变模型输出或损失定义。

DRL-TS 保留原生架构、activation checkpoint stride 1 和每 250 次全局优化更新的训练池
baseline probe；**epoch 1–2500 为 soft，2501 起为 hard**。soft 阶段使用已有
`drl_ts_soft_auxiliary_v1.json`，验证仍使用硬约束和统一成本口径。阶段按绝对 epoch 计算，
恢复任务不会重启 soft 阶段。边特征改为直接索引所需边，消除反向传播中
`batch × trajectories × nodes × nodes × edge_features` 的中间张量；前向与全部参数梯度
等价测试通过，保留原有边特征和解码行为。

统一目标仍是 `C_USD = 0.151750972762646 × D_km + 413.6331536717643 × K`。
共享 reward v3 使用 Cus500 的 objective scale `4238.927542618743`、failure base
`3.21013867342889`、unserved coefficient `1.0`。最终 best 的自动 test 不在此脚本中。

## CPU RAM 与参数覆盖

缓存是有界、按需的主机 RAM 缓存，避免每次验证重复读取相同 250 例分片；不会预载全量
10,000 个实例，也不会把缓存移入 GPU。远端可用 RAM 可能不同，预检记录实际 MemAvailable。
CPU 实测：一个新进程同时缓存 val 250 例和 train 256 例，RSS 为 **3.282 GiB**，
两 rank 基础进程与缓存合计外推约 **6.565 GiB**。单实例去重 ndarray 为 5.818 MiB；
本机加载 val 250 例需 5.34 秒、train 256 例需 4.88 秒。此测量不包含训练模型、PyTorch/NCCL
和验证工作内存，也不是正式训练峰值；远端可用 RAM 和磁盘速度需以实际机器为准。

若内存紧张，在首次启动前设置：

```bash
export CUS500_INSTANCE_CACHE_SIZE=64
```

`CUS500_BATCH_SIZE`、`CUS500_ACCUMULATION_STEPS` 分别覆盖每卡 batch 和梯度累积次数。
CPU 线程默认每进程 2，保留用户已设置的 OMP/MKL/OpenBLAS/Numba 环境变量。
改动 batch、GPU 拓扑、缓存或其他已记录配置后，应使用新的 `CUS500_OUTPUT_ROOT`，不能直接
恢复原实验。自定义输出目录请放在仓库外，或被 Git 忽略的 `results` 目录内。

## 进度与恢复

使用对应服务器的 shell 查看进度，不需要空闲 GPU：

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus500_dual_20260913/2080ti_4_1/full.sh --mode status
```

在 4_2 将路径改为 `2080ti_4_2/full.sh`。状态输出包含最新训练 epoch、最新验证、best
验证摘要、进程状态和输出目录。另可执行 `watch -n 5 nvidia-smi`。

默认输出根目录是 `EVRPTW_Benchmark/results/cus500_dual_20260913`：

```text
launchers/2080ti_4_1/rrnco/{launch_request.json,status.json,launcher.log}
launchers/2080ti_4_2/drl_ts/{launch_request.json,status.json,launcher.log}
runs/rrnco_road_cus500_seed1234/
runs/drl_ts_road_cus500_seed1234/
```

run 目录保存 `logical_epoch_history.jsonl`、`validation_history.jsonl`、
`validation_summary.json`、`best.ckpt`、`checkpoint_latest.pt`、`stdout.log` 和 `stderr.log`。
请求文件保存 GPU UUID、实际命令、环境版本、配置、数据与训练 stream 摘要、Git commit
以及实际源码 SHA。启动不会覆盖非空的旧 run。

中断后确认原训练进程已退出，使用**原 shell 和原环境变量**显式恢复：

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus500_dual_20260913/2080ti_4_1/full.sh --resume
```

恢复要求原配置、数据和源码摘要一致，恢复同一个全局 stream 位置和各 rank 的随机状态。
已经正常完成或早停的实验不能当作未完成任务再次启动。若进程退出但 status 仍为 running，
状态命令会标记 launcher 不存在；需结合 GPU 进程和 stderr 检查原因。
