# 2080ti_3_1：AM Road Cus500 多卡训练

这一轮只运行 **一个 AM / Road Cus500 / seed 1234 模型**，默认同时使用
2080ti_3_1 的 GPU 0、1、2。使用当前 `caliroute` 环境；与 Cus100 的结果目录、
启动器和训练 ID stream 分开，不读取或改写 Cus100 的 deployment manifest。

## 启动

在 **2080ti_3_1** 上执行：

```bash
cd /data/Maojie/ICLR/EVRPTW-DB
git fetch origin
git worktree add --detach /data/Maojie/ICLR/EVRPTW-DB-cus500-am origin/cus500-am-multigpu-2080ti-20260912
cd /data/Maojie/ICLR/EVRPTW-DB-cus500-am
conda activate caliroute

export CUS500_ROAD_ROOT="/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Dataset/Instances_v2/us_11city_full_clean_v7_bbde5db_20260823"
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus500_am_multigpu_20260912/2080ti_3_1/full.sh
```

独立 worktree 保留原仓库的本地修改和已有实验；无需 stash 或切换原仓库分支。
不需要重新解压 Cus100 包。新 worktree 通过上面的绝对路径读取原仓库已有的完整
Road release，按 Cus500 筛选，不复制或重新生成原始数据。若新 worktree 目录已经
存在，先进入该目录检查当前实验状态，不要覆盖或重复创建。

默认命令先检查环境和数据，并在 CPU 上准备训练 ID stream，然后在后台启动。
终端显示 launcher PID、全局 batch、完整命令与 status 路径之后可以退出 SSH。
`CUS500_PYTHON` 可指定 Python 的绝对路径；默认使用当前 conda 环境的 Python。

只预检、不准备 stream 或启动训练：

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus500_am_multigpu_20260912/2080ti_3_1/full.sh --mode preflight
```

若需要把 GPU 0 完全留给桌面，启动前设置：

```bash
export CUS500_GPUS=1,2
```

GPU 0 上现有的 `/usr/libexec/gnome-remote-desktop-daemon` 可以保留：启动器检查
`/proc/PID/exe` 的精确路径、该进程显存低于 1 GiB，且卡上总占用低于 1 GiB。
其他 CUDA compute 进程会阻止启动。脚本不会停止任何已有进程。

## 多卡方法和训练口径

每张卡保存一个 AM 模型副本，读取全局 stream 中不同的实例，反向传播后同步梯度，
共同更新**同一个模型**。显存不跨卡合并；每张卡仍需独立放下本地 batch。
验证实例按固定顺序分给各个 rank，保持每实例的原始验证 seed，由 rank 0 合并
验证结果、执行统一的 checkpoint 选择并写入文件。

AM 编码器的 BatchNorm 在前向传播时使用每张卡本地 batch 的统计量，每次优化
更新后把 rank 0 的 running buffers 广播给其他卡。因此，梯度采用全局 batch
归一化并同步，但不能宣称它与相同全局 batch 的单卡 BatchNorm 训练逐步等价。

```text
全局 effective batch = 每卡 physical batch × GPU 数量 × 梯度累积次数
全局训练 stream 长度 = 全局 effective batch × 最大 epoch
```

默认参数在 [config.json](config.json)，实测与适用范围见 [校准记录](PROFILE_REPORT.md)。
`CUS500_BATCH_SIZE` 和 `CUS500_ACCUMULATION_STEPS` 可显式覆盖；新 batch 或卡数
需要新的输出目录，不能作为原实验直接 resume。一个 epoch 是一次全局优化更新，
不是完整遍历训练集。增加卡数也增加每个 epoch 的客户暴露量，比较实验时同时报告
全局 batch、epoch、客户暴露量和 wall time。

| 项目 | 设置 |
|---|---|
| 数据 | Road Cus500，10,000 train / 500 固定 val |
| 节点 | 500 客户 + 50 充电站 + 仓库 |
| 随机种子 | 1234 |
| 每卡 / 全局 batch | 4 / 12（三卡，累积 1 次） |
| 显存实测 | 双卡各 9.943 GiB，三张物理卡尚未实测 |
| 时间预估 | 约 18–36 小时，取决于早停和远端速度 |
| 训练/验证轨迹数 | 每实例 30 条 |
| 训练/验证步数上限 | 1700 / 2550 |
| epoch 预算 | 最少 5000，最多 10000 |
| 验证 | 每 100 epoch，500 个固定实例 |
| 早停 | epoch 5000 后连续 5 次验证未改善；最早 5500 |
| checkpoint 选择 | 先最大化完整可行率，再最小化 verified cost |
| 优化器 | AdamW，学习率 1e-4，weight decay 0.01 |
| AM baseline | 前 2500 次更新用 EMA，随后 greedy rollout；每 2500 次更新用 64 个训练实例做 paired test |
| CPU 线程 | 每进程 OMP/MKL/OpenBLAS/Numba 默认 2，保留已显式设置的环境变量 |

目标仍为 `C_USD = 0.151750972762646 × D_km + 413.6331536717643 × K`。
使用共享 reward v3 的 **Cus500** 校准值：objective scale 4238.927542618743，
failure base 3.21013867342889，unserved coefficient 1.0。
共享环境继续使用 FFP；这次没有加入欧式 Cus500 数据、测试集生成或自动 test。

## 看进度和恢复

```bash
watch -n 5 nvidia-smi
```

```bash
cd /data/Maojie/ICLR/EVRPTW-DB-cus500-am
python EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus500_am_multigpu_20260912/launch.py --mode status
tail -n 3 EVRPTW_Benchmark/results/cus500_am_multigpu_20260912/runs/am_road_cus500_seed1234/logical_epoch_history.jsonl
tail -n 3 EVRPTW_Benchmark/results/cus500_am_multigpu_20260912/runs/am_road_cus500_seed1234/validation_history.jsonl
tail -n 30 EVRPTW_Benchmark/results/cus500_am_multigpu_20260912/runs/am_road_cus500_seed1234/stderr.log
```

默认输出根目录为 `EVRPTW_Benchmark/results/cus500_am_multigpu_20260912`，
可通过 `CUS500_OUTPUT_ROOT` 改为新的绝对目录；自定义输出请放在仓库外，或已被 Git 忽略的 `results` 目录内。`launchers/2080ti_3_1/launch_request.json`
保存实际命令、Python/依赖版本、GPU UUID、batch、数据及 stream 摘要、Git commit
和实际源码 SHA；`status.json` 保存 PID、状态与退出码。训练日志、best checkpoint
在 `runs/am_road_cus500_seed1234`。成本均值仅统计独立验证器接受的成功实例。

同一输出根目录的第二次启动会被锁或现有输出检查拒绝。中断后确认相关 GPU
训练进程已经退出，且 checkpoint 存在，再显式恢复：

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus500_am_multigpu_20260912/2080ti_3_1/full.sh --resume
```

恢复要求原参数、数据和源码摘要一致；已正常结束的实验不能默认续跑。
调试时可用 `--foreground` 在当前终端运行同一启动流程。
