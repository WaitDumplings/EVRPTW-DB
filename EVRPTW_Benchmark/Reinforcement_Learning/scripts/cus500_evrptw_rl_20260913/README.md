# AM 结束后的 EVRPTW-RL Road Cus500 双卡实验

GPU1/2 的修复版直接重训入口见 [README_GPU12.md](README_GPU12.md)，使用 `evrptw_rl_gpu12.sh`。

本目录为本机 GPU **0、1** 的后续实验，训练一个 EVRPTW-RL Road Cus500 模型。
AM 当前训练继续使用原工作目录；后续调度器等待 AM 正常完成或早停、相关进程退出且
GPU 空闲，再执行 EVRPTW-RL 显存校准并用最终配置启动。不会因 AM 报错退出就自动
把报错当作完成，也不终止现有任务。等待与校准逻辑分别由 `watch_after_am.py` 和
`autocalibrate.py` 管理。

## 安装等待任务与查看状态

在本机已核对的独立工作目录中执行，默认后台运行；启动 watcher 本身不会占用 GPU：

```bash
cd /data/Maojie/ICLR/cus500-evrptw-rl-after-am
conda activate maojie
python EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus500_evrptw_rl_20260913/watch_after_am.py --mode start
```

默认仅跟随这次 AM 实验：

```text
/data/Maojie/ICLR/cus500-am-multigpu/EVRPTW_Benchmark/results/cus500_am_dual_2080ti_4_1_20260913
runs/am_road_cus500_seed1234_2gpu
```

每 30 秒检查一次，`watcher/status.json` 持续记录 heartbeat，校准期间也会更新。
状态依次可能为 `waiting_am`、`waiting_am_exit`、`waiting_gpus`、`calibrating`、
`launching_evrptw_rl`、`confirming_evrptw_rl_start`，最后为 `handed_off` 或 `failed`。
`handed_off` 表示 EVRPTW-RL 已实际完成至少一次更新或进入验证，不代表整个训练完成。

```bash
python EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus500_evrptw_rl_20260913/watch_after_am.py --mode status
```

只有 AM launcher 记录 `completed`、退出码为 0，结果为 `AM-EVRPTW` 的正式成功/早停，
完成 5000–10000 epoch，best/latest checkpoint 非空，且原 launcher/torchrun/所有 rank
均已退出，才允许进入 GPU 检查。PID 使用 Linux 启动时刻识别，避免误认复用的 PID。
其他任务占用 GPU 0/1 时继续等待，不会终止进程；AM 失败或 launcher 异常消失时记录
`failed` 并退出，绝不单凭 GPU 空闲启动下一个模型。

注册时固定 AM 启动请求 SHA、数据/目标口径、物理 GPU UUID、主机名和 EVRPTW-RL 源码 SHA。
校准与正式启动前再次检查源码；修改代码或模板后原等待任务会拒绝继续。不要在后台
等待期间改动此工作目录的源码。watcher 使用独立输出锁防止重复注册。

watcher 因断线或机器进程中断需要重新启动时，可在同一目录、同一参数下再次执行
`--mode start`。如果已有 EVRPTW-RL 启动记录，会接续观察它，不会再提交第二次训练；
已有下游失败结果不会被偷偷 resume。若状态已为 `failed`，应先检查其 `error` 和日志，
不要把新的输出目录当作绕过失败原因的手段。

watcher 的 `--config`、`--output-root`、`--am-root`、`--road-root` 可显式指定，
`--foreground` 用于前台观察。自定义输出时，查看状态需使用相同 `--output-root`。
`watcher/request.json` 保存冻结请求，`watcher/watcher.log` 保存后台输出，
`watcher/launch.log` 保存正式启动命令的输出。

## 配置与训练口径

[config.json](config.json) 是校准模板，**batch 1 只是占位值**，不是正式 batch。
自动校准默认从每卡 batch 4 开始倍增、回退和二分，最后执行包含 EMA → greedy baseline
切换的 6 步确认。目标为每卡 NVIDIA-SMI 训练进程峰值 9.5–10.3 GiB；若整数 batch 的
粒度无法达到 9.5 GiB 下限，会选择实测安全的最大 batch 并如实记录占用，不填充显存。
最终每卡 batch 和实测记录由 AM 结束后的校准产生。两卡共享同一次优化更新，显存不跨卡合并。

| 项目 | 配置 |
|---|---|
| 数据 | Road Cus500，10,000 train / 固定 500 val |
| 单例规模 | 500 客户 + 50 充电站 + 仓库，共 551 节点 |
| train / val seed | 1234 / 910001234 |
| 训练 / 验证采样数 | 每实例 30 条 |
| 训练 / 验证 rollout 上限 | 1700 / 2550，与 AM 一致 |
| epoch 预算 | min 5000 / max 10000 |
| 验证及早停 | 每 100 epoch 验证；5000 后连续 5 次未改善早停 |
| 优化器 | AdamW，原生学习率 **1e-3**，weight decay 0.01，clip norm 2 |
| baseline | 前 1000 次更新 EMA（decay 0.9），随后 greedy rollout；每 100 次更新用 64 个训练实例配对检验 |
| activation checkpoint | stride 1，保留循环梯度和 RNG |
| 主机缓存 | 每 rank 的 train/val pool 各最多 256 例，按需载入 |

保留 Cus100 TR17/TR18 正式配方：原生 EVRPTW-RL 架构、学习率、baseline 和
`configs/evrptw_rl_station_auxiliary_v1.json`。station auxiliary 为合法充电站访问次数
除以客户数、权重 0.3，只参与模型训练损失。它不是 DRL-TS 的 soft/hard 阶段；
EVRPTW-RL 在第 1001 次更新切换的是 baseline。

统一验证目标及 task reward 与 AM 相同：

```text
C_USD = 0.151750972762646 × D_km + 413.6331536717643 × K
reward objective scale = 4238.927542618743
failure base = 3.21013867342889
unserved coefficient = 1.0
```

验证把固定 500 例按顺序分给两 rank，每实例采样种子保持不变；由独立验证器检查完整
可行性，由 rank 0 汇总。best checkpoint 优先可行率，再比较成功实例的平均成本。
一次 epoch 是一次全局优化更新；不同模型的每卡 batch 不同，应同时报告 epoch、
全局 batch 和客户暴露量。

```text
全局 batch = 每卡 batch × 2 × 梯度累积次数
训练 stream 长度 = 全局 batch × 10000
客户暴露量 = 已完成 epoch × 全局 batch × 500
```

复用已有冻结 Road 数据，不复制原始实例，也不使用 Cus100 deployment manifest。
CPU 准备脚本核对 release SHA、10,000/500 数量、551 节点与训练/验证 family 不重叠，
再生成精确全局预算的确定性抽样 stream。共享 stream 的 parquet/manifest 成对创建加锁。

## 训练入口

调度器生成最终配置后，通过同一启动器后台运行。手动调用时应显式使用最终配置，
例如（路径以实际生成结果为准）：

```bash
conda activate maojie
export CUS500_ROAD_ROOT="/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Dataset/Instances_v2/us_11city_full_clean_v7_bbde5db_20260823"

bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus500_evrptw_rl_20260913/full.sh \
  --config /绝对路径/最终配置.json
```

`full.sh` 固定模型和物理 GPU 0/1，不受旧 `CUS500_GPUS` 或之前重复 CLI 参数影响。
它使用激活的 conda Python；也可指定 `CUS500_PYTHON`。`--config`、`--output-root`、
`--foreground`、`--resume` 和 `--mode preflight|status` 均可用。

保留 `CUS500_BATCH_SIZE`、`CUS500_ACCUMULATION_STEPS`、`CUS500_INSTANCE_CACHE_SIZE`
作为显式参数覆盖接口。自动校准后启动应清理历史覆盖变量，或同时传入最终
`--batch-size`、`--accumulation-steps`、`--instance-cache-size`，避免旧 AM 环境覆盖实测配置。
更改训练配置后应建立新输出目录，不能作为原实验直接 resume。

启动器保护现有 compute 进程，并使用输出锁和按 GPU UUID 的跨输出目录锁避免抢卡。
GPU 0 只允许保留精确识别的 `/usr/libexec/gnome-remote-desktop-daemon` 桌面进程，
还会检查启动前占用和模型显存余量；GPU 1 不允许现有 compute 进程。
原有 TERRAN GPU 2/3 不属于本实验。

## 输出、状态和恢复

默认输出根目录：`EVRPTW_Benchmark/results/cus500_evrptw_rl_20260913`，
可通过 `CUS500_OUTPUT_ROOT` 或 `--output-root` 改为新的独立目录。

```text
launchers/local_after_am_gpu01/evrptw_rl/launch_request.json
launchers/local_after_am_gpu01/evrptw_rl/status.json
launchers/local_after_am_gpu01/evrptw_rl/launcher.log
runs/evrptw_rl_road_cus500_seed1234/
```

请求记录保存实际代码 SHA、Git commit、GPU UUID、Python/依赖、数据和 stream 摘要、
完整训练命令；run 目录保存训练/验证历史、best 和 latest checkpoint、stdout/stderr。
只有正常退出且存在正式成功结果和 best checkpoint，启动器才记录 completed；短测结果
不能被当作正式完成。

无需空闲 GPU 即可查看训练状态：

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus500_evrptw_rl_20260913/full.sh \
  --config /绝对路径/最终配置.json --mode status
```

中断后，确认原训练进程退出并存在 checkpoint，再用同一最终配置和输出目录追加
`--resume`。恢复核对原配置、数据、源码，继续原 stream 位置、baseline 与各 rank RNG；
已正常完成或早停的任务不能默认续跑。

主机缓存是有界惰性缓存。相同 Road Cus500 数据的 CPU 测量中，单 rank 常驻 val250 +
train256 后 RSS 约 3.282 GiB，两 rank 基础进程和缓存外推约 6.565 GiB；训练模型、
PyTorch/NCCL 和验证工作内存另计。这不是 EVRPTW-RL 正式训练 RAM 峰值，实际可用内存
由预检记录。CPU 线程默认每进程 2，可用原线程环境变量调整。
