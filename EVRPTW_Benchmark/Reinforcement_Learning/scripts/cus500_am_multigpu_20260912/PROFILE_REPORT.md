# AM Road Cus500：多卡工程验证与显存校准

日期：2026-09-12（本机时区）；记录和测试产物的 UTC 时间为 2026-09-13。
完整机器可读记录见 [calibration.json](calibration.json)。

## 部署结果

2080ti_3_1 的 GPU 0/1/2 同步训练一个 AM 模型。每卡 batch 4，全局 batch 12，
累积次数 1，每实例 30 条采样轨迹。训练上限 1700 步，验证上限 2550 步。
已有 Road Cus500 数据为 10,000 train / 500 val；正式启动生成 120,000 条全局
训练 ID stream，最多消费 60,000,000 次客户暴露。这里不生成新的 Road 实例。

多卡采用各 rank 独立 rollout、按全局 batch 归一化损失、梯度 SUM 后裁剪和更新。
同步不放进长度可变的解码循环。全局 EMA、分片验证、统一模型选择和早停均已覆盖。
AM 的 BatchNorm 在各卡计算本地统计，每次更新后广播 rank 0 的 running buffers；
它不等价于相同全局 batch 的单卡 BatchNorm。

## 实测

本机仅使用空闲 GPU 0/1，两张卡均为 RTX 2080 Ti 11 GiB；原有 GPU 2/3 的
TERRAN 作业继续运行。Python 使用 maojie 环境，CUDA PyTorch/NCCL 和版本见 JSON。
每进程 OMP/MKL/OpenBLAS/Numba 为 2 线程。

| 测试 | 每卡 batch | 训练/验证上限 | 更新次数 | 训练进程峰值/卡 | 结果 |
|---|---:|---:|---:|---:|---|
| 单卡 EMA | 8 | 1200/1800 | 第一次未完成 | — | OOM，排除 |
| 单卡 EMA | 5 | 1200/1800 | 10 | 9.018 GiB | 正常 |
| 单卡 EMA | 4 | 1600/2400 | 10 | 9.332 GiB | 正常 |
| 双卡 NCCL EMA | 4 | 1600/2400 | 20 | 9.410 / 9.410 GiB | 正常 |
| 双卡 NCCL greedy baseline | 4 | 1700/2550 | 20 | **9.943 / 9.943 GiB** | 正常 |

最终双卡测试每卡训练进程峰值为 **10,182 MiB**，包含训练进程的 CUDA/NCCL
开销。GPU 0 加上本机桌面后的卡上总峰值为 10,344 MiB。增大步数上限的原因是
初始随机策略有较多轨迹截断，同时为每卡较大的 batch 留足显存。没有通过空分配
占显存，也没有把三张卡的显存当作一张卡。

两个双卡测试均执行两轮验证，每轮 10 个固定实例、每实例 30 个候选；均为 10/10
完整可行。**这是启动、显存和更新行为的检查，不是完成训练后的 benchmark 结果。**
各试验参数不同，也不能拿它们的 cost 比较方法优劣。正式配置的验证集是 500 例。
训练中的单条随机轨迹仍有截断，FFP 不保证所有采样轨迹都在上限内完成。

greedy 阶段测试使用一次性 CLI 参数 `--baseline-warmup-epochs 0
--baseline-eval-size 2 --steps-per-epoch 10`，验证 actor 计算图仍在显存时能运行
baseline，并实际执行两次 paired test。正式实验保留原 AM 的 2500 次更新 EMA
warmup、之后 greedy baseline、每 2500 次更新用 64 个训练实例做 paired test。
测试中不显著或非有限的 paired-test p 值均未触发 baseline 替换。

两个测试产物位于本机独立工作目录下的：

- `EVRPTW_Benchmark/results/cus500_am_multigpu_20260912/profiling/nccl_2gpu_b4_h1600_1789280181`
- `EVRPTW_Benchmark/results/cus500_am_multigpu_20260912/profiling/nccl_greedy_2gpu_b4_h1700_1789280393`

每个目录保存实际命令、stream、源码摘要、GPU 身份、日志和 checkpoint；训练权重
及原始日志不提交 Git。首次临时双卡配置错误地将 early-stop 起点设为最大 epoch，
被训练器拒绝；修正该一次性测试配置后，两次正式工程 smoke 均正常完成。

## 正确性与恢复

98 项不同的回归检查通过。CPU 隔离测试中另有 1 个 CUDA 专项测试跳过；实际
GPU 路径由上述两轮共 40 次同步更新覆盖。Shell 语法和 `git diff --check` 通过。

- 真实 2/3 进程 Gloo 对比串行全局梯度，覆盖累积、未使用参数和裁剪顺序。
- 全局 EMA、AM BN buffers、数据分片、固定全局验证种子、rank 0 单份输出。
- 早停、单 rank 训练/验证失败、模型/AdamW/RNG/baseline probe 的精确恢复。
- checkpoint 原子发布，未提交日志备份回滚，状态 JSON 滞后和 best 别名修复。
- GPU smoke 的 e10/e20 模型和优化器状态均有限，59 个浮点模型状态张量均变化；
  AdamW step 正确，160 个流位置及 view ID 无重复，best 别名逐字节匹配选中 checkpoint。
- 启动器保护已有 compute 进程，支持保留 GPU 0 的已识别桌面服务，验证全局预算、
  数据摘要、精确 stream 长度和重复启动锁。

**本机没有空闲的三张 GPU，未执行三张物理 GPU 的训练。** 三 rank 同步逻辑在
CPU Gloo 上通过，GPU/NCCL 路径在两张卡上通过；2080ti_3_1 的实际 caliroute
环境、三卡通信及显存占用须由远端启动预检和首轮运行确认。

## 时间估计

本机双卡短测 EMA 平均 7.91 秒/更新（1600 步），greedy 阶段平均 10.16 秒/更新
（1700 步，包含两次小型 paired test）。同样每卡 batch 4，增加到三卡主要增加
每次更新的全局样本量，不能把每个 epoch 的耗时再除以三。

将 EMA 步数按 1700/1600 缩放，并将小样本验证按 500 例、三卡分片外推：
最早 5500 epoch 早停约 **17 小时**，10000 epoch 预算约 **32 小时**。
安排时可先预留 **18–36 小时**；这是短测外推，实际取决于远端 CPU/I/O、三卡通信、
轨迹长度和早停。正式启动后应按实际日志更新 ETA。
