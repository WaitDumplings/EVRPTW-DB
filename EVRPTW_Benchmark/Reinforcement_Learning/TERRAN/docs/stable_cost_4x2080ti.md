# TERRAN stable-cost：四张 RTX 2080 Ti 共同训练 Cus500 / Cus1000

每台四卡机器运行一个同步训练任务。四张卡各保存完整 actor、critic 和优化器，分担不同训练实例；所有局部 rollout 完成后，一起更新同一套参数。可用两台四卡机器分别训练 Cus500 和 Cus1000，也可以在不同机器上运行不同 seed。本入口使用单机 `torchrun`，不包含跨机器通信。

## 参数与显存

| 规模 | GPU 数 | 每卡物理 batch | 每卡累积次数 | 全局有效实例 batch | 每实例轨迹数 | PPO 时间 chunk |
|---|---:|---:|---:|---:|---:|---:|
| Cus500 | 4 | 32 | 2 | 256 | 16 | 16 |
| Cus1000 | 4 | 8 | 8 | 256 | 16 | 16 |

全局 batch = 每卡物理 batch × 每卡累积次数 × GPU 数。每轮共有 4096 条训练轨迹。使用 FP32，保持现有 stable-cost 的 USD reward、模型、学习率和优化规则。配置位于 `../configs/stable_cost_v1/4x2080ti/`，与旧 `Reinforcement_Learning/configs/2080ti/` 的 Cus50/Cus100 协议分开。

四张卡的显存分别使用，每张卡都必须装下自己的模型、优化器和 replay chunk；不能把它们当作一张 44 GiB 卡。这里采用较小的每卡 batch，保留通信与 CUDA 工作区余量。旧 Cus50/Cus100 的 2080 Ti 校准不能证明新 Cus500/Cus1000 配置已在该型号通过，目标机器仍需先做短程资源检查。CPU rollout 与各进程的数据缓存也占用主机 RAM。

## 从新配置启动

在目标机器激活项目环境，并从仓库根目录执行：

```bash
python -m EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.stable_cli launch \
  --scale Cus500 --machine-profile 4x2080ti --gpus 0,1,2,3 \
  --dataset-root /path/to/us_11city \
  --output-dir /path/to/results/terran_cus500_4gpu
```

Cus1000 使用 `--scale Cus1000` 和另一个输出目录。`launch` 启动后台 `torchrun`，输出进程信息后返回；不传 checkpoint 就会从头训练。可用 `--warm-start-checkpoint /path/to/actor.ckpt` 只加载 actor；这会新建 critic 和优化器，不适用于保留完整训练进度。

建议首先用独立输出目录与 `--epochs 2` 做目标机器试跑；检查 `metrics.jsonl` 中显存、每轮耗时、完成率和 KL。该短程配置用于执行验证，正式恢复需要保留源 checkpoint 的总 epochs 等训练设置。短程成功不代表长期收敛。

## 从现有单卡训练完整恢复

把源实验的 `checkpoint_latest.pt` 和 `resolved_config.yaml` 复制到目标机器。数据须是同一份 train/val 索引与对应 materialized families。复用源配置，只改变卡数与 batch，可保留 actor、critic、两个 Adam、PopArt、乘子、训练阶段和全局采样游标：

```bash
python -m EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.stable_cli launch \
  --scale Cus500 --config /path/to/source/resolved_config.yaml \
  --resume /path/to/source/checkpoint_latest.pt \
  --allow-batch-resize-resume --gpus 0,1,2,3 \
  --physical-batch-size 32 --effective-batch-size 256 --ppo-step-chunk-size 16 \
  --dataset-root /path/to/us_11city --allow-dataset-relocation-resume \
  --output-dir /path/to/results/terran_cus500_resumed_4gpu
```

Cus1000 将 scale 改为 `Cus1000`、每卡物理 batch 改为 `8`，全局 batch 仍为 `256`。恢复时使用源 `--config`，不同时传 `--machine-profile`，以保留原学习率、缓存与其他训练设置。默认 seed 为 `1234`；其他 seed 的实验须显式传 `--seed 原seed`。已经因 KL 回退降低的实际 optimizer 学习率也原样恢复。

默认恢复仍严格校验配置签名。`--allow-batch-resize-resume` 只放行物理/有效 batch、累积次数、卡数和时间 chunk；`--allow-dataset-relocation-resume` 只允许数据换挂载路径，仍校验训练索引哈希，以及源配置保存的验证索引哈希；不会逐文件校验所有 materialized families。两个选项均记录在 provenance、contract 和 checkpoint。改变卡数会改变采样随机数的分配和浮点求和顺序，因此不保证与原硬件逐位一致。

## 同步与验证行为

- 每个 rank 从全局确定性实例序列读取独立连续块。恢复游标以全局已消费实例数计算，不会把同一批实例重复分配给四张卡。
- 所有 rank 使用全局轨迹数作为 loss 分母，累积完局部时间 chunk 后对梯度求和。没有在可变长度解码循环内执行 collective。
- PopArt 的加权矩、advantage RMS、KL、失败率和乘子基于所有 rank 的样本；所有 rank 同时接受或回退 actor 更新。
- rank 0 写训练日志与 checkpoint，并在原验证计划下验证 500 个实例、每实例 100 条候选。其他 rank 等待，验证不重复四遍。验证前先保存已完成 epoch。
- 默认进程组超时为 7200 秒，可通过 `TERRAN_DISTRIBUTED_TIMEOUT_S` 调整，以覆盖长验证。工作进程报错会使整个 `torchrun` 任务失败；不自动从不完整更新继续。

```bash
python -m EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.stable_cli status \
  --output-dir /path/to/results/terran_cus500_resumed_4gpu
```

`run_state.json` 记录训练 worker PID、全部 worker PID 和 world size，`launch_process.json` 记录 torchrun 父进程；`stdout.log` / `stderr.log` 收集所有进程输出。`metrics.jsonl` 中的 cost、成功率、有效 batch 和吞吐量是全局统计，显存峰值取各卡最大值。

实际速度取决于 2080 Ti 算力、CPU 环境推进、存储、PCIe 通信和每卡 batch，不能直接承诺四倍提速，或快于现有单张 RTX 6000 Ada。先用目标机器连续几轮的 `epoch_wall_s` 测量后再估算总训练时间。

## 本地验证范围

2/4 进程 CPU Gloo 测试覆盖全局 batch 更新等价、模型/两个 Adam/PopArt 的副本一致性、不同轨迹长度、KL 回退、错误传播、rank 0 独占 checkpoint/验证，以及从单卡 checkpoint 恢复到多卡。四进程 torchrun CLI 测试覆盖实际进程组启动、锁、PID、重复启动拒绝和数量不匹配时退出。

两张现有 RTX 6000 Ada 上的实际 NCCL 小批量测试也完成了两轮 PPO 更新，更新后模型副本逐元素一致。该测试验证 CUDA 通信路径，不用于估计大规模训练吞吐。

另在现有 RTX 6000 Ada 上限制 PyTorch 每进程分配上限为 10 GiB，使用实际 Cus500/Cus1000 数据完成一轮单物理 batch 采样与 PPO 更新：

| 规模 | 物理 batch | n_traj / chunk | allocated 峰值 GiB | reserved 峰值 GiB | 轨迹完成率 |
|---|---:|---|---:|---:|---:|
| Cus500 | 32 | 16 / 16 | 4.47 | 5.65 | 1.0 |
| Cus1000 | 8 | 16 / 16 | 3.47 | 3.96 | 1.0 |

该检查验证长序列执行和显存余量，未测量 2080 Ti 实际耗时，也不包含四卡 NCCL 通信的显存。现有两组正式单卡训练没有被替换。
