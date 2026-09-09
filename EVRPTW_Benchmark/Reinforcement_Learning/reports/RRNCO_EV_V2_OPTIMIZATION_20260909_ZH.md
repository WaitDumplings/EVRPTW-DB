# RRNCO-EV v2：图信息消融与 2080 Ti 训练优化

本轮实现用于检验：同一 RRNCO-EV backbone、同一训练/验证预算下，显式道路矩阵是否降低 canonical cost。当前完成了实现、梯度/消融测试和资源 gate；这些结果尚不能证明超过 full.sh 的四个 benchmark。

## 实现及兼容性

默认 `RRNCOEVPolicy()` 仍执行原 `legacy` AFT、随机距离摘要、完整 D/T/E 输入。与修改前 HEAD 模型比较，相同 seed 下初始参数、state-dict keys、row/col embedding、decoder logits 均逐元素一致。旧 checkpoint 不需要迁移，独立验证入口对新增参数使用兼容默认值。原 `run_single_seed_long_training.sh` 未修改默认行为。

v2 使用独立 `run_optimized_long_training.sh`，显式启用：

- `--aft-mode stable`：按 `bias_ij + key_jd` 的联合 logits 对候选节点 j 做 softmax；分块计算，并在反传时重新计算激活。
- `--relation-chunk-size 32 --checkpoint-bias`：同时控制 NAB 专家融合和 stable AFT。原共享 rollout 已经每次只 encode 一次，显存主要来自 6 层 × row/col 的稠密关系专家激活。
- `--distance-sampling nearest`：分别对 outgoing/incoming 距离取最近 k 个邻居摘要；小图不够 k 个时复制最远的已选邻居，不加入对角线。该模式没有摘要采样随机性。
- `--relation-temperature 5`：设置初始实际温度为 5；参数内部保存 log temperature。legacy 默认实际温度仍为 exp(5)，约 148.4。
- `--reinforce-baseline leave_one_out`：同实例其余轨迹的平均 cost 作为 baseline。至少需要 2 条轨迹。

stable AFT 的每个特征维度都使用如下候选权重：

```text
weight[i,j,d] = softmax_over_j(relation_bias[i,j] + projected_key[j,d])
mixed[i,d] = sum_j(weight[i,j,d] * projected_value[j,d])
```

原来的 `exp(softmax(bias))` 会压缩关系差异。修改后的计算保留其 logit 尺度，同时避免分拆 `exp(bias-max)` 与 `exp(key-max)` 时、两者偏好相反节点造成的下溢。±1000 对立 logits 测试验证输出与梯度有限、均值正确。中途测试过的 split-exp 版本已替换，其 stable 显存数字不作为当前实现证据。

## 消融边界与 CaliRoute 借鉴

`GRAPH_MODE` 控制所有显式道路矩阵路径，包括初始 ANE、encoder 关系 bias、decoder 当前边 bias：

| 模式 | 显式道路输入 | 坐标及其几何信息 |
|---|---|---|
| `full` | D/T/E | 保留坐标和 angle |
| `distance_time` | D/T | 保留坐标和 angle |
| `distance` | D | 保留坐标和 angle |
| `node_only` | 无；D/T/E 清零且 encoder 关系 bias 为零 | 保留初始节点坐标，禁用 encoder pairwise angle bias |

各组环境 action mask 始终由真实道路约束计算；`node_only` 是政策显式矩阵输入的消融，并不意味着环境不使用道路信息。各组保持 backbone 参数结构一致，训练时被禁用的关系分支不产生有效关系梯度。

本机 CaliRoute 的 `offline2online/models/nets/graph_model/encoder.py` 将距离 bias 直接用于 attention；`offline2online/trainer.py` 的 POMO trajectory 分支在同一实例内比较多条轨迹。这里借鉴“同实例 baseline 降低实例难度差异”的原则，采用其他轨迹均值：

```text
baseline[b,s] = sum(cost[b,t] for t != s) / (K - 1)
loss = mean((cost - baseline).detach() * log_likelihood)
```

当前 LOO 实现没有复制 CaliRoute 的 reward 重算、成功样本筛选和标准差归一化，保留 EVRPTW-DB 的 canonical objective、failure contract 与 route verifier。模型也没有直接移植 CaliRoute 的整套 dynamic KV 或 graph token，因此不能声称复现了 CaliRoute。

## 已测资源与正确性

完整小型指标保存在 [RRNCO_EV_V2_RESOURCE_GATE_20260909.json](RRNCO_EV_V2_RESOURCE_GATE_20260909.json)。GPU 为 RTX 2080 Ti，Torch 2.5.1+cu121。

| 实验 | 峰值 allocated | 用途及限制 |
|---|---:|---|
| Cus100，121 节点，batch6×5，legacy | 8057.02 MiB | 未训练模型单次 rollout+backward，无 optimizer step |
| 同实例、同模型，仅 legacy NAB chunk32+checkpoint | 342.13 MiB | distance 专家梯度范数从 0.0049594878 变为 0.0049594771；数值等价 |
| 最终 joint-softmax v2，Cus50 batch128×5，3 个 optimizer updates + 10×100 验证 | 3187914240 bytes，2.969 GiB | 真实训练步骤与验证已完成；baseline 使用 paper |

Cus100 前两组都是 0/30 完成，仅说明等价 activation 重计算节省内存，不能证明解质量或稳定训练吞吐。两次顺序执行还存在缓存/预热差异，不将其耗时差声明为加速比。

Cus50 gate 处理 384 个实例、1920 条训练轨迹，平均 64.27 步，最大 65 步；训练器报告 19.78 秒，验证报告 1.90 秒。验证 9/10 完成，环境与 verifier 无判定分歧。均值 objective 985.82 USD 只对 9 个成功实例计算，不能与完整成功的 500-instance 结果直接比较。3 步训练的 rollout 截断率约 88.8%，并非收敛状态。

现有 13 项模型测试通过，包括禁用矩阵扰动不改变 embedding/logits、chunk/checkpoint 输出与梯度等价、完整 rollout 反传与 canonical verifier。Cus100 batch32 的最终实现资源 gate 尚待在目标 GPU 验证。LOO 的独立测试由共享训练协议测试覆盖；上述 paper baseline 资源 gate 不冒充 LOO 学习验证。

## 新 long launcher

默认设置：

| 项目 | Cus50 | Cus100 |
|---|---:|---:|
| Physical/effective batch | 128/128 | 32/32，待 gate |
| 每实例训练轨迹 | 5 | 5 |
| 训练/验证 max steps | 65/98 | 120/180 |
| 最大/最低 logical epochs | 10000/5000 | 10000/5000 |
| 最低预算前/后验证间隔 | 100/250 | 100/250 |
| 每次验证实例/候选 | 500/100 | 500/100 |
| 最早 early stop | 5000 | 5000 |
| Early-stop patience | 10 次验证 | 10 次验证 |

这里每个 logical epoch 对应一个 effective batch，不代表遍历整个训练集。默认有 70 次计划验证；Cus50 最大实例 exposure 为 1280000，customer exposure 为 64000000。`BATCH_SIZE`、`EFFECTIVE_BATCH_SIZE` 可分别设置，后者必须为前者整数倍；同 backbone 对照应保持相同 effective batch 和冻结 stream。`TRAIN_TRAJECTORIES` 可设置为 16 等配置，但只有经过目标 GPU gate 的组合才能据实报告资源表现。默认 CPU 线程数为 2，四个常用线程环境变量均可覆盖。

新输出位于 `EVRPTW_Benchmark/results/RRNCO_EV_optimized_long_v2/`。冻结 stream 使用现有 `scripts/build_training_stream.py` 构造，保存 ordered view ID 内容摘要和源 index SHA256，运行前验证 scale/seed/exposure/source digest，并将 contract SHA256 写入训练命令。full/node_only 共享同一 ordered stream，但使用独立输出目录。文件锁防止并发重建，已存在的 run 输出拒绝覆盖。

先查看命令，不创建输出或启动训练：

```bash
DRY_RUN=1 SCALE=Cus50 GPU=1 \
  bash EVRPTW_Benchmark/Reinforcement_Learning/RRNCO_EVRPTW/run_optimized_long_training.sh
```

启动 full；GPU 编号由当前服务器资源调度指定：

```bash
PYTHON_BIN=/home/npg/miniconda3/envs/maojie/bin/python \
SCALE=Cus50 GPU=1 GRAPH_MODE=full RUN_TAG=long_seed1234 \
  bash EVRPTW_Benchmark/Reinforcement_Learning/RRNCO_EVRPTW/run_optimized_long_training.sh
```

相同参数改 `GRAPH_MODE=node_only` 并选择另一可用 GPU 即是对应消融。`PREPARE_ONLY=1` 仅生成并核验冻结 stream、写入 `command.sh`、源码 commit/patch 和 launcher snapshot；之后可执行该 `command.sh`。本次 launcher 开发只运行 CPU preparation 和 dry-run，没有启动新训练。

要检验“图信息带来改善”，应先比较同 backbone 两组的完整成功率与成功条件一致时的 canonical cost，再与四个 benchmark 的相同实例、candidate budget、训练 exposure/compute 报告比较。单 seed 的 checkpoint 选择结果仍是探索性证据，test split 不用于参数或 checkpoint 选择。
