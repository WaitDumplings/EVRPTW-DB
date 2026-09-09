# stable_cost_v1 实施验证（2026-09-09）

实现与配置说明见 [stable_cost_v1.md](stable_cost_v1.md)。本次保留 TERRAN 构造式模型，新增原始 USD reward、动态剩余任务状态、独立成本/失败 critic、共享 PopArt、CPU rollout 缓冲、逻辑 batch PPO、更新后经验 KL 回退，以及可恢复的确定性采样。

## 自动检查

`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/exx/anaconda3/envs/maojie/bin/python -m pytest -q EVRPTW_Benchmark/Reinforcement_Learning/TERRAN/tests`

240 项通过，包含：旧接口兼容、新旧 checkpoint 的两个评估入口、原始成本账本、终止/截断/零需求与初始不可行边界、CPU replay 一致性、跨节点数模型调用、PopArt 保值与加权统计、critic 梯度隔离、分块梯度等价、双优化器恢复、KL 拒绝后的参数/Adam 状态恢复、跨数据 pass 的采样游标恢复和 CLI 数据身份校验。

原 HEAD 中已有的 5 个 canonical 测试失败已在独立进程复现：一处预期未包含原有 critic 配置字段，四处假 rollout 未使用真实的三维轨迹形状。只更新了对应测试预期/fixture，没有为这些旧失败更改生产行为。

## 实际数据与 GPU 集成

使用当前 Stage-2 train/val 数据和 RTX 6000 Ada GPU。Cus100、Cus500、Cus1000 均完成两轮小批训练；第二轮覆盖成本训练路径。Cus500 和 Cus1000 各额外验证了两个真实 validation 实例，每实例两个候选，独立 verifier 全部通过。该小批检查验证执行与约束语义，不用于宣称长期收敛或成本改善。

物理 batch 测试使用各规模原 best checkpoint 的 actor 权重，新 critic、PopArt 和优化器重新初始化：

| 规模 | 物理 batch | 每实例轨迹 | 时间 chunk | 单轮完成率 | 更新后经验 KL | GPU allocated 峰值 MiB | GPU reserved 峰值 MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cus500 | 64 | 16 | 16 | 1.0 | 0.01133 | 8999 | 11468 |
| Cus1000 | 32 | 16 | 16 | 1.0 | 0.000134 | 14054 | 18098 |

正式配置保持有效 batch 128，分别累积 2 和 4 个物理 batch；所有 rollout 收集完成后才更新策略。每个 epoch 原子保存最新 checkpoint，验证仍使用独立 verifier。原运行日志不覆盖，原 best/latest checkpoint 另存初始化快照；新实验记录 actor 来源和对应 hash。

正式新实验与旧 `drl_rq_protocol_frozen_v1` 的 reward、实例曝光量和更新几何不同，需作为独立方法版本比较。
