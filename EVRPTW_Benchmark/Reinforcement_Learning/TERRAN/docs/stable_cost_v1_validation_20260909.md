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

## 扩大 batch 与恢复验证

用户反馈 GPU 利用率偏低后，将 Cus500/Cus1000 的物理 batch 分别提高到 128/64，有效独立实例 batch 都提高到 256。每实例轨迹数仍为 16，PPO chunk 仍为 16。关闭 stable trainer 不使用的旧 reward 分项汇总；固定种子测试确认动作、reward、return 和终止语义保持一致。

每个规模先完成一个物理 batch 的真实数据采样与 PPO 更新，再用于正式多 microbatch 配置。下表时间为单物理 batch 试跑，不能直接当作有效 batch 256 的正式 epoch 时间。

| 规模 | 新物理 batch | 正式累积次数 | 正式有效 batch | GPU allocated 峰值 GiB | GPU reserved 峰值 GiB | 试跑更新后 KL | 单物理 batch 试跑秒数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cus500 | 128 | 2 | 256 | 17.49 | 22.29 | 0.019313 | 73.1 |
| Cus1000 | 64 | 4 | 256 | 27.40 | 35.17 | 0.000145 | 138.5 |

两组试跑均成功完成，轨迹完成率均为 1.0，无 OOM、非有限 loss 或梯度。nvidia-smi 的短时更新阶段采样平均利用率分别约 88%（24 个样本）与 91%（11 个样本）；这不是整个 epoch 的平均利用率，也不用于声称固定的加速比例。采样仍涉及 CPU 环境推进和每步设备同步。

全套 TERRAN 测试更新为 **261 passed**。新增测试覆盖显式 batch 恢复白名单、原 checkpoint 签名完整性、拒绝其他训练设置变化、真实 trainer 恢复后的模型/两个 Adam/PopArt/StableState 精确一致，以及恢复到已保存的采样游标。

原运行在完整 checkpoint 后迁移：Cus500 epoch 3（cost 阶段，384 个实例），Cus1000 epoch 1（feasibility 阶段，128 个实例）。快照位于 results/TERRAN_stable_cost_v1/batch_resize_20260909；新配置通过原 resolved_config 与 --allow-batch-resize-resume 生成，保留已有 optimizer 学习率及所有训练状态。原输出目录保留。
