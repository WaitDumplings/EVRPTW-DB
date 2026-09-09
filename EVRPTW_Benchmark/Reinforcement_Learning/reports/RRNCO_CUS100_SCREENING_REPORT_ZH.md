# RRNCO-EV Cus100 受控 Screening 报告

日期：2026-09-08
状态：单 seed、500 updates 的 exploratory road-aware baseline comparison；Cus100 已完成，Cus50 待运行。

## 结论

**RRNCO-EV 在本次 Cus100 受控实验中比 AM-EVRPTW 更好。** 全部 500 个 validation instances 均通过 canonical verifier；RRNCO-EV 的平均 objective 为 705.742 USD，AM 为 918.506 USD，降低 23.16%。平均道路距离降低 29.82%，平均车辆数降低 22.42%。

这是很强的候选信号，但目前只是单 seed、500 updates 和 validation 集结果，不能替代多 seed、5,000-update 和 test-set 正式评估。

## 适配方法

RRNCO-EV 是对上游 RRNCO road-relation architecture 的 EVRPTW-DB 适配，不是未修改的官方模型。

- row/outgoing 与 column/incoming 分离表示；
- inverse-distance expert（`k=25`）；
- row/column AFT encoder；
- 显式有向 `D/T/E/bearing` relation bias；
- decoder 显式使用 current-to-candidate `D/T/E`；
- 充电、SOC、TW、capacity、depot return、CS 重访和 route verifier 全部使用共享 canonical EVRPTW environment；
- 训练 baseline、optimizer 和数据协议与 AM 匹配，用于受控架构 screening。

上游参考 commit：`823d510dadf4dd711730ec4fbf337c356a0de6ae`，license 为 MIT。RRNCO-EV 的 energy channel 和 EVRPTW decoder state 是本项目扩展；训练不是上游 POMO recipe。
已核对 selected checkpoint 内的实际快照：`profile_id=rivian_energy_vehicle_cost_v2`、固定车辆成本 `413.6331536717643 USD`、电价 `0.39 USD/kWh`、能耗率 `0.38910505836575876 kWh/km`，训练和评估均引用同一 objective/reward 配置。

## 实验配置

| 项目 | 值 |
|---|---|
| scale / split | Cus100 / core train + core val |
| training seed | 1234 |
| logical updates | 500 |
| physical/effective batch | 4 / 4 |
| train trajectories | 5 per instance |
| train / validation max steps | 120 / 180 |
| validation cadence | every 100 updates |
| checkpoint selection | 50 fixed instances × 100 candidates |
| selected-checkpoint audit | 500 fixed instances × 100 candidates |
| optimizer | AdamW, lr=1e-4, weight decay=0.01 |
| objective | `413.6331536717643*K + 0.151750972762646*D` |
| verification | canonical EVRPTW verifier |

## Correctness 和 memory gate

- 定向关系、mask、sampling/greedy rollout、non-finite matrix rejection、baseline schedule 均有测试；
- 新测试加 AM/common/validation 回归合计 **213 passed**；
- pointer 的 key/value/logit 三组 projection 均有非零梯度；
- 2-epoch Cus100 train + full validation memory gate 通过，peak allocated **5.303 GiB**；
- `git diff --check`、Python compile 和 launcher `bash -n` 通过。

## 结果

### Checkpoint-selection subset

| 方法 | selected epoch | feasible | objective USD | distance km | vehicles | wall time | peak allocated |
|---|---:|---:|---:|---:|---:|---:|---:|
| RRNCO-EV | 500 | 50/50 | 762.112 | 388.368 | 1.700 | 385.85 s | 5.303 GiB |
| AM-EVRPTW | 400 | 50/50 | 916.417 | 532.964 | 2.020 | 260.35 s | 0.082 GiB |

注意：本次已完成的 v2 run 中，这 50 个实例由 `first(50)` 取得，全部属于 New York。launcher 已改为 v3、默认使用完整 500-instance validation，避免后续继续产生该 selection bias；若将来缩小 validation，必须实现城市分层固定抽样。

### 十城全量 validation

| 方法 | verifier PASS | objective USD | distance km | vehicles | eval wall time |
|---|---:|---:|---:|---:|---:|
| RRNCO-EV | 500/500 | 705.742 | 425.764 | 1.550 | 125.10 s |
| AM-EVRPTW | 500/500 | 918.506 | 606.698 | 1.998 | 103.88 s |

| 城市 | RRNCO-EV objective | AM objective | RRNCO-EV 降幅 | RRNCO 逐实例胜/总数 |
|---|---:|---:|---:|---:|
| Chicago | 591.659 | 902.837 | 34.47% | 46/50 |
| Dallas | 680.467 | 888.104 | 23.38% | 43/50 |
| Fort Worth | 756.931 | 957.279 | 20.93% | 47/50 |
| Houston | 835.080 | 995.698 | 16.13% | 42/50 |
| Los Angeles | 806.839 | 1,078.136 | 25.16% | 46/50 |
| New York | 762.112 | 916.417 | 16.84% | 45/50 |
| Philadelphia | 612.439 | 752.462 | 18.61% | 45/50 |
| Phoenix | 698.264 | 901.232 | 22.52% | 44/50 |
| San Antonio | 702.519 | 857.272 | 18.05% | 44/50 |
| San Diego | 611.107 | 935.625 | 34.69% | 49/50 |

剔除选模使用的 New York 50 例后，九城 held-back 450 例上：

- objective：699.478 vs 918.738，降低 23.87%；
- distance：429.919 vs 614.891，降低 30.08%；
- vehicles：1.533 vs 1.996，降低 23.16%；
- RRNCO-EV 逐实例 objective 胜出 406/450。
同车数配对分析：在 276/500 个两模型使用相同车辆数的实例上，RRNCO-EV 平均距离为 417.508 km，AM 为 525.882 km，降低 **20.61%**，并在 234/276 个实例上更短。九城 held-back 部分的同车数子集为 242 个，距离降低 **20.68%**（205/242 胜出）。因此，本次改善不仅来自减少车辆；但总体 objective 节省仍主要来自车辆数下降。

因此，“只在选模城市上过拟合”不能解释本次优势。

## 完成度与限制

已完成：adapter、共享环境接入、梯度/mask/verifier 测试、2-epoch train+val memory gate、500-update 同预算训练、500-instance 全量 validation 复验。

范围外项目（不作为本轮截止前必做项）：

- 多 seed 和 5,000-update 主实验；
- node-only / D / D+T / D+T+E 同 backbone ablation；
- Test1/Test2/Test3；
- 与 DRL-TS、TERRAN、ALNS/VNS/Gurobi 的正式统一比较；
- 参数量差异控制（RRNCO-EV 约 4.52M，AM 约 0.743M）。

当前结果冻结为 **单 seed、500 updates 的 road-aware baseline exploratory comparison**。正文优先报告未参与 checkpoint 选择的九城 450 例，完整 500 例和 New York 选模重叠放入附录；不称为正式独立测试集、未见城市泛化或 road-relation injection 的单因素因果证明。
