# ablation 分支：非学习求解器检查

本分支用于后续 final 实验。Gurobi、ALNS、VNS-TS 的新成本实验统一为：

```text
C(x) = 0.15175097276264593 × D_time_km(x) + 413.6331536717643 × K(x)
```

成本与能耗均来自最快时间路径：成本距离为 `running_time_path_distance_km`，
时间为 `running_time_shortest_matrix_s`，能耗为对应的
`running_time_path_energy_kwh`。共同 loader 在创建 adapter、缓存及独立 replay
前完成成本矩阵映射；不改写发布实例，不重解释历史 D_dist 结果。

## 这次继续修正的两处 ALNS 候选规则

此前 incumbent、SA 接受、自适应算子奖励和插入解的最终评分已使用美元成本，
但仍存在以下候选集裁剪：

1. 初始化先按最近节点距离截断待合并路线，可能丢失插入成本更低的路线。
   现在先枚举允许的插入位置，按 USD 边际确定路线 shortlist，再做有预算的完整
   可行性检查。是否合并与保留单客路线的完整成本比较，其中包含车辆固定费用。
2. 名称为 time 的 repair 先按时间增量取 top-k。成本模式现改为按 USD 插入
   增量排序；时间仅用于约束检查。显式 legacy distance 模式保留历史规则。

新的 ALNS profile 是 `alns_stage2_cost_operators_v4`，元数据中的算子评分版本为
`monetary_marginals_and_shortlists_with_dispatch_v2`。固定的候选预算仍存在；
站点修复前的金额增量是候选排序代理，不能解释为穷举所有修复后的全局最优插入。

VNS-TS 已使用完整 USD 移动增量（受影响源/目标路线、清理后的站点与路线数变化）
进行成本候选排序；初始化也先比较货币增量，最终 tabu/SA/最好解均以同一成本
计算。Gurobi 优化同一电费加车辆费目标，无启发式 distance-only 更新路径。

Shaw relatedness、时间窗紧迫程度、地理区域和随机破坏保留为邻域多样化机制，
不作为 incumbent 目标或算子奖励。包含 `distance` 的历史函数/算子名称不表示
新成本运行使用旧 D_dist。显式 legacy distance 分支仅用于历史复现；本分支
`scripts/` 的 cohort shell 显式传入货币 profile，不会进入该分支。

## 输出与预算

所有 cohort shell 默认输出到：

```text
EVRPTW_Benchmark/results/ablation_final_nonlearning_dtime/
```

可用 `EVRPTW_TEST_RESULTS_ROOT` 自定义新目录。运行契约同时记录 objective
profile、矩阵来源、算法版本及预算；不把历史 D_dist 或旧算法结果作为相同运行
静默续跑。默认 30 分钟预算、1/5/15/30 分钟 checkpoints 保持原设置。

单实例 `Gurobi/Cus500/random_one.sh` 仍为单独入口，保留原生 Gurobi 线程及
停止参数，记录每次回调可见的 gap/objective/bound 变化及相应 Runtime。
它不继承 cohort shell 的线程数与 30 分钟预算。

## 验证记录

- MetaHeuristics 全目录：`126 passed, 4 skipped`。跳过项为不可用的旧源码 oracle；
  legacy golden 测试通过。
- 新回归明确构造“时间更短但 USD 更贵”及“地理更近但插入 USD 更贵”的候选，
  验证成本路径不会按上述旧代理裁剪；legacy 时间排序单独验证。
- 32 个 shell 全部通过 `bash -n`。
- ALNS Cus100、VNS-TS Cus500、Gurobi Cus1000 T1 shell 的 dry-run 均确认货币
  profile、30 分钟预算及 ablation 独立结果目录；没有启动正式非学习求解。
- 初次审计及 D_time loader/replay、车辆费、非度量站点清理等回归依据见
  [此前审计](OBJECTIVE_AUDIT_20260919_ZH.md)。
