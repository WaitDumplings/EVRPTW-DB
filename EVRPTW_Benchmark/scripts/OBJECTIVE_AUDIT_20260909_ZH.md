# 非学习 Solver 与 DRL 目标一致性审计（2026-09-09）

结论：当前 `scripts/` 包装器启动的 Gurobi、ALNS、VNS-TS，与显式采用 `rivian_energy_vehicle_cost_v2` 的 DRL，优化和导出的经济成本公式一致。没有发现“非学习方法仍优化距离、只在导出时换成美元”的问题。这个结论不涵盖未传配置的直接 Python CLI、旧 distance_v1 结果，也不表示各方法的可行域或计算预算完全一致。

## 共同目标与配置入口

`C = 0.151750972762646 × D_km + 413.6331536717643 × K`，单位 USD。

- `D` 为完整路线的有向距离之和，包含返回 depot；`K` 为 depot 到非 depot 的有效出发次数。每次出车收取一次固定费用。
- 电费系数来自 `0.39 USD/kWh × 0.38910505836575876 kWh/km`。经济电费按距离和统一能耗系数计算，包含初始电池提供的行驶能量，不另外按充电站购电量收费；实例的 `E` 矩阵用于 SOC、充电时间和可行性。
- 配置：[rivian_energy_vehicle_cost_v2.json](../Reinforcement_Learning/configs/rivian_energy_vehicle_cost_v2.json)。共同入口 [scripts/common.sh](common.sh#L28) 引入 [test_scripts/common.sh:20](../test_scripts/common.sh#L20)，三个分支均传入同一个 `--objective_config`（[common.sh:116](common.sh#L116) 起）。
- 两套目标实现的有效公式一致：[Core objective.py:47](../../EVRPTW_Core/evrptw_core/objective.py#L47)、[DRL objective.py:47](../Reinforcement_Learning/common/objective.py#L47)。显式读取 v2 后配置字段也一致。

## 实际优化与导出

| 方法 | 搜索/训练中的目标 | 最终成本口径 |
| --- | --- | --- |
| Gurobi | [gurobi_solver.py:385](../Exact/Gurobi_Solver/gurobi_solver.py#L385) 构造距离表达式；[:423](../Exact/Gurobi_Solver/gurobi_solver.py#L423) 最小化距离费用与 depot 出边车辆费用之和。v2 不使用额外车辆数词典序目标，包装器也显式禁用该 tie-break。 | 提取路线后计算 `fields(D, len(routes))` 并独立验证（[:198](../Exact/Gurobi_Solver/gurobi_solver.py#L198)）；[run_gurobi.py:476](../Exact/Gurobi_Solver/run_gurobi.py#L476) 分开导出公里与 USD。旧 reference CSV 的 `objective` 列取有效 `objective_value`（[:605](../Exact/Gurobi_Solver/run_gurobi.py#L605)），并非固定取距离。 |
| ALNS | [solver.py:812](../MetaHeuristics/ALNS_Solver/solver.py#L812) 对可行解计算 `distance_unit_cost × D + vehicle_fixed_cost × len(routes)`，无效解为无穷大；实际接受准则使用该成本。部分修复/移除启发式按距离构造候选，不改变 incumbent 的评价目标。 | [run_alns.py:197](../MetaHeuristics/ALNS_Solver/run_alns.py#L197) 先 canonical replay，再用验证距离重算经济成本并记录 incumbent；[:281](../MetaHeuristics/ALNS_Solver/run_alns.py#L281) 保存目标配置和费用字段。 |
| VNS-TS | [solver.py:582](../MetaHeuristics/VNS_TS_Solver/solver.py#L582) 的搜索函数可加入容量/时间/电量违约及多样化惩罚；[可行解接受比较 :643](../MetaHeuristics/VNS_TS_Solver/solver.py#L643) 和[最佳可行解 :1199](../MetaHeuristics/VNS_TS_Solver/solver.py#L1199) 使用关闭惩罚的经济成本。 | [run_vns_ts.py:246](../MetaHeuristics/VNS_TS_Solver/run_vns_ts.py#L246) canonical replay 后重算无惩罚成本；[:373](../MetaHeuristics/VNS_TS_Solver/run_vns_ts.py#L373) 保存距离、车辆和 USD。 |
| DRL | [env.py:355](../Reinforcement_Learning/EVRPTW_Env/env.py#L355) 每个有效动作累计距离，depot 出发累计车辆，并使用相同增量经济成本生成奖励；训练中的缩放、未完成惩罚和可选 PBRS 不属于最终 USD。Fast 环境复用相同 `_apply_action`。 | [env.py:771](../Reinforcement_Learning/EVRPTW_Env/env.py#L771) 汇总实际成本；[evaluation.py:52](../Reinforcement_Learning/common/evaluation.py#L52) 按成功候选的完整路线重算 D 和出发次数，再验证与导出。失败轨迹不能通过补一条虚拟回库边变为可行结果。 |

共享 canonical replay 位于 [route_validator.py:71](../Exact/Gurobi_Solver/route_validator.py#L71)。ALNS/VNS 使用 [IncumbentReplayCache](../MetaHeuristics/benchmark_common.py#L846) 接入它。公里字段 `objective_distance_km` 与 `objective_value`/`objective_cost_usd` 分开保存；比较时应同时核对 profile、单位和 feasibility，不能将距离列与成本列混用。

正常 solver 输出每条路线只有首尾 depot，因此 `len(routes)` 等于 DRL 的 depot 出发次数。DRL 的 [route_dispatch_count](../Reinforcement_Learning/common/objective.py#L148) 对出发边直接计数。手工将多趟路线拼成含内部 depot 的单条列表会破坏 `len(routes)` 口径；ALNS/VNS 内部验证禁止这种路线，Gurobi 模型也不生成它。

## 直接 Python CLI 的默认值不同

三个入口的 `--objective_config` 均默认为空字符串，未提供时执行 `ObjectiveConfig()`，目标为 **distance_v1（km）**，固定车辆费用系数为零：

- [run_gurobi.py:791](../Exact/Gurobi_Solver/run_gurobi.py#L791)，加载分支 [:817](../Exact/Gurobi_Solver/run_gurobi.py#L817)。
- [run_alns.py:451](../MetaHeuristics/ALNS_Solver/run_alns.py#L451)，加载分支 [:490](../MetaHeuristics/ALNS_Solver/run_alns.py#L490)。
- [run_vns_ts.py:588](../MetaHeuristics/VNS_TS_Solver/run_vns_ts.py#L588)，加载分支 [:641](../MetaHeuristics/VNS_TS_Solver/run_vns_ts.py#L641)。

因此绕过包装器直接调用时，必须显式传 `--objective_config EVRPTW_Benchmark/Reinforcement_Learning/configs/rivian_energy_vehicle_cost_v2.json`，才能使用这里审计的经济目标。历史 distance_v1 解即使事后折算成本，也不能视作已按 v2 优化的解。

## 实例回放复验

CPU 上读取 Cus50 Test1 的实际实例 `iv_9389260a5c89a45cbd076a52`，使用实例附带的 50 条单客可行证书路线，canonical replay 通过。随后直接调用 ALNS/VNS 成本函数，并在真实 DRL Fast 环境逐步回放相同路线；所有动作通过 mask，100 步后成功。

| 项目 | 数值 |
| --- | ---: |
| 距离 D / km | 2686.613366127014 |
| 出发次数 K | 50 |
| 电费 / USD | 407.69619174690104 |
| 车辆费用 / USD | 20681.657683588215 |
| Core、DRL objective、ALNS、VNS 无惩罚目标、DRL env 总成本 | **21089.353875335117** |

这是成本核算验证，**不是优化后的 benchmark 结果**。上述证书回放没有使用 Gurobi 或 GPU；另已通过真实 Gurobi 小型实例测试，验证 v2 模型目标与路线重放成本一致。回放没有施加生产评估的截断步数；若配置 rollout cap 为 98，这条需要 100 步的证书路线会被截断，属于评估可行域差异。

团队已完成成本相关 **123 项测试通过**（一次 VNS 测试的初始失败由遗漏 `PYTHONPATH` 引起，补齐路径后通过），时间口径相关 **52 项测试通过**。52 项时间及求解器测试中的 Exact 17 项全部通过、无跳过，其中包括 [test_gurobi_cost_v2_objective_matches_replayed_route](../Exact/Gurobi_Solver/tests/test_stage2_gurobi.py#L179) 的真实 Gurobi 小型实例求解；没有启动正式 benchmark。

## 与目标函数分开判断的差异

- **可行域**：包装器设置 Gurobi `cs_copies=2`；[每个充电站副本最多使用一次](../Exact/Gurobi_Solver/gurobi_solver.py#L394)，即一个物理充电站在整份解中最多访问两次。DRL [每条路线禁止重复访问充电站](../Reinforcement_Learning/EVRPTW_Env/env.py#L456)，返回 depot 后重置；另有动作 mask、最大步数与候选采样数。共享 replay 的接受规则本身不能消除各搜索器的这些限制。
- **惩罚**：VNS 可暂时搜索不可行解，DRL 有失败惩罚和奖励塑形。应比较经过相同验证的成功解之原始 USD，而非训练 reward、带惩罚 generalized cost 或所有轨迹平均值。
- **计算预算**：非学习包装器采用每实例 1800 秒及 60/300/900/1800 秒 checkpoint；DRL 的训练时间、推理时间、candidate 数和 horizon 是另一组预算。成本公式相同不等于预算相同，预算不属于 objective function。

可在仓库根目录执行以下轻量命令，确认三个实际启动命令均携带 v2 配置；不会启动求解器：

```bash
EVRPTW_DRY_RUN=1 bash EVRPTW_Benchmark/scripts/Gurobi/Cus50/test1.sh
EVRPTW_DRY_RUN=1 bash EVRPTW_Benchmark/scripts/ALNS/Cus50/test1.sh
EVRPTW_DRY_RUN=1 bash EVRPTW_Benchmark/scripts/VNSTS/Cus50/test1.sh
```
