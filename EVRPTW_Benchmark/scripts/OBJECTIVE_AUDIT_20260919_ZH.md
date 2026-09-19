# 非学习求解器成本与算子审计（2026-09-19）

本次按确认后的论文口径，将**新非学习成本实验**改为最快时间路径距离 D_time。
审计并非确认旧实现全部正确：之前最终 incumbent 按美元比较，但 ALNS/VNS-TS
若干候选生成、筛选和边际评分仍使用距离代理，且可能遗漏增减车辆的固定费用。
这些路径现已修正。

## 统一计算口径

```text
C = (0.39 × 100/257) × D_time_km + 413.6331536717643 × K  [USD]
```

- 成本距离：`running_time_path_distance_km`。
- 时间传播：`running_time_shortest_matrix_s`。
- 电池消耗：`running_time_path_energy_kwh = (100/257) × D_time`。
- K：有效 depot 出发次数；在禁止内部 depot、禁止无客户路线的解接口下等于路线数。
- 约束惩罚、接受温度和多样化机制不属于最终报告成本。

`EVRPTW_Core/evrptw_core/objective.py::select_objective_distance` 为新成本运行
创建实例副本，将 D_time 供给模型、启发式 adapter 和独立 replay。原始 D_dist
保留，发布实例及旧结果文件不修改。缺少 D_time 时明确报错，不静默回退。
三个 Python CLI 默认读取 `rivian_energy_vehicle_cost_v2.json`；显式
`--objective_config ""` 才选择 legacy distance 模式。数值系数 profile 与
`objective_distance_source` 必须一起记录，不能仅凭 v2 名称判断路径口径。

## 审计与修正

| 方法/模块 | 已核查及修正 |
|---|---|
| Gurobi | 模型优化电费加车辆费，输入距离切换为 D_time；解后重放同源。保留真实 ObjBound/MIPGap，不因 OPTIMAL 状态强制 gap 为 0。MIPSOL 候选与 incumbent 分开处理。 |
| ALNS | greedy/regret/time-zone 插入改为美元边际，新路线加固定费；worst removal 计入站点清理、消失路线和节省的固定费；站点增删及构造期合并按成本比较。自适应算子奖励、SA 接受和 incumbent 比较均用成本。 |
| VNS-TS | 成本模式的候选排序/截断改为完整货币增量，涵盖源/目标路线、站点清理和路线数改变；不再先按近邻距离截断成本候选。合并拒绝成本增加的可行移动。最终广义成本、tabu/SA 和最好可行解一致使用成本。 |
| Replay/运行契约 | Exact 入口补齐内部 depot 和无客户路线拒绝；成本距离 source 写入解与运行契约。新 D_time 输出与旧 D_dist 输出隔离，禁止把不兼容契约当成已完成记录续跑。 |

D_time 不保证满足距离三角不等式，删除站点或合并路线不能假设必然减少成本。
相关评分与候选回归已覆盖这一点。

ALNS 的 Shaw relatedness、地理/时间相似性以及随机扰动仍可用于决定破坏哪些客户，
它们是邻域构造和多样化机制，不是目标函数或算子优劣评分。保留这些机制不代表
搜索以 D_dist 为目标。显式 legacy distance 模式仍保留旧算法路径供历史复现。

## 实例及回归验证

本轮 MetaHeuristics 全目录回归：**123 passed，4 skipped**（旧源码 oracle 不可用，
legacy golden 仍通过）；Exact 全目录：**29 passed**，包含真实小实例 Gurobi 求解。

- Exact 测试包含真实 Gurobi 优化、D_time/D_dist 不同的实例、独立重放、原生默认
  参数、MIPGap 与实时日志、同一秒内多次 gap/objective/bound 变化逐次落盘、
  无 incumbent/无穷界、回调较差候选、随机抽样。
- ALNS 测试覆盖新路线费用、清空源路线收益、非度量站点清理、成本/距离排序相反
  的移动，以及 legacy 行为；VNS-TS 覆盖对应货币候选与 legacy golden。
- 共享测试核查 D_time 映射的幂等性、原实例不变、显式 legacy 恢复、缺字段报错、
  loader/replay 一致和 resume 契约隔离。
- 真实 T1 Cus500 `iv_732358e746ca89a83d8409f5` 的同一组单客证书路线：
  D_dist = 20065.332339286804 km，D_time = 24458.522798538208 km；
  K = 500。重放成本从 209861.5105371747 变为 210528.1814628977 USD。
  这是**核算检查，不是求解成绩**，没有将生成期证书当作 BKS。
- 单实例 shell 已在真实 Cus500 上 dry-run：固定种子 20260919 选择
  `iv_a7379bb54d164041134f0760`，未启动完整 Cus500 MILP。

## 启动与时间记录

单实例 native-default Gurobi 入口：

```bash
conda activate maojie
bash EVRPTW_Benchmark/scripts/Gurobi/Cus500/random_one.sh
```

[完整说明与输出字段](Gurobi/Cus500/README_random_one.md)。此入口不设置 Threads、
MIPGap 或默认 TimeLimit，不使用多实例 worker pool。现有整组 benchmark 包装器
仍有各自的 30 分钟/每实例单线程预算，两种运行配置不可混淆。Gurobi 优化器
Runtime 和包括模型构建的 wall time 分别保存，不能互换。

本次没有重跑或重算历史 DRL archive，没有冻结新的 BKS，也不把新非学习 D_time
结果视为与旧 DRL D_dist 结果已经对齐。论文最终比较需统一评测协议。
