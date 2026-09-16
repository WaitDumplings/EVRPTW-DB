本次修复使 ALNS/VNSTS 在搜索候选构造、可行最优解保存时一致考虑新版经济成本，并修正 Gurobi 的 incumbent 记录、恢复契约和最优性报告。

成本公式保持为 `C = 0.15175097276264593 × D_km + 413.6331536717643 × K`；车辆数按完整路线的出车次数计。充电和容量/时间窗可行性仍使用各求解器的原有模型。

- ALNS：greedy/regret 插入按 `a*ΔD+b*ΔK` 比较；新增车辆计固定费用。时间/随机类探索算子保留各自作用。checkpoint 保存成本契约，恢复时核对系数、best 路线重算成本与温度，拒绝将旧公里值作为美元 best。
- VNSTS：fast 的候选预筛计入源路线删除、目标插入及车辆变化，保留可行性恢复候选。fast/full 独立保存已评估的更优可行解，探索不可行解不再丢失这一 incumbent。full 枚举使用规范化候选评分，消除空路线误计车辆。
- Gurobi：MIPSOL 回调仅用更优目标更新 incumbent，并保留独立路线复验；旧的距离目标、不同时间预算/求解设置的结果不能凭 instance_id 被 `skip_completed` 复用。保留真实 bound/gap，非零 MIPGap 容差终止不能伪报 gap=0 或严格最优。建模计入 solve 总预算，优化仅使用剩余时间，checkpoint 使用相同墙钟。

三个 Python runner 默认使用 `rivian_energy_vehicle_cost_v2.json`。要显式做旧距离实验，传入：

```bash
--objective_config EVRPTW_Benchmark/Reinforcement_Learning/configs/distance_v1.json
```

ALNS/VNSTS 算法 profile 已提高；默认结果目录改为：

```text
EVRPTW_Benchmark/results/CLE_EVRPTW_v2_test_1m_5m_15m_30m_cost_v2_searchfix_20260916
```

原有 shell 路径保持不变，例如：

```bash
bash EVRPTW_Benchmark/scripts/ALNS/Cus500/test1.sh
bash EVRPTW_Benchmark/scripts/VNSTS/Cus500/test1.sh
bash EVRPTW_Benchmark/scripts/Gurobi/Cus500/test1.sh
```

每个命令会启动对应正式实验；核对命令但不启动时使用 `EVRPTW_DRY_RUN=1`。预算仍为每实例30分钟，在1/5/15/30分钟记录 checkpoint。以上三个命令是单独选择使用的示例；默认各30个CPU worker，实际部署应按服务器资源安排。

旧结果的美元核算并未全部失效，但搜索行为有区别，应保留旧结果并使用新目录重跑。不要把修复前后结果混成同一算法版本。如果自定义 `EVRPTW_TEST_RESULTS_ROOT`，也应选择新的目录。

回归包括三客户时间窗反例：两车24km成本830.908331美元，一车44km成本420.310196美元；成本模式应偏好后一方案，显式距离模式应偏好前一方案。还覆盖 VNS 已找到可行改进但选择不可行探索的情况、full 空路线、跨目标 checkpoint，以及 Gurobi 回调、恢复隔离和非零最优性 gap。

本次未启动完整 benchmark；小实例测试证明目标和更新逻辑得到修复，不预先承诺整套测试集的平均提升幅度。

验证结果：相关联合回归 **161 passed、4 skipped**；跳过项为已有旧 HEAD 对照测试。30 个正式 shell 通过语法和 dry-run 配置检查。三个真实 CLI 在独立 canonical 三客户夹具上各运行默认 v2、显式 distance 两种模式，共6次通过：v2 均输出44km/1车/420.310196美元，distance均输出24km/2车/24km，路线通过回放。

VNS 更准确的候选评分会增加少量 CPU 开销：本机每路线约10客户的微基准中，Cus500候选生成约0.091→0.142秒，Cus1000约0.202→0.305秒。该微基准不代表完整30分钟实验的解质量；正式实验继续使用相同墙钟预算。

执行验证时使用 `maojie` Python、CPU和单线程数值库；未占用训练GPU。小实例 CLI 输出与脚本保存在本机 `/data/solver_cost_fixes_20260916/`，未将测试夹具混入正式数据集。
