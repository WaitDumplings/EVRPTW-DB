# RRNCO × EVRPTW-DB 深入静态审计

日期：2026-09-08
状态：静态审计已完成；后续 RRNCO-EV 适配和 Cus100 screening 见 `RRNCO_CUS100_SCREENING_REPORT_ZH.md`；尚非正式 benchmark 结论
审计对象：

- EVRPTW-DB：`/data/Maojie/ICLR/EVRPTW-DB`，分支 `drl-benchmark-adapters`，审计时 HEAD `4bb433c31a163fcce7d7b852676edd948f8cc06e`，工作树干净。
- RRNCO：`/data/Maojie/Github2/RRNCO/real-routing-nco`，官方基线 HEAD `823d510dadf4dd711730ec4fbf337c356a0de6ae`；工作树已有未提交修改和未跟踪的 Routing-D 适配文件。
- 论文：RRNCO arXiv v2（2026-03-14），ICLR 2026 接收版本。[^rrnco-paper] [^rrnco-iclr]

## 1. 结论先行

用户提出的核心判断——“真实有向道路关系没有被策略充分、直接地建模，是当前部分 DRL 弱于非学习求解器的重要原因”——**方向正确，但不能表述为已经证明的唯一原因**。

更准确的结论是：

1. AM-EVRPTW 和 TERRAN 的静态编码是 node-centric；道路矩阵参与环境转移、可行性 mask 和最终 verifier，但没有作为显式边关系进入策略编码器。
2. EVRPTW-RL 虽接收 travel-time matrix，却把每个起点的整行压成一个 row sum，丢失了“从 i 到哪个 j”的成对关系和大部分有向结构。
3. DRL-TS 是重要反例：它把完整 `distance / travel time / energy` 作为有向边特征，并在解码时取当前节点的整行边 embedding。因此不能再笼统地说“四个 DRL 都没有道路矩阵”。
4. 非学习方法的优势不只来自“看到了 D/T/E”，还来自实例级搜索、精确目标增量、精确可行性重放、充电站修复以及可行初始解。RRNCO 本身仍是 autoregressive construction model，并不会自动获得 ALNS/VNS 的局部搜索和修复能力。
5. RRNCO 是一个**高度相关、值得做受控 pilot 的候选架构**，因为它正面解决 node-only encoder 对真实非对称道路矩阵建模不足的问题；但当前官方代码只支持 ATSP、ACVRP、ACVRPTW，不支持 EVRPTW。
6. 当前本地 Routing-D converter 会舍弃充电站、电池、能耗、充电功率、车辆固定成本和 EVRPTW verifier 语义。直接用它跑出的结果不能称为 EVRPTW-DB 上的 RRNCO。
7. 推荐状态是：**允许设计并实现语义适配与小规模 correctness pilot；暂不批准正式训练，更不批准 Cus500/Cus1000。**

论文中建议统一使用“非学习求解器（non-learning solver）”，不要使用 “unlearning”。后者通常指 machine unlearning，是另一个研究方向。

## 2. 用户假设中，哪些已经被代码支持？

### 2.1 四个现有 DRL 的实际信息路径

| 模型 | 环境是否使用 D/T/E | 策略静态编码 | 解码时直接读取当前有向边 | 判断 |
|---|---:|---|---:|---|
| AM-EVRPTW | 是 | 坐标、需求、TW、service、充电速度、node type | 否 | node-centric |
| TERRAN | 是 | depot/customer/CS 坐标、需求、TW、充电速度 | 否 | node-centric |
| EVRPTW-RL | 是 | node features + `sum_j T[i,j]` | 否 | edge-summary，不保留 pairwise destination |
| DRL-TS | 是 | 完整 `D[i,j], T[i,j], E[i,j]` + adjacency | 是 | edge-aware |

证据：

- AM 的 initial embedding 只拼 node attributes，没有矩阵输入；动态 context 只有当前节点 embedding、load、battery、time：[AM model](/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Benchmark/Reinforcement_Learning/AM_EVRPTW/model.py:161)、[initial embeddings](/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Benchmark/Reinforcement_Learning/AM_EVRPTW/model.py:217)。
- TERRAN 的 node embedding 由坐标、需求、TW 和 charger ratio 组成：[TERRAN embedding](/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Benchmark/Reinforcement_Learning/TERRAN/models/nets/attention_model/embedding.py:14)。
- EVRPTW-RL 在 `_edge_message` 中对 travel-time matrix 做 `sum(dim=-1)`：[EVRPTW-RL model](/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Benchmark/Reinforcement_Learning/EVRPTW_RL/model.py:170)。两个目的地排列完全不同、但 row sum 相同的路网会得到相同摘要。
- DRL-TS 明确接收三张完整矩阵并做 edge-node updates：[DRL-TS encoder](/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Benchmark/Reinforcement_Learning/DRL_TS/model.py:230)；其边特征为 D/T/E/adjacency：[DRL-TS edge features](/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Benchmark/Reinforcement_Learning/DRL_TS/model.py:351)。解码器还按 last node gather 当前出边 embedding。

因此，应把讨论从“是否是 graph model”改成两个可审计的问题：

1. `D/T/E` 是否作为 pairwise edge features 直接进入 policy？
2. 当前状态下，decoder 是否能读取 `current node -> candidate node` 的对应边信息？

### 2.2 环境使用矩阵，不等于策略理解矩阵

EVRPTW-DB 的共享环境确实用真实道路矩阵执行：

- 原始节点顺序为 `[depot, customers, charging stations]`；读取真实 D/T/E 和逐站点充电功率：[environment setup](/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Benchmark/Reinforcement_Learning/EVRPTW_Env/env.py:123)。
- action mask 精确检查电池、容量、TW、工作时域和可返回 depot 条件：[action mask](/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Benchmark/Reinforcement_Learning/EVRPTW_Env/env.py:425)。
- 目标统计使用真实道路距离和 `vehicles_started`：[objective ledger](/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Benchmark/Reinforcement_Learning/EVRPTW_Env/env.py:771)。

这能阻止很多非法动作，但 node-centric policy 看到的主要是“现在什么不可选”，而不是“为什么不可选、未来选谁会更好”。Mask 是 hard feasibility oracle，不是完整的道路关系 representation。

### 2.3 当前有限结果与假设一致，但不能当因果证明

同一个旧运行快照 `9c2173a...` 的 Cus50 / seed 1234 validation 中，四个模型最佳记录为：

| 模型 | 最佳 epoch | mean verified cost (USD) | mean distance (km) | feasibility |
|---|---:|---:|---:|---:|
| DRL-TS | 1,300 | 433.657 | 131.949 | 1.0 |
| AM-EVRPTW | 1,200 | 436.110 | 142.662 | 1.0 |
| TERRAN | 1,500 | 436.517 | 150.800 | 1.0 |
| EVRPTW-RL | 100 | 1,379.266 | 742.804 | 1.0 |

这组结果与“保留 pairwise road relations 有帮助”相容，因为 DRL-TS 最好、row-sum 的 EVRPTW-RL 最差；但它只有一个 seed、不同架构/训练目标/课程，并且 EVRPTW-RL 的异常平坦表现可能还有训练问题。因此只能作为**支持性观察**，不能作为 causal ablation。

## 3. 为什么非学习求解器仍可能明显更强？

### 3.1 它们直接优化精确目标，而不是从 reward 间接学习

当前冻结目标是：

```text
C = 413.6331536717643 × vehicles_started
  + 0.151750972762646 × distance_km
```

配置见 [rivian_energy_vehicle_cost_v2.json](/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Benchmark/Reinforcement_Learning/configs/rivian_energy_vehicle_cost_v2.json:1)。增加一辆车相当于约 **2,725.736 km** 的距离成本。这是一个强离散目标：模型不仅要学习短边，还要学习全局 customer packing、route closure 和“是否值得开启新车”。RRNCO 官方原始 reward 是负总距离，不含该固定车辆成本。

ALNS 的候选目标直接计算 `c × total_distance + F × route_count`：[ALNS objective](/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Benchmark/MetaHeuristics/ALNS_Solver/solver.py:812)。它不需要从 noisy policy gradient 中学会 F 的语义。

### 3.2 非学习求解器有 repair/search，而现有 DRL 多为一次构造

ALNS 会：

- 重放 Stage-2 singleton feasibility certificate 或构造确定性可行 singleton routes；
- 对 route 做精确、缓存的 feasibility replay；
- 通过 remove/insert/charging-station repair 持续改进完整解。

对应入口见 [singleton construction](/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Benchmark/MetaHeuristics/ALNS_Solver/solver.py:1064) 和 [cached feasibility](/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Benchmark/MetaHeuristics/ALNS_Solver/solver.py:888)。

RRNCO 改善的是 construction policy 的表示能力；它没有自动获得 ALNS 的实例级邻域搜索。因此合理预期是缩小差距、提升跨城市鲁棒性，而不是自然超过 30 分钟 ALNS/VNS。

### 3.3 计算预算也不对等

RRNCO 论文在 ACVRPTW N=100 上相对 PyVRP 的 gap 仍约 3.74%–4.29%；其优势是 1,280 个实例约 52 秒的快速推理，而 PyVRP 是每实例 20 秒、合计约 7 小时。[^rrnco-results] [^rrnco-protocol]

EVRPTW-DB 当前非学习基线记录 5/30 分钟快照。如果只比较 objective、不同时报告每实例 wall time，就会把“求解预算优势”误写成“架构优势”。

## 4. RRNCO 到底怎样使用道路关系？

RRNCO 的论文动机与用户假设高度一致：真实路网提供非对称 D/T，而 node-only transformer 不适合直接编码这些 edge features。作者提出 ANE 和 NAB。[^rrnco-paper]

### 4.1 ANE：把非对称距离注入 node initialization

对每个节点，`DistanceExpert` 按反距离概率采样 k 个邻居，分别从 D 的 row 和 column 收集、排序并线性投影，形成 outgoing/incoming 双 embedding：[RRNCO ANE](/data/Maojie/Github2/RRNCO/real-routing-nco/rrnco/models/env_embeddings/rcvrptw.py:127)。再通过 contextual gate 与 coordinate expert 融合。

这比 row sum 强得多：它保留每个节点的局部距离分布及 incoming/outgoing 差异。但排序后的 sampled values 并不保留被采样邻居的 identity，所以 ANE 本身仍不是完整 pairwise reasoning。

### 4.2 NAB/AAFM：把 D/T/方向角作为全局 pairwise bias

NAB 分别把 D、T 和方向角 Phi 嵌入为 `B × N × N × E`，学习三通道 gate，再投影成 `B × N × N` 的 adaptive bias：[DistAngleFusion](/data/Maojie/Github2/RRNCO/real-routing-nco/rrnco/models/nn/attn_freenet.py:201)。每个 encoder layer 对 row 和 column 两个方向各做一次 AAFM：[Attn-Free layer](/data/Maojie/Github2/RRNCO/real-routing-nco/rrnco/models/nn/attn_freenet.py:444)。

因此 RRNCO 的确存在全图信息路径：任意 `D[i,j]/T[i,j]` 可以影响全局 node representation，不只是当前一步的 mask。

### 4.3 Decoder：当前节点的 D/T row 直接影响候选 logits

解码器 gather 当前节点的 distance row 和 duration row，以可学习 alpha/beta 加到 inductive bias，再作用于 logits：[RRNCO decoder](/data/Maojie/Github2/RRNCO/real-routing-nco/rrnco/models/decoder.py:183)。这补上了“当前 i 到候选 j”的直接短程决策信号。

### 4.4 训练/推理不是单一 greedy

底层 policy 默认 train sampling、val/test greedy；但 `RRNet` 会给 train/val/test 加 multistart，使用 shared POMO baseline，并在 val/test 做 8-fold coordinate augmentation：[RRNet training wrapper](/data/Maojie/Github2/RRNCO/real-routing-nco/rrnco/models/rl.py:74)。所以其评估实际上是“多个强制起点 × greedy continuation × augmentation 后取最好”，不能简单写成单轨迹 greedy。

## 5. 官方 RRNCO 对 EVRPTW-DB 缺什么？

官方 README 只列出 ATSP、ACVRP、ACVRPTW，未列 EVRPTW。[^rrnco-repo]

| EVRPTW-DB 必需语义 | 官方 RRNCO ACVRPTW | 当前本地 Routing-D converter | 结论 |
|---|---|---|---|
| depot/customers | 有 | 有 | 可复用 |
| charging stations 作为动作节点 | 无 | 无 | 必须新增 |
| 逐站点充电功率/充电时间 | 无 | 无 | 必须新增 |
| battery/SOC 动态状态 | 无 | 无 | 必须新增 |
| directed energy matrix E | 无 | 无 | 环境必须使用；是否进 encoder 要做版本化设计 |
| CS 当前 route 的访问规则 | 无 | 无 | 必须服从共享 EVRPTW env |
| 多车辆 dispatch/return 语义 | CVRP depot return | 部分 | 必须对齐 `vehicles_started` |
| `F × K + c × D` | 负距离 | 负距离 | 必须替换 |
| canonical verifier | 内部 checker 未完成 | 无 | 必须调用 EVRPTW-DB verifier |
| provenance/view split | 不同数据系统 | 未保留完整信息 | 必须直接读 Stage-2 contract |

官方 RMTVRP environment 的 reward 是负距离，[reward implementation](/data/Maojie/Github2/RRNCO/real-routing-nco/rrnco/envs/rmtvrp/env.py:459)；其 `check_solution_validity` 直接抛出 `NotImplementedError`：[validity checker](/data/Maojie/Github2/RRNCO/real-routing-nco/rrnco/envs/rmtvrp/env.py:486)。这意味着不能通过“给它一个 NPZ”就获得可信 EVRPTW 结果。

## 6. 本地 `/data/Maojie/Github2/RRNCO` 的当前状态

本地 official commit 与 `origin/main` 都是 `823d510...`，但工作树不是 pristine：

```text
modified:
  rrnco/envs/rcvrp/{__init__.py,env.py}
  rrnco/envs/rmtvrp/{__init__.py,env.py}
  rrnco/models/env_embeddings/{rcvrp.py,rcvrptw.py}

untracked:
  configs/env/routing_d_rcvrp.yaml
  configs/env/routing_d_rcvrptw.yaml
  rrnco/envs/rcvrp/fixed_generator.py
  rrnco/envs/rmtvrp/fixed_generator.py
  scripts/convert_routing_d.py
  scripts/time_cvrp_cus100_eval.py
  scripts/watch_cus50_cvrp_then_vrptw_cus50.sh
```

这些改动主要解决固定 NPZ 数据、sample size 大于 N，以及 batched depot mask；不是 EVRPTW 适配。现有 converter 只输出 depot/customers、demand、D、T、TW、service：[converter](/data/Maojie/Github2/RRNCO/real-routing-nco/scripts/convert_routing_d.py:117)。它会丢掉所有 EV-specific contract。

任何后续工作都应先：

1. 保留当前 dirty tree，不覆盖用户已有 Routing-D 工作；
2. 从 `823d510...` 建独立 worktree/branch；
3. 把 EVRPTW adapter 明确命名为 `RRNCO-EV` 或 `RRNCO-EVRPTW adapter`，不能声称是原论文无需修改的直接复现。

## 7. 论文与代码配置核对

论文报告 Adam、lr 4e-4、每 epoch 100,000 instances、batch 256、embedding 128、FF 512、12 个 AAFM layers；约 24 小时使用 4×A100 40GB。[^rrnco-hparams]

仓库 `configs/experiment/rrnet.yaml` 写 batch 64、6 encoder layers：[RRNCO config](/data/Maojie/Github2/RRNCO/real-routing-nco/configs/experiment/rrnet.yaml:22)。这两处大概率不是实质冲突：

- 4 GPU × per-device batch 64 = global batch 256；
- 一个 `Attn_Free_Layer` 内有 row/column 两个 AAFM block，6 个双向层可被计作 12 个方向块。

测试前仍应把 per-device/global batch 和 layer/block 的计数口径写入 manifest，避免复现歧义。

需要单独冻结的一点是：论文公式直接使用 `exp(A)`，代码先对 `adapt_bias` 做 softmax，再做 `exp`：[AFTFull](/data/Maojie/Github2/RRNCO/real-routing-nco/rrnco/models/nn/attn_freenet.py:292)。这不一定是 bug，可能是实现稳定化或论文省略，但在做“paper-faithful”声明前必须记录并决定沿用官方代码还是按公式实现，不能静默改变。

## 8. 规模与显存风险

RRNCO 的 ANE 只采样 k=25，看似近似线性；但 NAB 仍显式构造 `B × N × N × E` 的 D/T/Phi embeddings 和 `3E` concat。其核心复杂度仍至少为 `O(BN²E)`。

以 `E=128` 估算，一个 `N × N × E` tensor 的大小为：

| 总节点 N | FP16 | FP32 |
|---:|---:|---:|
| 100 | 2.44 MiB | 4.88 MiB |
| 500 | 61.04 MiB | 122.07 MiB |
| 1,000 | 244.14 MiB | 488.28 MiB |

一次三通道 fusion 可能同时保留 D/T/Phi 三个 embedding、`3E` concat 和 fused embedding，约相当于 7 个 E-width tensor；还未计 row/column 双方向、6 层 autograd、decoder multistart、optimizer states 和 charging-station nodes。

因此：

- 论文的主要公开证据是 N=100；
- EVRPTW-DB 中 `N = 1 + customers + charging stations`，实际 N 大于 Cus 标签；
- Cus500 training 即使 batch 1 也可能达到多 GB activation；
- Cus1000 在 2080 Ti 上很可能不可行，在 A6000 上也需先做 chunking/checkpointing/sparse bias 设计，而不是直接增加 batch。

这也是为什么第一轮只应考虑 Cus50/Cus100。

## 9. 推荐的正确接入方式

### 9.1 不要把 EVRPTW-DB 压进 RRNCO 的 ACVRPTW env

最稳妥的方向是：

```text
EVRPTW-DB Stage-2 data contract
        ↓
现有 EVRPTW_Env（唯一 transition/mask/reward 语义）
        ↓
RRNCO ANE + NAB/AAFM + decoder adapter
        ↓
现有 candidate selection + canonical verifier
```

也就是复用当前已冻结的 EVRPTW environment，只移植 RRNCO 的 representation/policy。这能避免两个环境分别实现 battery、charging、route closure 后产生语义漂移。

### 9.2 最小 RRNCO-EV 输入合同

Static node features：

- normalized coordinates；
- node type（depot/customer/CS）；
- demand、TW、service time；
- charging power 或 full-charge-time ratio。

Static edge features：

- D：distance matrix；
- T：canonical EV transition time matrix；
- Phi：有向坐标角；
- E：至少用于 environment/action mask；若作为 NAB 第四通道，必须命名为 RRNCO-EV 扩展并单独 ablate。

Dynamic decoder context：

- last node；
- current time/load/SOC；
- vehicles started / 当前 route 是否已服务 customer；
- candidate action mask；
- 当前节点对应的 D/T/E row。

Objective/reward：

- 成功解：严格使用 `F × vehicles_started + c × distance_km`；
- 未完成解：沿用当前冻结的 failure/unserved reward contract；
- checkpoint selection：feasible-first，再按 verified USD cost；
- 最终数字只能来自 canonical verifier。

### 9.3 归一化原则

官方 RRNCO environment 对每个 instance 的 D 做 min-max，而 T 来自另一套时间尺度。EVRPTW-DB 不能盲目复用：

- D/T/E 必须各自有明确单位、归一化分母和逆变换；
- 训练/val/test 的 normalization constants 必须从 training split 冻结，不能看 test；
- 不建议独立对每个 instance 做无记录 min-max，因为它会抹去不同城市/instance 的绝对 travel scale，而 absolute T/E 对 TW 和 SOC 有意义；
- Phi 建议用 `sin(phi), cos(phi)` 或明确保留论文 atan2 版本，做 versioned choice。

## 10. 真正能检验用户假设的实验

仅把 RRNCO 与 AM/TERRAN/DRL-TS 比较，不能严格证明“road-relation injection 是原因”，因为 backbone、训练法和 decoder 都不同。建议在同一个 RRNCO-EV backbone 中做受控 ablation：

| 版本 | 输入 | 回答的问题 |
|---|---|---|
| A | node-only | 同 backbone 不看矩阵时的下界 |
| B | D only | 显式 road distance 是否贡献主要增益 |
| C | D + T + Phi | paper-faithful RRNCO road representation |
| D（可选） | D + T + E + Phi | 能耗作为独立 edge modality 是否继续增益 |

所有版本固定：

- 同一个 Stage-2 train/val/test split；
- 同 seed、instance exposure、optimizer budget；
- 同共享 environment/reward；
- 同 validation instances 和随机种子；
- 同总 candidate budget；
- 同 verifier；
- 同硬件和 wall-time reporting。

DRL-TS 应作为现有 edge-aware 外部参照，而不是唯一 causal control。

## 11. 如果之后批准测试，建议按这四级 gate

### Gate 0：代码/语义审计，无 GPU

- 在 clean upstream worktree 实现 adapter；
- D/T/E/node order 单位逐项断言；
- depot/customer/CS action mapping round-trip；
- CS 访问与 charging time 完全服从共享 env；
- objective 与现有 common objective 逐 route 一致；
- RRNCO 输出 route 经 canonical verifier 复验。

### Gate 1：tiny deterministic correctness

- 人工小实例与少量真实 Cus50 views；
- 检查每一步 SOC/time/load/vehicles_started；
- environment success 必须与 verifier 逐条一致；
- 任一 mismatch 立即 STOP。

### Gate 2：2-epoch train + validation memory pilot

- 先 Cus50，再 Cus100；
- 必须包含一次完整 validation，不能只测 train；
- 记录 peak allocated/reserved GPU memory、step time、NAB 占比、rollout truncation、feasibility；
- 不复用当前 2080Ti batch，RRNCO 单独标定。

### Gate 3：有限 Cus100 scientific pilot

- 一个 seed；
- A/B/C 三个受控表示版本；
- 固定 validation 500 instances；
- 候选预算与现有 benchmark 对齐；
- 同时报 objective、distance、vehicles、feasibility、GPU-hours 和 inference wall time；
- 完成后停下审阅，不自动扩到 Cus500/1000。

## 12. 最终建议

### 建议批准的事项

- 批准把 RRNCO 作为“edge-aware real-road construction baseline candidate”；
- 批准 clean worktree 下的 RRNCO-EV adapter 设计；
- 批准 tiny correctness 与 Cus50/Cus100 memory pilot；
- 批准同 backbone 的 node-only / D / D+T+Phi ablation。

### 目前不建议批准的事项

- 不直接运行当前 Routing-D converter 作为 EVRPTW 结果；
- 不把原始 RRNCO checkpoint 直接用于 EVRPTW-DB；
- 不在 verifier/目标/CS 语义未接入前训练；
- 不先跑 Cus500/Cus1000；
- 不预先宣称 RRNCO 会超过 ALNS/VNS/Gurobi；
- 不把单个 RRNCO 对比写成道路关系的因果证明。

一句话判断：

> RRNCO 非常适合验证“显式有向道路关系能否改善神经构造器”，但它目前是 ACVRPTW road-topology model，不是 EVRPTW solver；正确的下一步是把其 encoder/decoder 接到现有冻结 EVRPTW environment，并先做受控 Cus50/Cus100 pilot，而不是直接跑现有本地转换代码。

## Sources

[^rrnco-paper]: [RRNCO arXiv v2 full text](https://arxiv.org/html/2503.16159v2), sections 4–6 and appendix C.
[^rrnco-iclr]: [ICLR 2026 proceedings entry](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9446c291a8744a125a0bda5b18f4d5a1-Abstract-Conference.html).
[^rrnco-repo]: [Official ai4co/real-routing-nco repository](https://github.com/ai4co/real-routing-nco), supported environments and run instructions.
[^rrnco-results]: [RRNCO paper, Table 1](https://arxiv.org/html/2503.16159v2#S6), ACVRPTW solution quality and runtime.
[^rrnco-protocol]: [RRNCO paper, sections 6.1 and C.3](https://arxiv.org/html/2503.16159v2#S6.SS1), testing hardware, augmentation and traditional-solver budgets.
[^rrnco-hparams]: [RRNCO paper, Table 6](https://arxiv.org/html/2503.16159v2#A3.SS1), optimizer and training hyperparameters.
