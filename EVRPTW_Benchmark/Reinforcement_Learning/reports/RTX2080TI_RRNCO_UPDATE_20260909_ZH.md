# 2080 Ti 配置更新与 RRNCO-EV 对照实验（2026-09-09）

已合并 origin/drl-benchmark-adapters 最新代码，保留原本四个本地 RRNCO 提交。三台 2080 Ti 的 16 个任务均通过生产 dry-run；本机 4_1 实际检查了四张 RTX 2080 Ti，其他两台只检查本地配置和数据，未远程启动。

## 配置与训练修复

- 三台常用 `scripts/2080ti_*/full.sh` 等入口现在转发实际 rq_v1 队列。TERRAN 的 Cus50/Cus100 独立配置由 job 中的路径和 SHA256 固定，启动、训练签名、结果核验使用同一配置。
- 保留此前实测的四模型 batch；TERRAN 为 Cus50 480×50、Cus100 280×50，4 minibatches、chunk64。采用拉取代码的 SmoothL1 critic、vf_coef=0.1、critic backbone 梯度系数0.1。PBRS 保持 gamma=1、progress/repair=0.5、5000轮退火到0.2、成功奖励0；reward 已按各规模独立归一化，不能简单再按客户数缩放这些参数。
- 不把独立 stable_cost_v1 实验偷换进 frozen 四 benchmark 队列。它改变了训练目标/critic/采样协议，不适合直接替代现有 G/E/support 实验。
- 修复 common trainer 完成结果漏写 stream_integrity_mode；修复 TERRAN 无注册 stream 时将 None 转成字符串造成的启动失败；修复4_2同一stream被G/E任务重复写入marker的问题。严格校验保留，旧结果没有改写。
- RRNCO 使用可选 stable AFT、关系计算分块及activation重计算、确定性进出方向距离摘要，以及同实例 leave-one-out baseline。原默认模型及旧checkpoint语义保留；详细公式及测试见 RRNCO_EV_V2_OPTIMIZATION_20260909_ZH.md。

## 本轮已完成的 Cus50 结果

三组 seed1234，300 updates，batch24，每实例5轨迹，同一个新建7200-view有序训练stream；训练/验证步数65/98；最后一次统一使用完整500个validation实例，每实例100候选，canonical verifier和相同USD成本公式。没有使用test集调参。

| 模型 | verifier通过 | 平均cost USD | 平均道路距离km |
|---|---:|---:|---:|
| 旧RRNCO（legacy AFT、random、paper baseline） | 500/500 | 451.6054 | 195.7126 |
| 优化版，无显式道路输入 | 500/500 | 456.2523 | 220.8833 |
| 优化版，完整D/T/E道路输入 | 500/500 | **442.3199** | **167.2327** |

完整图比同backbone无图cost降低3.0537%，逐实例469胜、31负；491个同车辆数实例中，距离166.23 vs219.15 km。完整优化版比旧实现cost降低2.0561%，436胜、64负；该比较同时改变AFT、距离摘要和baseline，不能把全部改进单独归因于某一项。

当前仍未超过已有Cus50 DRL-TS的433.6565、TERRAN的435.2899或AM的435.8491。DRL-TS用了39.6M客户曝光，本轮短实验只有0.36M；训练预算不同。旧RRNCO长训练还存在后期退化，尤其Cus100曾从最佳612.9167退化到0/500可行；长期稳定性必须由新实验验证。

配对明细：`RRNCO_EV_V2_CUS50_SCREEN_20260909.json`。此轮旧exploratory stream没有预先写入正式contract SHA，工具明确报告这个缺项；三组使用同一文件，但本轮不称正式多seed因果认证。未来screen v3和long v2已使用内容/源index/seed/规模/数量的严格SHA校验。

三个Python训练均写出status=passed、300更新及完整验证结果；执行期间原地编辑Bash launcher导致wrapper尾部解析失败、exit2。这是本轮操作问题，不是训练成功退出码；独立记录在 `results/2080ti_update_20260909/graph_screen/wrapper_execution_audit.json`，没有将wrapper伪装为exit0。新screen会执行不可变snapshot，后续编辑用原子替换。

## 资源验证与长实验调度

资源数据：`RTX2080TI_FINAL_RESOURCE_GATES_20260909.json`。

- TERRAN Cus50：480×50、2轮，峰值9.68GiB，10/10验证可行。
- TERRAN Cus100：280×50、2轮且每轮开启critic梯度诊断，峰值9.65GiB，10/10验证可行。整体GPU利用率仍受CPU环境采样影响，显存接近满载不代表持续满算力。
- RRNCO Cus50：256×16、2轮，峰值5.85GiB；只7/10验证可行，属于显存与有限梯度gate。64×16的3轮gate为10/10；完整300轮结果为500/500。
- RRNCO Cus100：32×16、2轮，峰值1.91GiB，但只有4/10验证可行，大量初期轨迹耗尽120步。没有把这个结果标为可行性通过，也不立即启动Cus100长训练。

本机长期调度使用 `scripts/2080ti_4_1/rrnco_then_full.sh`：GPU0/3立即启动新版AM/TERRAN benchmark；GPU1/2分别运行Cus50 RRNCO full/node_only长期对照，各自结束后自动接续对应full.sh benchmark队列。长对照采用相同batch256、16轨迹、LR1e-4、5000最低/10000最多更新，前100/后250更新验证一次，500实例×100候选。两组共享严格校验stream，分别保存best/checkpoint/日志。模型从头训练，未混入旧权重；最大客户曝光128M，两组一致，跨benchmark仍需明确曝光与训练耗时。

旧AM/TERRAN任务的best/latest/selected及状态和provenance已独立保全，见 `results/2080ti_update_20260909/preserved_old_runs/REPLACEMENT_CHECKLIST.md`。调度脚本不会擅自结束其他进程；启动前由本轮操作核对并替换旧任务。延迟benchmark启动时要求原clean commit仍在，避免从另一个未审核版本执行。

核心组合回归216项通过；模型和独立结果导出/配对23项通过（其中配对测试与前者有重叠）。pipeline的5项mock集成检查覆盖GPU交接、已有占用拒绝、RRNCO失败记录和源码变更阻断。正式训练尚需长期监控、多个seed和独立test评估，不能据本轮短实验宣布胜过全部四个benchmark。
