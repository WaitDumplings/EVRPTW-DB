# Cus500 双卡实测报告（2026-09-13）

RRNCO 与 DRL-TS 均通过本机 GPU 0/1 的实际 NCCL 双卡训练、验证及 checkpoint 检查。
正式配置为 RRNCO 每卡 22、DRL-TS 每卡 2；本轮仅做短测试，未启动两组正式训练。

## 最终参数与测量

| 模型 | 每卡 batch | 全局 batch | 每卡进程峰值 | 短测试更新数 | 总耗时 |
|---|---:|---:|---:|---:|---:|
| RRNCO full graph | 22 | 44 | 9.588 / 9.588 GiB | 6 | 429.60 秒 |
| DRL-TS | 2 | 4 | 9.285 / 9.285 GiB | 12 | 139.76 秒 |

两组均保持每实例 30 条轨迹、551 节点、训练 horizon 1700、验证 horizon 2550、
FP32、原模型宽度/层数和统一 energy+vehicle cost。显存是 NVIDIA-SMI 训练进程值，
不包括桌面；约每 0.5 秒采样，另记录 PyTorch allocated 峰值。驱动和PyTorch版本见
[机器可读报告](measured_profiles.json)。

RRNCO 达到 9.5–10.3 GiB 目标。DRL-TS 略低于目标，但每卡 batch 3 在 greedy baseline
encoder 阶段实际 OOM，因此定为 batch 2，不用无效分配填显存。

## 显存优化及排除的配置

- RRNCO 开启 decoder activation checkpoint stride 1，并保留已有 relation checkpoint、
  relation chunk 32。单卡 batch 2 不开启 decoder checkpoint 时占 7.641 GiB；开启后
  batch 4 为 2.377 GiB、batch 20 为 8.818 GiB，最终双卡 batch 22 为 9.588 GiB。
  重计算保留完整递归梯度，未截断计算图或减少道路信息。
- DRL-TS 原先先按轨迹展开全部边再 gather，反向传播需要 B×T×N×N×E 的临时梯度。
  Cus500、batch 2、30 轨迹、128 维时仅该缓冲就申请约 8.69 GiB并OOM。
  现在直接按起点索引边行，重复索引的梯度正确累加；保持模型结构和输出不变。
  修改后单卡 batch 2 的完整优化步骤峰值 9.186 GiB；batch 3 的 baseline encoder
  仍会OOM。正式配置继续使用 decoder checkpoint stride 1。

CPU 等价测试覆盖两模型的 checkpoint stride 1/2：路径、log likelihood、全参数梯度、
随机状态及 encoder BatchNorm/call 次数。DRL-TS直接边索引另与原 decoder 参考实现比较
前向及全部梯度，包含同一来源节点的重复选择。

## 双卡实际执行范围

- RRNCO：6 次同步更新，44×6=264 个训练实例位置；e3/e6各验证10例，两次均10/10可行。
  e3→e6 所有572个参数张量更新且有限，L2变化0.268334；分片、两rank RNG、best/latest
  与验证摘要一致。训练 baseline 为每实例其余29轨迹的 leave-one-out。
- DRL-TS：12 次同步更新，4×12=48 个训练实例位置；测试中soft为e1–6、hard为e7–12。
  baseline probe在e6按soft、e12按hard运行；e6/e12验证始终使用硬约束，各10/10可行。
  为覆盖阶段切换，短测将 baseline 间隔/样本数压缩到6/2；正式为250/64，soft至2500。
- 两组实际执行 launcher 的数据检查、stream生成、torchrun、验证分片合并、checkpoint保存
  和正常完成判定。小样本可行率只用于工程检查，不能据此比较模型效果或判断收敛。

正式训练仍为固定500例验证，每100次全局更新一次；最少5000、最多10000次更新，
5000后连续5次验证无改善早停。完整数据审计确认10,000个train、500个val、551节点，
train/val的view和family均无重叠，test未读。440000/40000长度的两份正式stream已验证
生成后复用一致；其合同和数据SHA见机器可读报告。不同batch意味着不同样本暴露量，
评估时应同时报告batch、客户暴露量和GPU-hours。

## CPU RAM

默认每rank的train/val pool各缓存至多256例，按需加载，val实际各250例。
独立CPU进程加载val250后RSS1.710 GiB，再加载train256后RSS3.282 GiB，
双rank基础缓存外推6.565 GiB；另需模型、torch、NCCL和临时工作内存。
短测实际峰值RSS：RRNCO两rank为6.833/6.697 GiB，DRL-TS为2.569/2.552 GiB；
短测尚未填满正式train/val缓存，不能把这些数值当作长期RAM上限。

## 时间估计

RRNCO短测每次更新46–70秒，10例验证9.4–15.9秒；DRL-TS普通更新约7–8秒，
10例验证约6.1秒。将验证量外推至500例，并计入保存及DRL训练池baseline probe，
RRNCO到5500次更新粗估3.5–5天、跑满10000次约6–9天；DRL-TS分别约17–24小时、
30–42小时。仅为短测外推，实际需用前100–200epoch修正，且后续路线长度和服务器负载会改变耗时。

## 回归验证

CPU测试：新双卡协议/模型入口/边索引18项通过；已有AM双卡协议24项通过；
AM/RRNCO/DRL-TS模型、目标函数、诊断和论文接口115项通过；部署46项通过。
前两组中的边索引测试与模型组重复2项，合计201个不同测试。
覆盖全局梯度与串行microbatch参考、LOO、两阶段训练和精确恢复、训练stream/RNG、
缓存、GPU锁、旧run保护、退出失败识别和恢复签名。两个full.sh均通过bash语法检查。

原始日志、checkpoint和大体积数据保留在本地被Git忽略的results中；仓库只提交配置、
代码、测试和摘要报告。启动及恢复命令见[README](README.md)。
