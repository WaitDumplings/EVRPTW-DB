# 2080 Ti 4_1 既有训练与 RRNCO 对比审计

审计日期：2026-09-09；进程快照：17:54 UTC（10:54 America/Los_Angeles）。本报告只读旧进程与产物；未终止训练、覆盖 checkpoint 或修改旧结果。

## 目前结论

既有 RRNCO-EV 在 Cus50 尚未超过四个 benchmark。它优于退化的 EVRPTW-RL，但在相同 500 个 validation 实例上落后 AM 和 DRL-TS；TERRAN 汇总成本也更低。此前 Cus100 的 500-update matched-AM screening 是有价值的候选信号，不能替代正式预算比较或 graph 因果证明。

| 方法 / 既有 run | 完成或最近训练 epoch | 最佳 epoch | validation 成本 USD | 距离 km | 车辆数 |
|---|---:|---:|---:|---:|---:|
| RRNCO-EV long / Cus50 | 7500，early-stop | 2500 | 444.018945 | 172.977209 | 1.010 |
| AM-EVRPTW / Cus50 | 2100，仍运行 | 2000 | 435.849077 | 135.494292 | 1.004 |
| DRL-TS / Cus50 | 5500，early-stop | 1300 | 433.656522 | 131.948865 | 1.000 |
| EVRPTW-RL / Cus50 | 5500，early-stop | 100 | 1379.265991 | 742.804297 | 3.062 |
| TERRAN / Cus50 | 3175，仍运行 | 2500 | 435.289874 | 142.712231 | 1.000 |

这些选中 checkpoint 的验证汇总均为 500/500 可行。AM/TERRAN 尚未结束，表中数字是快照，不能当最终成绩。

## 进程和调度状态

- 父调度 PID `3370690`：旧 `drl_job_runtime full`，manifest 是 `scripts/rq_v1/2080ti_4_1/jobs_preverified.jsonl`，运行约 32.5 小时。
- AM PID `3370835`、GPU 0：训练日志从审计初期 epoch 2093 持续增长至 2100，CPU 利用率约 121%；`stdout.log` 为空不是卡死，实际增量写入 `logical_epoch_history.jsonl` 与 `reward_diagnostics.jsonl`。
- TERRAN PID `3370840`、GPU 3：`logs/train_log.csv` 持续增长至 3175，CPU 利用率约 106%；最近验证 epoch 3100。`reward_diagnostics.jsonl` 也持续更新。
- DRL-TS、EVRPTW-RL 的训练都正常结束于 epoch 5500，但 `job_result.json.status=failed`。共同原因是 `TRAINING_STREAM_CONTRACT_MISMATCH`，具体消息为 `completed training did not preserve the no-rehash stream integrity mode`。
- 两个失败 job 的 `training_result.json` 保留正确的 contract SHA 和 snapshot，`best.ckpt.args.stream_integrity_mode` 也正确；缺的是 terminal JSON 的 `stream_integrity_mode` 字段。这是结果写出遗漏，不是训练中途 OOM 或 stream 内容失配证据。
- 已在 `common/protocol_trainers.py` 修复未来结果写出，并用训练真正生成的 checkpoint/result 通过原样 runtime gate 的回归验证。没有修改旧产物，也没有放宽 gate。已运行 Python 进程不会自动载入该修复。

旧 benchmark 产物根目录：

`EVRPTW_Benchmark/results/DRL_rq_v1/G/Full-support/{am_evrptw,drl_ts,evrptw_rl,terran}/Cus50/seed_1234/9c2173a83d12cc253e9e667112b81234c11f6937/`

每个目录的 `best.ckpt` / `checkpoint_selected.pt` 对应最佳验证；`checkpoint_latest.pt` 对应最近保存的验证点。TERRAN 的逐 epoch 存档在 `checkpoints/` 子目录，其余直接位于 run 根目录。

## 同实例配对结果

机器可读证据见 [RRNCO_2080TI_EXISTING_RUN_AUDIT_20260909.json](RRNCO_2080TI_EXISTING_RUN_AUDIT_20260909.json)。新工具 `scripts/audit_rrnco_comparison.py` 读取汇总的 `rows`，按 `view_id` 或 canonical `instance_id` 配对，检查重复 ID、非有限成本、成本公式、objective 配置、candidate 数、horizon、验证 seed、scale 和 split。

RRNCO long 与 AM、DRL-TS、EVRPTW-RL 的 500 个 ID 完全一致，objective 配置一致，sampling 100 candidates、horizon 98、seed 910001234 一致；三个配对的 500 例均共同可行。

| 对手 | RRNCO 成本相对变化 | RRNCO 逐实例胜 / 负 | 同车数子集：RRNCO 距离 / 对手距离 km |
|---|---:|---:|---:|
| AM | +1.8745% | 3 / 497 | 497 例：172.1381 / 135.0492 |
| DRL-TS | +2.3895% | 0 / 500 | 495 例：171.5752 / 131.0751 |
| EVRPTW-RL | −67.8076% | 500 / 0 | 12 例：109.3279 / 301.9635 |

TERRAN 的现有 validation summary 没有逐实例 `rows`，因此工具明确标记不能配对；不以相同 `instances=500` 自动证明 ID 相同。需要用 `TERRAN.eval_stage2` 导出逐实例结果再统一配对。

正式成本为同一路线上的：`413.6331536717643 × K + 0.151750972762646 × D`，单位 USD。Cus50 车辆成本占很高比例；仅看总成本会弱化几十公里的路线差异，因此同时报告同车数距离。

## RRNCO 长训练存在退化

RRNCO long 结果位于：

`EVRPTW_Benchmark/results/RRNCO_EV_single_seed_long_v1/{Cus50,Cus100}/seed_1234/473a8a6df6e27f1ae5859598f2638d2722cf6b0b/`

| 规模 | 最佳验证 | epoch 5000 | epoch 7500 |
|---|---|---|---|
| Cus50 | epoch 2500：444.0189，500/500 | 484.4095，500/500 | 520.5193，500/500 |
| Cus100 | epoch 1750：612.9167，500/500 | 1271.7287，488/500 | 无有限汇总成本，0/500 |

这说明长训练不只是进步慢，还存在策略退化。必须监控完整学习曲线、可行率、advantage 和梯度，保留 best checkpoint；不能只报最佳数值而忽略结尾崩溃。

旧配置的 `baseline_eval_size=0` 关闭 baseline 更新。RRNCO 使用 AM 风格的前 2500 update EMA，之后转 greedy baseline；没有更新时该 baseline 仍是初始网络。此次新增 RRNCO 显式 `reinforce_baseline=leave_one_out` 实验路径，用同实例其他采样轨迹成本作 detached baseline：`(sum(cost) - cost) / (K - 1)`，要求 K≥2。它不适用于自动修改四个 benchmark 的论文配方；默认仍为 `paper`。是否改善真实 RRNCO 需看新实验，不能由单元测试推断。

EVRPTW-RL 的最后 epoch 训练可行率仅 1.79%，98.21% 轨迹耗尽 horizon；其最佳 checkpoint 停在 epoch 100。应独立修复/校准其训练，再把它作为有效 benchmark。

## 预算与 graph 因果验证

RRNCO long Cus50 完成 180,000 训练实例、9,000,000 customer exposures、7500 updates、约 3.10 GPU 小时；DRL-TS 是 792,000 实例、39,600,000 exposures、5500 updates、约 9.00 小时。AM 每 update 2304 实例，RRNCO 每 update 24 实例。仅匹配 update 数不是匹配训练数据或算力。旧 RRNCO long stream 还没有正式 contract SHA，应另建带 provenance 的实验。

要验证提升来自 graph 信息，至少需要以下受控训练与评估：

1. 同一 RRNCO backbone 的 node-only 与完整 graph 模式：参数量、初始化、训练 ID 顺序、batch、轨迹数、optimizer、baseline、reward、PBRS 和训练预算保持一致。node-only 应同时去除 relation embedding、encoder bias、decoder D/T/E 特征等政策输入；环境仍按真实 road D/T/E 判断可行性与成本。
2. 对 graph 来源拆分：仅 D、D+T、D+T+E；同时记录 angle 是否保留，避免“node-only”偷偷保留关系输入。检查该数据集的 E 是否与 D 共线，再解释 energy channel 独立贡献。
3. 对称化有向关系、置乱关系的负控制，用于区分真实有向结构收益与额外参数/训练正则化。先做训练 ablation；仅推理时屏蔽 graph 会引入分布偏移，不能当完整因果证据。
4. 预先固定独立 test 集与候选预算，validation 只用于选超参数和 checkpoint。至少多个训练 seed，报告逐实例 paired gap、可行率、车辆数、同车数距离和跨城市结果。
5. 与四个 benchmark 比较须同时给出数据曝光和 GPU 时间曲线；架构因果比较采用完全匹配的训练 stream，系统性能比较采用预先固定算力/曝光预算。不能把更大的模型或更多训练归因于 graph。

## 可执行对比路径

在仓库根目录运行以下示例，输入也可以直接指定 `validation_summary.json`。脚本不会运行 GPU 或改动原产物：

```bash
python -m EVRPTW_Benchmark.Reinforcement_Learning.scripts.audit_rrnco_comparison \
  --run graph=/path/to/graph_run \
  --run node=/path/to/node_run \
  --run am=/path/to/am_run \
  --output /path/to/new_comparison.json
```

工具会显式列出缺失/不同的评估配置和训练预算；缺失逐实例记录不能认证全 cohort 配对。`full_cohort_comparison_verified` 只表示评估记录的 ID/配置一致，不表示训练公平或 graph 因果性。所有 comparison 的 `causal_graph_claim_supported` 保持 false。

现有 `RRNCO_EVRPTW.evaluate_checkpoint` 可对 AM/RRNCO checkpoint 做 canonical 500-instance validation；四个 benchmark 各有 eval 模块，TERRAN 为 `TERRAN.eval_stage2`。新 graph ablation checkpoint 的构造参数需由对应 evaluator 从 checkpoint args 恢复，不能拿旧 evaluator 忽略模式强行加载。若重新评估，必须保留模型模式、horizon、candidate、seed、完整 objective 配置及逐实例 ID 到输出。
