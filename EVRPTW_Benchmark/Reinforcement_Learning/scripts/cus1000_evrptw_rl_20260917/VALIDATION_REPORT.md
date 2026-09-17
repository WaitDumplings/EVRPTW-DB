# Cus1000四卡工程验证报告

分支`cus1000-evrptw-rl-4gpu-20260917`，代码基线`684b1ca`。没有本机正式训练结果。

## 验证范围

CPU测试74项通过：58项原有回归、11项部署集成、4项固定batch校验、1项入口锁定。恢复相关测试为2-rank CPU验证；**未做四卡NCCL中断后恢复实测**。

真实RTX2080Ti四卡短训独立于上述CPU测试：固定每卡12、全局48、候选数30、动作上限1800/2700；完成六次更新、两次10例验证及两次2例baseline probe。第2次更新完成EMA到greedy的baseline同步，后4次使用greedy。正式配置的EMA时长1000、probe间隔100和probe大小64保持不变；这里只缩短校准预算与cohort。

## 实测记录

| 更新 | Baseline | 更新秒数 | mean loss |
| --- | --- | --- | --- |
| 1 | paper_ema | 265.60 | -223.618 |
| 2 | paper_ema | 266.33 | 3170.18 |
| 3 | greedy_rollout | 267.12 | -16890.7 |
| 4 | greedy_rollout | 250.16 | -15048.3 |
| 5 | greedy_rollout | 219.21 | -13606.4 |
| 6 | greedy_rollout | 211.62 | -12382.1 |

| 验证更新 | 实例数 | 完整可行实例数 | 验证秒数 |
| --- | --- | --- | --- |
| 3 | 10 | 10 | 19.31 |
| 6 | 10 | 10 | 16.67 |

| Rank | NVIDIA-SMI训练进程峰值GiB |
| --- | --- |
| 0 | 9.197 |
| 1 | 9.197 |
| 2 | 9.197 |
| 3 | 9.197 |

GPU0桌面占用不计入训练进程峰值。六次更新退出码为0，无OOM；校准审计确认模型、baseline及optimizer浮点张量有限，共15个浮点模型张量相对参考checkpoint发生变化，latest/best文件存在。checkpoint审计说明保存内容通过检查，不等同于完成四卡恢复验证。

## 正式预算与ETA

每卡batch=12，GPU0/1/2/3同步训练一个模型，全局实例batch=48，梯度累积1。训练和验证每实例各30条候选，候选数不计入实例batch；动作上限为1800/2700。默认2000次全局optimizer更新，其中前1000次为EMA warmup，后1000次使用greedy baseline。训练stream为96,000次实例抽样，客户曝光预算96,000,000。一次logical epoch表示一次更新，不代表扫完整个训练集。

EMA均值265.96秒/更新、greedy均值237.03秒/更新。训练部分计算为`1000×EMA均值 + 1000×greedy均值`；验证部分为`两次10例验证均值×50×20`。训练更新部分外推约5.82天；将10例验证耗时线性放大到500例并计入20次验证后约6.03天（规划量级约6–7天）。

GPU0/3存在实测热降频。只有六次更新样本，greedy样本含2例probe而正式使用64例，不同频率、数据加载、checkpoint写入和路线长度变化均限制ETA精度。2000次正式更新及500例全量验证未在本机执行，不把该估算作为工期保证。

## 证据

校准source hash：`c154b4193d5fd79bf7cc729a38716a8193b341175f9d67e3f5eefce42a178e51`。该hash对应校准阶段工作树。校准后仅整理部署配置、固定batch入口、校准选择逻辑、测试和文档；模型、rollout及checkpoint审计逻辑未变。

- `calibration_report`：`/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Benchmark/results/cus1000_evrptw_rl_4gpu_profile_20260917/calibration/report.json`
- `formal_config`：`/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Benchmark/results/cus1000_evrptw_rl_4gpu_profile_20260917/calibrated_config.json`
- `confirmation_config`：`/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Benchmark/results/cus1000_evrptw_rl_4gpu_profile_20260917/calibration/target_band_1789630771663585007/batch12_confirmation/config.json`
- `training_result`：`/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Benchmark/results/cus1000_evrptw_rl_4gpu_profile_20260917/calibration/target_band_1789630771663585007/batch12_confirmation/run/training_result.json`
- `logical_epoch_history`：`/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Benchmark/results/cus1000_evrptw_rl_4gpu_profile_20260917/calibration/target_band_1789630771663585007/batch12_confirmation/run/logical_epoch_history.jsonl`
- `validation_history`：`/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Benchmark/results/cus1000_evrptw_rl_4gpu_profile_20260917/calibration/target_band_1789630771663585007/batch12_confirmation/run/validation_history.jsonl`
- `baseline_history`：`/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Benchmark/results/cus1000_evrptw_rl_4gpu_profile_20260917/calibration/target_band_1789630771663585007/batch12_confirmation/run/baseline_history.jsonl`
- `latest_checkpoint`：`/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Benchmark/results/cus1000_evrptw_rl_4gpu_profile_20260917/calibration/target_band_1789630771663585007/batch12_confirmation/run/checkpoint_latest.pt`
- `best_checkpoint`：`/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Benchmark/results/cus1000_evrptw_rl_4gpu_profile_20260917/calibration/target_band_1789630771663585007/batch12_confirmation/run/best.ckpt`

精简配置、显存峰值、耗时及checkpoint审计另存[measured_profile.json](measured_profile.json)，未复制搜索阶段错误日志。校准只证明工程可运行，不评价模型收敛或算法优劣；测试集没有参与调试。
