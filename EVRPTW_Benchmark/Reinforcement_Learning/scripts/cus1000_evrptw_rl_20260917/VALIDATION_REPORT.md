# Cus1000四卡工程验证报告

分支`cus1000-evrptw-rl-4gpu-20260917`，代码基线`684b1ca`。没有本机正式训练结果。

## 验证范围

CPU测试78项通过：58项原有回归、11项部署集成、4项固定batch校验、1项入口锁定、4项正式配置回归。恢复相关测试为2-rank CPU验证；**未做四卡NCCL中断后恢复实测**。

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

每卡batch=12，GPU0/1/2/3同步训练一个模型，全局实例batch=48，梯度累积1。训练和验证每实例各30条候选，候选数不计入实例batch；动作上限为1800/2700。默认2000次全局optimizer更新，固定预算关闭早停（patience=0、start=0），其中前1000次为EMA warmup，后1000次使用greedy baseline。训练stream为96,000次实例抽样，客户曝光预算96,000,000。一次logical epoch表示一次更新，不代表扫完整个训练集。

EMA均值265.96秒/更新、greedy均值237.03秒/更新。训练部分计算为`1000×EMA均值 + 1000×greedy均值`；验证部分为`两次10例验证均值×50×20`。训练更新部分外推约5.82天；将10例验证耗时线性放大到500例并计入20次验证后约6.03天（规划量级约6–7天）。

GPU0/3存在实测热降频。只有六次更新样本，greedy样本含2例probe而正式使用64例，不同频率、数据加载、checkpoint写入和路线长度变化均限制ETA精度。2000次正式更新及500例全量验证未在本机执行，不把该估算作为工期保证。

## 正式启动边界修复（2026-09-17）

初版`51ba4a2`的六次短训通过，但缩短预算时没有覆盖正式`max=min=early_start=2000`组合。2080ti_4_2在第一次更新前因`early-stop start must precede maximum training epochs`退出；没有发生OOM。当前配置将patience/start都设为0，显式关闭早停，max/min保持2000；启动器提前校验同样的边界。

新增4项CPU回归直接覆盖真实训练器的旧配置拒绝、新正式配置接受及关闭早停后连续劣化验证仍完成固定预算。另使用实际full.sh后台启动，完整保留2000/min2000、batch12/global48、30 trajectories、500例验证与EMA1000，在本机四张2080 Ti完成第1次更新，耗时251.68秒、mean loss=-223.618149。测试随后主动停止，全部测试GPU进程已释放；不是一次完成2000轮的正式训练。首更新没有到达epoch100验证。

首更新证据：`/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Benchmark/results/cus1000_production_startup_fix_1789633481272246299/startup_verification.json`。该测试工作树source hash：`65a2086816e4ac516647a6d80b41d99b08690e9d8b92c968a9d3d797e9d2cb39`。早期校准目录中的`calibrated_config.json`是修复前的历史记录；部署以当前仓库`config.json`为准。

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
