# Cus1000 T1：从已有 BKS 继续 ALNS 2 小时

在两台 Linux 服务器分别运行 `bks_T1_upper.sh` 和 `bks_T1_lower.sh`。
按原始 `BKS_Cus1000_T1.jsonl` 的行顺序，上半为第 1–250 个实例，下半为第 251–500 个；两部分无重叠且覆盖全部 500 个。

每台最多同时运行 **30 个 ALNS 计算进程**，另有一个轻量调度进程。
每个实例创建一个独立新进程，单计算线程；进程完成后自动补上下一个实例。
**每个实例独立计时 7200 秒**，不是整批共享 2 小时。每台 250 个实例约需 9 批，即约 18 小时，再加数据加载、检查和输出时间。
不使用 GPU。

## 准备

复制完整仓库（包括本目录新增文件）、BKS 目录和对应的 Stage-2 数据集到另一台服务器。
只复制两个 shell 文件不够：它们会调用同目录的 `bks_refine.py` 及仓库中的 ALNS、Core、数据加载代码。

默认 BKS 根据脚本自身位置定位：先找到仓库根目录，再到其上一级 `ICLR` 目录查找 BKS 文件夹。与启动命令所在的工作目录无关，也不依赖仓库文件夹的名称。

```text
ICLR/
  bks_cus1000_T1_three_models_20260923/
    BKS_Cus1000_T1.jsonl
  EVRPTW-DB/                         # 或其他仓库文件夹名称
    EVRPTW_Benchmark/scripts/ALNS/Cus1000/
      bks_refine.py
      bks_T1_upper.sh
      bks_T1_lower.sh
```

例如仓库位于 `/data/Maojie/ICLR/EVRPTW-DB` 时，默认读取 `/data/Maojie/ICLR/bks_cus1000_T1_three_models_20260923/BKS_Cus1000_T1.jsonl`。仍可使用 `--bks /其他路径/BKS_Cus1000_T1.jsonl` 覆盖默认路径。

建议数据集布局：

```text
/data/us_11city_full_clean_v7_bbde5db_20260823/
  generation_plan/core/test/test1_new_seed/view_index.parquet
  materialized/families/...
```

需要完整的匹配 family/view 文件和矩阵，BKS 中的 routes 本身不包含这些数据。
自动查找时同时支持恢复版目录名 `us_11city` 和原始目录名 `us_11city_full_clean_v7_bbde5db_20260823`。查找范围包括 `/data`、仓库根目录、仓库上一级 `ICLR` 下的 `EVRPTW_Dataset/Instances_v2/`，以及仓库上一级至上三级的 `evrptw_runtime/EVRPTW_Dataset/Instances_v2/`，兼容现有 benchmark 启动脚本的恢复布局。
找不到时会列出已检查的索引路径；数据集本身需要另行恢复或复制，Git 仓库和 BKS 文件夹不能替代数据集。
如已有数据放在其他位置：

```bash
export EVRPTW_DATASET_ROOT=/data/你的数据集根目录
# 仅当 family 目录与 index 分离存放时设置：
# export EVRPTW_FAMILY_ROOT=/data/你的materialized/families
```

使用已有 benchmark Python 环境（Python 3.10+），或在独立环境中安装：

```bash
python -m pip install -r EVRPTW_Benchmark/MetaHeuristics/ALNS_Solver/requirements.txt
export EVRPTW_PYTHON="$(command -v python)"
```

脚本默认使用 `python3`。可将 `EVRPTW_PYTHON` 设为其他 Python 的绝对路径。

## 两台服务器分别启动

以下命令从仓库根目录执行。可先用 `--dry-run` 检查实例数量、路径和参数，不启动优化。

服务器 A：

```bash
bash EVRPTW_Benchmark/scripts/ALNS/Cus1000/bks_T1_upper.sh --dry-run
nohup bash EVRPTW_Benchmark/scripts/ALNS/Cus1000/bks_T1_upper.sh \
  > /data/alns_bks_cus1000_T1_upper_2h.log 2>&1 < /dev/null &
```

服务器 B：

```bash
bash EVRPTW_Benchmark/scripts/ALNS/Cus1000/bks_T1_lower.sh --dry-run
nohup bash EVRPTW_Benchmark/scripts/ALNS/Cus1000/bks_T1_lower.sh \
  > /data/alns_bks_cus1000_T1_lower_2h.log 2>&1 < /dev/null &
```

可选参数：`--bks /data/其他BKS.jsonl`、`--dataset-path /data/数据集根目录`、`--output /data/其他输出目录`。
正式实验保留默认 `--workers 30`、`--time-limit-s 7200` 和 8 个检查点。

## 状态和解

默认输出目录：

- 上半：`/data/alns_bks_cus1000_T1_upper_2h/`
- 下半：`/data/alns_bks_cus1000_T1_lower_2h/`

目录内：

- `run.json`：全部选定 instance ID、输入与代码 SHA256、目标函数、种子和时间配置。
- `status.json`：已完成、失败、排队实例数，以及正在运行的实例和 PID。
- `instances/<instance_id>/initial_validation.json`：BKS 可行性与成本一致性检查。
- `instances/<instance_id>/status.json`、`worker.log`：单实例状态和错误信息。
- `instances/<instance_id>/best_at_900s.json`、`best_at_1800s.json`、…、`best_at_7200s.json`：分别对应 **15、30、45、60、75、90、105、120 分钟**。
- `instances/<instance_id>/result.json`：最终最优解、成本、改进额、实际搜索时长和迭代计数。
- 全部结束后产生 `summary.csv` 和 `validation_report.json`。

每份时间点 JSON 含完整 `routes`、`objective_value` / `objective_cost_usd`、车辆数、电费、距离、相对初始 BKS 的改进额，以及 `incumbent_event_time_s`。
从各实例开始搜索时计时；数据加载、初始 BKS 核验和 solver 构造不占用 7200 秒，ALNS 的初始后处理和搜索中 incumbent 验证计入预算。
计时器在运行中写文件，可能有很小的调度/IO 延迟，但每份文件严格使用该时间点之前已通过验证的最优解，不会用之后的解回填。
即使没有改进，也保存初始 BKS；最佳成本不会上升。

目标函数固定为 `rivian_energy_vehicle_cost_v2`，距离使用 `running_time_path_distance_km`：

```text
USD = distance_km × 0.38910505836575876 × 0.39
      + vehicle_count × 413.6331536717643
```

每个实例先通过独立路线验证器和 ALNS 可行性检查，并以绝对误差 `1e-6 USD` 核对输入成本；失败实例不进入搜索，并记录错误。
ALNS 关闭常规迭代次数上限（改用极大的迭代上限），按 7200 秒墙钟预算持续搜索；Linux 定时信号会中断仍在执行的算子，保留最近的已验证 incumbent。

## 重跑 / 中断后继续

重复使用已有输出目录会报错，避免误覆盖。原命令追加 `--resume` 时，配置和代码必须一致：

- 已完成实例跳过。
- 未完成/失败实例的旧文件存入其 `previous_attempts/`，从输入 BKS **重新跑满 2 小时**。
- 不是从中断时的 ALNS 内部状态继续。
- 仍有实例进程运行时，目录锁会阻止启动重复任务。

更改代码或实验参数时使用新的 `--output`。

## 已完成的验证

本机完整输入验证：**500/500 同时通过独立验证器和 ALNS 可行性检查；ALNS、独立重算与提供成本的最大绝对差均为 0 USD**。
平均初始成本约为 **1997.5131394582577 USD**。

上下部分各取 3 个真实实例，分别以 2 个并发进程、每实例 24 秒、每 3 秒保存一次进行实跑，检查排队补位与独立进程。
6 个实例均跑满预算并生成 8 份解；全部 48 份快照经独立 replay 和 ALNS 再次检查。
短时测试有 3 个实例产生改进。这是短时功能验证，未在本机启动正式 500 实例的 2 小时实验。

验证汇总见同目录 `bks_T1_verification.json`。本机详细结果：

```text
/data/alns_bks_cus1000_T1_validation_20260924/
/data/alns_bks_cus1000_T1_smoke_upper_20260924/
/data/alns_bks_cus1000_T1_smoke_lower_20260924/
/data/alns_bks_cus1000_T1_smoke_verification_20260924.json
```

快速回归测试：

```bash
python -m unittest discover -s EVRPTW_Benchmark/scripts/ALNS/Cus1000/tests -v
```
