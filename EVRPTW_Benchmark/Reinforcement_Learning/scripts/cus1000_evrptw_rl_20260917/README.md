# EVRPTW-RL Road Cus1000：2080ti_4_2四卡部署

分支：`cus1000-evrptw-rl-4gpu-20260917`；源代码基线：`684b1ca`。本机只进行了工程短训校准，未启动正式训练。

每卡batch=12，GPU0/1/2/3同步训练一个模型，全局实例batch=48，梯度累积1。训练和验证每实例各30条候选，候选数不计入实例batch；动作上限为1800/2700。默认2000次全局optimizer更新，其中前1000次为EMA warmup，后1000次使用greedy baseline。训练stream为96,000次实例抽样，客户曝光预算96,000,000。一次logical epoch表示一次更新，不代表扫完整个训练集。

数据为冻结Road发布 `us_11city_full_clean_v7_bbde5db_20260823`：5000个Cus1000训练实例、500个验证实例，每例1051节点。每100次更新验证全部500例，sampling seed 910001234。模型为Structure2Vec mean，使用既有USD objective、reward contract和station辅助项。正式训练从seed1234随机初始化，不使用校准checkpoint。

## 部署与运行

在2080ti_4_2建立独立worktree；以下新目录应尚不存在，已有实验目录保持原状：

```bash
cd /data/Maojie/ICLR/EVRPTW-DB
git fetch origin
git worktree add --detach ../cus1000-evrptw-rl-4gpu-deploy origin/cus1000-evrptw-rl-4gpu-20260917
cd ../cus1000-evrptw-rl-4gpu-deploy
conda activate maojie
export CUS1000_ROAD_ROOT=/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Dataset/Instances_v2/us_11city_full_clean_v7_bbde5db_20260823
cus1000_entry=EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus1000_evrptw_rl_20260917/2080ti_4_2/full.sh
bash "$cus1000_entry" --mode preflight
bash "$cus1000_entry"                 # 默认后台启动，可退出终端
bash "$cus1000_entry" --mode status   # 查看进度
```

中断后恢复原实验可运行 `bash "$cus1000_entry" --resume`。

`CUS1000_PYTHON`可指定Python，`CUS1000_OUTPUT_ROOT`可指定独立输出根；启动、status和resume使用相同输出根。默认输出为`EVRPTW_Benchmark/results/cus1000_evrptw_rl_20260917`。数据路径也兼容`CUS500_ROAD_ROOT`、`CUS100_ROAD_ROOT`，显式无效路径会拒绝。

`full.sh`固定batch=12、梯度累积1，覆盖遗留的batch环境变量及同名命令行参数。GPU固定0/1/2/3，任意卡存在其他计算任务则拒绝启动，允许的GPU0桌面进程保留。启动器验证数据及源码hash，并持有输出锁和与旧Cus500实验共享的GPU锁。已有run不会自动覆盖；恢复要求源码、数据、卡数、batch与配置一致，训练期间不要修改工作树。恢复验证范围为2-rank CPU，尚未实测四卡NCCL恢复。

## 实测与工期

固定B12六次四卡更新已通过：2次EMA、4次greedy，两次10例验证和第4/6次更新的baseline probe。四rank训练进程峰值9.197GiB，无OOM。GPU0/3观察到热降频。

EMA更新均值266.0秒，greedy更新均值237.0秒。训练更新部分外推约5.82天；将10例验证耗时线性放大到500例并计入20次验证后约6.03天（规划量级约6–7天）。这是短训外推，未实跑2000次更新；greedy样本中的probe为2例，正式probe为64例且频率不同，路线长度与温度也会影响总工期。

CPU验证74项（58原有+11部署+4固定batch+1入口锁定）与真实四卡短训分别记录。详见[验证报告](VALIDATION_REPORT.md)及[精简实测数据](measured_profile.json)。校准不构成收敛或算法性能结论，未使用测试集。
