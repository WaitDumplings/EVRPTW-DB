# Curriculum stage 1: Cus100 continuation

GitHub branch: **`ablation`**. These scripts load the ten archived Cus100 policies
from `/data/best_ckpt` and train **2000 additional logical epochs on one GPU**.
`G` means Road-100 checkpoint → Road Cus100 training; `E` means Euclidean-100
checkpoint → the frozen TERRAN synthetic Cus100 corpus. This stage does not
train Cus500/Cus1000 or convert synthetic inputs into Road matrices.

## Start

Activate the environment containing this project's dependencies (for example
`conda activate maojie`; `caliroute` is also usable if those dependencies are
installed), and run from the repository root:

```bash
git fetch origin
git switch ablation
git pull --ff-only origin ablation

./script_curriculum/am.sh G 1
```

The last positional argument is the **physical GPU index shown by nvidia-smi**.
Only that GPU is visible to the trainer. The script checks that it has no other
compute job and holds a GPU lock while running. It does not stop existing jobs.
Use an available GPU for each simultaneous run.

The five executable entry points are:

```bash
./script_curriculum/am.sh G 1
./script_curriculum/evrptw_rl.sh G 1
./script_curriculum/drl_ts.sh G 1
./script_curriculum/terran.sh G 1
./script_curriculum/rrnco.sh G 1
```

These are **alternative commands**, not five simultaneous jobs on GPU 1.
Replace `G` with `E` for the Euclidean counterpart. `G, 1,` is also accepted.
The command runs in the foreground; keep it in `tmux` for remote sessions:

```bash
tmux new -s curriculum-am
./script_curriculum/am.sh G 1
# Detach with Ctrl-b, then d; later: tmux attach -t curriculum-am
```

Preview a command without starting training:

```bash
./script_curriculum/am.sh E 1 --dry-run
```

This checks checkpoint identity and dataset location; full index/stream preflight
runs when starting training. Checkpoints and data are external assets, not Git
payloads. `checkpoints.json` pins all ten relative paths, SHA256 values, saved
epochs and architectures. It selects EVRPTW-RL Cus100 from NPG5; the retired
NPG6 sum-aggregation Cus500 checkpoint is never used.

## One command per server

After the first checkout/update containing these scripts, run the matching entry
point from the repository root. Each script first runs
`git pull --ff-only origin ablation`, selects its conda environment, checks all
assigned checkpoint hashes/data locations and GPU availability, then starts its
jobs under `nohup`. It prints the launcher PIDs and a unique log directory.

| Entry point | GPU 0 | GPU 1 | GPU 2 | GPU 3 | Default environment |
|---|---|---|---|---|---|
| `./script_curriculum/2080ti_4_1.sh` | AM G | AM E | EVRPTW-RL G | EVRPTW-RL E | maojie |
| `./script_curriculum/2080ti_4_2.sh` | DRL-TS G | DRL-TS E | TERRAN G | TERRAN E | maojie |
| `./script_curriculum/2080ti_3_1.sh` | RRNCO G | RRNCO E | unused | — | caliroute |

These server entries default to `/data/curriculum_stage1` for results. Submission
logs and `jobs.tsv` go to `launchers/<server>/<UTC timestamp>_<PID>` under that
root; each trainer prints its own result directory in its submission log.
`CURRICULUM_OUTPUT_ROOT`, data/checkpoint overrides and `CURRICULUM_PYTHON` are
honored. Set `CURRICULUM_CONDA_ENV` to choose a different named environment when
an explicit Python executable is not supplied. Conda must be initialized or its
executable available through `CONDA_EXE`/`PATH` for automatic activation.

```bash
# Check without submitting training (still pulls ablation unless disabled):
./script_curriculum/2080ti_4_1.sh --dry-run

# Run using an explicit interpreter and the current checkout:
CURRICULUM_SKIP_PULL=1 CURRICULUM_PYTHON=/path/to/env/bin/python \
  ./script_curriculum/2080ti_4_1.sh
```

Physical GPU IDs are checked against any existing `CUDA_VISIBLE_DEVICES` setting;
use `unset CUDA_VISIBLE_DEVICES` first if a previous command restricted this
terminal to a smaller GPU set. A busy GPU or failed preflight aborts before this
server script submits any jobs. The per-job launcher also checks its GPU lock to
handle concurrent submissions. These are background submissions, not completion
claims; inspect the printed logs for subsequent training errors. GPU/data checks
are repeated by each trainer, including full index/stream checks.

## Data and path overrides

Road defaults search the current checkout, then the sibling `EVRPTW-DB` checkout
for `EVRPTW_Dataset/Instances_v2/us_11city_full_clean_v7_bbde5db_20260823`.
The `us_11city` alias is also supported in the current checkout. Euclidean defaults
search both checkouts for `EVRPTW_Dataset/TERRAN_synthetic100_feasible4_20260911`.
Absolute paths are accepted:

```bash
export CURRICULUM_CKPT_ROOT=/data/best_ckpt
export CURRICULUM_ROAD_ROOT=/path/to/us_11city_full_clean_v7_bbde5db_20260823
export CURRICULUM_SYNTHETIC_ROOT=/path/to/TERRAN_synthetic100_feasible4_20260911
export CURRICULUM_OUTPUT_ROOT=/data/curriculum_stage1
./script_curriculum/rrnco.sh E 0
```

`CUS100_ROAD_ROOT`/`EVRPTW_DATASET_ROOT` and `CUS100_SYNTHETIC_ROOT` are fallback
aliases. `CURRICULUM_PYTHON=/path/to/env/bin/python` selects an interpreter.
Explicit flags `--road-root`, `--synthetic-root`, `--checkpoint-root` and
`--output-root` take priority. Neither corpus is generated by this launcher.
Both have 50,000 train and 500 validation views. Index IDs, parent separation,
and Euclidean manifest/index checks are performed; Euclidean payload presence
is checked, but every compressed payload is not rehashed on each launch.

## Training and reporting

- Monetary objective: vehicle dispatch cost **413.6331536717643 USD** plus
  **0.39 × (100/257) USD/km × D_time**. Time uses `T_time`; energy uses
  `(100/257) × D_time`. The revised verifier rejects internal-depot and
  customer-free trips. This does not change the historic source checkpoint's
  original objective or its selection history.
- Stage counters run **1–2000**, independent of the archived checkpoint epoch.
  A logical epoch is one global instance batch/update cycle, not a corpus pass.
  No early stopping. Seed 1234; a fresh deterministic shuffled full-pool stream
  is recorded for this stage. Source sample counters are not carried forward.
- Validation at **100, 200, …, 2000**: 500 instances in the current training
  domain, 30 sampled candidates each, seed 910001234, independent verification.
  Selection first maximizes verified feasible rate, then minimizes feasible
  mean USD cost. No test data is used. Training also samples 30 trajectories per
  instance; rollout caps are 240 for training and 360 for validation.
- Policy weights and semantic architecture are checked strictly. The target
  objective/reward, optimizer, baseline history, validation best and stopping
  state are fresh. This is explicit **weights-only continuation**, not an
  optimizer-state resume across incompatible objectives.
- TERRAN uses the existing legacy PPO architecture and loads its policy
  backbone; its separate critic head and optimizer are reset. PPO has 3 passes,
  4 minibatches, actor LR1e-4, value coefficient 0.1 and the existing PBRS schedule.
- DRL-TS stays **hard** from stage epoch 1: both source checkpoints completed
  their soft stage. EVRPTW-RL retains mean graph aggregation. RRNCO retains full
  relations and stable AFT. Architecture and all non-batch defaults match the
  audited Cus100 configurations.
- Baseline schedules restart with stage counters: AM uses its default
  2500-update EMA warmup (so this 2000-update stage remains EMA), EVRPTW-RL uses
  its 1000-update EMA warmup, and RRNCO retains leave-one-out. TERRAN's PBRS
  annealing likewise starts from stage epoch 1. These are recorded settings,
  not a claim that the old optimizer/phase state was resumed.
- Road and Euclidean retain their separate prior training-only reward
  normalizers; the Road D_time normalizer is a fixed reused hyperparameter,
  not a new D_time calibration. Euclidean D_dist and D_time coincide numerically.

RRNCO G now defaults to **batch 72**, as requested on 2026-09-20; RRNCO E
remains batch 50. The RRNCO G batch-82 memory measurement in
[SMOKE_REPORT.md](SMOKE_REPORT.md) is historical; batch 72 has not been GPU
profiled separately. Other defaults retain the measured batches in that report. To explicitly lower a batch if another
machine has less free memory, use `--batch-size 32`; the physical and effective
batch both change, and the requested exposure is recorded. Batch is not the
trajectory count. A smoke test with a separate fresh directory is:

```bash
./script_curriculum/am.sh G 1 --epochs 2 --validation-every 1 --validation-limit 2
```

Short overrides are diagnostics, not the default final experiment.

Every invocation creates a new timestamped output directory and prints its path.
An explicit `--run-dir` must not already exist. Original checkpoints are never
overwritten. In that directory:

- `training.log`, `status.json`: trainer output, PID, progress and exit state.
- `validation_history.jsonl`, `validation_summary.csv`: per-checkpoint eval
  records and an unrounded CSV extract (epoch, instance count, feasible count,
  mean verified cost, best flag).
- `best_overall.ckpt` / `best.ckpt` and latest trainer checkpoints: stage results;
  names follow each trainer's existing conventions.
- `request.json`, trainer config/state, `artifacts/training_stream.parquet`:
  command, source checkpoint/hash/epoch, target protocol, GPU, code hashes,
  dataset hashes and exact sampled instance stream.

To read the log while it runs: `tail -f /path/printed/by/launcher/training.log`.
This delivery configures the 2000-epoch experiments; local GPU validation consists
of the short runs documented in the smoke report.

## Next stage: AM Road Cus500

[`cus500_curr/am.sh`](cus500_curr/am.sh) loads the frozen best AM Road Cus100
stage-1 checkpoint and trains one policy synchronously on GPU 0/1 for 3000
additional epochs. Defaults are 12 instances per GPU (global 24), 30 trajectories,
and 500-instance validation every 100 updates. See the
[stage-2 instructions](cus500_curr/README.md) for source identity and outputs.

### 五模型 Road Cus100 → Cus500 双卡入口

将第一阶段的 Road best 分别放到 `/data/cus100_ckpt/{am,evrptw_rl,drl_ts,terran,rrnco}.ckpt`，
运行 `./script_curriculum/cus500_curr/am_cus100_to_500.sh 0 1`（其他模型替换 am）。
均新增3000轮，每100轮验证一次。完整配置和各模型 batch 的实测范围见
[cus500_curr/README.md](cus500_curr/README.md)。
