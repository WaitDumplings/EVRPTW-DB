# DRL RQ server launch bundles

These four bundles generate training checkpoints only. All defaults are derived
from the checked-out repository; no machine-specific absolute data path is
embedded. Dataset discovery defaults to the archive restore tree relative to the checked-out repository: `../../../evrptw_runtime/EVRPTW_Dataset/Instances_v2/us_11city`. The canonical and frozen v7 directory names are supported below that restore root. Standalone artifact preparation and all server launchers use the same resolver; no server-specific absolute path is committed.

Activate any Python environment containing the project's required dependencies
(for example `maojie` or `caliroute`) and run exactly one pilot launcher on each
server. The launcher records the actual Python executable, prefix, and active
Conda environment in every job provenance; it does not require a particular
environment name.

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/2080ti_4_1/pilot.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/2080ti_4_2/pilot.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/2080ti_3_1/pilot.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/a6000_2_1/pilot.sh
```

Each command prepares deterministic local artifacts if needed and then launches
its GPU queues through `nohup`/`setsid`. `status.sh` and `logs.sh` do not start
work. `resume.sh` resumes only committed formal checkpoints.

Each `full.sh` is unlocked independently by the pilot jobs in the same server
manifest. All of those pilots must pass under the same executable commit as the
full run. No shared `pilot_gate_report.json` or cross-server wait is required.
The aggregate pilot report remains reviewer evidence, not a runtime dependency.

No SHA-256 or per-file content hashing is performed by these scripts.

## Active seed-wise Cus50/Cus100/Cus500 candidate

The generated manifests use runtime budget
`drl_rq_runtime_budget_v5_seedwise_pow2_fulltrain_val500_es3`. Cus1000 is
not scheduled. The launcher defaults to seed 1234; later seeds are opt-in through
`DRL_SEEDS`.

| Scale | Epochs | Environments/epoch | Total environments | Customer exposures |
|---|---:|---:|---:|---:|
| Cus50 | 1,000 | 1,024 | 1,024,000 | 51,200,000 |
| Cus100 | 1,000 | 256 | 256,000 | 25,600,000 |
| Cus500 | 1,000 | 64 | 64,000 | 32,000,000 |

Training views follow a deterministic city-by-day stratified shuffle cycle. A
stratum is exhausted before any view in that stratum is reused. All four methods
therefore consume the same seed-specific logical stream within a scale. Physical
batches differ only to fit GPU memory; exact accumulation preserves the common
logical batch.

| Scale | AM physical | EVRPTW-RL physical | DRL-TS physical | TERRAN physical | Microbatches AM/RL/TS/TERRAN |
|---|---:|---:|---:|---:|---:|
| Cus50 | 1,024 | 128 | 128 | 128 | 1 / 8 / 8 / 8 |
| Cus100 | 256 | 64 | 32 | 128 | 1 / 4 / 8 / 2 |
| Cus500 | 4 | 2 | 1 | 32 | 16 / 32 / 64 / 2 |

Each formal job requests at most 1,000 logical epochs. At epochs 50, 100, ...,
1000 it evaluates the same fixed 500 validation views. Validation and test both
use stochastic decoding with exactly 100 seeded candidates per instance.
Checkpoint selection first maximizes independent-verifier feasibility rate and
then minimizes mean verified directed distance over the feasible candidates.
Validation/test reward and penalties are not added to the reported distance.
The 100-candidate count is a common benchmark inference budget supported by all
four methods, not a claim that it reproduces every paper's original candidate
count. The selected checkpoint is written as both `best.ckpt` and the
backward-compatible `checkpoint_selected.pt`; all checks are retained in
`validation_history.jsonl`.

TERRAN uses 100 parallel POMO-style trajectories per training instance. The
other three REINFORCE adapters retain one sampled rollout per training instance;
their batch dimensions continue to represent different training instances.

Early stopping uses validation-check patience, because no metric exists between
50-epoch validation points. Three consecutive validation checks without a
lexicographic improvement stop the job, equivalent to 150 training epochs since
the last improvement. With the first checkpoint at epoch 50, the earliest stop
is epoch 200. An early-stopped run is a valid completed training outcome and
retains its best checkpoint and terminal report.

The default one-seed schedule uses all four servers and all 13 GPUs. The current
measurement-based no-early-stop makespan estimate is about five to six days,
dominated by DRL-TS Cus500. Early stopping can shorten this but is not assumed in
the conservative estimate.

Every server wrapper accepts `--seed`; omitting it defaults to 1234. The pilot
jobs themselves remain the frozen seed-1234 code/hardware gate, while the
argument selects which formal stream is prepared and which full/resume/status
jobs are addressed.

```bash
# Default seed 1234:
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/<server>/pilot.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/<server>/status.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/<server>/full.sh

# Equivalent explicit form:
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/<server>/full.sh --seed 1234
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/<server>/status.sh --seed 1234
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/<server>/resume.sh --seed 1234
```

Standalone preparation accepts the same argument:

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/prepare_artifacts.sh --seed 1234
```

For a later registered seed, replace 1234 with 2345 or 3456 consistently on
all servers. `--seeds 1234,2345` and the legacy `DRL_SEEDS` environment variable
remain available for intentional multi-seed queueing.

If the restore directory has a different relative location, override only its
root; the dataset remains below `EVRPTW_Dataset`:

```bash
export EVRPTW_RESTORE_ROOT="../../../evrptw_runtime"
```
