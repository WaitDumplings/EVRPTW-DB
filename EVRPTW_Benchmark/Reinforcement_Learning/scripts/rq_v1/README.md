# DRL RQ server launch bundles

These four bundles generate training checkpoints only. All defaults are derived
from the checked-out repository; no machine-specific absolute data path is
embedded. Dataset discovery first checks the archive restore tree relative to the checkout (`../../../evrptw_runtime`) and then the repository-local `EVRPTW_Dataset` tree. Canonical and frozen-v7 directory names are supported in both locations. A candidate is accepted only when its core training index exists; no server-specific absolute path is committed.

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

## Active single-seed four-scale candidate

The generated manifests use runtime budget
`drl_rq_runtime_budget_v6_scale_aware_gpu_val500_es3`. The generated manifests
contain seed 1234 only. Scale ownership is strict: the three 2080 Ti servers run
only Cus50/Cus100, while the two RTX 6000 Ada GPUs run only Cus500/Cus1000.

| Scale | Hardware | Epochs | Environments/epoch | Total environments | Customer exposures |
|---|---|---:|---:|---:|---:|
| Cus50 | 2080 Ti | 1,000 | 1,024 | 1,024,000 | 51,200,000 |
| Cus100 | 2080 Ti | 1,000 | 256 | 256,000 | 25,600,000 |
| Cus500 | RTX 6000 Ada | 1,000 | 64 | 64,000 | 32,000,000 |
| Cus1000 | RTX 6000 Ada | 1,000 | 2 | 2,000 | 2,000,000 |

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
| Cus1000 | 1 | 1 | 1 | 1 | 2 / 2 / 2 / 2 |

The Cus50/Cus100 rows were measured on 11 GiB RTX 2080 Ti cards with the
complete two-epoch pilot followed by sampled best-of-100 validation. Observed
process peaks were 4.72--8.88 GiB for Cus50 and 6.32--8.49 GiB for Cus100.
Every listed 2080 Ti batch therefore completed below 10 GiB. The Cus500 and
Cus1000 values are conservative Ada defaults, not final 48 GiB calibration;
they may be increased only after running the local Ada pilot.

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

The one-seed schedule uses all four servers and all 13 GPUs. The three 2080 Ti
queues contain no Cus500/Cus1000 work. Their pilot/formal job counts are 13, 9,
and 6 for `2080ti_4_1`, `2080ti_4_2`, and `2080ti_3_1`, respectively. The Ada
queue contains all larger-scale work: eight pilots and eight formal jobs, split
evenly across its two GPUs. Its final makespan depends on the later 48 GiB batch
calibration and is therefore not frozen here.

Every server wrapper accepts `--seed`; omitting it defaults to 1234. The pilot
jobs remain the frozen seed-1234 code/hardware gate. No other seed is registered
in this manifest version, so another seed cannot be launched until the config
and generated manifests are explicitly revised and reviewed.

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

The `--seed`/`--seeds` interface and legacy `DRL_SEEDS` environment variable are
retained for future extensions, but currently only seed 1234 matches a job.

If the restore directory has a different relative location, override only its
root; the dataset remains below `EVRPTW_Dataset`:

```bash
export EVRPTW_RESTORE_ROOT="../../../evrptw_runtime"
```
