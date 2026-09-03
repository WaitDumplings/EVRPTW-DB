# DRL RQ server launch bundles

These four bundles generate training checkpoints only. All defaults are derived
from the checked-out repository; no machine-specific absolute data path is
embedded. Dataset discovery is relative to the checked-out repository. The launcher checks, in order, an explicit `EVRPTW_DATASET_ROOT`, `EVRPTW_Dataset/Instances_v2/us_11city`, the frozen `us_11city_full_clean_v7_bbde5db_20260823` directory, and the same two locations in a sibling `EVRPTW-DB` checkout. No server-specific absolute path is committed.

Activate the `maojie` environment and run exactly one pilot launcher on each
server:

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/2080ti_4_1/pilot.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/2080ti_4_2/pilot.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/2080ti_3_1/pilot.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/rq_v1/a6000_2_1/pilot.sh
```

Each command prepares deterministic local artifacts if needed and then launches
its GPU queues through `nohup`/`setsid`. `status.sh` and `logs.sh` do not start
work. `resume.sh` resumes only committed formal checkpoints.

`full.sh` is intentionally present but blocked. It cannot start the 72 formal
runs until both the pilot gate report and a versioned G1--G8 PASS authorization
exist. Candidate exposure and batch settings are pilot inputs, not silently
accepted formal hyperparameters.

No SHA-256 or per-file content hashing is performed by these scripts.

## Active 24-hour anchored candidate

The generated manifests use runtime budget
`drl_rq_runtime_budget_v2_cus1000_24h_anchor`:

| Scale | Epochs | Environments/epoch | Effective batch | Total environments | Customer exposures |
|---|---:|---:|---:|---:|---:|
| Cus50 | 100 | 200 | 200 | 20,000 | 1,000,000 |
| Cus100 | 200 | 50 | 50 | 10,000 | 1,000,000 |
| Cus500 | 500 | 4 | 4 | 2,000 | 1,000,000 |
| Cus1000 | 1,000 | 1 | 1 | 1,000 | 1,000,000 |

The supplied 24-hour value is a per-job Cus1000 planning target on one A6000,
not an enforced timeout. With 12 jobs and two A6000 GPUs, the idealized
training-only queue is six waves, or about six days. A6000 pilot evidence is
still mandatory before `full` can be authorized.

For the common audit-checkout/original-data layout, the automatic sibling
discovery is sufficient. An explicit portable override is also accepted:

```bash
export EVRPTW_DATASET_ROOT="../EVRPTW-DB/EVRPTW_Dataset/Instances_v2/us_11city_full_clean_v7_bbde5db_20260823"
```
