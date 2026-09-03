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

Each `full.sh` is unlocked independently by the pilot jobs in the same server
manifest. All of those pilots must pass under the same executable commit as the
full run. No shared `pilot_gate_report.json` or cross-server wait is required.
The aggregate pilot report remains reviewer evidence, not a runtime dependency.

No SHA-256 or per-file content hashing is performed by these scripts.

## Active Cus1000 batch-two candidate

The generated manifests use runtime budget
`drl_rq_runtime_budget_v4_cus1000_b2_val100`:

| Scale | Epochs | Environments/epoch | Effective batch | Total environments | Customer exposures |
|---|---:|---:|---:|---:|---:|
| Cus50 | 1,000 | 200 | 200 | 200,000 | 10,000,000 |
| Cus100 | 1,000 | 50 | 50 | 50,000 | 5,000,000 |
| Cus500 | 1,000 | 4 | 4 | 4,000 | 2,000,000 |
| Cus1000 | 1,000 | 2 | 2 | 2,000 | 2,000,000 |

The effective batch is the number of distinct base instances consumed per
logical epoch. Physical batches are only a memory implementation detail; exact
gradient accumulation preserves the effective batch for the three REINFORCE
models. TERRAN receives the same base-instance count before its paper-specific
PPO minibatch updates.

| Scale | AM physical | EVRPTW-RL physical | DRL-TS physical | TERRAN physical | Microbatches AM/RL/TS/TERRAN |
|---|---:|---:|---:|---:|---:|
| Cus50 | 200 | 100 | 50 | 200 | 1 / 2 / 4 / 1 |
| Cus100 | 50 | 25 | 10 | 50 | 1 / 2 / 5 / 1 |
| Cus500 | 4 | 2 | 1 | 4 | 1 / 2 / 4 / 1 |
| Cus1000 | 1 | 1 | 1 | 1 | 2 / 2 / 2 / 2 |

Every formal job runs 1,000 logical epochs. Validation is evaluated at epochs
50, 100, ..., 1000. Cus50/100/500 use a fixed 500-view selection set; Cus1000
uses a fixed 100-view selection set. Selection first maximizes independent-
verifier feasibility rate and then minimizes mean verified directed distance.
The winning checkpoint is written as both `best.ckpt` and the backward-
compatible `checkpoint_selected.pt`; all 20 records remain in
`validation_history.jsonl`. After selection, Cus1000 runs one full 500-view val
audit and writes `validation_final_audit.json` without changing the selected
checkpoint.

The linear Cus1000 planning value is now approximately 48 hours per model/seed
job on one A6000, not an enforced timeout. With 12 jobs and two A6000 GPUs, the
idealized training-only queue is six waves, or about 12 days. A6000 pilot
evidence is still mandatory before `full` can be authorized.

For the common audit-checkout/original-data layout, the automatic sibling
discovery is sufficient. An explicit portable override is also accepted:

```bash
export EVRPTW_DATASET_ROOT="../EVRPTW-DB/EVRPTW_Dataset/Instances_v2/us_11city_full_clean_v7_bbde5db_20260823"
```
