# Frozen benchmark launchers by solver, scale, and test

These are the canonical, explicit entry points for the Stage-2 v7 test set.
Every leaf shell runs exactly one `solver × scale × test` combination. The
older `EVRPTW_Benchmark/test_scripts/` launchers remain available as combined
compatibility entry points.

## Actual test matrix

| Scale | Customers | Test entry points | Views per entry point |
|---|---:|---|---:|
| Cus50 | 50 | Test1 only | 500 |
| Cus100 | 100 | Test1, Test2, Test3 | 500 each |
| Cus500 | 500 | Test1, Test2, Test3 | 500 each |
| Cus1000 | 1,000 | Test1, Test2, Test3, scalability | 500 each |
| Cus2000 | 2,000 | scalability only | 500 |

The three core tests are:

- `test1.sh`: new seeds on the seen cities;
- `test2.sh`: held-out customer/charger locations;
- `test3.sh`: held-out city.

Cus2000 does have a test split. It is the unseen-scale, same-cities
scalability/scale-transfer test rather than one of the three core tests. Its
view index contains 500 Cus1000 controls and 500 Cus2000 unseen-scale views, so
both `Cus1000/scalability.sh` and `Cus2000/scalability.sh` are provided.

## Directory layout

```text
scripts/
├── Gurobi/{Cus50,Cus100,Cus500,Cus1000,Cus2000}/
├── ALNS/{Cus50,Cus100,Cus500,Cus1000,Cus2000}/
└── VNSTS/{Cus50,Cus100,Cus500,Cus1000,Cus2000}/
```

## Examples

```bash
bash EVRPTW_Benchmark/scripts/Gurobi/Cus500/test2.sh
bash EVRPTW_Benchmark/scripts/ALNS/Cus100/test3.sh
bash EVRPTW_Benchmark/scripts/VNSTS/Cus2000/scalability.sh
```

All shells use the frozen contract: checkpoints at 300, 1800, 3600, and 7200
seconds; a 7200-second limit; 30 workers by default; seed 2026 for ALNS/VNS-TS;
and `cs_copies=2`, `mip_gap=0`, and one Gurobi thread per worker for Exact.

The launchers automatically find the canonical restored root at
`EVRPTW_Dataset/Instances_v2/us_11city`. Set `EVRPTW_DATASET_ROOT` only when the
dataset is elsewhere.

## Multi-server shards

Each index/scale pair has 500 views. With ten servers, assign a unique
zero-based shard index to every server:

```bash
EVRPTW_SHARD_COUNT=10 EVRPTW_SHARD_INDEX=3 \
  bash EVRPTW_Benchmark/scripts/Gurobi/Cus500/test2.sh
```

The same shard variables select the identical view IDs for Gurobi, ALNS, and
VNS-TS. Do not change the shard count while resuming an existing result tree.

Useful controls:

```text
EVRPTW_TEST_WORKERS       default 30
EVRPTW_MAX_IN_FLIGHT      default 2 × workers for ALNS/VNS-TS
EVRPTW_CSV_FLUSH_INTERVAL default 25
EVRPTW_TEST_RESULTS_ROOT  custom output root
EVRPTW_SHARD_COUNT        total server shards
EVRPTW_SHARD_INDEX        zero-based shard index
EVRPTW_MAX_INSTANCES      pilot limit within the selected shard
EVRPTW_START_INDEX        manual inclusive filtered position
EVRPTW_END_INDEX          manual exclusive filtered position
EVRPTW_CONDA_ENV          default maojie
EVRPTW_DATASET_ROOT       restored dataset root
EVRPTW_DRY_RUN=1          print the command without solving
```

Every launcher resumes terminal instances through `--skip_completed`. Output
directories include solver, scale, test, and shard identity.
