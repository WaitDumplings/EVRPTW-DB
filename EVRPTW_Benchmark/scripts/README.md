# Frozen traditional-solver benchmark launchers

These are the canonical, explicit entry points for the Stage-2 v7 test set.
Every leaf shell runs exactly one `solver × scale × test` combination. The
older `EVRPTW_Benchmark/test_scripts/` launchers remain available as combined
compatibility entry points.

## Approved execution matrix

| Scale | Available shell entry points | Views per test | Traditional-solver role |
|---|---|---:|---|
| Cus50 | Test1 | 500 | required compatibility experiment |
| Cus100 | Test1, Test2, Test3 | 500 each | required main experiment |
| Cus500 | Test1, Test2, Test3 | 500 each | required main experiment |
| Cus1000 | Test1, Test2, Test3 | 500 each | optional diagnostic only |
| Cus2000 | none | 500 paired views | learning-method scale transfer only |

The three core tests are:

- `test1.sh`: new seeds on the seen cities;
- `test2.sh`: held-out communities/locations in the same ten cities;
- `test3.sh`: held-out city.

Cus1000 Test1/2/3 shells are deliberately retained in case later diagnostic
budgets produce useful results. They are not part of the mandatory
traditional-solver cohort or the full-set ranking. Use
`EVRPTW_MAX_INSTANCES` or sharding for a declared subset.

Cus2000 is a paired, same-city unseen-scale evaluation: a Cus1000-trained
learning method is evaluated on deterministic Cus1000 controls and Cus2000
views without fine-tuning. Gurobi, ALNS, and VNS-TS launchers are intentionally
not provided for this track.

## Directory layout

```text
scripts/
├── Gurobi/{Cus50,Cus100,Cus500,Cus1000}/
├── ALNS/{Cus50,Cus100,Cus500,Cus1000}/
└── VNSTS/{Cus50,Cus100,Cus500,Cus1000}/
```

## Examples

```bash
bash EVRPTW_Benchmark/scripts/Gurobi/Cus500/test2.sh
bash EVRPTW_Benchmark/scripts/ALNS/Cus100/test3.sh
EVRPTW_MAX_INSTANCES=20 bash EVRPTW_Benchmark/scripts/VNSTS/Cus1000/test1.sh
```

All shells use the frozen contract: checkpoints at 300, 1800, 3600, and 7200
seconds; a 7200-second limit; 30 workers by default; seed 2026 for ALNS/VNS-TS;
and `cs_copies=2`, `mip_gap=0`, and one Gurobi thread per worker for Exact.

The default dataset root is repository-relative:
`EVRPTW_Dataset/Instances_v2/us_11city`. Extract/restore the dataset there and
the same clone works on every server. `EVRPTW_DATASET_ROOT` and
`EVRPTW_TEST_RESULTS_ROOT` may also be repository-relative; absolute
overrides remain supported.

## Unified checkpoint output

All three runners emit the same time-trace CSV columns at 300, 1800, 3600, and
7200 seconds. Each row contains best-so-far objective, routes, flattened route
sequence, incumbent/status timing, validation status, solver identity, and
source metadata. Gurobi additionally populates its valid lower bound and MIP
gap; these two fields are blank for ALNS and VNS-TS because heuristics do not
provide certified bounds.

The filenames remain solver-specific
(`gurobi_time_trace.csv`, `alns_time_trace.csv`, and
`vns_ts_time_trace.csv`) so existing resume trees remain compatible. If a
run finishes early, later checkpoints repeat the final incumbent; only Gurobi
may label a solution proven optimal.

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
EVRPTW_TEST_RESULTS_ROOT  custom output root (repository-relative or absolute)
EVRPTW_SHARD_COUNT        total server shards
EVRPTW_SHARD_INDEX        zero-based shard index
EVRPTW_MAX_INSTANCES      pilot limit within the selected shard
EVRPTW_START_INDEX        manual inclusive filtered position
EVRPTW_END_INDEX          manual exclusive filtered position
EVRPTW_CONDA_ENV          default maojie
EVRPTW_DATASET_ROOT       restored dataset root (repository-relative or absolute)
EVRPTW_DRY_RUN=1          print the command without solving
```

Every launcher resumes terminal instances through `--skip_completed`. Output
directories include solver, scale, test, and shard identity.
