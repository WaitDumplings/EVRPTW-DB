# Cus50 Test-1 benchmark shells

This folder freezes the directly comparable Test-1 run for the three classical
baselines. All three consume the same 500-view Stage-2 v7 index:

```text
generation_plan/compatibility_cus50/test/
  test1_new_seed_same_cities/view_index.parquet
```

The benchmark contains 500 `Cus50` views over the ten seen cities. Gurobi is
kept on this small-scale compatibility track; the separate Core and
Scalability tracks contain Cus100, Cus500, Cus1000, and Cus2000 views and are
not silently mixed into this comparison.

## Frozen timing contract

Every solver receives a 7,200-second per-instance wall-clock budget and writes
the best feasible objective and complete route at:

```text
300, 1800, 3600, 7200 seconds
5 min, 30 min, 1 h, 2 h
```

If Gurobi proves optimality before two hours, the certified optimal objective
and route are copied to every later checkpoint. If ALNS or VNS-TS terminates
naturally before two hours, its final best validated incumbent is copied to
later checkpoints; heuristic incumbents are not mislabeled as certified
optimal. A solution found after a checkpoint is never copied backward into
that earlier checkpoint.

Gurobi always uses `cs_copies=2`, `mip_gap=0`, and one Gurobi thread per worker.
ALNS and VNS-TS use seed 2026 and one BLAS/OpenMP thread per worker. The default
process count is 30 for all three solvers.

## Run

From the repository root:

```bash
bash EVRPTW_Benchmark/test_scripts/run_gurobi_cus50_test.sh
bash EVRPTW_Benchmark/test_scripts/run_alns_cus50_test.sh
bash EVRPTW_Benchmark/test_scripts/run_vnsts_cus50_test.sh
```

To run them sequentially:

```bash
bash EVRPTW_Benchmark/test_scripts/run_all_cus50_tests.sh
```

The combined launcher is deliberately sequential: running all three process
pools simultaneously would oversubscribe CPU and memory.

The restored release location is detected automatically at
`EVRPTW_Dataset/Instances_v2/us_11city`. On the generation server, the frozen
v7 source root is used as a fallback. A different restored location can be
selected explicitly:

```bash
EVRPTW_DATASET_ROOT=/data/EVRPTW_Dataset/Instances_v2/us_11city \
  bash EVRPTW_Benchmark/test_scripts/run_alns_cus50_test.sh
```

Useful non-contract execution controls are:

```text
EVRPTW_TEST_WORKERS       default 30
EVRPTW_MAX_IN_FLIGHT      default 2 * workers (ALNS/VNS-TS)
EVRPTW_CSV_FLUSH_INTERVAL default 25 completed views (ALNS/VNS-TS)
EVRPTW_TEST_RESULTS_ROOT  custom output root
EVRPTW_MAX_INSTANCES      small pilot/diagnostic limit
EVRPTW_START_INDEX        inclusive stable view-index row
EVRPTW_END_INDEX          exclusive stable view-index row
EVRPTW_CONDA_ENV          default maojie
EVRPTW_DRY_RUN=1          print commands without executing solvers
```

Changing worker count, range, or output location does not change the published
per-instance timing contract. Keep the default checkpoint list, two-hour
budget, seed, algorithm profile, and Gurobi CS-copy count for paper results.

## Outputs

Outputs are written below:

```text
EVRPTW_Benchmark/results/CLE_EVRPTW_v2_test_2h/
  compatibility_cus50/test1_new_seed_same_cities/
```

Each solver writes a summary CSV, a time-trace CSV containing objective and
full `routes_json` at all four checkpoints, final solution files, and one
checkpoint solution file for every checkpoint with an incumbent. Both
metaheuristics independently replay every published route; Exact does the same
after Gurobi extraction.
