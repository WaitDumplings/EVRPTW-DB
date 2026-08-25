# Test benchmark shells

The canonical explicit `solver × scale × test` entry points now live in
`EVRPTW_Benchmark/scripts/`. This folder retains the earlier combined Cus50 and
Cus500 launchers for compatibility.

This folder contains frozen Test launchers for the Exact, ALNS, and VNS-TS
baselines. All runs use the same per-instance timing contract:

```text
checkpoints: 300, 1800, 3600, 7200 seconds
             5 min, 30 min, 1 h, 2 h
time limit:  7200 seconds
workers:     30 by default
Gurobi:      cs_copies=2, mip_gap=0, one thread per worker
ALNS/VNS-TS: seed=2026, one BLAS/OpenMP thread per worker
```

A solution found after a checkpoint is never copied backward. If Gurobi proves
optimality early, its certified optimum and route are copied to every later
checkpoint. If ALNS or VNS-TS terminates naturally early, its final best
validated incumbent is copied forward, but is not mislabeled as certified
optimal.

## 1. Restore the dataset on every server

Copy the 6.9-GiB slim archive to the server, enter the cloned repository, and
start the persistent restore job:

```bash
./restore_dataset_archive.sh start \
  --archive /path/to/EVRPTW_Dataset_us11city_full_clean_v7_bbde5db.tar.zst \
  --destination "$PWD" \
  --workers 30
```

The command starts a detached tmux job by default. Monitor and wait with:

```bash
./restore_dataset_archive.sh status --destination "$PWD"
./restore_dataset_archive.sh logs --destination "$PWD" --follow
./restore_dataset_archive.sh wait --destination "$PWD"
```

Do not start benchmarks until the report below has `passed=true` and
`selected_family_count=7500`:

```text
EVRPTW_Dataset/Instances_v2/us_11city/matrix_restore_report.json
```

The restored tree is roughly 175 GiB including regenerated matrices; allow at
least 200 GiB of free disk per server. The validated 30-worker restore took
about 5.3 hours on the generation server. No SHA256/content-hash pass is used.

## 2. Cus50 compatibility Test-1

Cus50 contains one 500-view Test-1 index over the ten seen cities:

```bash
bash EVRPTW_Benchmark/test_scripts/run_gurobi_cus50_test.sh
bash EVRPTW_Benchmark/test_scripts/run_alns_cus50_test.sh
bash EVRPTW_Benchmark/test_scripts/run_vnsts_cus50_test.sh
```

Run all three sequentially with:

```bash
bash EVRPTW_Benchmark/test_scripts/run_all_cus50_tests.sh
```

## 3. Cus500 Core tests

Cus500 covers three independent 500-view indices, for 1,500 views per solver:

```text
core/test/test1_new_seed
core/test/test2_heldout_locations
core/test/test3_heldout_city
```

Each solver shell runs Test-1, Test-2, and Test-3 sequentially:

```bash
bash EVRPTW_Benchmark/test_scripts/run_gurobi_cus500_tests.sh
bash EVRPTW_Benchmark/test_scripts/run_alns_cus500_tests.sh
bash EVRPTW_Benchmark/test_scripts/run_vnsts_cus500_tests.sh
```

Run all three solvers sequentially with:

```bash
bash EVRPTW_Benchmark/test_scripts/run_all_cus500_tests.sh
```

At 30 workers, the theoretical upper bound is about 100 hours per solver when
all 1,500 instances consume the full two-hour budget. The combined launcher can
therefore take about 300 hours on one server. Multi-server partitioning is
strongly recommended, especially for time-limited Cus500 Gurobi.

## 4. Multi-server partitioning

Use zero-based shard indices. For example, with ten servers, run the following
on server 3:

```bash
EVRPTW_SHARD_COUNT=10 EVRPTW_SHARD_INDEX=3 \
  bash EVRPTW_Benchmark/test_scripts/run_all_cus500_tests.sh
```

This assigns the same non-overlapping row range from every 500-view test index
to all three solvers. With ten servers, each server handles 50 views per test,
150 views per solver, and at most about ten hours per solver at 30 workers.
Shard outputs are isolated under directories such as `shard_3_of_10`, so they
are safe even on a shared results filesystem.

Every server must use the same:

- Git commit;
- restored release ID;
- `EVRPTW_SHARD_COUNT`;
- checkpoint/time-limit contract;
- seed and solver profile.

The shard indices across servers must cover exactly `0..count-1` once each.

For a one-instance command audit without solving:

```bash
EVRPTW_DRY_RUN=1 EVRPTW_MAX_INSTANCES=1 \
  EVRPTW_SHARD_COUNT=10 EVRPTW_SHARD_INDEX=0 \
  bash EVRPTW_Benchmark/test_scripts/run_all_cus500_tests.sh
```

## Execution controls

```text
EVRPTW_TEST_WORKERS       default 30
EVRPTW_MAX_IN_FLIGHT      default 2 * workers (ALNS/VNS-TS)
EVRPTW_CSV_FLUSH_INTERVAL default 25 completed views (ALNS/VNS-TS)
EVRPTW_TEST_RESULTS_ROOT  custom output root
EVRPTW_SHARD_COUNT        number of servers
EVRPTW_SHARD_INDEX        zero-based server index
EVRPTW_MAX_INSTANCES      small pilot/diagnostic limit within a shard
EVRPTW_START_INDEX        manual inclusive row, alternative to shard variables
EVRPTW_END_INDEX          manual exclusive row, alternative to shard variables
EVRPTW_CONDA_ENV          default maojie
EVRPTW_DATASET_ROOT       custom restored dataset root
EVRPTW_DRY_RUN=1          print commands without running solvers
```

The canonical restored dataset location is detected automatically at
`EVRPTW_Dataset/Instances_v2/us_11city`. The frozen generation root is used as
a fallback on the generation server.

## Outputs

Outputs are written below `EVRPTW_Benchmark/results/CLE_EVRPTW_v2_test_2h`.
Each solver writes a summary CSV, a time-trace CSV containing objective and full
`routes_json` at all four checkpoints, final solution files, and checkpoint
solution files. Every published route is independently replayed.

All launchers use `--skip_completed`. Re-running the same command safely resumes
completed output. Do not change the run contract inside an existing output
directory.
