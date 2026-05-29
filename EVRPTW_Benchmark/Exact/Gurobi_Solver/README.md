# Gurobi_Solver

Exact small-scale EVRP-TW-D solver using Gurobi and the canonical pickle instance schema.

## Instance Format

The new benchmark path reads `.pkl` daily instances through `EVRPTW_Core`. We do **not** use the legacy `.txt` parser as the primary format. A txt exporter can be added later only for backward compatibility.

## Run

```bash
python run_gurobi.py \
  --dataset_path ../../../EVRPTW_Dataset/AC_v1/AC_Tiny_5 \
  --save_path ../../results/AC_v1/AC_Tiny_5/Gurobi_Solver \
  --time_limit_s 900 \
  --checkpoints_s 60,300,900 \
  --cs_copies 3
```

The solver writes:

- `gurobi_summary.csv`: one row per instance, including final objective, final route sequence, final gap, and `first_feasible_time_s`;
- `gurobi_time_trace.csv`: one row per `instance_id + checkpoint_s`, including incumbent objective, best bound, gap, route sequence, and checkpoint solution path;
- `solutions/*.pkl`: final solution per instance;
- `solutions/checkpoints/*.pkl`: incumbent solution snapshots for requested checkpoints.
- when `--reference_save_path` is provided, `reference_save_path/<split>/solutions.csv` plus per-instance `routes/<scale>/*.json` files matching the reference-solution template.

Completed instances are flushed incrementally: after each instance finishes, the runner atomically rewrites the summary CSV, time-trace CSV, optional reference `solutions.csv`, and route JSON. Existing rows for the same `instance_id` are replaced and rows are kept sorted by `instance_id`, so long multi-day runs remain inspectable and restart-friendly.
Use `--skip_completed` to resume from an existing `gurobi_summary.csv` without re-solving instances that already have a non-error status row.

Checkpoint semantics: at checkpoint `t`, the row stores the best incumbent solution available at or before runtime `t`. If the model proves optimal before a later checkpoint, that checkpoint row records the final solution with `reached_checkpoint=false`.

If `--checkpoints_s` is omitted or empty, the runner defaults to a 7200-second exact solve and records one checkpoint at 7200 seconds. If checkpoints are provided and `--time_limit_s` is omitted, the time limit defaults to the largest checkpoint. `vehicle_count` is reported in both the final summary and every checkpoint row. Charging-station dummy copies are controlled by `--cs_copies` and default to 3.

For the packaged eval split, run from the repository root with the local Gurobi
13 package/license:

```bash
GRB_LICENSE_FILE=/home/exx/anaconda3/envs/maojie/lib/gurobi.lic \
PYTHONPATH=/home/exx/anaconda3/envs/maojie/lib/python3.11/site-packages \
python EVRPTW_Benchmark/Exact/Gurobi_Solver/run_gurobi.py \
  --dataset_path EVRPTW_Dataset/dataset_v1/dataset/eval \
  --save_path EVRPTW_Benchmark/results/dataset_v1/eval/Gurobi_Solver \
  --reference_save_path EVRPTW_Dataset/dataset_v1/reference_solutions \
  --reference_split eval \
  --workers 4 \
  --threads 1 \
  --time_limit_s 900 \
  --checkpoints_s 60,300,900
```

Use `--scales Cus5,Cus15`, `--limit N`, or an instance suffix range for smoke tests and distributed runs. The range is half-open: `--start_index 750 --end_index 1000` runs instance ids ending in `000750` through `000999`. With `--workers > 1`, the runner defaults to one Gurobi thread per worker unless `--threads` is set.

For a second server that should only fill the final quarter of `val/Cus15`, use the slice helper from the repository root. It defaults to `Cus15`, `cs_copies=2`, `time_limit_s=7200`, `skip_completed`, and the half-open range `[750, 1000)`:

```bash
GRB_LICENSE_FILE=/home/exx/anaconda3/envs/maojie/lib/gurobi.lic \
PYTHONPATH=/home/exx/anaconda3/envs/maojie/lib/python3.11/site-packages \
python EVRPTW_Benchmark/Exact/Gurobi_Solver/run_val_cus15_slice.py \
  --workers 16 \
  --verbose
```

Override `--start_index` and `--end_index` to assign a different shard. Prefer writing each server to a separate output directory and merging by `instance_id` later if the machines do not share a filesystem with safe file locking.


## Gurobi Environment

`gurobipy` must match the installed Gurobi license major version. If the Python package is newer than the license, Gurobi will fail before model construction. For example, a Gurobi 12 license requires a Gurobi 12-compatible Python environment or an updated Gurobi 13 license.

The runner records per-instance environment/model errors in `gurobi_summary.csv` instead of stopping the whole benchmark batch. Use `--save_traceback` when debugging local solver installations.
