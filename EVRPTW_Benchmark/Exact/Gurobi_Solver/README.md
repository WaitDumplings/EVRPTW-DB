# Gurobi_Solver

Exact small-scale EVRP-TW-D solver using Gurobi and the canonical pickle instance schema.

## Instance Format

The new benchmark path reads `.pkl` daily instances through `EVRPTW_Core`. We do **not** use the legacy `.txt` parser as the primary format. A txt exporter can be added later only for backward compatibility.

## Run

```bash
python run_gurobi.py \
  --dataset_path ../../../EVRPTW_Dataset/Amazon_Calibrated_v1/Cus_5/CS_2 \
  --save_path ../../results/Amazon_Calibrated_v1/Cus_5/CS_2/Gurobi_Solver \
  --time_limit_s 900 \
  --checkpoints_s 60,300,900 \
  --cs_copies 3
```

The solver writes:

- `gurobi_summary.csv`: one row per instance, including final objective, final route sequence, final gap, and `first_feasible_time_s`;
- `gurobi_time_trace.csv`: one row per `instance_id + checkpoint_s`, including incumbent objective, best bound, gap, route sequence, and checkpoint solution path;
- `solutions/*.pkl`: final solution per instance;
- `solutions/checkpoints/*.pkl`: incumbent solution snapshots for requested checkpoints.

Checkpoint semantics: at checkpoint `t`, the row stores the best incumbent solution available at or before runtime `t`. If the model proves optimal before a later checkpoint, that checkpoint row records the final solution with `reached_checkpoint=false`.

If `--checkpoints_s` is omitted or empty, the runner defaults to a 7200-second exact solve and records one checkpoint at 7200 seconds. If checkpoints are provided and `--time_limit_s` is omitted, the time limit defaults to the largest checkpoint. `vehicle_count` is reported in both the final summary and every checkpoint row. Charging-station dummy copies are controlled by `--cs_copies` and default to 3.


## Gurobi Environment

`gurobipy` must match the installed Gurobi license major version. If the Python package is newer than the license, Gurobi will fail before model construction. For example, a Gurobi 12 license requires a Gurobi 12-compatible Python environment or an updated Gurobi 13 license.

The runner records per-instance environment/model errors in `gurobi_summary.csv` instead of stopping the whole benchmark batch. Use `--save_traceback` when debugging local solver installations.
