# ALNS_Solver

Pickle-native multiprocess ALNS benchmark solver for EVRP-TW-D instances.

This module adapts the legacy `ALNS_Solver_MULTI` implementation to the canonical `EVRPTW_Core` instance schema. It does not read legacy txt files. The solver uses the instance `distance_matrix_km` road-distance matrix as the objective metric, so objective values are comparable with `Exact/Gurobi_Solver`.

## Run

```bash
python run_alns.py \
  --dataset_path ../../../EVRPTW_Dataset/AC_v1/AC_Tiny_5 \
  --save_path ../../results/AC_v1/AC_Tiny_5/ALNS_Solver \
  --num_workers 4 \
  --seed 2026
```

Optional iteration controls:

- `--max_iters`: override the ALNS maximum iterations.
- `--delta_iters`: run only a limited number of iterations.
- `--scales`: optional comma-separated scale filter, for example `Cus5,Cus15`.

Smoke test on the packaged validation split:

```bash
python EVRPTW_Benchmark/MetaHeuristics/ALNS_Solver/run_alns.py \
  --dataset_path EVRPTW_Dataset/dataset_v1/dataset/val \
  --save_path EVRPTW_Benchmark/results/dataset_v1/val/ALNS_Solver_one_instance \
  --scales Cus5 \
  --max_instances 1 \
  --num_workers 1 \
  --seed 2026 \
  --delta_iters 20
```

Outputs:

- `alns_summary.csv`: one row per instance, including objective, vehicle count, runtime, routes, and solution path.
- `alns_routes.csv`: one row per route.
- `solutions/*.pkl`: canonical `EVRPTWSolution` pickles.
