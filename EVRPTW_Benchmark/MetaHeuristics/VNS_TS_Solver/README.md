# VNS_TS_Solver

Pickle-native VNS + Tabu Search benchmark module. The solver is adapted from the legacy `VNS_TabuSearch_Solver` and writes the shared `EVRPTWSolution` schema.

```bash
python run_vns_ts.py \
  --dataset_path ../../../EVRPTW_Dataset/AC_v1/AC_Tiny_5 \
  --save_path ../../results/AC_v1/AC_Tiny_5/VNS_TS_Solver \
  --num_workers 4 \
  --seed 2026
```

Important parameters:

- `--predefine_route_number`: route count used by the initial sweep construction.
- `--eta_feas`, `--eta_dist`: outer VNS budgets for feasibility and distance phases.
- `--tabu_iter`: inner tabu-search iterations per VNS perturbation.
- `--scales`: optional comma-separated scale filter, for example `Cus5,Cus15`.

The adapter keeps objective units in kilometers and time units in seconds. Energy consumption is computed as `distance_km * kWh_per_km`, matching the canonical dataset schema.

Smoke test on the packaged validation split:

```bash
python EVRPTW_Benchmark/MetaHeuristics/VNS_TS_Solver/run_vns_ts.py \
  --dataset_path EVRPTW_Dataset/dataset_v1/dataset/val \
  --save_path EVRPTW_Benchmark/results/dataset_v1/val/VNS_TS_Solver_one_instance \
  --scales Cus5 \
  --max_instances 1 \
  --num_workers 1 \
  --seed 2026 \
  --predefine_route_number 3 \
  --eta_feas 5 \
  --eta_dist 5 \
  --tabu_iter 5 \
  --search_mode fast
```


## Search Modes

Default `search_mode=fast` uses candidate-budgeted Tabu Search: it screens relocate/exchange/2-opt/station moves by local road-distance deltas and evaluates only the best bounded candidate set. This keeps VNS-TS practical as a benchmark baseline while preserving the VNS perturbation + Tabu improvement structure.

Use `--search_mode full --eta_feas 60 --eta_dist 60 --tabu_iter 30` to run the legacy-style full-neighborhood Tabu Search on small instances.

Default fast profile:

```text
eta_feas=20
eta_dist=20
tabu_iter=10
move_candidate_limit=40
route_neighbor_limit=4
position_neighbor_limit=4
exchange_neighbor_limit=6
station_candidate_limit=5
```
