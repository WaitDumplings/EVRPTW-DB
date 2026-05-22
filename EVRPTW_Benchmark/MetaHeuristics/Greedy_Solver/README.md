# Greedy_Solver

Pickle-native constructive Greedy baseline for EVRP-TW-D instances.

The solver is intentionally simple and deterministic. It uses the canonical `distance_matrix_km`, second-level time windows, full-charge station semantics, and the shared `EVRPTWSolution` output schema. It is suitable both as a fast benchmark baseline and as an independent sanity check for generated instances.

```bash
python run_greedy.py \
  --dataset_path ../../../EVRPTW_Dataset/Amazon_Calibrated_v1/Cus_5/CS_2 \
  --save_path ../../results/Amazon_Calibrated_v1/Cus_5/CS_2/Greedy_Solver \
  --num_workers 4 \
  --customer_order nearest \
  --seed 2026
```

Customer ordering modes:

- `nearest`: choose the nearest feasible unserved customer from the current node.
- `earliest_due`: prioritize feasible customers with earlier due times.
- `hybrid`: nearest-distance score with a small due-time term.

The benchmark writes:

```text
greedy_summary.csv
greedy_routes.csv
solutions/*.pkl
```

Reviewer-facing role:

```text
Greedy is a constructive feasibility baseline and vehicle upper-bound estimator.
It is not an optimization label for the dataset generator.
```
