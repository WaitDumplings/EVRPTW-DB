# EVRPTW-DB

EVRPTW-DB is a dataset-and-benchmark repository for Electric Vehicle Routing Problems with Time Windows (EVRP-TW).

The project is organized around two deliverables:

- **D: Dataset** - a real-data-calibrated hierarchical generator and generated EVRP-TW-D instances.
- **B: Benchmark** - exact, metaheuristic, and reinforcement-learning solvers evaluated on the same instance schema.

## Repository Layout

```text
EVRPTW-DB/
  EVRPTW_Core/                  # shared schema, loaders, validation, metrics
  EVRPTW_Dataset_Generator/     # hierarchical mother-board / active-day generator
  EVRPTW_Dataset/               # generated datasets, grouped by dataset name / Cus / CS
  EVRPTW_Benchmark/             # solver implementations and benchmark runner
    Exact/
      Gurobi_Solver/
    MetaHeuristics/
      Greedy_Solver/
      VNS_TS_Solver/
      ALNS_Solver/
    Reinforcement_Learning/
      TERRAN/
  docs/
```

## Current Status

- `EVRPTW_Dataset_Generator` contains the Amazon-calibrated hierarchical generator.
- `EVRPTW_Dataset` is intentionally empty in git; generated instances should be produced locally or released separately.
- `EVRPTW_Benchmark` currently contains the planned solver layout. Solver adapters and validation will be added next.

## Example Dataset Generation

```bash
cd EVRPTW_Dataset_Generator
python instance_generate.py \
  --config_path configs/amazon_hierarchy.yaml \
  --save_path ../EVRPTW_Dataset/Amazon_Calibrated_v1/Cus_1800/CS_12 \
  --num_instances 1000 \
  --num_customers 1800 \
  --num_charging_stations 12 \
  --num_regions 10 \
  --mother_num_customers 5000 \
  --mother_num_charging_stations 120 \
  --region_reuse_limit 200 \
  --seed 20260522
```

## Design Notes

- Dataset generation is two-stage: mother board = region/service territory, instance = one operating day.
- Benchmark solvers must only read exported daily instances from `EVRPTW_Dataset`; they must not access inactive mother-board customers or inactive charging stations.
- Shared validation and metrics will live in `EVRPTW_Core` to avoid schema drift between generator and solvers.
