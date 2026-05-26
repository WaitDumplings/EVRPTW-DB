# EVRPTW-DB

EVRPTW-DB is a dataset-and-benchmark repository for Electric Vehicle Routing Problems with Time Windows (EVRP-TW).

The project is organized around two deliverables:

- **EVRPTW-D**: real-data-calibrated generated EVRP-TW datasets.
- **EVRPTW-B**: exact, metaheuristic, and reinforcement-learning solvers evaluated on the same instance schema.

The first released dataset profile is **AC-v1**: Amazon-Calibrated v1. This keeps the dataset family name stable (`EVRPTW-D`) while allowing later calibrated profiles such as `AC-v2` or non-Amazon profiles.

## Repository Layout

```text
EVRPTW-DB/
  EVRPTW_Core/                  # shared schema, loaders, validation, metrics
  EVRPTW_Dataset_Generator/     # service-territory / operating-day generator
  EVRPTW_Dataset/               # generated EVRPTW-D releases, e.g. AC_v1
  EVRPTW_Benchmark/             # solver implementations and benchmark runners
    Exact/
    MetaHeuristics/
    Reinforcement_Learning/
  docs/
```

## Dataset Framing

Generation is two-stage:

1. A **service territory graph** represents a stable city/region/delivery-station territory.
2. An **operating-day instance** activates customers and charging stations from that territory, then samples demand, service time, time windows, and active travel matrices.

Internal code still preserves legacy `mother_board_*` fields for backward compatibility, but public documentation and release manifests use service-territory terminology.

## AC-v1 Layout

```text
EVRPTW_Dataset/
  AC_v1/
    train/service_territory_pool.pkl        # 1024 training service territories
    eval/service_territory_pool.pkl         # held-out evaluation service territories
    eval/AC_Tiny_5/instances.pkl
    eval/AC_Small_15/instances.pkl
    eval/AC_Medium_50/instances.pkl
    eval/AC_Large_100/instances.pkl
    eval/AC_XLarge_1000/instances.pkl
    generation_timing.csv
    dataset_manifest.json
```

## Example

```bash
python EVRPTW_Dataset_Generator/prepare_ac_benchmark_suite.py \
  --train-territories 1024 \
  --eval-territories 256 \
  --num-instances 1000 \
  --seed 20260525
```

Benchmark solvers must only read exported operating-day instances from `EVRPTW_Dataset`; they must not access inactive territory customers or inactive charging stations.
