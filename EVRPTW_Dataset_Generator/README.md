# EVRPTW-D Dataset Generator

This package implements the EVRPTW-D two-stage generator. The current release profile is **AC-v1** (Amazon-Calibrated v1).

1. A **service territory graph** represents a stable city/region/delivery-station territory. It stores a road graph, depot, latent customer pool, community structure, and charging-station candidate pool.
2. An **operating-day instance** activates customers and charging stations from a territory, then samples demand, service time, time windows, and active EV travel data.

The benchmark framing is:

> We first sample a city/region-level service territory, then sample daily operating instances from that territory.

Legacy code fields such as `mother_board_id` are retained for compatibility, but public manifests and documentation use service-territory terminology.

## Prepare AC-v1

```bash
conda run -n maojie python -m EVRPTW_Dataset_Generator.prepare_ac_benchmark_suite \
  --train-territories 1024 \
  --eval-territories 256 \
  --num-instances 1000 \
  --seed 20260525
```

This creates single-file pickle bundles:

```text
EVRPTW_Dataset/AC_v1/
  train/service_territory_pool.pkl
  eval/service_territory_pool.pkl
  eval/AC_Tiny_5/instances.pkl
  eval/AC_Small_15/instances.pkl
  eval/AC_Medium_50/instances.pkl
  eval/AC_Large_100/instances.pkl
  eval/AC_XLarge_1000/instances.pkl
  generation_timing.csv
  dataset_manifest.json
```

## Reusable Service-Territory Pools

For repeated RL training runs, territory generation can be moved offline and reused across customer scales:

```bash
conda run -n maojie python -m EVRPTW_Dataset_Generator.prepare_region_pool \
  --num-territories 1024 \
  --latent-customer-pool-size 5000 \
  --cs-candidate-pool-size 120 \
  --seed 20260525
```

Default output:

```text
EVRPTW_Dataset/AC_v1/train/service_territory_pool.pkl
```

Downstream trainers can pass this path as `territory_pool_path`. If the pool cannot be read or contains fewer territories than requested, training falls back to online generation.

## Documentation

- [Design rationale](docs/design_rationale.md) explains the two-stage service-territory / active-day model.
- [Calibration guideline](docs/calibration_guideline.md) explains parameter meanings and how to calibrate the generator to a new real last-mile dataset.
- [Amazon depot-day scale reference](docs/amazon_depot_day_scale.md) records observed station-territory and daily active-customer scales.

## Stored Matrices

Service territories do not store active-day time windows or charging-aware shortest-time matrices. Each operating-day instance computes these after active customers and active charging stations are selected, so inactive charging stations cannot leak into instance travel times. By default, instance pickle files persist `distance_matrix_km` and `cs_time_to_depot_s`; raw travel-time, EV transition-time, and shortest-time matrices in seconds are computed during generation/audit but not saved unless enabled in `storage`.

Amazon historical route size is used only as a proxy for active community demand scale. Amazon route count, actual route sequence, route duration, and route cost are never used as generated vehicle targets or solution priors.
