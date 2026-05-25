# EVRP-TW-Hierarchy-D

This project implements a two-stage EVRP-TW-D generator.

1. A **mother board** represents a stable city/region/station service territory. It stores a road graph, depot, latent customer pool, community structure, and charging-station candidate pool.
2. A **daily instance** represents one operating day sampled from that territory. It activates customers and charging stations, then samples demand, service time, time windows, and computes the active EV shortest-time matrix.

The benchmark framing is:

> We first sample a city/region-level service territory, then sample daily operating instances from that territory.

## Example

```bash
python instance_generate.py \
  --config_path configs/amazon_hierarchy.yaml \
  --save_path dataset/Cus_1800_CS_12 \
  --num_instances 1000 \
  --num_customers 1800 \
  --num_charging_stations 12 \
  --num_regions 10 \
  --mother_num_customers 5000 \
  --mother_num_charging_stations 120 \
  --region_reuse_limit 200 \
  --seed 20260522
```

If `--mother_num_customers` is omitted, the default is read from
`region.default_mother_num_customers` in the config. The Amazon-calibrated
default is 5000 latent customers for efficient generation/training. Amazon-scale studies can override this to 8000+ latent customers, matching the lower end of observed station
mother-board sizes and covering the largest observed station-day active count
in the training set.

## Documentation

- [Design rationale](docs/design_rationale.md) explains the two-stage mother-board / active-day model.
- [Calibration guideline](docs/calibration_guideline.md) explains the algorithm self-audit, parameter meanings, and how to calibrate the generator to a new real last-mile dataset.
- [Amazon depot-day scale reference](docs/amazon_depot_day_scale.md) records the station/mother-board and daily active-customer scales used by the default config.

## Shortest-Path Acceleration

The generator keeps Dijkstra as the exact positive-edge shortest-path primitive. The default `shortest_path.oracle_mode: source_cache` caches source-to-all road distances only for nodes actually used by sampled daily instances. This is the measured faster default for the current Cus1800/CS12 setting because each day activates only part of the mother board.

For heavy reuse of the same region, `shortest_path.oracle_mode: terminal_matrix` can precompute road shortest distances among:

```text
depot + all latent customers + all charging-station candidates
```

Daily active instances then build `distance_matrix_km` by slicing this matrix. The tradeoff is a fixed memory and precomputation cost; in the benchmark report at `analysis_outputs/shortest_path_oracle_benchmark.md`, terminal precomputation used about 100 MB for 5121 terminals and was slower for 1, 3, and 10 sampled days.

## Output Layout

```text
save_path/
  regions/
    region_000_board.pkl
  instances/
    Cus_1800_CS_12/
      instance_000000.pkl
  metadata/
    region_usage.csv
    generation_summary.csv
    failed_attempts.csv
  analysis_outputs/
    daily_instance_vs_amazon.md
    plots/
```

Mother boards do not store active-day time windows or charging-aware shortest-time matrices. Each daily instance computes these after active customers and active charging stations are selected, so inactive charging stations cannot leak into instance travel times. By default, instance pickle files persist `distance_matrix_km` and `cs_time_to_depot_s`; raw travel-time, EV transition-time, and shortest-time matrices in seconds are computed during generation/audit but not saved unless enabled in `storage`.

Amazon historical route size is used only as a proxy for active community demand scale. Amazon route count, actual route sequence, route duration, and route cost are never used as generated vehicle targets or solution priors.
