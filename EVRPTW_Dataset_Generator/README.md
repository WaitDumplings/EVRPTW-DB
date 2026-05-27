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
- [Geo-AC-v1 geospatial profile](docs/geospatial_profile.md) documents county containers, source-file conventions, and depot catchment semantics.
- [Geo-AC-v1 public data pipeline](docs/data_pipeline.md) documents the Census/ACS/OSM/AFDC ETL, API keys, optional subterritory slicing, and release layout.

## Stored Matrices

Service territories do not store active-day time windows or charging-aware shortest-time matrices. Each operating-day instance computes these after active customers and active charging stations are selected, so inactive charging stations cannot leak into instance travel times. By default, instance pickle files persist `distance_matrix_km` and `cs_time_to_depot_s`; raw travel-time, EV transition-time, and shortest-time matrices in seconds are computed during generation/audit but not saved unless enabled in `storage`.

Amazon historical route size is used only as a proxy for active community demand scale. Amazon route count, actual route sequence, route duration, and route cost are never used as generated vehicle targets or solution priors.

## Prepare Geo-AC-v1

Geo-AC-v1 is the real-geography semi-synthetic profile. It uses county
containers for public geospatial source data, optional subterritory slicing for
large counties, road-frontage latent customers, and depot-centered road
catchments for operating-day instances.

The spatial layer comes from public data. Amazon-derived calibration is used
only for daily operating attributes such as demand, service time, time windows,
and activation rates; it does not determine customer locations.

First prepare normalized public geodata CSVs:

```bash
export CENSUS_API_KEY=...
export NREL_API_KEY=...
conda run -n maojie python EVRPTW_Dataset_Generator/prepare_public_geodata.py \
  --city-config EVRPTW_Dataset_Generator/configs/geo_ac_v1_us10.yaml \
  --output-root EVRPTW_Dataset/Geo_AC_v1/source_data \
  --config-out EVRPTW_Dataset_Generator/configs/geo_ac_v1_us10.with_sources.yaml
```

For the official NA-US-20 profile, split large county containers and generate
road-frontage latent customers:

```bash
conda run -n maojie python -m EVRPTW_Dataset_Generator.prepare_geospatial_subterritories \
  --source-config EVRPTW_Dataset_Generator/configs/geo_ac_v1_us10.with_sources.yaml \
  --slice-config EVRPTW_Dataset_Generator/configs/geo_ac_v1_na_us20_slices.yaml \
  --output-root EVRPTW_Dataset/Geo_AC_v1/source_data_na_us20 \
  --config-out EVRPTW_Dataset_Generator/configs/geo_ac_v1_na_us20.with_sources.yaml
```

Then generate service-territory pools and fixed evaluation operating-day
instances from those CSV files:

```bash
conda run -n maojie python -m EVRPTW_Dataset_Generator.prepare_geospatial_benchmark_suite \
  --city-config EVRPTW_Dataset_Generator/configs/geo_ac_v1_na_us20.with_sources.yaml \
  --output-root EVRPTW_Dataset/Geo_AC_v1/eval_standard_20 \
  --require-real-sources \
  --instances-per-scale 20

conda run -n maojie python -m EVRPTW_Dataset_Generator.prepare_geo_ac_release_metadata \
  --source-root EVRPTW_Dataset/Geo_AC_v1/source_data_na_us20 \
  --eval-root EVRPTW_Dataset/Geo_AC_v1/eval_standard_20
```

The official fixed evaluation split is `20 territories x 4 scales x 20
instances = 1600` operating-day instances. The source data directory remains
the reusable geospatial dataset; the eval directory is a fixed benchmark split.

The ETL writes standardized local CSV inputs for roads, customer occupancy
seeds, public charging stations, and depot candidates. The downstream generator
still supports a deterministic county-shaped scaffold for development, but
publication runs should pass `--require-real-sources`.

To prepare a new North American county/city, copy
`configs/template_new_city.yaml`, set the county FIPS and display metadata, run
`prepare_public_geodata.py`, and then run the benchmark suite builder. If the
county is very large or internally heterogeneous, create a slice config and run
`prepare_geospatial_subterritories.py` as an optional step before generating
instances.
