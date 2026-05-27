# EVRPTW-DB

EVRPTW-DB is a dataset-and-benchmark repository for Electric Vehicle Routing Problems with Time Windows (EVRP-TW).

The project is organized around two deliverables:

- **EVRPTW-D**: real-data-calibrated generated EVRP-TW datasets.
- **EVRPTW-B**: exact, metaheuristic, and reinforcement-learning solvers evaluated on the same instance schema.

The first released dataset profile is **AC-v1**: Amazon-Calibrated v1. This keeps the dataset family name stable (`EVRPTW-D`) while allowing later calibrated profiles such as `AC-v2` or non-Amazon profiles.

The geospatial profile is **Geo-AC-v1**: a real-geography,
semi-synthetic North American benchmark that uses public road networks,
Census/ACS occupancy, public charging infrastructure, depot/industrial
candidates, road-frontage latent customers, and the same Amazon-calibrated
operating-day sampler. Amazon calibration affects daily demand, service time,
time windows, and activation behavior; it does not determine customer
locations.

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

Geo-AC-v1:

```bash
export CENSUS_API_KEY=...
export NREL_API_KEY=...
conda run -n maojie python EVRPTW_Dataset_Generator/prepare_public_geodata.py \
  --city-config EVRPTW_Dataset_Generator/configs/geo_ac_v1_us10.yaml \
  --output-root EVRPTW_Dataset/Geo_AC_v1/source_data \
  --config-out EVRPTW_Dataset_Generator/configs/geo_ac_v1_us10.with_sources.yaml

conda run -n maojie python -m EVRPTW_Dataset_Generator.prepare_geospatial_subterritories \
  --source-config EVRPTW_Dataset_Generator/configs/geo_ac_v1_us10.with_sources.yaml \
  --slice-config EVRPTW_Dataset_Generator/configs/geo_ac_v1_na_us20_slices.yaml \
  --output-root EVRPTW_Dataset/Geo_AC_v1/source_data_na_us20 \
  --config-out EVRPTW_Dataset_Generator/configs/geo_ac_v1_na_us20.with_sources.yaml

conda run -n maojie python -m EVRPTW_Dataset_Generator.prepare_geospatial_benchmark_suite \
  --city-config EVRPTW_Dataset_Generator/configs/geo_ac_v1_na_us20.with_sources.yaml \
  --output-root EVRPTW_Dataset/Geo_AC_v1/eval_standard_20 \
  --require-real-sources \
  --instances-per-scale 20

conda run -n maojie python -m EVRPTW_Dataset_Generator.prepare_geo_ac_release_metadata \
  --source-root EVRPTW_Dataset/Geo_AC_v1/source_data_na_us20 \
  --eval-root EVRPTW_Dataset/Geo_AC_v1/eval_standard_20
```

The official Geo-AC-v1 release profile is **NA-US-20**: 20 service territories,
fixed `Cus5/Cus15/Cus50/Cus100` evaluation suites, and 20 operating-day
instances per territory-scale pair. The large generated CSVs, QA maps, and
instance pickle files are ignored by git and should be published through a
dataset hosting service or release artifact. The GitHub repository keeps the
generator, configs, documentation, and small reproducibility assets.

Benchmark solvers must only read exported operating-day instances from `EVRPTW_Dataset`; they must not access inactive territory customers or inactive charging stations.
