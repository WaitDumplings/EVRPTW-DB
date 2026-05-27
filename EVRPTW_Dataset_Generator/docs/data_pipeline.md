# Geo-AC-v1 Public Data Pipeline

Geo-AC-v1 is a real-geography, semi-synthetic EVRPTW benchmark. The spatial
layer is produced from public geospatial data; Amazon-derived calibration is
used only for daily operating attributes such as demand, service time, time
windows, and activation rates. It does not determine customer locations.

## What Is Published

The recommended public release has two parts:

```text
Geo_AC_v1_NA_US20/
  source_data/
    <territory_id>/
      normalized/
        road_nodes.csv
        road_edges.csv
        customer_seed.csv
        latent_customer.csv
        charging_station.csv
        depot_candidate.csv
      qa/
        qa_summary.json
        qa_report.md
        preview_layers.geojson
  eval_standard_20/
    service_territories/
    eval/
      <territory_id>/
        Cus_5/
        Cus_15/
        Cus_50/
        Cus_100/
  qa_maps/
  qa_maps_latent/
  metadata/
```

`source_data` is the reusable geospatial dataset. `eval_standard_20` is the
fixed evaluation split with 20 operating-day instances per territory and scale.

## Public Data Sources

- Census TIGER/Line county and block-group geometry.
- ACS 5-year occupied housing units (`B25002_002E`) for community occupancy
  weights.
- OpenStreetMap/OSMnx drive roads for routable road graphs.
- NREL/AFDC public EV charging station records.
- OSM/Overture warehouse, logistics, freight, and industrial features for depot
  candidates, with marked fallback depots if open depot-like features are too
  sparse.

Customer seeds are Census block-group representative points with ACS occupied
housing counts. Latent customers are generated from those seeds by
occupancy-weighted road-frontage placement, then snapped to the road graph with
short connector distances. A service territory is the stable set of road nodes,
latent customers, charging stations, and depot candidates. Each instance is one
operating day that activates a subset of customers and charging stations.

## API Keys

Two public API keys are needed for the full ETL.

`CENSUS_API_KEY`:

- Sign up at <https://api.census.gov/data/key_signup.html>.
- The key is used for ACS occupied housing units (`B25002_002E`).
- A Census key is free. Do not commit it to git.

`NREL_API_KEY`:

- Sign up through NREL/developer API access, commonly via
  <https://developer.nrel.gov/signup/>.
- The key is used for the AFDC alternative fuel stations API.
- The key is free for normal research/API usage. Do not commit it to git.

Set keys as environment variables:

```bash
export CENSUS_API_KEY="your-census-key"
export NREL_API_KEY="your-nrel-key"
```

For local development, using a `.env` file is fine, but `.env` files must stay
untracked. The repository `.gitignore` excludes `.env`.

OSMnx uses public OpenStreetMap/Overpass infrastructure and does not require an
API key. Be mindful of rate limits; cached source files are reused unless
`--force` is passed.

## Install

The local development examples use the `maojie` conda environment. Replace it
with your own environment name if needed.

```bash
conda activate maojie
python -m pip install -r EVRPTW_Dataset_Generator/requirements.txt
```

## Prepare the Official NA-US-20 Dataset

The official release is built in two steps. First, create county-level public
source CSVs:

```bash
conda run -n maojie python -m EVRPTW_Dataset_Generator.prepare_public_geodata \
  --city-config EVRPTW_Dataset_Generator/configs/geo_ac_v1_us10.yaml \
  --output-root EVRPTW_Dataset/Geo_AC_v1/source_data \
  --config-out EVRPTW_Dataset_Generator/configs/geo_ac_v1_us10.with_sources.yaml
```

Second, split large county containers into service territories, clean sparse
remote community/depot outliers, generate road-frontage latent customers, and
write the NA-US-20 source manifest:

```bash
conda run -n maojie python -m EVRPTW_Dataset_Generator.prepare_geospatial_subterritories \
  --source-config EVRPTW_Dataset_Generator/configs/geo_ac_v1_us10.with_sources.yaml \
  --slice-config EVRPTW_Dataset_Generator/configs/geo_ac_v1_na_us20_slices.yaml \
  --output-root EVRPTW_Dataset/Geo_AC_v1/source_data_na_us20 \
  --config-out EVRPTW_Dataset_Generator/configs/geo_ac_v1_na_us20.with_sources.yaml
```

Then build fixed evaluation instances:

```bash
conda run -n maojie python -m EVRPTW_Dataset_Generator.prepare_geospatial_benchmark_suite \
  --city-config EVRPTW_Dataset_Generator/configs/geo_ac_v1_na_us20.with_sources.yaml \
  --output-root EVRPTW_Dataset/Geo_AC_v1/eval_standard_20 \
  --require-real-sources \
  --instances-per-scale 20
```

This creates `20 territories x 4 scales x 20 instances = 1600` evaluation
instances.

Finally write release metadata and README files for the ignored dataset
directories:

```bash
conda run -n maojie python -m EVRPTW_Dataset_Generator.prepare_geo_ac_release_metadata \
  --source-root EVRPTW_Dataset/Geo_AC_v1/source_data_na_us20 \
  --eval-root EVRPTW_Dataset/Geo_AC_v1/eval_standard_20
```

## Prepare an Additional North American City

1. Copy `configs/template_new_city.yaml`.
2. Change `territory_id`, `display_name`, `county_name`, `state`, and
   `county_fips`.
3. Optionally change `bbox_lonlat`, `area_size_km`, depot catchment radii, and
   latent customer pool size.
4. Run the public geodata ETL:

```bash
conda run -n maojie python -m EVRPTW_Dataset_Generator.prepare_public_geodata \
  --city-config EVRPTW_Dataset_Generator/configs/my_city.yaml \
  --output-root EVRPTW_Dataset/Geo_AC_v1/source_data_my_city \
  --config-out EVRPTW_Dataset_Generator/configs/my_city.with_sources.yaml
```

5. Run a source-only smoke build:

```bash
conda run -n maojie python -m EVRPTW_Dataset_Generator.prepare_geospatial_benchmark_suite \
  --city-config EVRPTW_Dataset_Generator/configs/my_city.with_sources.yaml \
  --output-root EVRPTW_Dataset/Geo_AC_v1/my_city_smoke \
  --require-real-sources \
  --skip-instances
```

6. Generate a small instance smoke test:

```bash
conda run -n maojie python -m EVRPTW_Dataset_Generator.prepare_geospatial_benchmark_suite \
  --city-config EVRPTW_Dataset_Generator/configs/my_city.with_sources.yaml \
  --output-root EVRPTW_Dataset/Geo_AC_v1/my_city_eval_smoke \
  --require-real-sources \
  --scales 5,15 \
  --instances-per-scale 2
```

## Optional Territory Slicing

Subterritory slicing is optional but recommended when a county container is too
large, disconnected, or mixes very different delivery regimes. Examples include
Los Angeles County, Harris County, Maricopa County, Cook County, and King
County.

Use slicing when QA maps show:

- one depot catchment would unrealistically cover the whole county;
- sparse remote communities are isolated from the urban/suburban service core;
- depot candidates cluster in only one part of the county;
- a county naturally contains multiple logistics/service markets.

The slicer uses named anchor points and county-clipped Voronoi cells. The
configuration lives in `configs/geo_ac_v1_na_us20_slices.yaml`. For a new large
county, create a similar slice config with child `territory_id`,
`display_name`, `anchor_lonlat`, and `depot_candidate_count`, run
`prepare_geospatial_subterritories.py`, inspect QA maps, and adjust anchors or
manual depot exclusion boxes if needed.

For a compact county or city-county such as San Francisco, the county-level
container is usually sufficient, so slicing can be skipped.

## QA and Visualization

Create maps for all latent customers, charging stations, depots, and roads:

```bash
conda run -n maojie python -m EVRPTW_Dataset_Generator.plot_latent_customer_qa_maps \
  --source-root EVRPTW_Dataset/Geo_AC_v1/source_data_na_us20 \
  --output-dir EVRPTW_Dataset/Geo_AC_v1/source_data_na_us20/qa_maps_latent
```

Publication QA should check road connectedness, snap distances, customer
spacing/connector distances, depot-catchment reachability, charger coverage,
finite instance distance matrices, and a greedy feasibility audit.
