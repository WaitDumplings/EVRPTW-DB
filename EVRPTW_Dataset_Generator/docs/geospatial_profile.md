# Geo-AC-v1 Geospatial Profile

Geo-AC-v1 is a real-geography semi-synthetic EVRPTW profile. County boundaries
are used as data containers, while actual operating-day instances use an
anonymous depot candidate and a road-distance depot catchment.

## Default Territories

The initial acquisition config (`geo_ac_v1_us10.yaml`) uses 10 county
containers:

| territory_id | county container |
|---|---|
| `los_angeles_ca_los_angeles_county` | Los Angeles County, CA |
| `seattle_wa_king_county` | King County, WA |
| `austin_tx_travis_county` | Travis County, TX |
| `chicago_il_cook_county` | Cook County, IL |
| `boston_ma_middlesex_county` | Middlesex County, MA |
| `new_york_ny_queens_county` | Queens County, NY |
| `dallas_tx_dallas_county` | Dallas County, TX |
| `houston_tx_harris_county` | Harris County, TX |
| `atlanta_ga_fulton_county` | Fulton County, GA |
| `phoenix_az_maricopa_county` | Maricopa County, AZ |

The official release config (`geo_ac_v1_na_us20.with_sources.yaml`) keeps
compact counties unchanged and splits large county containers into 20
service-territory units. Large-county slicing is optional for a new city, but it
is recommended when the county is too large for one realistic depot-centered
service area.

## Standard Source Files

Use `prepare_public_geodata.py` to create one normalized source-data directory
per territory and a generated `geo_ac_v1_us10.with_sources.yaml` config.

Required for true geospatial board construction:

- `road_nodes_csv`: columns `node_id,x_km,y_km`, or `node_id,lon,lat`.
- `road_edges_csv`: columns `u,v`, optionally `length_km`, referencing road node
  IDs.
- `customer_seed_csv`: columns `x_km,y_km` or `lon,lat`, plus optional
  `occupancy` and `community_id`.

Optional but supported:

- `charging_station_csv`: public EV charging candidates.
- `depot_candidate_csv`: warehouse/logistics/industrial depot candidates.
- `latent_customer_csv`: fixed road-frontage latent customer positions.

If optional charging or depot files are missing, the builder uses documented
fallback candidates from the road graph. If the required files are missing, it
uses the existing synthetic service-territory generator as a county-shaped
scaffold and annotates the manifest with `source_mode`.

Publication runs should pass `--require-real-sources` to
`prepare_geospatial_benchmark_suite.py`. That mode requires all five normalized
CSV files to exist and contain rows, and rejects the synthetic scaffold path.

## Public Geodata ETL

The ETL uses public sources only:

- Census TIGER/Line county and block-group geometries.
- ACS 5-year occupied housing units (`B25002_002E`) for occupancy weights;
  `CENSUS_API_KEY` is required.
- OpenStreetMap/OSMnx drive roads for the routable graph; Overture
  transportation is attempted as a secondary QA export when `--road-source both`.
- AFDC/NREL public EV charging stations; `NREL_API_KEY` is required.
- OSM warehouse/logistics/industrial tags for depot candidates, with marked
  center-region fallback candidates if public depot-like features are sparse.

API keys are supplied through environment variables:

- `CENSUS_API_KEY` from <https://api.census.gov/data/key_signup.html>.
- `NREL_API_KEY` from <https://developer.nrel.gov/signup/>.

Do not commit keys. OSMnx/OpenStreetMap does not require a key, but public
Overpass endpoints should be used respectfully and cached outputs should be
reused.

```bash
export CENSUS_API_KEY=...
export NREL_API_KEY=...
conda run -n maojie python EVRPTW_Dataset_Generator/prepare_public_geodata.py \
  --city-config EVRPTW_Dataset_Generator/configs/geo_ac_v1_us10.yaml \
  --output-root EVRPTW_Dataset/Geo_AC_v1/source_data \
  --config-out EVRPTW_Dataset_Generator/configs/geo_ac_v1_us10.with_sources.yaml
```

For the official NA-US-20 source data, run the subterritory slicer:

```bash
conda run -n maojie python -m EVRPTW_Dataset_Generator.prepare_geospatial_subterritories \
  --source-config EVRPTW_Dataset_Generator/configs/geo_ac_v1_us10.with_sources.yaml \
  --slice-config EVRPTW_Dataset_Generator/configs/geo_ac_v1_na_us20_slices.yaml \
  --output-root EVRPTW_Dataset/Geo_AC_v1/source_data_na_us20 \
  --config-out EVRPTW_Dataset_Generator/configs/geo_ac_v1_na_us20.with_sources.yaml
```

## Instance Semantics

For geospatial boards, each active day:

1. samples one depot candidate;
2. builds a customer pool from its road catchment;
3. activates customers through the existing active-community sampler;
4. activates public charging candidates through the existing graph
   facility-location objective;
5. samples Amazon-calibrated demand, service time, and time windows;
6. runs the existing greedy feasibility audit.

This should be described as a real-geography semi-synthetic benchmark, not as a
reconstruction of real Amazon routes or real depot operations.
