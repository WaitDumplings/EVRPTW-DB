# EVRPTW City Logistics Environment Generator

This directory builds the static, city-level geospatial substrate used by
EVRPTW-DB. We call that Stage-1 artifact a **City Logistics Environment
(CLE)**. A CLE is not an EVRPTW instance: it contains the reusable physical
environment from which many operating-day instances can later be sampled.

The implementation is split into a country-independent core and a documented
U.S. reference adapter. The adapter demonstrates one reproducible realization
for ten U.S. cities; another country can provide different source adapters
while preserving the canonical CLE schema.

## Stage boundary

| Stage 1: CLE (this package) | Stage 2: instance generation |
| --- | --- |
| Frozen service boundary and routing envelope | Active customer subset |
| Directed OSM topology and real road geometry | Package count, volume, and demand |
| Latent residential service locations and types | One time window per active location |
| Candidate depots and public charging sites | Service time and vehicle/fleet policy |
| Legal and reference running speed per directed edge | Weekday/weekend static speed realization |
| Source hashes, QA flags, and release gates | Distance/time/path matrices and feasibility checks |

Package count is intentionally absent from a CLE. `Cus100` means 100 distinct
active physical service locations; one apartment location may later receive
many packages and therefore have larger demand and service time.

## U.S. reference cohort

The frozen profile `configs/us_top10_cle_v1.json` contains city-proper service
areas for New York City, Los Angeles, Chicago, Houston, Phoenix, Philadelphia,
San Antonio, San Diego, Dallas, and Fort Worth. It does not silently replace a
city with its metropolitan area.

The included boundary assets are land-only service masks derived from 2025
U.S. Census TIGER/Line Place and Area Hydrography data. Water is excluded from
service-location placement. Real OSM roads outside the service boundary may be
retained only as `transit_only` routing edges when needed for connectivity.

## Install

The recommended environment includes `osmium-tool`, which is required for
reproducible extraction from frozen PBF files.

```bash
cd EVRPTW_Dataset_Generator
conda env create -f environment.yml
conda activate evrptw-cle
```

Alternatively, install Python dependencies with `pip install -e .`; install
`osmium` separately through the operating system.

## Required source layout

Large source and generated files are not committed. Prepare this layout:

```text
data/
  sources/
    geofabrik/
      new-york-latest.osm.pbf
      socal-latest.osm.pbf
      illinois-latest.osm.pbf
      texas-latest.osm.pbf
      arizona-latest.osm.pbf
      pennsylvania-latest.osm.pbf
    microsoft-us-building-footprints/
      NewYork.geojson
      California.geojson
      Illinois.geojson
      Texas.geojson
      Arizona.geojson
      Pennsylvania.geojson
    afdc/
      afdc_us_public_available_electric.csv
      afdc_census_address_anchors.csv
      afdc_us_public_available_electric_resolved_v2.csv # preferred, generated below
    osm/
      osm_charging_pois_top10.csv
    hpms-edge-matches/                                 # optional
      san-diego.parquet
      ...
```

The six Microsoft state files are sufficient for this ten-city cohort. The
building registry stores their expected file sizes and SHA-256 hashes. The
files provide footprint geometry, not house/apartment labels; NSI supplies the
residential occupancy and unit evidence.

### 1. Freeze OSM PBFs

```bash
python scripts/fetch_pbf_sources.py \
  --preset configs/top10_us_cities_population_v1.json \
  --manifest data/sources/geofabrik/source_manifest.json
```

The preset contains the Geofabrik URLs and reuses four state/regional files
across cities. Every build records file hashes and the PBF replication
timestamp when available.

### 2. Place Microsoft US Building Footprints

Download the six state GeoJSON files from Microsoft USBuildingFootprints and
place them under `data/sources/microsoft-us-building-footprints/` with the exact
names shown above. Do not edit the files in place. If Microsoft publishes a new
snapshot whose hash differs from `configs/top10_building_extraction_v1.json`,
validate the schema and create a new registry version rather than silently
changing the old one.

### 3. Freeze and resolve AFDC charging sites

Get a free NREL API key, then download the U.S. public, available electric-site
snapshot:

```bash
export NREL_API_KEY=YOUR_KEY
python scripts/download_afdc_snapshot.py
```

If a complete AFDC CSV export has already been frozen, normalize it with the
same declared filters instead of downloading it again:

```bash
python scripts/filter_afdc_snapshot.py \
  --input /path/to/alt_fuel_stations.csv \
  --output data/sources/afdc/afdc_us_public_available_electric.csv
```

Extract exact-address OSM charging POIs from the same PBF snapshots and obtain
address anchors from the public Census batch geocoder:

```bash
python scripts/extract_osm_charging_pois.py
python scripts/geocode_afdc_addresses_census.py \
  --afdc data/sources/afdc/afdc_us_public_available_electric.csv
```

Resolve coordinates without discarding provenance:

```bash
python scripts/resolve_afdc_coordinates.py \
  --afdc data/sources/afdc/afdc_us_public_available_electric.csv \
  --census-results data/sources/afdc/afdc_census_address_anchors.csv \
  --osm-pois data/sources/osm/osm_charging_pois_top10.csv \
  --output data/sources/afdc/afdc_us_public_available_electric_resolved_v2.csv
```

Resolution precedence is: reviewed manual override, exact normalized OSM
charging-POI address match, then raw AFDC coordinate. Census output is retained
as an address-access anchor and QA comparison; it is not mislabeled as the
exact charger location. An optional manual override CSV must contain
`afdc_id,resolved_longitude,resolved_latitude,review_note`.

The U.S. preflight requires both Census and OSM evidence in the resolution
manifest. A file is not accepted merely because its name contains `resolved`.
Every station receives one of four explicit tiers: reviewed exact geometry,
OSM exact-address geometry, Census-address-corroborated but exact geometry
unverified, or uncorroborated source coordinate. The last tier is retained for
audit but excluded from the default benchmark candidate pool.

### 4. Optional HPMS edge evidence

The core does not hide raw HPMS-to-OSM conflation. A U.S. speed adapter may
provide one normalized file per city with:

```text
edge_id,F_SYSTEM,SPEED_LIMIT,match_confidence
```

`edge_u,edge_v,edge_key` may replace `edge_id`. `SPEED_LIMIT` is in mph, and
only `high`, `verified`, `accepted`, or numeric confidence at least 0.8 can
affect the canonical layer. When this optional table is absent, the pipeline
uses OSM `highway=*` for H/M/U classification and OSM `maxspeed` plus transparent
within-city hierarchical imputation for legal speeds.

### 5. NSI

No manual NSI download is required. The build queries the public USACE NSI API
in deterministic 5 km tiles, caches the raw compressed responses, and hashes
them. Re-running from the frozen cache does not call the API again.

## Preflight

Run the read-only source and configuration gate before a long build:

```bash
python scripts/preflight_cle_sources.py \
  --profile configs/us_top10_cle_v1.json \
  --output work/us-top10-v1/qa/preflight.json
```

Missing required sources, boundary/hash mismatches, incompatible AFDC columns,
or a missing `osmium` executable fail the gate. Missing HPMS matches generate a
warning because the documented OSM fallback remains valid.

## Build all ten CLEs

After preflight passes:

```bash
bash scripts/build_top10_cle.sh
```

The runner is resumable and supports explicit stages:

```bash
bash scripts/build_top10_cle.sh --stages roads buildings depots cles package index
bash scripts/build_top10_cle.sh --cities san-diego --continue-on-error
```

After a directed-connectivity or facility-evidence policy migration, refresh
the frozen access ledgers without repeating NSI download or footprint matching:

```bash
python scripts/build_top10_cle.py \
  --stages cles package index \
  --refresh-protected-connectivity \
  --replace-release-package
```

The default profile separates rebuildable work artifacts from portable dataset
artifacts:

```text
EVRPTW_Dataset_Generator/
  data/sources/                 # frozen source inputs
  work/us-top10-v1/             # roads, caches, intermediate layers, debug CLEs
EVRPTW_Dataset/
  CLE_v1/us_top10/
    cle_index.{json,csv}
    cities/<city>/              # self-contained portable CLE packages
```

The `cles` stage assembles and technically verifies one debug CLE per city under
`work/us-top10-v1/cles/`. The `package` stage copies the verified operational
GraphML and road manifest into the CLE, removes machine-local runtime references,
runs strict portability verification in a staging directory, and atomically
promotes the result to `EVRPTW_Dataset/CLE_v1/us_top10/cities/`.

`work/us-top10-v1/top10_cle_index.{json,csv}` reports technical work-artifact
status. `EVRPTW_Dataset/CLE_v1/us_top10/cle_index.{json,csv}` reports strict
portable-package status. A portable package may still have open scientific
release gates; `portable_package_verified` and `release_eligible` are deliberately
separate fields.

An existing verified package is reused. The packager refuses to overwrite an
existing invalid or stale package; use a new versioned release root after inputs
or policies change.

Generate the numeric tables used by the dataset/benchmark appendix directly
from the portable cohort manifests:

```bash
python scripts/build_cle_appendix_tables.py --replace
```

The generated CSV, JSON, and Markdown files are stored under
`EVRPTW_Dataset/CLE_v1/us_top10/appendix_tables/`. They distinguish source
locations, evidence-qualified candidate pools, distance QA tails, and SCC
quarantines.

For a small, version-controlled paper snapshot, write the same tables to the
documentation tree after the ten portable packages have been verified:

```bash
python scripts/build_cle_appendix_tables.py \
  --output-dir docs/generated/us_top10_cle_v1 \
  --replace
```

The generated manifest records every source city-manifest hash, so a later
rebuild can be distinguished from the exact cohort used for a paper table.

One package can be checked independently of all source and work directories:

```bash
python scripts/package_cle.py \
  --destination-cle ../EVRPTW_Dataset/CLE_v1/us_top10/cities/san-diego \
  --verify-only
```

## Build a road graph for another city

The core road module can resolve an arbitrary qualified city name through OSM:

```bash
evrptw-cle build \
  --city "Boston, Massachusetts, USA" \
  --slug boston \
  --output-root data/custom-cities
```

For a reproducible release, provide a frozen administrative boundary,
land/service mask, and regional PBF with `--boundary-file`,
`--query-mask-file`, and `--pbf-file`. A complete CLE for a city outside the
U.S. ten-city registry additionally needs compatible building, residential,
facility, and speed adapters described in `docs/PORTABILITY.md`.

## Frozen policies that are easy to misread

- Connectivity is measured against the city-proper road graph. If the largest
  weak component covers less than 99% of city nodes or less than 99.5% of city
  physical-road length, the routing envelope is tried at 1, 2, 5, 10, then 20
  km. Only real OSM roads are added. Outside-city roads are `transit_only`.
- Customer-to-road 200 m and facility-to-road 250 m values are QA references,
  not deletion thresholds. All anchorable rows are retained and the distance
  tail is flagged for review.
- Depot Tier A is explicit dispatch/carrier-facility evidence. Tier B is a
  warehouse/logistics proxy. Tier C is excluded. Building area is retained as
  a continuous feature; 1,000 m2 is a sensitivity flag, not a hard truth rule.
- NSI ordinary residential records (`RES1`, `RES2`, `RES3`) are grouped by a
  shared NSI structure identifier and classified from residential-unit
  evidence as `house`, `manufactured_home`, `small_apt`, `medium_apt`, or
  `large_apt`. Microsoft polygons supply geometry. Nearby polygons are not
  automatically merged into an apartment complex without independent site
  evidence.
- Customer, depot, and charger access is projected to the nearest eligible
  physical road. The new connector is bidirectional and symmetric, while the
  original OSM one-way topology is preserved.
- Stage 1 stores legal speed and reference running speed, not time-dependent
  traffic. Stage 2 will create one static directed weekday or weekend speed
  realization per instance.

Detailed contracts are in:

- `docs/PIPELINE.md`
- `docs/DATA_SOURCES.md`
- `docs/OUTPUT_SCHEMA.md`
- `docs/PORTABILITY.md`
- `docs/LEGACY_STAGE2.md`

## Tests

```bash
pytest
python -m compileall -q src scripts tests
```

The unit suite tests connectivity, boundary and source gates, building
extraction, NSI classification, geometry matching, directed access anchors,
facility policies, AFDC coordinate resolution, speed semantics, assembly, and
visualization.

## Legacy Stage-2 compatibility

`evrptw_hierarchy/`, `prepare_region_pool.py`,
`prepare_ac_benchmark_suite.py`, and `instance_generate.py` are retained only
because existing TERRAN experiments import them. They are not the new CLE
pipeline and should not be used to describe the future real-road instance
generator. See `docs/LEGACY_STAGE2.md`.
