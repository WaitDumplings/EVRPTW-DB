# EVRPTW CLE and Instance Generator

This directory contains two deliberately separate pipelines:

1. **Stage 1** builds the static city-level **City Logistics Environment
   (CLE)**.
2. **Stage 2** consumes only a portable CLE package and creates deterministic
   operating-day matrix families plus scale views.

A CLE is not an EVRPTW instance. It contains the reusable physical environment;
active orders and daily road conditions exist only in Stage 2.

The implementation is split into a country-independent core and a documented
U.S. reference adapter. The adapter demonstrates one reproducible realization
for ten U.S. cities; another country can provide different source adapters
while preserving the canonical CLE schema.

## Stage boundary

| Stage 1: CLE | Stage 2: instance generation |
| --- | --- |
| Frozen service boundary and routing envelope | Active customer subset |
| Directed OSM topology and real road geometry | Package count, volume, and demand |
| Latent residential service locations and types | One time window per active location |
| Candidate depots and public charging sites | Service time and vehicle/fleet policy |
| Legal and reference running speed per directed edge | Weekday/weekend static speed realization |
| Source manifests, QA flags, and release gates | Distance/time/energy matrices and feasibility checks |

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
  traffic. Stage 2 creates one static directed weekday or weekend speed
  realization per instance.

Detailed Stage-1 contracts are in:

- `docs/PIPELINE.md`
- `docs/DATA_SOURCES.md`
- `docs/OUTPUT_SCHEMA.md`
- `docs/PORTABILITY.md`
- `docs/LEGACY_STAGE2.md`

## Stage 2: CLE to operating-day instances

The new CLE-backed implementation is `src/evrptw_stage2/`. It does not call or
silently fall back to the legacy synthetic `evrptw_hierarchy` generator.

### Current release status

Two independent gates are enforced:

- `official` requires every CLE manifest to have `release_eligible=true` and
  uses only customer/depot/charger release-eligible rows.
- `non_release_pilot` may use existing candidate/default eligibility rows, but
  every plan, family, view, report, and warning is stamped as non-release.

At the current repository snapshot, the ten portable CLEs pass technical and
portability checks but still have scientific release blockers. The U.S.
operations profile is also `development_calibration`. Therefore the commands
below use `non_release_pilot`; the code refuses to relabel these artifacts as
official data.

### Stage-2 configuration files

- `configs/cle_evrptw_stage2_v1.json` freezes cities, split tracks, scales,
  seeds, the 08:00--24:00 horizon, 5:2 weekday/weekend ratio, volume units, and
  the matrix-family/view contract.
- `configs/us_reference_instance_profile_v1.json` is the replaceable U.S.
  parameter adapter for activation, road-state variation, the Rivian reference
  energy/charging model, packages, service time, and time windows.
- `configs/us_census_block_groups_v1.json` maps the ten training cities plus
  Jacksonville to seven public Census TIGER/Line block-group files.

The protocol config and U.S. profile are separate on purpose. Another country
can preserve the canonical CLE/instance schemas while supplying different
building, charger, speed, community, and operations adapters.

### 1. Download community boundaries

The customer split uses complete Census block group plus road-SCC subgroups.
Download all seven state archives once:

```bash
python scripts/fetch_census_block_groups.py \
  --preset configs/us_census_block_groups_v1.json \
  --output-dir data/sources/census_block_groups_2025
```

The research downloader records URL, vintage, byte count, and ZIP integrity.
It intentionally does not run a full release checksum audit during routine
pilot iteration.

### 2. Preflight a portable CLE

```bash
evrptw-stage2 preflight \
  --config configs/cle_evrptw_stage2_v1.json \
  --cle-root ../EVRPTW_Dataset/CLE_v1/us_top10 \
  --mode non_release_pilot \
  --cities san-diego
```

The reader rejects absolute or escaping manifest paths, unsupported schemas,
unverified portable packages, missing speed fields, and insufficient eligible
customer/depot/charger pools. Requesting `official` additionally enforces all
scientific release gates.

### 3. Freeze the 80/20 complete-community split

```bash
evrptw-stage2 build-customer-split \
  --config configs/cle_evrptw_stage2_v1.json \
  --cle-root ../EVRPTW_Dataset/CLE_v1/us_top10 \
  --mode non_release_pilot \
  --city san-diego \
  --block-groups data/sources/census_block_groups_2025/tl_2025_06_bg.zip \
  --output-dir work/stage2-pilot-v1/customer_splits/san-diego
```

The assignment unit is a complete `Census block group x anchor SCC` community.
No community can cross train and held-out pools. The deterministic allocator
first constrains held-out service-location count to the requested 20%, then
chooses among close candidates using house/apartment and unit-band balance.
`customer_split_manifest.parquet` is the only location-pool ledger consumed by
Stage 2.

### 4. Build the family/view plan

```bash
evrptw-stage2 plan \
  --config configs/cle_evrptw_stage2_v1.json \
  --cle-root ../EVRPTW_Dataset/CLE_v1/us_top10 \
  --mode non_release_pilot \
  --cities san-diego \
  --tracks train validation test1_new_seed test2_heldout_locations unseen_scale_same_cities \
  --pilot-families-per-city 2 \
  --output-root work/stage2-pilot-v1/generation_plan
```

Official planning requires all ten training cities and Jacksonville. Pilot
planning requires an explicit reduced family count, so an accidental pilot can
never inherit official counts.

| Scale | Role | Train views | Validation | Tests |
| --- | --- | ---: | ---: | --- |
| Cus50 / CS10 | compatibility and budgeted-MIP | 100,000 | 500 | Test-1: 500 |
| Cus100 / CS20 | core | 50,000 | 500 | Test-1/2/3: 500 each |
| Cus500 / CS50 | core | 10,000 | 500 | Test-1/2/3: 500 each |
| Cus1000 / CS50 | core parent | 5,000 | 500 | Test-1/2/3: 500 each |
| Cus2000 / CS50 | unseen-scale scalability | 0 | 0 | same-city test: 100 |

Each training scale has exactly five million active-customer exposures. One
Cus1000 parent permutation produces 20 disjoint Cus50 blocks, 10 Cus100 blocks,
2 Cus500 blocks, and 1 Cus1000 view. CS10 and CS20 are prefixes of the selected
CS50 order. Cus50 has its own consumer folders while retaining the same parent
family ownership, so a matrix family never belongs to more than one split.

Test meanings are fixed:

- Test-1: new seeds, same cities, train location pool;
- Test-2: ten cities, complete held-out communities only;
- Test-3: Jacksonville, never used by the ten-city training cohort;
- Cus2000: same-city unseen-scale test only.

### 5. Materialize one matrix family

```bash
evrptw-stage2 materialize-family \
  --config configs/cle_evrptw_stage2_v1.json \
  --profile configs/us_reference_instance_profile_v1.json \
  --cle-root ../EVRPTW_Dataset/CLE_v1/us_top10 \
  --mode non_release_pilot \
  --plan-root work/stage2-pilot-v1/generation_plan \
  --family-id <family_id> \
  --customer-split work/stage2-pilot-v1/customer_splits/san-diego/customer_split_manifest.parquet \
  --output-root work/stage2-pilot-v1/materialized
```

Family materialization performs these steps in order:

1. deterministically samples weekday/weekend at 5:2;
2. samples one eligible Tier-A/B depot so depot identity is not fixed;
3. constructs an expandable depot catchment from the split-eligible latent
   location pool;
4. before daily customer activation, greedily orders 50 compatible CS
   candidates against complete-community centroids in that catchment; this
   does not use the day's exact customer IDs, and the first 10/20/50 form a
   nested sequence;
5. activates communities, then locations within them using a
   residential-unit-aware order probability and depot-distance decay;
6. realizes every directed road edge as
   `min(legal speed, reference speed x hierarchical day/corridor/segment/direction factor)`;
7. computes speed-sensitive Rivian reference energy per edge;
8. uses each terminal's directed edge projection offset plus its bidirectional
   connector, without converting OSM one-way edges to two-way;
9. evaluates geometry-only straight/right/left/U-turn penalties and excludes
   signal timing;
10. builds six parent matrices: distance-path distance/time/energy and
   running-time-path time/distance/energy;
11. samples packages, volume demand, service time, and one customer time window;
12. applies an unlimited-fleet single-customer constructive feasibility gate
    with zero or more optional full-charge CS visits before and after service;
    and
13. writes lower-scale index views without runtime masks or copied matrices.

The current U.S. development profile starts the depot catchment at 40 km and
expands it in 10 km increments only when the eligible pool is too small. This
is a configurable pilot parameter, not an asserted industry-standard radius;
official calibration and sensitivity reporting remain part of the profile
release gate.

The running-time path currently minimizes directed edge running time and then
evaluates turn penalties on that selected path. The manifest states
`turn_penalty_in_running_time_path_optimization=false`; it must not be described
as an exact turn-expanded shortest path until that optional model is added.

AFDC station power is used when reported, otherwise the city/mode median is
used. If an entire city/mode has no reported power, official generation fails.
The current San Diego pilot has no reported power in its frozen AFDC table, so
pilot mode visibly falls back to the Rivian AC 11 kW or DC 100 kW cap.

Each view stores a compact feasibility certificate per customer: the selected
route's service-arrival time, return duration, charging-visit count, inbound
full-battery terminal, first post-customer charger, and customer-transition
energy margin. It is a sufficient unlimited-fleet certificate, not an optimizer
hint or an action mask.

The gate does not run Greedy, ALNS, or another optimization solver. For each
view it first builds a directed graph over the depot and active charging
stations, where an arc into a CS includes the time needed to restore the energy
used since the preceding full-battery state. Forward Dijkstra caches the
shortest depot-to-every-full-state durations; reverse Dijkstra caches the
shortest every-full-CS-to-depot durations and automatically permits multiple
CS hops. For customer `c`, the gate enumerates the last inbound full state `p`
and the first outbound CS `q`. The shared battery segment is accepted only if
`energy(p,c) + energy(c,q) <= battery_capacity`; charging at `q` restores that
entire accumulated amount before the cached full-state return is used. A
direct customer-to-depot case is evaluated separately. Waiting, service time,
the customer time window, return before the operating-horizon end, and the
one-customer volume-capacity condition are then checked. Under the frozen
unlimited-fleet and infinite-port contract, one certified route per customer
is a constructive feasible solution for the complete instance.

### 6. Verify and load a view

```bash
evrptw-stage2 verify-family \
  --family-dir work/stage2-pilot-v1/materialized/families/<family_id>
```

Python consumers use:

```python
from evrptw_stage2.artifacts import load_materialized_view

instance = load_materialized_view(
    "work/stage2-pilot-v1/materialized/families/<family_id>",
    "<view_id>",
)
```

The loader slices parent matrices by `terminal_parent_indices.npy` and returns
depot/customer/CS coordinates, package counts, volume demand, service time,
time windows, CS power, the compact constructive feasibility certificate, and
all six matrices. `runtime_mask` is always `None`; environments compute masks
from state and stored time/energy data.

### Batch runner

The resumable runner joins all preceding steps. This command reproduces a
small San Diego vertical slice and reuses already verified families:

```bash
python scripts/build_stage2_instances.py \
  --config configs/cle_evrptw_stage2_v1.json \
  --profile configs/us_reference_instance_profile_v1.json \
  --cle-root ../EVRPTW_Dataset/CLE_v1/us_top10 \
  --block-group-preset configs/us_census_block_groups_v1.json \
  --block-group-source-dir data/sources/census_block_groups_2025 \
  --output-root work/stage2-pilot-v1 \
  --mode non_release_pilot \
  --cities san-diego \
  --tracks train validation test1_new_seed test2_heldout_locations unseen_scale_same_cities \
  --pilot-families-per-city 2 \
  --max-families 1
```

The run report records materialization and verification wall time, matrix
bytes, terminal-pair throughput, and process peak RSS. In the current
non-release San Diego pilot, a newly materialized 1,051-terminal six-matrix
family took 39.46 seconds, occupied 26,511,192 matrix bytes, and the runner's
process peak RSS was 2,314,223,616 bytes. These are local pilot measurements,
not a cross-platform performance guarantee.

A failed family attempt never leaves a partial family directory. The runner
writes `rejections/<family_id>.json` with the reason, attempt number/seed, and
deterministic next-attempt seed. By default one new attempt is made per run;
`--max-attempts-per-family` may explicitly allow more. An accepted replacement
keeps the planned family/view IDs and records both the base seed and accepted
attempt seed in its manifests.

For an official run, omit pilot counts and city/track reductions and use
`--mode official`. That command remains blocked until all eleven CLE inputs and
the operations profile are release eligible.

### Stage-2 artifact layout and storage

```text
CLE_EVRPTW_v1/
  customer_splits/<city>/
    customer_split_manifest.parquet
    community_manifest.parquet
    customer_split_report.json
  generation_plan/
    split_registry.json
    core/.../{family_index,view_index}.parquet
    compatibility_cus50/.../view_index.parquet
    scalability_cus2000/.../{family_index,view_index}.parquet
  materialized/families/<family_id>/
    family_manifest.json
    terminal_index.parquet
    matrices/*.npy
    views/<view_id>/
      view_manifest.json
      terminal_parent_indices.npy
      customer_attributes.npz
      charging_power_kw.npy
```

The official plan contains 7,100 parent families and 172,100 logical views.
Six uncompressed float32 parent matrices are projected at 195,668,810,400 bytes
(182.23 GiB); the corresponding three-matrix design is 97,834,405,200 bytes
(91.12 GiB). `split_registry.json` records both numbers. Choosing whether all
six matrices belong in the published training release is therefore an explicit
benchmark-design decision, not a hidden implementation detail.

## Tests

```bash
pytest
python -m compileall -q src scripts tests
```

The unit suite tests connectivity, boundary and source gates, building
extraction, NSI classification, geometry matching, directed access anchors,
facility policies, AFDC coordinate resolution, speed semantics, assembly,
visualization, Stage-2 release gates, complete-community splits, family/view
counts, edge-projection routing, view attributes, and feasibility.

## Legacy Stage-2 compatibility

`evrptw_hierarchy/`, `prepare_region_pool.py`,
`prepare_ac_benchmark_suite.py`, and `instance_generate.py` are retained only
because existing TERRAN experiments import them. They are not the new CLE
pipeline and must not be used to describe the CLE-backed real-road instance
generator. See `docs/LEGACY_STAGE2.md`.
