# EVRPTW CLE and Instance Generator

This package builds a reusable real-geography **City Logistics Environment
(CLE)** and then samples classical static EVRPTW instances from that
environment.

- **Stage 1** freezes the city boundary, directed road graph, latent service
  locations, depot/charging candidates, and legal/reference road speeds.
- **Stage 2** activates daily customers and generates package volume, service
  time, one time window per location by matching observed Amazon stop
  templates, a weekday/weekend road state, terminal matrices, and feasibility
  certificates.

A CLE is not an instance. In particular, package count, realized demand,
service time, time windows, and active-customer flags do not belong in Stage 1.
`Cus100` means 100 distinct active physical service locations.

## Documentation map

- [Stage-1 pipeline](docs/PIPELINE.md): rigorous CLE construction logic.
- [Stage-2 instance model](docs/STAGE2_INSTANCE_MODEL.md): submodels, formulas,
  parameter sources, assumptions, and literature.
- [Stage-2 中文审阅版](docs/STAGE2_INSTANCE_MODEL_CN.md): 与英文规范对应的
  中文模型、参数和 release-gap 说明。
- [Data sources](docs/DATA_SOURCES.md): provenance and source limitations.
- [Amazon source contract](docs/AMAZON_LAST_MILE_2021.md): official download,
  local/release storage, and CC BY-NC attribution boundary.
- [Output schema](docs/OUTPUT_SCHEMA.md): CLE and instance artifact contracts.
- [Portability](docs/PORTABILITY.md): adapters for other cities/countries.
- [11-city build report](docs/US_11CITY_BUILD_REPORT.md): completed CLE and
  Stage-2 engineering evidence and resource measurements.

The README is intentionally operational. Scientific justification and exact
model semantics are maintained in the linked documents.

## U.S. reference cohort

The reference cohort has ten training city-proper service areas: New York City,
Los Angeles, Chicago, Houston, Phoenix, Philadelphia, San Antonio, San Diego,
Dallas, and Fort Worth. Jacksonville is the eleventh CLE and held-out Test-3
city.

The reference adapter integrates Census, OSM, Microsoft US Building
Footprints, USACE NSI, AFDC, HPMS, EPA MOVES5, Amazon Last Mile Routing
Research Challenge 2021 templates, and official Rivian Commercial Van
specifications. It is one
portable U.S. implementation of the canonical schemas, not a requirement that
another country use the same sources.

## Install

```bash
cd EVRPTW_Dataset_Generator
conda env create -f environment.yml
conda activate evrptw-cle
```

`osmium-tool` is required for reproducible extraction from frozen PBF files.
`pip install -e .` is also supported when system dependencies are already
available.

## Generate any U.S. city by name

The paper cohort and the reusable city adapter are intentionally separate:

- `./generate_cle.sh` rebuilds the frozen 11-city benchmark profile.
- `./generate_us_city_cle.sh --city NAME --state STATE` creates a new
  single-city profile without editing the frozen cohort.

From the repository root:

```bash
conda activate evrptw-cle
export NLR_API_KEY=YOUR_FREE_NLR_DEVELOPER_KEY

./generate_us_city_cle.sh --city "San Antonio" --state TX
```

The command performs these steps in order:

1. Resolve `city + state` to exactly one 2025 Census Place GEOID. If the name
   is ambiguous, stop and report candidates; `--census-place-geoid` is the
   explicit disambiguation interface.
2. Download Census Place, County, and AREAWATER inputs and construct the
   city-proper administrative and land-only service boundaries.
3. Download/reuse the official Microsoft state building archive.
4. Download/reuse an official Geofabrik OSM PBF. The generic default is the
   state extract; `--geofabrik-region california/socal` or `--pbf-url URL`
   may select a smaller official extract that fully contains the city.
5. Query the official FHWA 2018 HPMS public FeatureServer only over a padded
   city bounding box. HPMS remains evidence for functional class and, after
   direction verification, missing posted speed; it is never the routing graph.
6. Download/reuse the national public/available AFDC electric-station snapshot.
   A free `NLR_API_KEY` is required only when that shared file is absent.
   The legacy environment name `NREL_API_KEY` remains accepted for compatibility.
7. Extract OSM charging POIs for the city, batch-geocode the state's AFDC
   addresses with the Census Geocoder, and build a city-specific resolved AFDC
   evidence table. Raw AFDC coordinates are always retained. An address match
   is labeled as an address anchor, not as exact EVSE geometry.
8. Run the normal road, building, depot, NSI/customer, facility, speed,
   connectivity, packaging, and portable-verification stages.

Generated configs and resumable work stay under
`work/us-city-adapter/<city>/`; the portable result is
`../EVRPTW_Dataset/CLE_v1/us_custom/<city>/`. Public state/national inputs stay
under `data/sources/` and are reused by later cities. Useful modes are:

```bash
# Resolve the Census Place and inspect the generated contract only.
./generate_us_city_cle.sh --city "Austin" --state TX --prepare-only

# Prepare configs and all sources but stop before the expensive CLE build.
./generate_us_city_cle.sh --city "Austin" --state TX --sources-only

# Force fresh public-source downloads (normally unnecessary).
./generate_us_city_cle.sh --city "Austin" --state TX --force-downloads
```

Research-mode custom building registries do not pre-scan a multi-gigabyte state
file solely to pin a hash and feature count. Those values are recorded during
the one extraction pass. The final public release workflow may pin and verify
the resulting source snapshot. This optimization changes provenance timing,
not the extracted locations or CLE schema.

## Two production commands

Run both commands from the repository root:

```bash
export NLR_API_KEY=YOUR_FREE_NLR_DEVELOPER_KEY
./generate_cle.sh
./generate_instances.sh
```

The first command runs `scripts/prepare_us11_sources.py` before the CLE build.
It downloads only missing fixed-cohort sources and reuses every existing
nonempty input. The second command uses 12 workers by default and downloads the
three Amazon model-build JSON files automatically when its compact calibration
artifact and raw inputs are both absent.
For the exact server execution order, background commands, success criteria,
and restart policy, follow the root README section
[`Server-agent production runbook`](../README.md#server-agent-production-runbook).

`generate_instances.sh` accepts only the current Stage-1 speed contract:
`evrptw_directed_speed_profiles_v6` with a versioned reference profile ID.
After changing the legal-speed evidence or MOVES profile, rerun
`generate_cle.sh` first and then regenerate all matrix families. The reader
rejects legacy single-column `reference_speed_kph` CLEs so old and new speed
semantics cannot be mixed silently.

The first command generates all eleven CLEs and writes them only below
`EVRPTW_Dataset/CLE_v2/us_11city/`. Stage-2 is currently authorized only through
the C1/C2 gates, Los Angeles smoke, and 140-family non-release pilot. Before
generation it converts the
three Amazon model-build JSON files into a compact, reusable artifact layer.
Both runners are resumable. The full
completed corpus uses about 154.79 GiB for the four parent matrices alone. A
slim export omits that deterministic cache and is about 4.09 GiB before
compression.

Useful environment overrides are:

```bash
WORKERS=8 ./generate_instances.sh
KEEP_CLE_WORK=1 ./generate_cle.sh
PREPARE_CLE_SOURCES=0 ./generate_cle.sh  # only for an already complete frozen bundle
INSTANCE_MODE=non_release_pilot WORKERS=1 \
AMAZON_MODEL_BUILD_INPUTS=/data/almrrc2021-data-training/model_build_inputs \
  ./generate_instances.sh --cities san-diego --tracks train --max-families 1
```

`research` remains accepted only by lower-level exploratory APIs; the production
runner forbids a research-mode full corpus. A full run requires an advisor-signed
promotion to a `release_calibrated` profile, `official` mode, a clean acceptance
commit, a new output root, and `--full-run-approved`.

## Stage 1 quick start

### Automatically prepared source layout

`generate_cle.sh` invokes `scripts/prepare_us11_sources.py` first. The preparer
checks this contract, downloads only missing public inputs, and reuses existing
nonempty files. The resulting layout is:

```text
data/sources/
  geofabrik/
    new-york-latest.osm.pbf
    socal-latest.osm.pbf
    illinois-latest.osm.pbf
    texas-latest.osm.pbf
    arizona-latest.osm.pbf
    pennsylvania-latest.osm.pbf
    florida-latest.osm.pbf
  microsoft-us-building-footprints/
    NewYork.geojson
    California.geojson
    Illinois.geojson
    Texas.geojson
    Arizona.geojson
    Pennsylvania.geojson
    Florida.geojson
  afdc/
    afdc_us_public_available_electric.csv
    afdc_census_address_anchors.csv
    afdc_us_public_available_electric_resolved_us_11city_v1.csv
  osm/
    osm_charging_pois_us_11city.csv
  nsi-us-11city/
    <city>/raw_tiles/*.geojsonseq.gz
  census_block_groups_2025/
    tl_2025_{04,06,12,17,36,42,48}_bg.zip
  hpms/
    new-york.geojson                 # one bounded official city window
    los-angeles.geojson
    chicago.geojson
    houston.geojson
    phoenix.geojson
    philadelphia.geojson
    san-antonio.geojson
    san-diego.geojson
    dallas.geojson
    fort-worth.geojson
    jacksonville.geojson
  moves5/                             # optional reproducibility input
    movesdb20241112.sql
  amazon-last-mile-2021/              # may instead live outside the repo
    model_build_inputs/
      route_data.json
      package_data.json
      travel_times.json
```

When frozen NSI API tile responses are present, the eleven-city build reuses
them without a network request. Otherwise the same code queries and caches NSI
again in deterministic city tiles.

Inspect source readiness without downloading anything:

```bash
python scripts/prepare_us11_sources.py --check-only
```

The integrated Stage-2 Amazon downloader uses the public AWS Open Data bucket with
`--no-sign-request`; no AWS account or API key is required. Raw Amazon JSON is
kept under the ignored `data/sources/` tree and is not committed to Git. See
[the Amazon source contract](docs/AMAZON_LAST_MILE_2021.md) before publishing
the compact artifact or generated instances.

The HPMS files are bounded public FHWA geospatial extracts. Their exact
extension is not fixed; the city-to-source mapping is versioned in
`configs/us_11city_hpms_sources_v1.json`. The integrated preparer produces the
eleven city-window files above, then the CLE builder creates per-city match tables under
`work/us-11city-v1/hpms_edge_matches/`. A single-city matcher can also be run
directly:

```bash
PYTHONPATH=src python scripts/build_hpms_edge_matches.py \
  --city-slug san-diego \
  --hpms data/sources/hpms/california.parquet \
  --graph work/us-11city-v1/cities/san-diego/graph_operational.graphml \
  --boundary boundaries/us-11city-2025/san-diego/land_boundary.geojson \
  --output work/us-11city-v1/hpms_edge_matches/san-diego.parquet
```

Build all eleven CLEs from the repository root:

```bash
./generate_cle.sh
```

The repository includes the compact, frozen
`configs/us_moves5_speed_profile_v1.json` used by CLE generation. It was
derived from EPA database `movesdb20241112`; the raw 373 MiB SQL dump is not a
runtime dependency. To reproduce the profile from the official database:

```bash
PYTHONPATH=src python scripts/derive_moves5_speed_profile.py \
  --moves-sql data/sources/moves5/movesdb20241112.sql \
  --output configs/us_moves5_speed_profile_v1.json
```

MOVES is not a road network. Its default tables provide national distributions
by broad vehicle/road/day/hour strata. OSM/HPMS still supplies every edge,
direction, class, and legal-speed scale; MOVES supplies only two urban
speed-retention priors. See [the Stage-1 speed model](docs/PIPELINE.md#62-legal-free-flow-proxy-and-reference-speed).

The underlying runner is composable for debugging:

```bash
CLE_WORK_ROOT=/tmp/cle-work CLE_RELEASE_ROOT=/tmp/cle-release \
  ./generate_cle.sh --cities san-diego --continue-on-error
```

Work products stay under `work/us-11city-v1/`. Self-contained packages are
written under `../EVRPTW_Dataset/CLE_v2/us_11city/cities/<city>/`.

Check one portable CLE without the work tree:

```bash
python scripts/package_cle.py \
  --destination-cle ../EVRPTW_Dataset/CLE_v2/us_11city/cities/san-diego \
  --verify-only
```

Generate the paper/appendix cohort tables from portable manifests:

```bash
PYTHONPATH=src conda run -n maojie --no-capture-output \
  python scripts/build_cle_appendix_tables.py \
  --cle-root ../EVRPTW_Dataset/CLE_v2/us_11city \
  --replace
```

Important Stage-1 rules:

- road-envelope extension uses real OSM roads only;
- the city node and physical-road coverage gates are 99% and 99.5%;
- the full real-OSM buffer ladder is exhausted before the generic residual
  fallback may exclude still-uncovered weak components with fewer than 100
  nodes from the effective gate denominator; raw coverage and skipped shares
  remain in the manifest;
- 200 m customer and 250 m facility distances are QA flags, not deletion
  thresholds;
- original OSM one-way direction is never relaxed;
- outside-boundary roads are transit-only;
- technical portability and scientific release eligibility are separate.

## Stage 2 quick start

The canonical implementation is `src/evrptw_stage2/`. It consumes only a
portable CLE package and never silently falls back to
`evrptw_hierarchy/`.

The protocol and adapter are separate:

- `configs/cle_evrptw_stage2_v2.json` freezes splits, scales, horizon,
  matrix-family layout, and benchmark policies.
- `configs/us_reference_instance_profile_v2.json` supplies replaceable U.S.
  activation, speed, order, vehicle, charging, and feasibility parameters.

Build the compact Amazon artifact layer once (the production shell performs
this automatically when `manifest.json` is absent):

```bash
PYTHONPATH=src python scripts/build_amazon_stage2_artifacts.py \
  --model-build-inputs /data/almrrc2021-data-training/model_build_inputs \
  --output-dir ../EVRPTW_Dataset/Calibration_v2/amazon_stage2_v3
```

The artifact layer records single-day and same-station multi-day-composite
support, route/decile spatial references, stop-level order templates, and
preprocessing attrition. It does not export Amazon coordinates.

Download Census block groups, preflight one CLE, and freeze its community
split:

```bash
python scripts/fetch_census_block_groups.py \
  --preset configs/us_census_block_groups_v1.json \
  --output-dir data/sources/census_block_groups_2025

evrptw-stage2 preflight \
  --config configs/cle_evrptw_stage2_v2.json \
  --cle-root ../EVRPTW_Dataset/CLE_v2/us_11city \
  --mode non_release_pilot \
  --cities san-diego

evrptw-stage2 build-customer-split \
  --config configs/cle_evrptw_stage2_v2.json \
  --cle-root ../EVRPTW_Dataset/CLE_v2/us_11city \
  --mode non_release_pilot \
  --city san-diego \
  --block-groups data/sources/census_block_groups_2025/tl_2025_06_bg.zip \
  --output-dir work/stage2-pilot-v1/customer_splits/san-diego
```

Build a small plan and materialize one family:

```bash
evrptw-stage2 plan \
  --config configs/cle_evrptw_stage2_v2.json \
  --cle-root ../EVRPTW_Dataset/CLE_v2/us_11city \
  --mode non_release_pilot \
  --cities san-diego \
  --tracks train validation \
  --pilot-families-per-city 2 \
  --output-root work/stage2-pilot-v1/generation_plan

evrptw-stage2 materialize-family \
  --config configs/cle_evrptw_stage2_v2.json \
  --profile configs/us_reference_instance_profile_v2.json \
  --cle-root ../EVRPTW_Dataset/CLE_v2/us_11city \
  --mode non_release_pilot \
  --plan-root work/stage2-pilot-v1/generation_plan \
  --family-id <family_id> \
  --customer-split work/stage2-pilot-v1/customer_splits/san-diego/customer_split_manifest.parquet \
  --community-adjacency work/stage2-pilot-v1/customer_splits/san-diego/community_adjacency.parquet \
  --amazon-artifact-root ../EVRPTW_Dataset/Calibration_v2/amazon_stage2_v3 \
  --output-root work/stage2-pilot-v1/materialized
```

Or run the repository-level vertical-slice shell:

```bash
cd ..
INSTANCE_MODE=non_release_pilot WORKERS=1 \
AMAZON_MODEL_BUILD_INPUTS=/data/almrrc2021-data-training/model_build_inputs \
  ./generate_instances.sh --cities san-diego --tracks train --max-families 1
```

To rebuild both stages in isolated directories and verify one complete San
Diego family, run from the repository root:

```bash
./validate_san_diego.sh
```

This validation requires the same frozen Stage-1 U.S. source bundle as the
eleven-city build. It writes the CLE, one family, the Stage-2 run report, and
the Phase-1 metric bundle below
`EVRPTW_Dataset/Validation/san-diego/`; it does not replace the production
US-11-city release tree.

The production shell defaults to 12 processes and one BLAS/OpenMP thread per
process. A worker keeps one city's immutable routing topology across a
25-family chunk. The runner conservatively budgets 5 GiB per worker and
reserves 4 GiB for the parent/OS. Completed families are verified and reused
on restart; the process queue is bounded, so the full plan is not serialized
into memory at once.

Verify and load:

```bash
evrptw-stage2 verify-family \
  --family-dir work/stage2-pilot-v1/materialized/families/<family_id>
```

```python
from evrptw_stage2.artifacts import load_materialized_view

instance = load_materialized_view(
    "work/stage2-pilot-v1/materialized/families/<family_id>",
    "<view_id>",
)
```

Each parent family stores four matrices: distance-shortest distance and
zero-turn time, plus exact zero-turn fastest time and its path distance. The
edge-state graph still enforces the virtual-split immediate-reversal ban. The loader derives
both energy matrices from the frozen linear coefficient
`h = 100/257 kWh/km`. Lower scales store only parent indices and daily
attributes; runtime masks are never stored.

Every completed parent family also stores `phase1_metrics.json`, one row per
parent customer in `phase1_observations.parquet`, and per-region pairwise
diagnostics. A complete unsharded run aggregates them under
`reports/phase1/`. For multi-server shards, aggregate after all family folders
have been merged:

```bash
PYTHONPATH=src python scripts/aggregate_phase1_metrics.py \
  --instance-root ../EVRPTW_Dataset/Instances_v2/us_11city
```

### Slim instance distribution and reconstruction

The dense parent matrices are a reproducible local cache. To distribute the
complete benchmark as CLE plus lightweight family/view parameters, export a
slim tree:

```bash
PYTHONPATH=src python scripts/reconstruct_stage2_instances.py export-slim \
  --source-root ../EVRPTW_Dataset/Instances_v2/us_11city \
  --output-root ../EVRPTW_Dataset/Instances_v2_slim/us_11city \
  --cle-root ../EVRPTW_Dataset/CLE_v2/us_11city \
  --profile configs/us_reference_instance_profile_v2.json
```

On another server, place the slim tree at the desired final instance path and
restore every family:

```bash
CLE_ROOT=/data/EVRPTW_Dataset/CLE_v2/us_11city \
INSTANCE_ROOT=/data/EVRPTW_Dataset/Instances_v2/us_11city \
WORKERS=12 scripts/restore_stage2_instances.sh
```

Or restore only the parent family required by one or more instance IDs:

```bash
scripts/restore_stage2_instances.sh \
  --view-id iv_005ce24ec5949cef5ae4690e

scripts/restore_stage2_instances.sh --view-id-file instance_ids.txt
```

The restore uses stored family road-state factors rather than replaying RNG,
checks the CLE and profile hashes, and accepts the output only when all four
`.npy` checksums match the original export. It never overwrites a partial or
conflicting matrix directory. See [OUTPUT_SCHEMA.md](docs/OUTPUT_SCHEMA.md)
for the reconstruction contract.

At repository level, the production archive creator performs the slim export,
copies CLE, enforces the CLE/Stage-2/Phase-1 acceptance reports, and publishes
the archive plus checksum atomically:

```bash
cd ..
./auto.sh archive create \
  --archive /data/EVRPTW_Dataset_us11city_research_slim_v1.tar.zst \
  --compression-threads 12
```

If CLE and slim parameters are distributed together as the supported
`EVRPTW_Dataset` archive layout, use the repository-level background workflow
instead of unpacking by hand:

```bash
cd ..
./auto.sh archive start \
  --archive /data/EVRPTW_Dataset_us11city_research_slim_v1.tar.zst \
  --destination /data \
  --workers 12 \
  --families-per-worker-task 25

./auto.sh archive status --destination /data
./auto.sh archive logs --destination /data --follow
./auto.sh archive wait --destination /data
```

The required checksum defaults to `<archive>.sha256`; pass
`--sha256-file FILE` only when it has another name. The workflow rejects
absolute paths, `..` traversal, links, special files, duplicate members,
unexpected archive roots, insufficient disk space, and unrelated existing
targets. It extracts into private staging and publishes the tree atomically
before invoking the same exact reconstruction path shown above. A repeated
`start` for the same archive safely resumes and reuses complete family caches.
The job state and log live under
`<destination>/.evrptw_restore_us11city/`. For this release, budget at least
about 170 GiB free and wait for phase `succeeded` before starting a benchmark.

## Frozen Stage-2 benchmark sizes

| Scale | CS | Train | Validation | Test |
| --- | ---: | ---: | ---: | --- |
| Cus50 | 10 | 100,000 | 500 | Test-1: 500 |
| Cus100 | 20 | 50,000 | 500 | Test-1/2/3: 500 each |
| Cus500 | 50 | 10,000 | 500 | Test-1/2/3: 500 each |
| Cus1000 | 50 | 5,000 | 500 | Test-1/2/3: 500 each |
| Cus2000 | 50 | 0 | 0 | same-city unseen-scale: 500 |

Each training scale has five million active-customer exposures. Complete
communities are held out before generation: Test-1 uses new seeds on the train
pool, Test-2 uses held-out communities in the ten training cities, and Test-3
uses Jacksonville.

Logical train/validation/test directories store family/view indices. Parent
matrix families live once under `materialized/families/`; split indices point
to them. Test is subdivided into `test1_new_seed`,
`test2_heldout_locations`, and `test3_heldout_city`. Compatibility Cus50 and
scalability Cus2000 keep their own consumer cohorts without duplicating a
shared parent matrix.

## Stage-2 V2.1 candidate status

The 2026-08-10 build report and its outputs are Legacy evidence and are not
inputs to this candidate. V2.1 rebuilds CLE and instances from the current
sources into `Calibration_v2/` and `Instances_v2/` only.

The authorized calibration pilot covers the ten training cities and only the
train/validation tracks. Test2, Jacksonville/Test3, Cus2000, and the complete
7,500-family corpus are blocked until the M1-M5/Q90 pilot evidence has been
reviewed and explicitly approved. The legacy performance report remains useful
only as historical engineering context; it is not V2.1 acceptance evidence.

## Generation modes

- `research` is the default complete 11-city dataset build used in the current
  research phase. It keeps public-source candidate semantics explicit.
- `non_release_pilot` requires bounded family counts and is only for smoke
  tests or engineering experiments.
- `official` retains the stricter final-release eligibility checks and is not
  silently substituted by either mode.

## Tests

```bash
PYTHONPATH=src python -m pytest -q
```

The Stage-2 suite covers split isolation, family/view counts, Amazon artifact
preprocessing, controlled two-margin rounding, road-contiguous activation,
global customer uniqueness, nested views, order-template covering,
projected-edge routing, exact zero-turn path choice and reversal validation,
linear energy derivation,
multi-hop CS return caching, Phase-1 metrics, and family verification.

The previous synthetic `evrptw_hierarchy` pipeline has been retired. The
supported generator consists only of the CLE builder and CLE-backed Stage 2
implementation documented above.
