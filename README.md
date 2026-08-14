# EVRPTW-DB

EVRPTW-DB is a dataset and benchmark project for practical Electric Vehicle
Routing Problems with Time Windows. Its central design is to separate a
reusable real-geography environment from the operating-day instances sampled
on top of it.

## Two-stage data model

### Stage 1: City Logistics Environment (CLE)

A CLE freezes the static city-level substrate:

- a land-only city service boundary and a real directed OSM routing graph;
- latent residential service locations with house/apartment evidence;
- candidate depot and public charging-site layers;
- edge-level legal and commercial-vehicle reference running speeds;
- road-access connectors, provenance, checksums, and QA/release gates.

It deliberately contains no active customer, package count, demand, service
time, or realized time window.

### Stage 2: CLE-EVRPTW instances

An instance selects a depot and exactly N active locations, chooses relevant
real charging sites, attaches distinct feasible Amazon stop-level order
templates, chooses a vehicle/fleet policy, creates one static directed weekday
or weekend road state, and exports paired distance-shortest and exact
turn-aware fastest-path matrices.
The four dense matrices are a deterministic local cache: a portable release
may omit them and reconstruct them bit-for-bit from the frozen CLE, stored
family road-state factors, and terminal projections. The two energy matrices
are derived from path distance and the frozen linear vehicle coefficient.

This separation lets many instances reuse one large physical graph without
copying it into every instance and prevents latent future information from
leaking into solver inputs.

## Repository layout

```text
EVRPTW-DB/
  EVRPTW_Core/                  # shared schemas, loaders, validation, metrics
  EVRPTW_Dataset_Generator/     # CLE generator and source-adapter pipeline
  EVRPTW_Dataset/               # versioned CLE/instance release artifacts
  EVRPTW_Benchmark/             # exact, metaheuristic, and RL solvers
    Exact/
    MetaHeuristics/
    Reinforcement_Learning/
```

## U.S. reference implementation

The reference cohort contains ten training-city CLEs—New York City, Los
Angeles, Chicago, Houston, Phoenix, Philadelphia, San Antonio, San Diego,
Dallas, and Fort Worth—plus Jacksonville as the held-out Test-3 city. Every
boundary is a Census city-proper service area with water removed; Jacksonville
is not used by training, validation, Test-1, Test-2, or same-city Cus2000.

It integrates public/freely accessible sources with distinct roles:

- Census TIGER/Line Place and Area Hydrography boundaries;
- OpenStreetMap roads and facility tags from frozen Geofabrik PBFs;
- Microsoft USBuildingFootprints geometry;
- USACE National Structure Inventory residential occupancy/unit evidence;
- NREL Alternative Fuels Data Center charging sites;
- FHWA HPMS functional-class and legal-speed evidence, conflated to OSM by the
  Generator with an auditable direction-aware matcher;
- EPA MOVES5 national light-commercial-truck speed distributions, converted
  into weekday/weekend road-class retention factors rather than treated as
  edge observations; and
- Amazon Last Mile Routing Research Challenge 2021 depot-day route structure
  and stop-level package/service/time-window templates, without transferring
  its obfuscated coordinates to CLE cities.

The result is best described as a **real-geography, public-data-integrated,
semi-synthetic benchmark substrate**. Public data grounds topology, geometry,
facility candidates, and statistical priors; Stage 2 still generates
operating-day demand and time windows. The project does not claim that every
OSM warehouse is a verified Amazon depot, every public charger can serve a
Rivian EDV, or every generated order is an observed delivery.

## Quick start

Source preparation, exact required filenames, data licenses, frozen thresholds,
and one-command generation are documented in
[`EVRPTW_Dataset_Generator/README.md`](EVRPTW_Dataset_Generator/README.md).
The rigorous Stage-2 submodels, formulas, parameter provenance, assumptions,
and literature are in
[`docs/STAGE2_INSTANCE_MODEL.md`](EVRPTW_Dataset_Generator/docs/STAGE2_INSTANCE_MODEL.md).
Optimized local workers, memory limits, exact-output benchmarks, resume, and
multi-server sharding are documented in
[`docs/STAGE2_PERFORMANCE.md`](EVRPTW_Dataset_Generator/docs/STAGE2_PERFORMANCE.md).

The current speed schema is `evrptw_directed_speed_profiles_v6`. CLEs generated
under the previous NREL-anchor profile must be regenerated before producing
matrices with the MOVES5 reference model; old matrices must not be mixed with
v6 CLEs.

### One city by name (not limited to the reference 11)

The frozen 11-city cohort remains the paper benchmark. A separate U.S. city
adapter accepts any Census Place name plus its state and constructs a compatible
single-city profile. It does not silently add the city to the paper cohort.

```bash
conda activate evrptw-cle
export NLR_API_KEY=YOUR_FREE_NLR_DEVELOPER_KEY

# State-wide OSM extract (generic default).
./generate_us_city_cle.sh --city "San Antonio" --state TX

# An official smaller Geofabrik extract may be selected when available.
./generate_us_city_cle.sh \
  --city "San Diego" --state CA \
  --geofabrik-region california/socal
```

The adapter resolves the name inside the specified state against 2025 Census
Places, builds the city-proper land boundary, downloads/reuses OSM, Microsoft
building, bounded FHWA HPMS, and national AFDC inputs. It then derives OSM POI
and Census-address evidence for AFDC coordinate QA before invoking the same CLE
builder and verifier. Raw AFDC coordinates remain in the evidence table, and a
Census address anchor is not labeled as exact EVSE geometry. NSI is queried and
cached by the existing customer stage.
Ambiguous place names fail with candidate GEOIDs; they are never geocoded by a
best-effort web search. Outputs are written under
`EVRPTW_Dataset/CLE_v1/us_custom/<city>/`.

After unpacking the server bundle:

```bash
cd EVRPTW_Dataset_Generator
conda env create -f environment.yml
conda activate evrptw-cle
cd ..
export NLR_API_KEY=YOUR_FREE_NLR_DEVELOPER_KEY
./generate_cle.sh
./generate_instances.sh
```

`generate_cle.sh` first checks every fixed-cohort public input. Missing OSM,
Microsoft building, bounded HPMS, AFDC/coordinate-evidence, and Census
block-group files are downloaded or derived; existing nonempty files are
reused. NSI is cached during the CLE customer stage. `generate_instances.sh`
uses 12 workers by default and downloads only the three public Amazon training
JSON files used by Stage 2 when neither those files nor the compact artifact
already exists. AWS access is unsigned, so no AWS account is required. Raw
Amazon files stay outside Git; see
[`AMAZON_LAST_MILE_2021.md`](EVRPTW_Dataset_Generator/docs/AMAZON_LAST_MILE_2021.md)
for the compact-artifact and release-license policy.

### Server-agent production runbook

The server agent should execute the production build in this order. Do not run
CLE and instance generation concurrently.

1. Start from the repository root, record the code revision, and confirm the
   host has sufficient resources. The complete matrix cache is about 155 GiB;
   at least 300 GiB free disk is recommended for sources, work files, CLEs,
   matrices, and reports.

   ```bash
   # Fresh host:
   # git clone git@github.com:WaitDumplings/EVRPTW-DB.git /data/Maojie/ICLR/EVRPTW-DB
   cd /data/Maojie/ICLR/EVRPTW-DB
   git pull --ff-only origin main
   git status --short
   git rev-parse HEAD
   free -h
   df -h .
   ```

2. Create the environment on a new host, or update it after pulling a newer
   commit. The environment includes `osmium` and the AWS CLI.

   ```bash
   conda env create -f EVRPTW_Dataset_Generator/environment.yml
   # For an existing environment instead:
   # conda env update -n evrptw-cle -f EVRPTW_Dataset_Generator/environment.yml --prune
   conda activate evrptw-cle
   export NLR_API_KEY=YOUR_FREE_NLR_DEVELOPER_KEY
   mkdir -p logs
   ```

3. Generate all eleven CLEs. This command first downloads only missing source
   inputs, reuses existing inputs, and then runs the resumable CLE builder.

   ```bash
   nohup bash -c './generate_cle.sh; code=$?; printf "%s\n" "$code" > logs/generate_cle.exit; exit "$code"' \
     > logs/generate_cle.log 2>&1 &
   echo $! > logs/generate_cle.pid
   tail -f logs/generate_cle.log
   ```

4. After the CLE process ends, require exit code zero and verify the cohort
   index before starting Stage 2.

   ```bash
   cat logs/generate_cle.exit
   jq '{status, verified_cle_count, failures}' \
     EVRPTW_Dataset/CLE_v1/us_11city/cle_index.json
   ```

   Expected values are `status="complete"`, `verified_cle_count=11`, and
   `failures=[]`.

5. Generate the full research instance corpus. The runner uses 12 workers by
   default, downloads the three required Amazon files when absent, builds the
   compact calibration artifact, and reuses completed families on restart.

   ```bash
   nohup bash -c 'INSTANCE_MODE=research ./generate_instances.sh; code=$?; printf "%s\n" "$code" > logs/generate_instances.exit; exit "$code"' \
     > logs/generate_instances.log 2>&1 &
   echo $! > logs/generate_instances.pid
   tail -f logs/generate_instances.log
   ```

6. After Stage 2 ends, require exit code zero and inspect the authoritative run
   report and Phase-1 aggregate directory.

   ```bash
   cat logs/generate_instances.exit
   jq '{passed, unresolved_family_count: (.unresolved_family_ids | length), verified_family_count: (.verified | length)}' \
     EVRPTW_Dataset/Instances_v1/us_11city/stage2_run_report.json
   ls EVRPTW_Dataset/Instances_v1/us_11city/reports/phase1
   ```

   Expected values are `passed=true` and `unresolved_family_count=0`.

If either process is interrupted, preserve the source, work, and output trees
and rerun the same command. Do not delete partial downloads or completed family
folders: the acquisition and generation stages are designed to resume and
reuse them. Do not start solver benchmarks until both checks above pass.

For a complete isolated San Diego vertical slice after the source bundle is in
place:

```bash
./validate_san_diego.sh
```

For an existing Python 3.11 environment, the repository-level pip equivalent
is:

```bash
python -m pip install -r requirements.txt
```

Use the same `python` executable for installation and archive restoration. The
archive launcher checks the Stage-2 restore imports before checksum scanning so
a missing dependency fails immediately with the corresponding install command.

These are the two production entry points. `generate_cle.sh` writes the eleven
portable CLEs under `EVRPTW_Dataset/CLE_v1/us_11city/` and removes its
intermediate work tree after a complete successful run. `generate_instances.sh`
uses 12 processes by default and writes the split plan, community adjacency,
matrix families, views, verification reports, rejected-attempt records, and
Phase-1 metrics under
`EVRPTW_Dataset/Instances_v1/us_11city/`.

`generate_us_city_cle.sh` is the exploratory single-city entry point. A custom
city can later be promoted into a new frozen cohort only by versioning its
source snapshot and cohort configuration; changing a city name does not mutate
the 11-city benchmark.

The combined entry point exposes both supported CLE-backed acquisition modes:

```bash
# Build CLE and sample Stage 2 directly.
./auto.sh stage2

# Reconstruct after transferring CLE plus the slim instance-parameter tree.
# Add --view-id/--view-id-file to restore only selected parent families.
CLE_ROOT=/data/EVRPTW_Dataset/CLE_v1/us_11city \
INSTANCE_OUTPUT_ROOT=/data/EVRPTW_Dataset/Instances_v1/us_11city \
WORKERS=12 ./auto.sh restore
```

For a transferred slim release archive, one command verifies its SHA-256,
checks every archive member, unpacks it safely, and restores all matrix
families in a persistent background `tmux` session:

```bash
# Keep the required sidecar beside the archive as FILE.tar.zst.sha256.
./auto.sh archive start \
  --archive /data/EVRPTW_Dataset_us11city_research_slim_v1.tar.zst \
  --destination /data \
  --workers 12

./auto.sh archive status --destination /data
./auto.sh archive logs --destination /data --follow
./auto.sh archive wait --destination /data
```

The archive must contain one top-level `EVRPTW_Dataset/` directory; the final
tree is `/data/EVRPTW_Dataset` for the example above. `start` is resumable for
the same verified archive and reuses atomically completed matrix families, but
refuses to overwrite or adopt an unrelated existing dataset. Use
`--foreground` only for interactive debugging or CI. For the current
US-11-city release, reserve at least about 170 GiB of free space and do not run
benchmarks until `status` reports `succeeded`; the wrapper also performs an
exact manifest-based space check before extraction.

The build keeps raw sources, caches, and debug artifacts under
`EVRPTW_Dataset_Generator/`, then packages one self-contained CLE per city under
`EVRPTW_Dataset/CLE_v1/us_11city/`. The release-side verifier requires the
operational GraphML and all runtime paths to remain inside the city package.
Large source files and generated GraphML/GeoParquet artifacts are intentionally
excluded from ordinary Git history and should be distributed with versioned
checksum manifests.

## Current implementation status

The Generator implements both Stage-1 CLE construction and a separate
`evrptw_stage2` package. Stage 2 now includes a strict portable-CLE reader,
complete-community held-out splits, a directed road-community adjacency graph,
deterministic family/view plans, physical-facility depot grouping,
Amazon-envelope territory construction, route-by-decile controlled rounding,
road-contiguous region growth, globally unique customer assignment, customer-
relevant real-CS selection, observed Amazon order-template covering, exact
turn-aware dual-path matrices, a full-CS-to-depot multi-hop cache, and family-
plus-corpus Phase-1 metrics. The cache is static instance data, not a stored
runtime action mask.

All eleven CLE packages have passed technical and package-portability
verification. Stage-2 held-out-location/held-out-city, nested training-view,
and Cus2000 vertical slices also pass. The production instance shell defaults
to `research` mode: it permits the complete frozen benchmark plan while
retaining its research-generation label and the source provenance. The stricter
`official` mode remains available for a later final public release. The exact
build evidence is recorded in
[`US_11CITY_BUILD_REPORT.md`](EVRPTW_Dataset_Generator/docs/US_11CITY_BUILD_REPORT.md).

## Reproducibility and release policy

- Routine research runs record versioned source/profile IDs, seeds, and manifest
  references. A complete byte-level checksum audit is deferred to the final
  release workflow.
- Road extension uses real OSM roads only; outside-city roads are transit-only.
- Customer/facility access-distance values are QA references, not arbitrary
  hard-deletion thresholds.
- Every road projection is labeled by directed SCC. Default instance depots,
  customers, and charging stations must share the reference SCC; source rows
  outside it are retained with quarantine labels.
- AFDC coordinate evidence is tiered. Census validates an address anchor but is
  never presented as an exact charging-space coordinate.
- Legal speed, reference vehicle running speed, and Stage-2 operational speed
  are separate fields.
- Solver code may read only exported active instances, never inactive CLE
  locations or future scenario state.
- Generated OSM derivatives must retain attribution and comply with ODbL.
- An explicit repository code license must be selected before public release.
