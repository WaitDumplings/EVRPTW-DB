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

An instance selects active locations and facilities, samples package demand,
service time, and one time window per active location, chooses a vehicle/fleet
policy, creates one static directed weekday or weekend speed realization, and
exports paired distance-shortest and exact turn-aware fastest-path matrices.
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
- optional high-confidence FHWA HPMS edge evidence;
- an explicitly documented NREL Fleet DNA running-speed prior.

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

After unpacking the server bundle:

```bash
cd EVRPTW_Dataset_Generator
conda env create -f environment.yml
conda activate evrptw-cle
cd ..
./generate_cle.sh
./generate_instances.sh
```

These are the two production entry points. `generate_cle.sh` writes the eleven
portable CLEs under `EVRPTW_Dataset/CLE_v1/us_11city/` and removes its
intermediate work tree after a complete successful run. `generate_instances.sh`
uses 12 processes by default and writes the split plan, community ledgers,
matrix families, views, and verification reports under
`EVRPTW_Dataset/Instances_v1/us_11city/`.

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

The build keeps raw sources, caches, and debug artifacts under
`EVRPTW_Dataset_Generator/`, then packages one self-contained CLE per city under
`EVRPTW_Dataset/CLE_v1/us_11city/`. The release-side verifier requires the
operational GraphML and all runtime paths to remain inside the city package.
Large source files and generated GraphML/GeoParquet artifacts are intentionally
excluded from ordinary Git history and should be distributed with versioned
checksum manifests.

## Current implementation status

The new Generator implements both the Stage-1 CLE boundary and a separate
`evrptw_stage2` package. Stage 2 now has a strict portable-CLE reader,
official-versus-pilot gates, Census-block-group community partitioning,
deterministic family/view plans, unit-aware customer activation, depot-aware
catchments, nested charger selection independent of daily active-customer IDs,
directed edge-level weekday/weekend road states, projected-edge
routing, exact turn-aware dual-path matrix families, volume/package/service/time-window
attributes, a sufficient feasibility gate, a stored full-CS-to-depot
fastest-feasible-time cache for dynamic mask acceleration, a consumer loader,
and a structural verifier. The cache is static instance data, not a stored
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
