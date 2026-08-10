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
exports distance-path and running-time-path matrices. Scale views reuse one
parent matrix family and therefore do not copy the city graph or lower-scale
matrices.

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

The first CLE profile targets ten U.S. city-proper service areas: New York
City, Los Angeles, Chicago, Houston, Phoenix, Philadelphia, San Antonio, San
Diego, Dallas, and Fort Worth.

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

After sources pass preflight:

```bash
cd EVRPTW_Dataset_Generator
conda env create -f environment.yml
conda activate evrptw-cle
bash scripts/build_top10_cle.sh
```

The CLE-backed Stage-2 reference runner, its community split, and non-release
San Diego vertical-slice commands are documented in the Generator README under
**Stage 2: CLE to operating-day instances**. Official generation is deliberately
blocked while the CLE scientific release gates and U.S. operations-profile
calibration remain open.

The build keeps raw sources, caches, and debug artifacts under
`EVRPTW_Dataset_Generator/`, then packages one self-contained CLE per city under
`EVRPTW_Dataset/CLE_v1/us_top10/`. The release-side verifier requires the
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
routing, dual path matrix families, volume/package/service/time-window
attributes, a sufficient feasibility gate, a consumer loader, and a structural
verifier.

This is currently a **non-release vertical slice**, not an official dataset.
The ten existing CLE packages are technically portable but still declare open
scientific release blockers, and the U.S. instance profile is labeled
`development_calibration`. Both gates must pass before the runner accepts
`--mode official`.

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
