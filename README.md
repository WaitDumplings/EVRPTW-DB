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

### Stage 2: EVRPTW instances

An instance will select active locations and facilities, sample package demand,
service time, and one time window per active location, choose a vehicle/fleet
policy, create one static directed weekday or weekend speed realization, and
export shortest-distance plus fastest-time matrices and path references.

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
facility candidates, and statistical priors; Stage 2 will still generate
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

The build keeps raw sources, caches, and debug artifacts under
`EVRPTW_Dataset_Generator/`, then packages one self-contained CLE per city under
`EVRPTW_Dataset/CLE_v1/us_top10/`. The release-side verifier requires the
operational GraphML and all runtime paths to remain inside the city package.
Large source files and generated GraphML/GeoParquet artifacts are intentionally
excluded from ordinary Git history and should be distributed with versioned
checksum manifests.

## Current migration status

The new Generator implements Stage 1 and its U.S. reference adapter. The legacy
`evrptw_hierarchy` Stage-2 code remains only because existing TERRAN utilities
still import it. It is explicitly separated from the new CLE pipeline; the
future CLE-to-instance generator will replace it after its sampling semantics
and schemas are frozen.

## Reproducibility and release policy

- Every frozen input and generated layer is hash-addressed in manifests.
- Road extension uses real OSM roads only; outside-city roads are transit-only.
- Customer/facility access-distance values are QA references, not arbitrary
  hard-deletion thresholds.
- Legal speed, reference vehicle running speed, and Stage-2 operational speed
  are separate fields.
- Solver code may read only exported active instances, never inactive CLE
  locations or future scenario state.
- Generated OSM derivatives must retain attribution and comply with ODbL.
- An explicit repository code license must be selected before public release.
