# City Logistics Environment output schema

This document first defines the Stage-1 CLE. Stage-2 family/view artifacts are
defined at the end and their modeling semantics are in
[STAGE2_INSTANCE_MODEL.md](STAGE2_INSTANCE_MODEL.md).

Default U.S. work and release roots:

```text
EVRPTW_Dataset_Generator/work/us-11city-v1/
  qa/preflight.json
  cities/<city>/                       # road extraction and connectivity audit
  buildings/<city>/                    # city Microsoft footprints
  customers/<city>/                    # frozen NSI source and classifications
  customer_geometry/<city>_spatial_match/
  customer_access/<city>/
  depot_candidates/
  cles/<city>/
    manifest.json
    source_registry.json
    boundary/
      admin_boundary.geojson
      service_boundary.geojson
    graph/
      graph_reference.json             # may refer to work graph
    service_locations/
      latent_locations.parquet
      service_access_nodes.parquet
      road_projection_nodes.parquet
      service_access_connectors.parquet
    infrastructure/
      chargers.parquet
      depots.parquet
      facility_manifest.json
    profiles/
      directed_legal_speeds.parquet
      speed_manifest.json
    qa/
      cle_report.json
  cle_cohort_index.json
  cle_cohort_index.csv

EVRPTW_Dataset/CLE_v1/us_11city/
  cle_index.json
  cle_index.csv
  appendix_tables/                    # paper-ready cohort tables
  cities/<city>/
    manifest.json
    source_registry.json
    boundary/
      admin_boundary.geojson
      service_boundary.geojson
    graph/
      graph_reference.json             # relative package paths only
      graph_operational.graphml
      road_manifest.json
    service_locations/
      latent_locations.parquet
      service_access_nodes.parquet
      road_projection_nodes.parquet
      service_access_connectors.parquet
    infrastructure/
      chargers.parquet
      depots.parquet
      facility_manifest.json
    profiles/
      directed_legal_speeds.parquet
      speed_manifest.json
    qa/
      cle_report.json
```

## Entity semantics

- `latent_service_location_id`: fixed physical delivery opportunity. It is not
  an order, package, person, or active customer.
- `road_projection_node_id`: a deterministic point on an eligible physical
  road used to split the directed edge when that location is activated.
- `service_access_connector_id`: a symmetric bidirectional connector between a
  location and its road projection. Connector speed is assigned in Stage 2.
- `charger_id`: AFDC site identifier with raw/resolved coordinate provenance,
  connector/port evidence, and compatibility status.
- `candidate_id`: OSM-derived depot candidate with Tier A/B evidence and
  verification state.
- `edge_id`: directed OSM graph edge `u:v:key`; reciprocal directions remain
  separate.

## Required speed columns

`directed_legal_speeds.parquet` includes at least:

```text
edge_u, edge_v, edge_key, edge_id, physical_segment_id, corridor_id,
length_m, highway, operating_mode, operating_mode_source,
legal_speed_kph, legal_speed_source, legal_speed_confidence,
legal_speed_imputed, legal_travel_time_s,
v_model_nrel_kph, reference_speed_kph, reference_speed_source,
reference_travel_time_s
```

Raw directional/generic OSM values, parsed HGV evidence, optional HPMS fields,
match confidence, and conflict flags are retained for audit.

## Required latent-location concepts

The exact table contains additional source and QA columns, but downstream code
should rely on these concepts:

```text
latent_service_location_id, geometry, service_location_type,
residential_units, geometry_evidence_tier, physical_edge_id,
directed_edge_refs, road_access_distance_m,
road_access_distance_qa_flag, anchor_scc_id, reference_scc_id,
protected_roundtrip_eligible, cle_candidate_eligible,
cle_default_instance_eligible, active_customer
```

`active_customer` must be false for every Stage-1 row.

Facility tables use the same SCC fields. Charger rows additionally retain
`coordinate_validation_tier`, `coordinate_validation_status`, raw AFDC
coordinates, optional Census address anchors, optional exact-address OSM
geometry, and the final resolved geometry. Source retention and default
benchmark eligibility are separate flags.

## Manifest semantics

`manifest.json` is authoritative for paths, layer counts, status, and SHA-256
hashes. Work-artifact `source_registry.json` records local source paths and
hashes for rebuild auditing. A portable package omits machine-local build paths,
retains upstream hashes, and adds relative paths only for files included in the
package. A consumer must verify the manifest before using a CLE and must not
infer release eligibility merely from file presence.

The debug CLE references the work GraphML by path and hash. The release packager
copies that graph exactly once into the city package, records its hash as
`outputs.operational_graph`, and rewrites `graph_reference.json` using paths
relative to its own directory. Strict portability verification rejects missing
runtime files, absolute runtime paths, `..` path escapes, and graph-reference
hash mismatches.

Three statuses must remain distinct:

- `technical_verification_passed`: work tables and hashes are internally valid;
- `portable_package_verified`: the CLE can be copied and loaded without the
  Generator work tree;
- `release_eligible`: all declared scientific/evidence release gates are closed.

## Stage-2 matrix-family schema

The release root separates logical split indices from the deduplicated physical
family store:

```text
Instances_v1/us_11city/
  customer_splits/<city>/
  generation_plan/
    core/train/
    core/validation/
    core/test/test1_new_seed/
    core/test/test2_heldout_locations/
    core/test/test3_heldout_city/
    compatibility_cus50/
    scalability_cus2000/
  materialized/families/<family_id>/
  stage2_run_report.json
```

The train/validation/test folders contain Parquet indices. They do not copy
matrix files. A family or view is resolved by stable ID into the shared
`materialized/families/` store.

A `cle_evrptw_materialized_matrix_family_v2` directory contains:

```text
family_manifest.json
terminal_index.parquet
matrices/
  distance_matrix_km.npy
  distance_path_travel_time_s.npy
  running_time_shortest_matrix_s.npy
  running_time_path_distance_km.npy
views/<view_id>/
  view_manifest.json
  terminal_parent_indices.npy
  customer_attributes.npz
  charging_attributes.npz
```

The four parent matrices are square `float32` arrays ordered by
`terminal_index.parquet`. Energy matrices are not stored. The manifest records
`specific_energy_consumption_kwh_per_km`, and the consumer loader derives:

```text
distance_path_energy_kwh = distance_matrix_km * h
running_time_path_energy_kwh = running_time_path_distance_km * h
```

The loader exposes the two derived arrays for solver compatibility and labels
`metadata.energy_matrix_source=derived_from_path_distance`.

### Slim release and deterministic matrix cache

The four dense matrices are derived caches, not irreducible instance data.  A
`cle_evrptw_slim_instances_v1` export keeps the complete directory structure
above but omits every family `matrices/` directory.  It adds:

```text
Instances_v1/us_11city/
  _reconstruction/
    reconstruction_contract.json
    reference_profile.json
    instance_registry.parquet
```

`instance_registry.parquet` maps every stable `view_id` to its parent
`family_id`.  Reconstructing one view reconstructs the four shared parent
matrices for that family. `reconstruction_contract.json` freezes SHA-256 hashes
of the CLE operational graph and directed-speed layer, the exact reference
profile, and every omitted `.npy` file.

Reconstruction uses, in order:

1. `road_state_report.moves_road_type_baseline_factors` stored in the family
   manifest (it does not replay a random-number generator);
2. CLE `reference_speed_kph`, `legal_speed_kph`, and the frozen profile;
3. the complete `terminal_index.parquet` edge-access contract, especially
   `directed_projection_offsets` and `connector_length_m`; and
4. the same distance-shortest and turn-aware running-time routing code used by
   direct Stage-2 materialization.

A simple graph-node ID is insufficient because a terminal can lie partway
along a directed physical edge. A restored cache is accepted in strict mode
only when all four generated `.npy` SHA-256 hashes match the export contract.
The restore command writes a complete temporary directory and atomically moves
it into place; it never overwrites a partial or conflicting existing cache.

## Stage-2 view schema

A `cle_evrptw_materialized_view_v3` stores only parent indices and
view-specific attributes. `customer_attributes.npz` contains:

- `package_counts`, `demands_cm3`, and `service_time_s`;
- `time_windows_s[:,2]`;
- the one-customer arrival/return/charging feasibility certificate; and
- `order_sampling_attempts`, for auditing sample-then-validate replacements.

`charging_attributes.npz` contains effective charging power and
`full_cs_to_depot_time_s`. The latter permits zero or more intermediate CS
hops and excludes charging at its origin and at the depot.

`view_manifest.json` records the split/track/scale, operating horizon,
vehicle and charging policies, energy model, rejection summary, and source
parent matrices. No runtime action mask is stored.
