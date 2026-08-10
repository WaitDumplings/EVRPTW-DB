# City Logistics Environment output schema

Default U.S. work and release roots:

```text
EVRPTW_Dataset_Generator/work/us-top10-v1/
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
  top10_cle_index.json
  top10_cle_index.csv

EVRPTW_Dataset/CLE_v1/us_top10/
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
