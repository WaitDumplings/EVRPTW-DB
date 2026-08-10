# EVRPTW_Dataset

Generated datasets are stored here. The dataset family is **EVRPTW-D**. Large
generated artifacts are ignored by git and should be published separately
through a dataset hosting service, release artifact, or Git LFS-style workflow.

The current profiles are:

- **CLE-v1 / US Top 10**: portable city-level road, service-location,
  infrastructure, access, and reference-speed environments used as Stage-1
  inputs.
- **CLE-EVRPTW-v1**: the new CLE-backed matrix-family/view layout. The current
  generated artifact is a non-release San Diego vertical slice; official
  train/validation/test generation remains gated.
- **AC-v1**: Amazon-calibrated synthetic service-territory benchmark.
- **Geo-AC-v1 / NA-US-20**: real-geography semi-synthetic benchmark. Public
  geospatial data determines road networks, communities, latent customer
  positions, charging stations, and depot candidates. Amazon calibration is used
  only for operating-day demand, service-time, time-window, and activation
  behavior.

## CLE-v1 / US Top 10 Layout

```text
EVRPTW_Dataset/
  CLE_v1/
    us_top10/
      cle_index.json
      cle_index.csv
      cities/
        san-diego/
          manifest.json
          source_registry.json
          boundary/
          graph/
            graph_reference.json
            graph_operational.graphml
            road_manifest.json
          service_locations/
          infrastructure/
          profiles/
          qa/
```

Each city directory is a self-contained portable package. Its manifest records
technical verification, portability verification, and scientific
`release_eligible` status separately. Generator caches and intermediate layers
are not part of this directory.

## CLE-EVRPTW-v1 Layout

```text
EVRPTW_Dataset/
  CLE_EVRPTW_v1/
    customer_splits/<city>/
    generation_plan/
    materialized/families/<family_id>/
      family_manifest.json
      terminal_index.parquet
      matrices/
      views/<view_id>/
```

One family owns the parent terminal order and matrices. Cus50, Cus100, and
Cus500 views store only indices into the Cus1000 parent; Cus2000 is a separate
test-only parent. View artifacts store volume demand, package count, service
time, one time window per active location, selected CS power, and a compact
single-customer feasibility certificate with optional full-charge CS visits.
They do not store dynamic masks or a copy of the city graph.

Until all release gates close, development output belongs under
`EVRPTW_Dataset_Generator/work/` and must retain `non_release_pilot=true`.

## AC-v1 Layout

```text
EVRPTW_Dataset/
  AC_v1/
    train/
      service_territory_pool.pkl
      manifest.json
    eval/
      service_territory_pool.pkl
      manifest.json
      AC_Tiny_5/
        instances.pkl
        metadata/
        analysis_outputs/
      AC_Small_15/
        instances.pkl
      AC_Medium_50/
        instances.pkl
      AC_Large_100/
        instances.pkl
      AC_XLarge_1000/
        instances.pkl
    generation_timing.csv
    dataset_manifest.json
```

`train/service_territory_pool.pkl` stores the reusable training
service-territory pool as one pickle stream. Training instances are sampled
online from this pool and are not stored by default.
`eval/service_territory_pool.pkl` stores held-out service territories for fixed
evaluation suites. Each `eval/AC_*` directory stores its fixed operating-day
instances as one `instances.pkl` pickle stream.

## Geo-AC-v1 / NA-US-20 Release Layout

Recommended local/release layout:

```text
EVRPTW_Dataset/
  Geo_AC_v1/
    source_data_na_us20/
      README.md
      metadata/
        territory_table.csv
        dataset_summary.json
        source_versions.json
        normalized_schema.json
        file_inventory.csv
      <territory_id>/
        normalized/
          road_nodes.csv
          road_edges.csv
          customer_seed.csv
          latent_customer.csv
          charging_station.csv
          depot_candidate.csv
        qa/
          qa_summary.json
          qa_report.md
          preview_layers.geojson
      qa_maps/
      qa_maps_latent/
    eval_standard_20/
      README.md
      service_territories/<territory_id>/service_territory_pool.pkl
      eval/<territory_id>/Cus_5/instances.pkl
      eval/<territory_id>/Cus_15/instances.pkl
      eval/<territory_id>/Cus_50/instances.pkl
      eval/<territory_id>/Cus_100/instances.pkl
```

`source_data_na_us20` is the reusable geospatial source dataset. It can be used
to regenerate service-territory pools and new operating-day instances. The
standard evaluation split contains `20 territories x 4 scales x 20 instances =
1600` fixed operating-day instances.

Generated `.pkl`, `.csv`, `.geojson`, `.png`, and raw geospatial cache files are
ignored by git by default; release datasets separately.
