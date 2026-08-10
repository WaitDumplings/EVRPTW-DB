# Porting the CLE pipeline beyond the U.S.

The core consumes normalized evidence tables; it does not require U.S.-specific
source names. A new geography adapter must provide:

1. an administrative boundary and land/service boundary in WGS84;
2. a directed road graph with stable node/edge IDs, geometry, length, direction,
   access, and native road tags;
3. building or address geometry plus residential type/unit evidence;
4. charging-site evidence with access, connector, ports, power if known, and
   raw/resolved coordinates;
5. depot candidates with explicit evidence tiers;
6. a native-road-class to canonical H/M/U crosswalk;
7. legal speed evidence and a transparent imputation hierarchy;
8. an optional commercial-vehicle reference-speed model.

The adapter must emit both native and canonical fields. For example, a Canadian
road adapter may retain its provincial functional class while emitting the same
`operating_mode` values consumed by Stage 2. It may replace AFDC, HPMS, NSI, and
Microsoft data completely; only the normalized contracts remain fixed.

## Required validation

A new adapter is accepted only when it demonstrates:

- exact boundary/service-mask provenance and hashes;
- no invented intercomponent roads;
- explicit weak and directed connectivity statistics;
- stable nearest-road projection with original one-way direction preserved;
- no hard deletion caused solely by an arbitrary access-distance reference;
- legal/reference/operational speed semantics kept separate;
- source-level missingness and uncertainty preserved;
- a deterministic rebuild from frozen inputs;
- checksum and semantic verification of the final CLE;
- a self-contained operational graph and package-relative runtime paths.

## Configuration pattern

Create a new versioned profile beside `configs/us_top10_cle_v1.json`. Paths and
adapters belong in configuration; source-specific parsing belongs in adapter
code; the assembler should receive canonical paths and manifests only. Do not
add country branches directly to the CLE schema or Stage-2 solver.

Each profile defines a Generator-owned `work_root` and a dataset-owned
`release_root`. Country-specific raw sources and caches remain under the work
side. The final package must pass `verify_cle(..., require_portable=True)` after
the work tree is treated as unavailable; source registries may retain hashes and
public source identifiers, but must not require the original machine's absolute
paths for routing or instance generation.
