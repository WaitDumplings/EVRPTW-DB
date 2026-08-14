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
8. a versioned reference-speed adapter, or an explicit documented decision to
   use legal speed directly.

The adapter must emit both native and canonical fields. For example, a Canadian
road adapter may retain its provincial functional class while emitting the same
`operating_mode` values consumed by Stage 2. It may replace AFDC, HPMS, NSI, and
Microsoft data completely; only the normalized contracts remain fixed.

EPA MOVES5 is the bundled U.S. adapter, not a schema requirement. A Canadian or
other-country implementation may supply its own category/day profile, with its
own mapping from native road classes to canonical speed strata. The adapter
must keep edge legal speed, the low-flow/free-flow proxy, the category-level
operating prior, and final edge reference speed as separate fields. It must not
present a national category table as an observed speed for a particular edge.

## U.S. city-name adapter

The repository includes a concrete onboarding adapter for U.S. Census Places:

```bash
NLR_API_KEY=... ./generate_us_city_cle.sh --city "Austin" --state TX
```

This command materializes a one-city profile and calls the same canonical CLE
assembler used by the frozen reference cohort. The state registry supplies
only public source identifiers and filename conventions. City identity and
boundary membership come from Census, not from a hard-coded 11-city list.
Optional URL/subregion overrides are explicit adapter inputs and remain in the
generated contract.

For AFDC, the U.S. adapter also builds coordinate evidence automatically:
OSM charging POIs provide mapped-site evidence, while the Census Geocoder
provides address-anchor evidence. Both are provenance fields, not silent
coordinate truth. The raw AFDC coordinate remains recoverable, and exact-site
and address-only evidence stay distinguishable in the normalized facility
table.

This mechanism is not a claim that every possible U.S. city will have equally
complete OSM tags, NSI classifications, AFDC sites, HPMS matches, or depot
candidates. The existing preflight, coverage, SCC, missingness, and package
verification reports quantify those outcomes. A technically successful custom
CLE also does not become part of the paper benchmark until a new cohort version
is declared.

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

Create a new versioned profile beside `configs/us_11city_cle_v1.json`. Paths and
adapters belong in configuration; source-specific parsing belongs in adapter
code; the assembler should receive canonical paths and manifests only. Do not
add country branches directly to the CLE schema or Stage-2 solver.

Each profile defines a Generator-owned `work_root` and a dataset-owned
`release_root`. Country-specific raw sources and caches remain under the work
side. The final package must pass `verify_cle(..., require_portable=True)` after
the work tree is treated as unavailable; source registries may retain hashes and
public source identifiers, but must not require the original machine's absolute
paths for routing or instance generation.
