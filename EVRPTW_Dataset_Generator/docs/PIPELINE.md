# CLE construction pipeline

This document defines Stage 1. The Stage-2 operating-day model, formulas,
parameter provenance, and literature are specified separately in
[STAGE2_INSTANCE_MODEL.md](STAGE2_INSTANCE_MODEL.md).

## 1. Contract and execution graph

```text
frozen city/service boundaries + frozen OSM PBF
                         |
                         v
               directed road graph ------ HPMS/OSM legal speed evidence
                         |                           |
          +--------------+---------------+           v
          |              |               |    reference speed profile
          v              v               v
   OSM depot tags   AFDC charging   eligible physical roads
          |              |               |
          +-------+------+-------+-------+
                  |              |
       NSI residential data      |
                  |              |
      Microsoft footprint match  |
                  +--------------+
                         |
                   CLE assembler
                         |
             independent hash/semantic verifier
```

Every branch records native source fields, normalized fields, source paths,
SHA-256 hashes, row counts, and unresolved QA states before assembly.

## 2. Boundary and road graph

1. Freeze the Census Place polygon as the administrative boundary.
2. Subtract Census AREAWATER polygons to obtain the land-only service boundary.
3. Extract a directed `drive` graph from a frozen Geofabrik PBF. Preserve
   one-way direction, parallel edges, original OSM IDs/tags, and geometry.
4. Audit all weakly connected components in the exact city graph.
5. Let `G_city` be all in-boundary roads and `L_city` its largest weak
   component. Compute:

   ```text
   node_coverage = nodes(L_city) / nodes(G_city)
   physical_road_length_coverage = undirected_physical_length(L_city)
                                   / undirected_physical_length(G_city)
   ```

   The second metric deduplicates reciprocal directed representations of the
   same physical road so two-way streets are not counted twice.
6. If either coverage is below 0.99 or 0.995 respectively, test routing
   envelopes at 1, 2, 5, 10, and 20 km. Select the smallest envelope whose real
   OSM roads connect the retained city roads and pass both thresholds.
7. If the entire buffer ladder is exhausted without passing, apply the
   profile-defined residual rule. V1 may remove only still-uncovered weak
   components with fewer than 100 nodes from the *effective gate denominator*.
   It does not invent a road, edit an OSM direction, or delete the component
   from the raw audit. The manifest must retain raw coverage, effective
   coverage, every skipped component ID, and skipped node/physical-length
   counts and shares. Any uncovered component with at least 100 nodes remains
   a hard failure.
8. Mark all nodes/edges outside the service boundary `transit_only=true`. They
   can connect a route but cannot host a service location or facility.
9. Compute stable directed SCC labels. The largest directed SCC is the Stage-1
   reference service SCC; smaller SCCs remain in the road artifact for audit.

This is a weak-connectivity construction gate, not a claim that every directed
node pair is mutually reachable. Directed strong-component statistics are
still recorded, and every active Stage-2 OD pair must pass directed route
feasibility. The pipeline never makes one-way roads bidirectional and never
draws synthetic intercomponent roads.

## 3. Charging sites

1. Freeze AFDC rows with `Fuel Type Code=ELEC`, `Status Code=E`,
   `Access Code=public`, and `Country=US`.
2. Retain raw AFDC coordinates and address fields.
3. Produce optional Census address anchors and exact-address OSM charging-POI
   matches from the same PBF snapshot.
4. Resolve geometry with explicit precedence: reviewed manual override, OSM
   exact-address match, raw AFDC. Census is QA/address access only.
5. Assign a coordinate-evidence tier. Census corroboration does not convert an
   AFDC point into exact charger geometry; uncorroborated rows remain visible
   but cannot enter the default candidate pool.
6. Spatially retain sites inside the service boundary.
7. Store L2/DC-fast port counts, connector tokens, access restrictions, and
   maximum vehicle class separately. A Tesla/NACS site is not removed merely
   because the reference vehicle connector policy is unresolved.
8. Project every anchorable site to the nearest eligible physical road. Keep
   all distances; `>250 m` is a QA flag only.
9. Label the exact projection by directed SCC and quarantine it from the
   default pool unless it inherits the reference SCC.

AFDC does not consistently provide per-port power. Missing kW remains missing;
the CLE does not infer a charging curve or claim that every public site can
serve a medium-duty delivery van.

## 4. Depot candidates

Depot evidence is extracted from the same frozen OSM PBF with tags such as
`building=warehouse`, `industrial=warehouse`, `office=logistics`,
`amenity=post_depot`, accepted `depot=*`/`logistics=*` values, and relevant
industrial land use.

The evidence policy separates:

- Tier A: named carrier/logistics evidence tied to a physical warehouse,
  dispatch-function name, or explicit depot/logistics feature. Retail shipping
  counters are excluded.
- Tier B: warehouse/logistics proxy lacking verified current parcel-dispatch
  function. Included in the optional pool.
- Tier C: generic industrial or ambiguous carrier point. Excluded by default.

Area is measured for polygonal features and stored continuously. The 1,000 m2
reference only flags sensitivity; it neither creates nor deletes Tier B. Every
retained candidate is road-anchored, with long access distance flagged rather
than deleted.

## 5. Latent service locations

1. Query NSI over the exact land boundary in deterministic tiles and freeze the
   raw responses.
2. Keep ordinary residential occupancy families `RES1`, `RES2`, and `RES3`;
   exclude institutional residential families from the ordinary parcel pool.
3. Group NSI records by the first available shared structure identifier:
   `ftprntid`, `usastrucid`, `bid`, then record ID fallback.
4. Sum residential-unit evidence within the group and retain original NSI
   occupancy, footprint, floor, height, area, and mixed-use signals.
5. Classify the latent service location:

   | Class | Default evidence rule |
   | --- | --- |
   | `house` | fewer than 2 estimated units, ordinary residential fallback |
   | `manufactured_home` | `RES2` fallback when units do not imply multi-unit |
   | `small_apt` | 2-4 estimated units or lower `RES3` fallback |
   | `medium_apt` | 5-19 estimated units or mid `RES3` fallback |
   | `large_apt` | at least 20 estimated units or high `RES3` fallback |

6. Match each NSI structure point to Microsoft footprint geometry:
   - G1: NSI point is covered by a Microsoft polygon.
   - G2 candidate: nearest polygon is at most 10 m away and its area differs by
     at most a factor of four. G2 remains explicitly reviewable.
7. Do not infer apartment complexes merely because buildings are nearby. A
   future complex adapter may merge buildings only with independent parcel,
   address, parcel-boundary, or site evidence.
8. Project the resolved building boundary to the nearest eligible physical road
   and materialize a road projection node plus symmetric bidirectional access
   connector. `>200 m` is a QA flag, not a deletion rule.
9. Infer the SCC inherited by the exact directed-edge split. Retain all source
   locations, but only reference-SCC projections are default instance
   candidates. A bidirectional connector alone is not proof of road-network
   round-trip reachability.

The CLE stores `residential_units` and location type as activation covariates.
It does not store realized package count, demand, service time, or time window.

## 6. Speed profiles

### 6.1 HPMS-to-OSM conflation

The U.S. adapter reads a public HPMS line layer for the relevant state, clips
it to the service boundary, and projects it and the operational OSM graph into
the same local metric CRS. Directed OSM edges that represent the same physical
segment are grouped before matching. Candidate ranking uses route-number
compatibility, lateral distance, buffered line overlap, local orientation, and
H/M/U compatibility. Geometry orientation is evaluated modulo 180 degrees, so
the arbitrary coordinate storage order of either source is not interpreted as
travel direction.

The V1 profile generates candidates within 75 m and requires at least 20%
buffered overlap with an orientation difference no greater than 30 degrees. A
non-ambiguous candidate is high confidence only when it has an exact route
token or is nearly coincident, is within 25 m, has at least 50% overlap, and is
within 15 degrees. All tolerances are versioned matcher parameters and the
measured evidence is retained in the normalized output.

A high-confidence physical-corridor match may supply `F_SYSTEM`. HPMS
`SPEED_LIMIT` may fill a directed OSM edge only when that match is high
confidence and the matched OSM physical segment has a unique verified one-way
direction. Bidirectional or ambiguous corridor evidence can still classify the
road but cannot supply a directional speed.

### 6.2 Legal, free-flow proxy, and reference speed

Each directed physical edge keeps separate evidence and model fields:

1. `legal_speed_kph`: direction-applicable OSM `maxspeed`, generic OSM
   `maxspeed`, direction-verified high-confidence HPMS `SPEED_LIMIT` when OSM
   is missing, then transparent within-city class/mode/parent/global
   length-weighted-median imputation.
2. `operating_mode`: HPMS `F_SYSTEM` 1-2 -> H, 3-6 -> M, 7 -> U when a
   high-confidence normalized match exists; otherwise OSM `highway=*` fallback.
3. `moves_road_type`: an independent adapter. HPMS `F_SYSTEM` 1-2 maps to
   MOVES urban restricted access (`roadTypeID=4`), 3-7 to urban unrestricted
   access (`roadTypeID=5`). OSM fallback treats motorway/trunk and their links
   as restricted and other driveable classes as unrestricted.
4. `free_flow_speed_proxy_kph`: equal to the direction-applicable legal speed.
   It is explicitly a proxy, not an observed low-flow edge speed.
5. `reference_speed_weekday_kph` and `reference_speed_weekend_kph`: static
   edge running speeds produced by the model below.

MOVES5 is a model/database, not a spatial network. Its default
`AvgSpeedDistribution` is one national distribution by source type, broad road
type, day type, hour, and speed bin. `HourVMTFraction` is a normalized national
temporal allocation, not traffic volume on an OSM edge and not EDV-specific.
The U.S. adapter uses sourceTypeID 32 (Light Commercial Truck).

For each MOVES road type `r`, day type `d`, and hour `h`, first compute the
hourly mean from MOVES VHT fractions `p` and bin-center speeds `s`:

```text
v_hour(r,d,h) = sum_b p(r,d,h,b) * s(b)
```

Normalize `HourVMTFraction` inside the 08:00-24:00 service window. The window
effective speed preserves total travel time and is therefore VMT/VHT, a
VMT-weighted harmonic aggregation rather than an equal-hour arithmetic mean:

```text
w(r,d,h) = HourVMTFraction(r,d,h) / sum_{k=08:00}^{24:00} HourVMTFraction(r,d,k)
v_effective(r,d) = 1 / sum_{h=08:00}^{24:00} w(r,d,h) / v_hour(r,d,h)
```

Following FHWA's off-peak approach, the class-level low-flow benchmark is the
85th percentile of the VHT-weighted weekend 06:00-10:00 MOVES speed-bin
distribution. It is a national class benchmark, not an edge free-flow
observation. The transferable speed-retention factor is:

```text
rho(r,d) = v_effective(r,d) / Q85_low_flow(r)
reference_speed(edge,d) = legal_speed(edge) * rho(moves_road_type(edge),d)
```

The frozen `movesdb20241112` extraction is:

| MOVES urban class | Low-flow Q85 | Weekday effective | Weekend effective | Weekday rho | Weekend rho |
| --- | ---: | ---: | ---: | ---: | ---: |
| Restricted access | 70.00 mph | 53.115 mph | 58.902 mph | 0.758791 | 0.841456 |
| Unrestricted access | 45.00 mph | 24.528 mph | 25.384 mph | 0.545059 | 0.564078 |

Using the absolute MOVES effective speed directly would collapse the network
to two speeds. The ratio adapter instead preserves edge-specific legal-speed
scale and direction: a 45 mph arterial and a 25 mph residential edge share the
national unrestricted retention factor but do not receive the same reference
speed. This proportional transfer is a documented national-to-edge modeling
assumption; MOVES does not validate it on individual streets.

The compact derived table is versioned in
`configs/us_moves5_speed_profile_v1.json`, and
`scripts/derive_moves5_speed_profile.py` reproduces it from the official SQL
dump. Stage 1 does not write a stochastic speed-scenario bank; Stage 2 selects
the weekday or weekend reference column and the canonical residual factor is
fixed at 1.0.

## 7. Assembly, packaging, and verification

The assembler requires all layer manifests to use the same city slug, boundary,
and operational-graph hash. It copies small canonical tables into the CLE,
references the work GraphML artifact, writes a source registry, computes every
published output hash, and records unresolved evidence gates. This first CLE is
a Generator-owned technical/debug artifact.

The verifier recomputes hashes, counts, ID uniqueness, Stage-1 inactivity,
positive finite speed/travel-time values, reference-speed legal caps, and
facility/customer release semantics. Technical verification and external
real-world validation are intentionally separate statuses.

For each Stage-2 instance, the selected depot, active customers, and permitted
charging stations must inherit the same reference SCC. The current customer
access materializer independently checks `reference road -> customer` and
`customer -> reference road` after virtual edge splitting. The Stage-2 facility
materializer must apply the same post-materialization check to the selected
depot and charging stations before matrices are accepted.

After technical verification, the packager:

1. copies the work CLE into a release staging directory;
2. copies `graph_operational.graphml` and the road manifest into `graph/`;
3. replaces absolute graph references with package-relative paths;
4. removes machine-local paths from the release source registry while retaining
   source hashes;
5. recomputes every affected output hash;
6. runs the strict portable verifier; and
7. atomically promotes the staging directory into `EVRPTW_Dataset`.

The strict verifier fails on missing packaged runtime files, absolute output
paths, paths escaping the CLE root, or graph-reference hash mismatches. Passing
that check sets `portable_package_verified=true`; it does not override open
scientific gates or set `release_eligible=true`.
