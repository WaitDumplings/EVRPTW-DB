# U.S. 11-city CLE and Stage-2 build report

This report records the completed 2026-08-10 engineering build. It separates
technical completion from scientific release eligibility.

> Historical-build note: this report predates the direction-aware HPMS-to-OSM
> matcher and `evrptw_directed_speed_profiles_v5`. Its topology, facility, and
> service-location counts remain historical engineering evidence, but its speed
> provenance is not evidence for the current profile. The 11 CLEs and all
> Stage-2 matrices must be regenerated before updating this report.

## Stage 1 result

The cohort is New York City, Los Angeles, Chicago, Houston, Phoenix,
Philadelphia, San Antonio, San Diego, Dallas, Fort Worth, and Jacksonville.
The first ten cities are the training-city pool; Jacksonville is Test-3 only.

All 11 CLEs completed and passed technical and portable-package verification:

| Aggregate | Count |
| --- | ---: |
| Operational road nodes | 574,980 |
| Directed operational road edges | 1,444,732 |
| Latent service locations | 3,767,723 |
| Default service-location pool | 3,531,615 |
| Depot candidates | 9,723 |
| Charging-site candidates | 5,286 |

The paper-ready per-city tables are generated from the portable manifests under
`EVRPTW_Dataset/CLE_v1/us_11city/appendix_tables/`. They include road
connectivity, service-location composition, charging/depot evidence, speed
provenance, and release status.

### Jacksonville connectivity exception

Jacksonville did not pass the raw 99% node / 99.5% physical-road-length gate
after the complete 0, 1, 2, 5, 10, and 20 km real-OSM envelope ladder. The best
real-road envelope was 1 km, with 98.8159% raw node coverage and 99.4163% raw
physical-road-length coverage.

The profile-defined residual rule then excluded only still-uncovered weak
components with fewer than 100 nodes from the effective denominator. It
skipped 136 components, 461 nodes (1.1841%), and 49,557.49 m of physical road
(0.5837%). No uncovered component with at least 100 nodes remained. Effective
coverage was 100% for nodes and physical-road length. All raw values and every
skipped component ID remain in the road manifest. No synthetic road was added,
and no OSM one-way direction was changed.

## Stage 2 vertical-slice result

Three independent non-release pilots passed without a rejected attempt:

| Pilot | Families | Views | Parent terminals | Matrix bytes/family | Wall time | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Test-2 across 10 cities + Jacksonville Test-3 | 11 | 33 | 1,051 | 17,674,128 | 615.2 s | 6.59 GB |
| Nested train views | 1 | 33 | 1,051 | 17,674,128 | 50.4 s | 4.41 GB |
| Same-city unseen-scale Cus2000 | 1 | 1 | 2,051 | 67,306,128 | 133.7 s | 4.30 GB |

The nested train family produced Cus50 x20, Cus100 x10, Cus500 x2, and Cus1000
x1 without duplicating the parent matrices. All four stored matrices passed
shape, finite-value, directionality, turn-aware routing, view-isolation,
derived-energy, feasibility-certificate, and CS-to-depot cache checks. No
runtime action mask is stored.

After profiling, the runner was changed to reuse immutable city topology,
precompute projected-edge and turn-transition costs, vectorize destination
option evaluation, and materialize spawn-safe single-city chunks. On the same
16 GiB development machine, four New York Cus1000 families required 64.14 s
with two workers, versus 121.80 s in the serial comparison. Cus2000
materialization fell from 125.27 s to 44.87 s. Exact value comparisons covered
1,996 Cus1000 arrays plus four terminal tables and 19 Cus2000 arrays plus one
terminal table; no value changed. Detailed commands and sharding behavior are
in [STAGE2_PERFORMANCE.md](STAGE2_PERFORMANCE.md).

## Full-corpus resource envelope

The frozen contract contains 7,000 Cus1000 parent families: 5,000 train, 500
validation, and 500 each for Test-1, Test-2, and Test-3. Cus2000 adds 100 parent
families. Using observed pilot file sizes, parent matrices alone require:

```text
7,000 x 17,674,128 + 100 x 67,306,128
= 130,449,508,800 bytes
= 121.49 GiB
```

View attributes, split manifests, plans, QA reports, and filesystem overhead
are additional. The measured optimized steady state implies approximately
22-30 wall-clock hours on the tested 16 GiB machine with two workers, allowing
for verification and cross-city variation. Family-level resume and
deterministic multi-server sharding are available for production.

## Release boundary

This build is technically executable and portable, but it is not yet an
official benchmark release. Every current CLE reports six open gates:

- Microsoft/NSI geometry calibration;
- customer road-access review;
- depot release review;
- charging-coordinate validation;
- charging release review; and
- delivery-community calibration.

The U.S. operations profile is also `development_calibration` with
`official_generation_eligible=false`. Therefore `--mode official` correctly
refuses full materialization. Pilot artifacts must retain
`non_release_pilot=true`; changing that label without closing the gates would
invalidate the benchmark provenance contract.

## Verification command

```bash
cd EVRPTW_Dataset_Generator
MPLCONFIGDIR=/tmp/evrptw-mpl PYTHONPATH=src \
  conda run -n maojie --no-capture-output pytest -q
```

The completed build passed 79 tests. The only warnings were third-party
`pyproj`/NumPy deprecation warnings.
