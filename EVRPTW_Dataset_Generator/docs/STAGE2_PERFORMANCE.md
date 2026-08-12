# Stage-2 execution and performance

This document describes how the reference generator materializes the frozen
Stage-2 family plan efficiently without changing its routing or instance
semantics. The benchmark contract remains in
[STAGE2_INSTANCE_MODEL.md](STAGE2_INSTANCE_MODEL.md); this file is only the
execution contract.

## 1. Unit of work

A matrix family is the atomic, resumable unit. Each family selects one depot,
one parent customer superset, one fixed-size charging-station set, one
family-level road state, and one vehicle realization. It stores four parent
matrices and all lower-scale index views.

Families are deterministic functions of their recorded seeds. A successful
family is written to a staging directory, verified, and atomically promoted.
On restart, an existing family is verified and reused instead of recomputed.
Rejection ledgers remain family-specific.

## 2. Optimized routing path

The implementation keeps the exact V1 policies:

- directed shortest physical distance, with travel time accumulated along that
  same path;
- directed fastest running-time routing on the turn-aware line graph; and
- exact projected-edge access for depots, customers, and charging stations.

The performance changes are semantic-preserving:

1. Projection partial-distance and partial-time costs are computed once per
   terminal access option rather than inside every OD pair.
2. Valid turn transitions and their penalties are computed once per city
   topology.
3. A worker reuses immutable graph topology, node/edge mappings, distance
   adjacency, and turn-transition structure across family road states from the
   same city. Only family-dependent edge times and the time-weighted adjacency
   are rebuilt.
4. Destination-option evaluation for the turn-aware fastest-time matrix is
   vectorized while preserving lexicographic tie rules and option witnesses.
5. CLEs are loaded one city at a time in the parent runner rather than retaining
   all eleven cities simultaneously.

The generator does not replace exact routing with Euclidean distance, remove
turn penalties, approximate the directed graph, or reuse a time matrix across
different family road states.

## 3. Local multiprocessing

`scripts/build_stage2_instances.py` uses the `spawn` start method. A worker task
contains a single-city chunk, so topology reuse occurs inside each process.

```bash
PYTHONPATH=src python scripts/build_stage2_instances.py \
  --config configs/cle_evrptw_stage2_v1.json \
  --profile configs/us_reference_instance_profile_v1.json \
  --cle-root ../EVRPTW_Dataset/CLE_v1/us_11city \
  --block-group-preset configs/us_census_block_groups_v1.json \
  --block-group-source-dir data/sources/census_block_groups_2025 \
  --output-root <output-root> \
  --mode <official-or-non_release_pilot> \
  --workers 2 \
  --families-per-worker-task 25
```

The process queue keeps at most `2 * workers` materialization chunks and
`4 * workers` verification tasks in flight. This prevents the full family/view
plan from being copied into the multiprocessing queue.

The default is one worker. The runner estimates a safe upper bound from
physical memory using 5 GiB per routing worker plus a 4 GiB reserve. Requests
above that bound fail before generation. The override flag exists for measured
server configurations but should not be used merely to obtain more processes:

```text
--allow-memory-oversubscription
```

## 4. Multi-server sharding

Create the complete plan and customer splits once:

```bash
PYTHONPATH=src python scripts/build_stage2_instances.py \
  --config configs/cle_evrptw_stage2_v1.json \
  --profile configs/us_reference_instance_profile_v1.json \
  --cle-root ../EVRPTW_Dataset/CLE_v1/us_11city \
  --block-group-preset configs/us_census_block_groups_v1.json \
  --block-group-source-dir data/sources/census_block_groups_2025 \
  --output-root <shared-output-root> \
  --mode official \
  --stages preflight splits plan
```

Then assign each independent runner a zero-based shard. For five servers, use
the same command and output root on every server, changing only
`--shard-index` from 0 through 4:

```bash
PYTHONPATH=src python scripts/build_stage2_instances.py \
  --config configs/cle_evrptw_stage2_v1.json \
  --profile configs/us_reference_instance_profile_v1.json \
  --cle-root ../EVRPTW_Dataset/CLE_v1/us_11city \
  --block-group-preset configs/us_census_block_groups_v1.json \
  --block-group-source-dir data/sources/census_block_groups_2025 \
  --output-root <shared-output-root> \
  --mode official \
  --stages preflight materialize verify \
  --workers <safe-workers-on-this-server> \
  --families-per-worker-task 25 \
  --shard-count 5 \
  --shard-index <0-to-4>
```

Sharding is a deterministic stride over the sorted selected-family plan. Each
family belongs to exactly one shard. Reports are written separately as
`stage2_run_report.shard-XXX-of-YYY.json`, while family directories and
rejection ledgers remain uniquely keyed by `family_id`. A shared filesystem
must provide normal atomic rename semantics. Without shared storage, copy the
completed family directories and shard reports into one release tree before a
final whole-plan verification.

## 5. Measured results

Measurements below used New York, the frozen Stage-2 profile, an Apple M4
10-core system with 16 GiB RAM, and exact seeds shared with the earlier pilot.
They are engineering timings, not a cross-city runtime guarantee.

| Workload | Earlier time | Optimized time | Change |
| --- | ---: | ---: | ---: |
| Two Cus1000 families, serial materialization | 87.86 s | 45.20 s | -48.6% |
| Four Cus1000 families, serial versus final two-worker run | 121.80 s | 64.14 s | -47.3% wall time |
| One Cus2000 family, materialization | 125.27 s | 44.87 s | -64.2% |
| Resume four completed Cus1000 families | n/a | 5.87 s | verify and reuse |

The final two-worker Cus1000 materialization times were 21.32-31.22 s per
family while the processes competed for CPU and memory bandwidth. Individual
worker peak RSS was at most 2.18 GB in that run. The conservative 5 GiB worker
budget intentionally exceeds this observed value.

Correctness comparisons used values, shapes, dtypes, and view contents rather
than file hashes:

- four Cus1000 families: 1,996 arrays and four terminal tables were exactly
  equal between serial and optimized parallel outputs;
- one Cus2000 family: 19 arrays and one terminal table were exactly equal to
  the earlier pilot; and
- all materialized-family verifiers passed with no unresolved family.

## 6. Production estimate

The frozen plan has 7,000 Cus1000 parent families and 100 Cus2000 families.
Using the measured steady-state New York throughput, two local workers imply
approximately 22-30 hours including family verification. The range allows for
startup, city changes, graph-size variation, and storage overhead. It replaces
the earlier roughly 100-hour sequential estimate, but it is not a promise that
all cities have New York's routing cost.

Additional workers or shards reduce compute time only while memory bandwidth,
CPU contention, and filesystem throughput remain below saturation. Run a
small exact-seed pilot on each server class before choosing its worker count.
