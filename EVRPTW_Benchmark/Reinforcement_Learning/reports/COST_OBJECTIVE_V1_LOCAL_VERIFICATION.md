# Cost-objective v1: local integration verification

## Change under test

All four formal learning methods use `rivian_energy_vehicle_cost_v1` for fresh
training, validation/checkpoint selection and evaluation. The coefficients are
`0.1341 USD/kWh`, `100/257 kWh/km` and `33.56 USD/dispatch`. The implementation
starts from the preceding `2ec0b1c` undiscounted-TERRAN/cache revision; no model
architecture, dataset, sampling stream, training budget or feasibility mask is
changed by the objective revision.

## Checks completed

| Check | Result |
|---|---|
| Local DRL CPU regression suite, exclusions below | 409 passed, 10.38 s |
| RQ shell syntax checks (`bash -n`) | Passed |
| Diff whitespace/error check | Passed |
| Formal jobs bound to the same objective dictionary and JSON path | 24 / 24 |
| Generated manifest comparison against prior commit | 28 records unchanged outside objective fields; includes four Cus1000 scheduling projections |
| New GPU/server training launched by this patch | None |

The automated tests cover departure-fee accounting, return and padding behavior,
open/failed and charger-first trips, cost-versus-distance candidate ordering,
independent verification, cost-aware checkpoint selection, legacy distance
behavior, exact objective snapshots and cross-objective resume rejection.
Small real-policy gradient tests cover AM, EVRPTW-RL and DRL-TS hard/soft training;
TERRAN tests cover returns, PBRS, optimizer-update cache boundaries and cost
selection. Launcher tests include cost-config forwarding to both train and
eval/transfer commands, stale manifest rejection and completed-result provenance.

Reproduction of the local suite:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  EVRPTW_Benchmark/Reinforcement_Learning \
  --ignore=EVRPTW_Benchmark/Reinforcement_Learning/reference_materials \
  --ignore=EVRPTW_Benchmark/Reinforcement_Learning/tests/test_rq_server_environment.py \
  -q -p no:cacheprovider
```

The macOS host cannot validate the four Linux environment-wrapper tests that
depend on `flock` and GNU `realpath -m`. Vendored upstream reference material is
also excluded. Passing CPU tests is not evidence of GPU convergence, full-data
throughput, or a successful end-to-end Linux launch. No SHA256/file-hash
verification was added or performed.

## Server handoff

After stopping any older queue, pull the new `drl-benchmark-adapters` commit.
Use the corresponding `scripts/rq_v1/<server>/full.sh --seed 1234` without the
former `--methods terran` restriction. The new Git-commit output directory
separates all four models from historical distance training. Do not copy or load
old checkpoints. Shared v13 streams and the dataset need not be regenerated.

The four server bundles are `2080ti_4_1`, `2080ti_4_2`, `2080ti_3_1`, and
`a6000_2_1`. The separate Cus1000 priority launcher is an alternative to that
server's full queue, not an additional concurrent queue. Use `resume.sh` only
for an interrupted run of this same commit and exact objective.

See [the objective contract](../COST_OBJECTIVE_CONTRACT_V1.md) and
[the active runbook](../scripts/rq_v1/README.md) for accounting and launch details.
