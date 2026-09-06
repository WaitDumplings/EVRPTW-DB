# Local verification: no consecutive CS visits

Action contract: `drl_no_consecutive_cs_v1`.
Base revision: `eb796d26906ae82cfada71e9515836299b067413` (cost-only revision).

## Implemented scope

AM-EVRPTW, EVRPTW-RL, DRL-TS and TERRAN now prohibit consecutive selected
charging-station terminals in training, validation and test. The shared slow,
Fast NumPy and JIT masks agree. The DRL route selector separately reports
physical verification and the new policy check. Customer-interleaved charging
is not prohibited by this rule; bounded return lookahead is conservative, as
specified in [the action contract](../ACTION_CONSTRAINT_CONTRACT_V1.md).

Configuration, all checkpoint producers/consumers, training histories,
validation summaries and evaluation exports carry the action-contract ID.
Missing, different or conflicting checkpoint IDs are rejected before reuse,
including checkpoints from the preceding electricity-plus-vehicle-cost revision.

## Evidence

- Full local DRL regression suite: **523 passed in 10.66 seconds**.
- Includes slow/Fast/JIT action and transition equivalence, customer-bridge
  continuations, forbidden-action accounting, and station-history constraints.
- Includes physical-feasible but policy-invalid exported routes, candidate
  selection, and all four methods' checkpoint/resume/evaluation compatibility.
- Includes small real training updates for the three REINFORCE adapters;
  TERRAN's production checkpoint and resume/evaluation paths are regression-tested.
- Additional read-only randomized check: 20 toy instances, four trajectories
  each, 53 batch transitions; slow/Fast/JIT masks, rewards, states and routes
  agreed, with no exported CS-to-CS arcs.
- Parsed comparison against the base revision: all 28 manifest records
  (24 formal jobs plus four Cus1000 projection records) and all three formal
  YAML configs differ only by the added action-contract field. Economic
  coefficients, seeds, schedules, physical batches and exposure budgets are
  unchanged.
- RQ shell syntax and `git diff --check`: passed.

Command, from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest EVRPTW_Benchmark/Reinforcement_Learning \
  --ignore=EVRPTW_Benchmark/Reinforcement_Learning/reference_materials \
  --ignore=EVRPTW_Benchmark/Reinforcement_Learning/tests/test_rq_server_environment.py \
  -q -p no:cacheprovider
```

The excluded server-environment file requires Linux utilities unavailable on
this macOS host. No remote server queue, GPU training, convergence evaluation,
dataset regeneration or file-hash validation was run. Gurobi, ALNS, VNS-TS and
the shared physical verifier remain unchanged. CPU regressions do not establish
large-scale GPU performance or training quality.

## Restart

Pull the new revision and use the existing server-specific `full.sh` with a
fresh commit-scoped output root, without an old checkpoint. Dataset and shared
ID streams do not need rebuilding. See the
[server runbook](../scripts/rq_v1/README.md). Report the action restriction
explicitly when comparing DRL with the unchanged classical solvers.
