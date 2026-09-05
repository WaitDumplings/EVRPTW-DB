# TERRAN Cus1000 Reward-Scale Audit V1

Date: 2026-09-05

## Scope

This audit covers the formal Cus1000 TERRAN configuration on the RTX A6000,
including the distance objective, customer and repair-distance PBRS, terminal
heuristics, discount horizon, and dead-end penalty.  All before/after replay
comparisons used the same checkpoint, seed, instances, and sampled action
sequence.

## Failure in the previous formal run

The previous training-pool distance scale was the median one-customer repair
distance:

```text
reward_distance_scale_km = 40.757239
```

That made a typical customer edge about `-0.40` and a complete 18,000 km route
about `-442`, while the nominal customer and repair PBRS budgets were only
`0.5 + 0.5`.  After epoch 300 those auxiliary terms and the terminal heuristic
were all multiplied by `0.2`.

The formal log confirms that this was not merely a small-signal problem.  PPO
was numerically rewarded for ending early:

| Phase | Mean trajectory length | Mean undiscounted trajectory return |
|---|---:|---:|
| Epoch 1 | 525 | -151.74 |
| Epoch 500 collapse | 28.9 | -18.17 |
| Epoch 1000 collapse | 20.4 | -18.15 |
| Epoch 5000 recovery | 354 | -158.11 |
| Epoch 5200 recovery | 658 | -276.13 |

The old `gamma=0.99` also had an effective horizon of roughly 100 transitions;
`0.99^1200 = 5.78e-6`.  With the positive-progress potential, 899 of the 1000
customer-service transitions had a negative customer-PBRS value.  Terminal
potential was nonzero, and terminal success/failure was incorrectly annealed
together with auxiliary PBRS.

## Implemented calibration

The repair-sum scale is computed deterministically from the frozen training
pool and is proportional to problem size:

```text
reward_distance_scale_mode = dataset_single_customer_repair_sum
resolved scale             = 43160.153004 km
gamma                      = 0.999
invalid/dead-end penalty   = -1.0
PBRS annealing             = cosine 1.0 -> 0.2 over epochs 1..5000
```

Customer and repair potentials now use negative remaining work.  Their initial
combined potential is `-1`, every success or truncation uses terminal potential
zero, and the trainer uses the same gamma as PBRS.  Consequently their total
discounted contribution is a policy-independent constant within an epoch.
Terminal success/failure remains separate from PBRS and is no longer annealed.

The reward log now separates:

- normalized distance;
- non-distance base penalties;
- customer, repair-distance, and feasible-ratio PBRS;
- terminal heuristic;
- pure PBRS versus all shaping;
- per-transition absolute ratios;
- discounted and undiscounted per-trajectory totals;
- customer versus non-customer action means.

## Fixed-action replay

Epoch-250 checkpoint replay used 107,712 active transitions and completed all
100 trajectories.  Both old and new calculations had action checksum
`30174162772`.

Under the old configuration, mean absolute PBRS was only about 1.6% of the base
reward and its complete discounted trajectory contribution was about `2.0e-5`.
Under the calibrated configuration, a customer action had mean distance reward
`-2.38e-4` and mean customer-plus-repair PBRS `+1.54e-3`.  The discounted pure
PBRS trajectory total was exactly 1 up to floating-point tolerance.

## Real Cus1000 smoke

One full training epoch used the formal physical batch, trajectory count,
rollout horizon, and PPO chunk:

```text
physical environments = 2
n_traj               = 100
rollout steps        = 1200
PPO chunk            = 736
PPO update epochs    = 3
epoch wall time      = 18.385 s
observed GPU memory  = 42319 MiB
```

Key epoch-1 diagnostics:

```text
mean distance / transition                 = -0.000255506
mean non-distance base / transition        = -0.001466248
mean pure PBRS / transition                = +0.002491517
mean terminal heuristic / transition       = -0.000463677
undiscounted distance / trajectory         = -0.134179
undiscounted pure PBRS / trajectory        = +1.308420
discounted distance / trajectory           = -0.087270
discounted pure PBRS / trajectory           = +0.999997
discounted terminal heuristic / trajectory = -0.228475
train feasible rate                        = 0.225
mean served customers                      = 468.0
```

The last two values match the previous same-seed epoch-1 rollout, as expected:
reward changes cannot affect actions until the first optimizer update.

## Verification

The TERRAN test suite passes (`33 passed`).  Tests cover terminal telescoping,
post-terminal masking, non-annealed terminal objectives, distance/penalty
decomposition, reward aggregation identities, the 5000-epoch annealing boundary,
and the formal YAML calibration.
