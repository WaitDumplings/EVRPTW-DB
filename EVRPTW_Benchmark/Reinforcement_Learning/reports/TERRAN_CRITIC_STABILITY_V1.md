# TERRAN critic stability, 2026-09-09

This change addresses the observed Cus1000 instability without changing the
economic objective, gamma, PBRS, or terminal success/failure rewards. The
previous active-transition weighting correction remains in place.

## Training settings

The canonical Stage-2 TERRAN configuration is shared by Cus50, Cus100, Cus500
and Cus1000. Future launches use:

| Setting | Previous | New |
| --- | --- | --- |
| Value loss | MSE | Smooth-L1, beta 1 |
| Fixed residual divisor | 1 | 1 |
| Value-loss weight | 0.5 | 0.1 |
| Critic gradient multiplier into shared backbone | 1 | 0.1 |
| Gamma | 1 | 1 |

Value predictions, Monte-Carlo returns, and advantages retain the same
normalized economic-reward units. Smooth-L1 is a different regression loss,
not an equivalent rewrite of MSE. It limits the derivative with respect to
the residual, not the full network parameter gradient.

The gradient multiplier is an identity in the forward pass. It scales only
the critic-to-backbone backward path; critic-head gradients still receive
the configured value-loss weight, and the policy gradient is unaffected.
The actor is a parameter-free logits selector, so policy parameter gradients
are measured on the shared backbone rather than a nonexistent actor head.

## Diagnostics

The reward diagnostics include raw MSE alongside the configured value loss,
PPO probability-ratio/approximate-KL/clipping statistics, and value targets,
predictions and residuals grouped by rollout outcome. On the first epoch and
every 25 epochs, the first PPO optimizer update additionally measures policy
and weighted-value gradients on shared parameters, their cosine, and the
weighted-value gradient on the critic head. Gradient vectors are accumulated
across time chunks using the normal valid-transition weights. Measuring them
does not modify the optimizer gradients or parameters.

These observations distinguish scalar loss magnitude from actual parameter
gradient contributions. They do not establish that training has recovered;
that requires subsequent validation results.

## Frozen-rollout diagnostic

The September 9 diagnostic used the old Cus1000 epoch-800 checkpoint, the
registered training instances for epoch 782, batch 4, 50 trajectories, horizon
1250 and time chunks of 624. Both variants used the same 250000 transitions
without any optimizer step. The raw residual MSE was identical (93.152308).

| Measured gradient | Previous MSE configuration | New configuration |
| --- | ---: | ---: |
| Policy on shared backbone | 0.155585 | 0.155585 |
| Weighted value on shared backbone | 0.903633 | 0.001920 |
| Weighted value on critic head | 169.465800 | 1.915465 |
| Global norm before clipping | 169.468671 | 1.963075 |

This sample directly shows that critic-head gradients dominated the global
clipping norm, while the value contribution to the shared backbone exceeded
the policy contribution. It does not determine the initial historical trigger
or prove recovery of validation quality. Both passes used about 36.79 GiB of
peak allocated GPU memory; allocator reservation is a separate measurement.
The reproducible diagnostic entrypoint is `TERRAN.benchmark_critic_stability`.

## Relaunch

`scripts/retrain_terran_critic.py` reuses a previous fresh TERRAN run's launch
arguments and registered training stream, with the updated canonical config
and a new output directory. It starts from random initialization with a new
AdamW optimizer. It does not restart other scales automatically.

Example, run with the project's Python environment from the repository root:

```bash
python -m EVRPTW_Benchmark.Reinforcement_Learning.scripts.retrain_terran_critic \
  --source-run /absolute/path/to/previous/Cus1000/run \
  --output-dir /absolute/path/to/new/Cus1000/run \
  --gpu 1
```

Cus1000 retains batch 4, 50 sampled training trajectories, rollout horizon
1250, 100 validation candidates and horizon 1875. Validation runs every 100
epochs. Maximum training remains 10000 epochs, with early stopping starting
at epoch 5000 and patience of five validations.

The currently running Cus500 process retains its already loaded configuration
and continues independently. The shared update applies when future jobs start.
