# Training reward-scale diagnostics

These are observational logs for AM-EVRPTW, EVRPTW-RL, DRL-TS and TERRAN.
They do not change the objective, normalization, masks, auxiliary penalties,
curriculum, sampling stream, optimizer, gradient clipping or checkpoint ranking.
They are not a new acceptance gate or a convergence guarantee.

## Output and frequency

Each formal training run writes `reward_diagnostics.jsonl` in its existing
output directory. Existing training histories and CSV columns are unchanged.
The three REINFORCE methods write after each completed logical optimizer update;
TERRAN writes after each completed training epoch, which may contain several
PPO optimizer updates. Physical microbatches are combined for the corresponding
logical update/epoch. There is no per-action disk write or additional decoding.

Rows use `schema = drl_reward_diagnostics_v1`. They identify the method,
training position and logging session. A resumed run can repeat epochs newer
than its restored checkpoint; retain session/resume metadata when interpreting
the append-only history. A logged update is not a promise that its checkpoint
has already been saved. Historical runs do not gain these logs retroactively.

## Normalization provenance

Record the active objective and units, the distance normalization mode and
training-pool calibration metadata, and the resulting objective denominator
`reward_objective_scale`. For the economic track:

```text
C = distance_unit_cost * distance_km + vehicle_unit_cost * vehicles_started
S = distance_unit_cost * distance_reference_km + vehicle_unit_cost * K_reference
```

AM/EVRPTW-RL/DRL-TS retain a training-pool median singleton round-trip distance
with `K_reference = 1`. TERRAN retains a training-pool mean singleton-repair sum
with `K_reference = N`. No normalizer is fitted using validation/test outcomes.
These scales are not necessarily equal between methods or training conditions.
Normalization does not imply that rewards, returns or costs lie in `[0, 1]`.

## What the observations mean

| Family | Training quantity | Additional observations |
|---|---|---|
| AM-EVRPTW | Positive trajectory cost `(C + converted incomplete penalty) / S` | Incomplete penalty in training units; actual actor/baseline costs and cost advantage |
| EVRPTW-RL | Positive trajectory cost `C / S + station penalty + incomplete penalty` | Each auxiliary penalty separately; actual actor/baseline costs and cost advantage |
| DRL-TS | Positive trajectory cost `C / S + resource penalties + incomplete penalty` | Capacity/time/energy and incomplete penalties separately; actual actor/baseline costs and cost advantage |
| TERRAN | Negative step cost `-delta_C / S` plus native shaping/terminal terms | Active-step base/shaped rewards, return-to-go and initial trajectory return, advantages before/after PPO normalization, reward-component totals |

The REINFORCE cost advantage is `actor_cost - baseline_cost`: positive means
worse than the baseline. TERRAN's return advantage is `return - value`:
positive means better than the value estimate. Their signs and granularities
must not be compared as if they were the same metric. An EMA baseline is the
scalar actually used in that update, not another decoded solution.

Distance/vehicle objective components decompose the base cost; the base cost
must not be added to its own decomposition a second time. Legacy distance-mode
values must not be relabelled as electricity dollars. Auxiliary penalties and
PBRS terms have training units, not raw USD. Shaped returns are not the
benchmark's evaluation objective. Truncated trajectory returns remain partial
training returns, not verified complete-solution costs.

## Distributions and gradients

Distribution summaries contain active count, finite/nonfinite count, mean,
population standard deviation, extrema, P05/P50/P95 and quantile sample count.
Padding and post-termination positions are excluded with the rollout validity
mask; active terminal transitions remain included. Finite statistics are
reported separately from nonfinite observations. Empty finite populations use
JSON `null`, not a fabricated zero or nonstandard JSON NaN.

Means, standard deviations and extrema use the full eligible finite population.
Quantiles are exact when all observations fit the sample budget; otherwise
they are approximate diagnostics on a bounded deterministic sample. Read each
summary's sampling-method and sample-count fields. No training RNG is consumed.
These approximate quantiles are not publication-level population estimates.

Gradient statistics use the pre-clipping norm returned by the existing clipping
call. Clipping frequency is the fraction, among completed optimizer updates
with finite norms, exceeding the configured clipping threshold. Report
nonfinite norms separately. Microbatch accumulation is not counted as multiple
optimizer updates, and gradients are neither recomputed nor clipped again.

## Reading an early training run

Inspect raw USD cost, distance, vehicle count and verified validation feasibility
together with these diagnostics. A large reward or loss is not alone evidence
of a bug. Look for nonfinite values, unexpectedly dominant auxiliary penalties,
near-zero/very noisy advantages, and persistent heavy gradient clipping.
Separate curriculum stages and warmup periods. A short numerically healthy run
does not establish convergence or justify adjusting parameters on test results.

GPU summaries are detached and performed at logging boundaries. Large tensor
statistics use bounded chunks, and no full reward trajectory is exported to
disk. `diagnostics_compute_wall_time_s` measures summary construction, not
the full instrumentation or disk-write overhead. TERRAN preserves its existing
CSV epoch timer, which is captured before the new epoch-end diagnostic work;
total run wall time still includes this work. Diagnostic overhead still needs
to be measured on the actual server; local CPU tests do not establish GPU
throughput.

## Local verification

The logging-only revision based on `1f62ef3` passed **440 CPU tests**, with
**one CUDA-specific test skipped** on the macOS host. The existing exclusions
are vendored reference implementations and the Linux server-environment tests
(which require `flock` and GNU `realpath -m`). The final suite took 10.80 s.

New checks cover hand-computed masked distributions and cost decomposition,
empty/nonfinite populations, bounded statistics, actual pre-clipping norms,
both PPO accumulation paths, all three REINFORCE methods in fixed-budget and
data-pass modes, and append-only resume-session semantics. Enabling/disabling
diagnostics preserves the tested actions/rewards, model and optimizer updates,
rollout counts and RNG states. No server training or GPU throughput run was
launched by this change.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  EVRPTW_Benchmark/Reinforcement_Learning \
  --ignore=EVRPTW_Benchmark/Reinforcement_Learning/reference_materials \
  --ignore=EVRPTW_Benchmark/Reinforcement_Learning/tests/test_rq_server_environment.py \
  -q -p no:cacheprovider
```
