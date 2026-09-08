# Training reward-scale diagnostics

These are observational logs for AM-EVRPTW, EVRPTW-RL, DRL-TS and TERRAN.
They do not change the objective, normalization, masks, auxiliary penalties,
curriculum, sampling stream, optimizer, gradient clipping or checkpoint ranking.
They are not a new acceptance gate or a convergence guarantee.

## Output and frequency

Each formal training run writes `reward_diagnostics.jsonl` in its existing
output directory. Existing training histories remain append-only; TERRAN also
records its terminal success component in dedicated CSV columns.
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
C = 0.151750972762646 * distance_km
    + 413.6331536717643 * vehicles_started
S_N = median(C_ref) on the frozen, verified training-reference cohort
L_task = C / S_N + I[incomplete] * (b_N + lambda_u * unserved_fraction)
```

The frozen `drl_energy_vehicle_reference_scale_v3` contract, calibrated for
`rivian_energy_vehicle_cost_v2`, supplies one
`S_N`, `b_N`, and `lambda_u` per calibrated scale. The same values are used by
all four methods and conditions at that scale; distance and vehicle count in
each reference cost come from the same independently replayed route set. No
normalizer is fitted using validation/test outcomes, and normalization does not
imply that rewards, returns or total training costs lie in `[0, 1]`.

TERRAN may additionally freeze a method-specific completion reward `b_success` in
the same normalized-cost unit:

```text
r_task = -C / S_N
         + I[success] * b_success
         - I[incomplete] * (b_N + lambda_u * unserved_fraction)
```

The success and failure terms are mutually exclusive, charged only on the
first terminal transition, and never PBRS-annealed. `terminal_success_bonus`,
`terminal_failure_base`, and `terminal_unserved` are logged separately. The
constant success term does not change the ordering among feasible solutions.

The floor rule `b_N = Q_0.99(C_ref / S_N) + 1` is an empirical calibration
candidate, not a mathematical feasibility-first guarantee. A quantile-based
floor need not exceed the largest member of its 500-reference calibration
cohort. Diagnostics must not report it as a global upper bound on feasible
solution cost.

## What the observations mean

| Family | Training quantity | Additional observations |
|---|---|---|
| AM-EVRPTW | Shared positive task cost `C / S_N + terminal task cost` | Task base/failure terms; actual actor/baseline costs and cost advantage |
| EVRPTW-RL | Shared positive task cost plus its separately versioned station auxiliary | Task and station-profile terms separately; actual actor/baseline costs and cost advantage |
| DRL-TS | Shared positive task cost plus the separately versioned bounded Stage-1 soft auxiliary | Raw/clipped capacity, time, and energy observations; task and soft-profile terms separately; actual actor/baseline costs and cost advantage |
| TERRAN | Equivalent negative task reward `-delta_C / S_N` plus terminal task reward and native PBRS | Active-step base/shaped rewards, return-to-go and initial trajectory return, advantages before/after PPO normalization, reward-component totals |

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

For DRL-TS Stage 1, each normalized raw transition excess `x_j,t` is retained,
while the training component is

```text
v_bar_j = min(component_clip,
              sum_{t in A_j} min(x_j,t, step_clip) / N)
soft_auxiliary_total = alpha * v_bar_capacity
                     + beta  * v_bar_time
                     + gamma * v_bar_energy
```

`N` is the fixed instance customer count. `A_capacity` contains customer
arrivals; `A_time` and `A_energy` contain every valid travel transition. The
frozen profile uses `step_clip=1`, `component_clip=1`, and unit weights, so each
component is in `[0, 1]`. The applicable-transition counters are diagnostic
only: dividing by them would let extra zero-violation travel dilute an incurred
violation and is therefore forbidden.

The `soft_*_raw_sum` fields are unclipped and remain the feasibility/audit
signal. The `soft_*_clipped_sum` fields show the numerator used for training,
and `soft_*_component_unweighted` shows the fixed-`N`, component-clipped value
before its method weight. A structurally complete soft rollout with a nonzero
raw sum is labelled `completed_with_soft_violation`; it receives its soft
auxiliary but no `terminal_failure_base` or `terminal_unserved` term. Every
incomplete rollout receives those common task terms once. This distinction is
why raw feasibility, structural completion, and bounded training auxiliary must
not be inferred from one another.

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

Reward-contract checks cover the frozen common scale/floor provenance, task
component identities, and the rule that a completed DRL-TS soft violation does
not pay the hard failure floor. DRL-TS-specific checks cover hand-computed
per-step clipping, fixed-customer normalization, component clipping, resistance
to zero-violation action dilution, raw feasibility, profile loading, and
checkpoint/provenance fields.

The observational logging checks continue to cover masked distributions and
cost decomposition, empty/nonfinite populations, bounded statistics, actual
pre-clipping norms, both PPO accumulation paths, all three REINFORCE methods,
and append-only resume-session semantics. Enabling/disabling diagnostics must
preserve actions/rewards, model and optimizer updates, rollout counts, and RNG
states. CPU tests do not establish GPU convergence or throughput.

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  EVRPTW_Benchmark/Reinforcement_Learning \
  --ignore=EVRPTW_Benchmark/Reinforcement_Learning/reference_materials \
  --ignore=EVRPTW_Benchmark/Reinforcement_Learning/tests/test_rq_server_environment.py \
  -q -p no:cacheprovider
```
