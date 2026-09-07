# DRL-TS

This directory contains a paper-guided adaptation of Chen et al., *Deep
Reinforcement Learning with Two-Stage Training Strategy for Practical Electric
Vehicle Routing Problem with Time Windows* (PPSN 2022), DOI:
<https://doi.org/10.1007/978-3-031-14714-2_25>.

The full chapter supplied by the user was audited against the implementation.
The paper says source code is available on request, and no public author
repository was verified. This is not presented as official code or a numerical
reproduction.

## Method boundary

The implementation follows the paper's edge-aware GAT, simultaneous node/edge
updates, GRU/attention decoder, nearest-neighbor edge feature, two-stage
soft/hard training, violation terms, REINFORCE, and greedy rollout baseline.
EVRPTW-DB adds real directed-road distance/time/energy, service duration,
station power, normalized physical units, and independent verification.

The paper permits repeated station visits but masks station selection directly
from the depot or another station. Both training stages and evaluation enforce
that source-state rule; there is no charging-station visit penalty. Hard-stage
FFP is configured with `allow_consecutive_station_actions=False`, so its return
witness matches the paper mask: direct depot, or at most
`customer -> one station -> depot`. It cannot admit a customer on the strength
of a multi-station return path whose second charging action the paper mask
would subsequently remove. The shared/TERRAN default remains `True`.

See [ADAPTATION.md](ADAPTATION.md) for the equation-level correspondence and
explicit deviations, and
[../CHARGING_ADAPTER_CONTRACT.md](../CHARGING_ADAPTER_CONTRACT.md) for shared
physical semantics.

## Formal reward contracts

Formal economic-track runs keep two independently versioned layers. The shared
[`drl_energy_vehicle_reference_scale_v2`](../configs/drl_reward_contract_energy_vehicle_v2.json)
task contract supplies normalized electricity-plus-vehicle cost and the
terminal incomplete-rollout term. The DRL-TS method profile supplies only the
Stage-1 capacity, time-window, and energy auxiliary:

```text
v_bar_j = min(component_clip,
              sum_t min(normalized_raw_excess_j,t, step_clip) / N)
```

`N` is the fixed customer count. Capacity applies only to customer arrivals;
time and energy apply to all valid travel transitions. The frozen
[`drl_ts_soft_auxiliary_v1`](../configs/drl_ts_soft_auxiliary_v1.json) profile
uses per-step and per-component clips of 1 and unit weights. Raw, unclipped
violation sums still determine feasibility and remain in diagnostics; action
counts never replace `N` as the denominator.

A rollout that completes the soft stage after a resource violation pays the
bounded auxiliary but not the shared hard failure floor. An incomplete rollout
pays that terminal term exactly once. The frozen empirical calibration
candidate `Q_0.99(C_ref / S_N) + 1` is not a mathematical feasibility-first
guarantee, and its being above every normalized cost in the 500 calibration
references is only an in-sample observation. Formal manifests, checkpoints,
resume checks, and diagnostics record the shared task contract and the separate
method profile. See [ADAPTATION.md](ADAPTATION.md) for the full definitions and
applicability rules.

## Train

Run from the repository root. A training pool must contain one fixed scale and
therefore one fixed terminal count.

```bash
PYTHONPATH=EVRPTW_Core:EVRPTW_Dataset_Generator/src \
python -m EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.train \
  --dataset-path EVRPTW_Dataset/Instances_v2/us_11city \
  --scale Cus100 \
  --split-ids train \
  --track-ids train \
  --output-dir EVRPTW_Benchmark/results/DRL_TS/Cus100/seed1234
```

The standalone CLI defaults reproduce the paper's 200 epochs, 250 batches per
epoch, 0.5 soft-stage fraction, 128-dimensional embeddings, two encoder layers,
eight heads, ten nearest neighbors, unit violation weights, and Adam at
`1e-4`. RQ launchers may override the compute schedule; all resolved settings
are recorded in checkpoints and manifests.

## Evaluate

```bash
PYTHONPATH=EVRPTW_Core:EVRPTW_Dataset_Generator/src \
python -m EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.eval \
  --dataset-path EVRPTW_Dataset/Instances_v2/us_11city \
  --checkpoint EVRPTW_Benchmark/results/DRL_TS/Cus100/seed1234/checkpoint_latest.pt \
  --scale Cus100 \
  --split-ids test \
  --track-ids test1_new_seed \
  --decode-type greedy \
  --candidates 1 \
  --output-dir EVRPTW_Benchmark/results/DRL_TS/Cus100/test1
```

The paper compares greedy decoding and the best of 1,280 sampled solutions.
Use `--decode-type sampling --candidates 1280` when that registered protocol
and memory budget are intended. Candidate selection prefers completed rollouts
and then minimizes distance. Exported routes are independently replayed before
their directed-road distance is accepted.
