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
that rule; there is no charging-station visit penalty.

See [ADAPTATION.md](ADAPTATION.md) for the equation-level correspondence and
explicit deviations, and
[../CHARGING_ADAPTER_CONTRACT.md](../CHARGING_ADAPTER_CONTRACT.md) for shared
physical semantics.

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
