# DRL-TS

This directory contains a paper-guided reimplementation of Chen et al.,
*Deep Reinforcement Learning with Two-Stage Training Strategy for Practical
Electric Vehicle Routing Problem with Time Windows* (PPSN 2022), DOI:
<https://doi.org/10.1007/978-3-031-14714-2_25>.

The publisher page states that source code is available on request; no public
author repository was verified for this benchmark. This is therefore not
presented as official code or as a numerical reproduction of the paper.

## Method boundary

The implementation retains the paper's edge-aware graph-attention encoder,
GRU/attention decoder, nearest-neighbor edge feature, two-stage training
curriculum, violation penalties, REINFORCE training, and greedy rollout
baseline. Its objective-facing term is total directed-road distance. Stage 1
uses the paper's soft capacity, time-window, and energy constraints; Stage 2
and all evaluation use the shared canonical hard mask.

See [ADAPTATION.md](ADAPTATION.md) for exact correspondence and deviations, and
[../CHARGING_ADAPTER_CONTRACT.md](../CHARGING_ADAPTER_CONTRACT.md) for the
shared physical semantics.

## Train

Run from the repository root. A training pool must contain one fixed scale and
therefore one fixed terminal count.

```bash
PYTHONPATH=EVRPTW_Core:EVRPTW_Dataset_Generator/src \
python -m EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.train \
  --dataset-path EVRPTW_Dataset/Instances_v2/us_10city_release \
  --scale Cus100 \
  --split-ids train \
  --track-ids train \
  --output-dir EVRPTW_Benchmark/results/DRL_TS/Cus100/seed1234
```

The paper's defaults are registered by the CLI: 200 epochs, a 0.5 soft-stage
fraction, 250 batches per epoch, 128-dimensional embeddings, two encoder
layers, eight heads, ten nearest neighbors, and Adam at `1e-4`. Every command
line is stored in the checkpoint.

## Evaluate

```bash
PYTHONPATH=EVRPTW_Core:EVRPTW_Dataset_Generator/src \
python -m EVRPTW_Benchmark.Reinforcement_Learning.DRL_TS.eval \
  --dataset-path EVRPTW_Dataset/Instances_v2/us_10city_release \
  --checkpoint EVRPTW_Benchmark/results/DRL_TS/Cus100/seed1234/checkpoint_latest.pt \
  --scale Cus100 \
  --split-ids test \
  --track-ids test1_new_seed \
  --decode-type greedy \
  --candidates 1 \
  --output-dir EVRPTW_Benchmark/results/DRL_TS/Cus100/test1
```

For the paper's sampling protocol, use `--decode-type sampling --candidates
1280` when memory permits. Candidate selection prefers a complete environment
rollout and then minimizes distance. Exported routes are independently replayed
by the shared verifier before their distance is reported.
