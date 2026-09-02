# AM-EVRPTW

`AM-EVRPTW` is the generic learning-architecture anchor in EVRPTW-B.  It adapts
the ICLR 2019 Attention Model to the shared directed-road EVRP-TW environment
while retaining REINFORCE with a greedy rollout baseline.

Read [`ADAPTATION.md`](ADAPTATION.md) before using results in a paper.  Shared
charging and feasibility semantics are defined in
[`../CHARGING_ADAPTER_CONTRACT.md`](../CHARGING_ADAPTER_CONTRACT.md).

## Train

Run from the repository root.  `--dataset-path` may be an individual
`view_index.parquet` or a dataset root containing view indices.  Use the exact
split and track IDs present in the release index.

```bash
python -m EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.train \
  --dataset-path EVRPTW_Dataset/Instances_v2/RELEASE_ROOT/generation_plan \
  --scale Cus100 \
  --split-ids train \
  --output-dir EVRPTW_Benchmark/results/AM_EVRPTW/Cus100
```

The defaults follow the upstream AM architecture (`d=128`, three encoder
layers, eight heads, tanh clipping 10, Adam at `1e-4`).  Training-budget values
must be frozen in the experiment configuration rather than inferred from these
examples.

## Evaluate

```bash
python -m EVRPTW_Benchmark.Reinforcement_Learning.AM_EVRPTW.eval \
  --dataset-path EVRPTW_Dataset/Instances_v2/RELEASE_ROOT/generation_plan \
  --checkpoint EVRPTW_Benchmark/results/AM_EVRPTW/Cus100/checkpoint_latest.pt \
  --scale Cus100 \
  --split-ids test \
  --track-ids test1_new_seed \
  --decode-type greedy \
  --output-dir EVRPTW_Benchmark/results/AM_EVRPTW/Cus100/test1
```

For a frozen sampling budget, use `--decode-type sampling --candidates K`.
Every selected route is independently replayed by the shared route verifier;
`summary.csv` reports verifier status and `routes.jsonl` retains route-level
evidence.

## Status

The architecture adapter and canonical-environment smoke tests are implemented.
Large-scale training throughput and release-table reproduction remain separate
experimental gates; code availability is not evidence that those experiments
have completed.
