# EVRPTW-RL

Paper-guided reimplementation of the native single-stage EVRPTW policy by Lin,
Ghaddar, and Nathwani.  Read [`ADAPTATION.md`](ADAPTATION.md) before reporting
results; this is not author-released code.

## Train

```bash
python -m EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL.train \
  --dataset-path EVRPTW_Dataset/Instances_v2/RELEASE_ROOT/generation_plan \
  --scale Cus100 \
  --split-ids train \
  --track-ids train \
  --output-dir EVRPTW_Benchmark/results/EVRPTW_RL/Cus100
```

The paper defaults are registered where they were reported: 10,000 updates,
batch size 128, 128-dimensional states, Adam at `1e-3`, clip norm 2.0, EMA
warmup 1,000, and rollout-baseline checks every 100 updates.  The unpublished
Structure2Vec recursion count is an explicit run parameter rather than a hidden
claim about the original implementation.

## Evaluate

```bash
python -m EVRPTW_Benchmark.Reinforcement_Learning.EVRPTW_RL.eval \
  --dataset-path EVRPTW_Dataset/Instances_v2/RELEASE_ROOT/generation_plan \
  --checkpoint EVRPTW_Benchmark/results/EVRPTW_RL/Cus100/checkpoint_latest.pt \
  --scale Cus100 \
  --split-ids test \
  --track-ids test1_new_seed \
  --decode-type sampling \
  --candidates 100 \
  --output-dir EVRPTW_Benchmark/results/EVRPTW_RL/Cus100/test1
```

The paper reports greedy, stochastic-100, and beam-3 decoding.  This first
reimplementation supplies greedy and stochastic-`K`, which cover the canonical
benchmark protocol.  Beam search is not silently approximated; it remains an
explicitly documented unimplemented paper option.

Every selected route is replayed with the same independent verifier used for
the exact and metaheuristic baselines.  Training reward is not an evaluation
metric; the reported objective follows the frozen objective profile. The
Cus100 repair uses verified electricity plus vehicle fixed cost; its station
visit auxiliary is used only for training. Legacy distance profiles remain
available.

### Cus100 numerical stability repair

For the opt-in 2080ti_4_2 TR17/TR18 repair, use `--graph-aggregation mean`.
Legacy/default `sum` is preserved. The mean mode normalizes both neighbor and
edge aggregation by node degree to address the observed attention saturation;
it adds no learned layers. Its difference from the published sum equation is
recorded in [ADAPTATION.md](ADAPTATION.md).

Start a new experiment for the changed aggregation, and retain its explicit
configuration when evaluating its checkpoint. The repair deployment is
[`scripts/cus100_20260911/2080ti_4_2/evrptw_rl_stable.sh`](../scripts/cus100_20260911/2080ti_4_2/evrptw_rl_stable.sh).
It targets only TR17/TR18 in new output directories, first checking useful
numerical behavior and GPU memory before starting fresh formal training.
