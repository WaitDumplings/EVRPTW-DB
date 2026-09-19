# Legacy PPO synchronous multi-GPU entry

`TERRAN.train_distributed` uses the same `Agent` (legacy critic), PPO loss,
Monte Carlo returns, PBRS schedule, critic gradient scaling, objective mapping,
and independent validation code as `TERRAN.train`. It does not enable
`stable_cost_v1`, remaining-customer context, or warm-start weights.
The original single-GPU `TERRAN.train` training path is unchanged.

## Invocation

Use the ordinary TERRAN CLI with this module under torchrun:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m EVRPTW_Benchmark.Reinforcement_Learning.TERRAN.train_distributed \
  --expected-world-size 2 \
  --config <config.yaml> --seed 1234 --device cuda \
  --num-envs-per-gpu 16 --physical-batch-size 16 --effective-batch-size 32 \
  --training-epochs 300 --early-stop-patience-validations 0 \
  <the same dataset, objective, rollout, validation, and output arguments>
```

The ablation launcher constructs complete commands and chooses the GPU count.
`--physical-batch-size` is instances per rank; `--effective-batch-size` is the
GLOBAL instance batch, without multiplying by trajectories. Effective batch
must divide exactly by physical batch times world size. Larger effective
batches accumulate multiple local rollout buffers before a PPO update.
`--instance-cache-size` bounds each process's CPU instance cache, default 256.

This entry intentionally accepts only **fresh fixed-epoch Stage-2 runs**.
Resume, warm start, data-pass budgets, `stable_cost_v1`, and nonzero early-stop
patience fail explicitly. Existing outputs with training evidence are rejected;
use a new output directory for a new run. Checkpoints remain compatible with
the ordinary evaluation entry points.

## Synchronization contract

- Each process receives disjoint contiguous physical batches from the same
  seeded global shuffle stream (or frozen registered view-ID stream). Workers
  do not load instances assigned to another rank.
- Every worker initializes the same model. Trajectory random streams then
  differ by rank. Topology is part of the saved training signature; stochastic
  runs across different GPU counts are not claimed to be bit-identical.
- Advantage mean and population standard deviation use all valid transitions
  across ranks, with exact population weighting despite unequal route lengths.
- PPO losses use the global valid-transition denominator. Gradients are
  SUM-reduced once per optimizer update, then clipped and applied on all ranks.
  There is no collective inside variable-length decoding or backward chunks.
- With one physical rollout, corresponding local minibatch groups form the
  global minibatch. With multiple rollout buffers, the entire effective batch
  forms one update per PPO pass, matching the legacy accumulation branch.
- Model buffers are broadcast from rank zero after each update. Forward
  normalization remains local. Validation runs on rank zero over the complete
  frozen cohort, and all ranks wait for its result; this is not distributed
  validation or independent training with different seeds.
- Rank zero writes one global training/validation history and checkpoint set.
  Global sampled-instance counts and optimizer counts are checked each epoch.
  Local exceptions are gathered at rollout/backward boundaries; process death
  is handled by torchrun and the configured process-group timeout.

## Outputs and validation

`distributed_contract.json` records the topology and reduction semantics;
`resolved_config.json` records the configuration. `logical_epoch_history.jsonl`
contains global exposure, PPO losses, advantage moments, per-rank PPO
diagnostics, and trajectory outcomes. Checkpoint aliases and
`validation_history.jsonl` follow the single-GPU naming convention.

CPU/Gloo regression tests execute the actual TERRAN model and PPO update,
compare a two-rank update against a serial global-batch reference, check equal
parameters across ranks, and run two training/validation epochs using 2, 3,
and 4 processes. These tests establish synchronization and reporting behavior;
they do not establish large-scale CUDA memory usage or throughput. The
large-scale physical batches must still be profiled on the target GPUs.
