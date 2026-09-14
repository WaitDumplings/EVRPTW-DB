# EVRPTW-RL Road500: three GPUs after epoch 300

The watcher waits for the existing two-GPU run to finish epoch-300 validation and save its checkpoint, then continues on **physical GPUs 0/1/2** from **epoch 301**. The previous two-GPU watcher is superseded. EVRPTW-RL's phase change is EMA baseline → greedy rollout baseline; the environment uses hard constraints throughout.

Each worker keeps **batch 24** and **30 trajectories**, so global batch increases from **48 to 72**. Rollout caps remain **1700 training / 2550 validation**. Architecture (sum aggregation), learning rate, objective, station auxiliary and validation cohort remain the same. This is continuation of the original trained actor and AdamW state. Advancing the baseline schedule alone does not guarantee that cost will improve.

## Start and status

From the isolated `cus500-evrptw-stage2-3gpu` worktree:

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus500_evrptw_rl_20260913/watch_stage2_gpu012.sh
```

This registers a background watcher. Do not register a second copy. To inspect it:

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus500_evrptw_rl_20260913/watch_stage2_gpu012.sh --mode status
```

The default source request belongs to `/data/Maojie/ICLR/cus500-evrptw-rl-after-am`. The destination is this worktree's `EVRPTW_Benchmark/results/cus500_evrptw_rl_stage2_3gpu_20260914`.

Use `CUS500_STAGE2_PYTHON` for another compatible Python, or explicit `--source-request` and `--output-root` paths. This checkpoint migration is restricted to the audited Road500 source configuration and epoch 300. The older two-GPU behavior remains available with `watch_stage2_at300.sh --stage2-gpus 0,1` and a separate output root.

## Training state and data accounting

Epochs 1–300 consumed **14,400** instances (`300 × 48`). After migration:

- Epoch 301 reads stream positions `[14400, 14472)`, split into three consecutive 24-instance shards.
- Completed epoch `e ≥ 300` has consumed `14400 + (e − 300) × 72` instances. Cursor and customer exposure are never reconstructed as `e × 72`.
- The maximum remains 10,000 epochs, requiring **712,800** stream entries and **356,400,000** customer exposures; minimum epochs and early stopping remain 5,000 and the original settings.
- The extended training stream preserves the complete original **480,000-entry prefix**, verified by content hash. It then extends the same deterministic shuffle cycles.

The stage boundary copies the epoch-300 actor into its greedy baseline. Both original workers' RNG states, actor weights, AdamW moments/step, stream cursor, historical validation/best selections, and previous runtime accounting are preserved. New worker 2 receives independently seeded Python/NumPy/pool/CPU RNG state; its CUDA RNG starts from a documented copy of source worker 0 and is independently reseeded by the existing rank-specific actor rollout seed before its first action. This topology change is recorded explicitly and does not claim bitwise equivalence to the old two-worker trajectory.

The first greedy-baseline update is epoch 301. The first paired baseline comparison remains epoch 400 (`step > 300`, interval 100). Accumulated GPU hours retain the original two-worker hours and add only the new three-worker session hours.

## Safe handoff and records

1. Require matching epoch-300 validation, epoch artifact, latest checkpoint and committed state. A training/validation log row alone cannot trigger stopping.
2. Prepare an independent run, archive original checkpoint bytes with hashes, and explicitly migrate schedule/topology/stream signatures. Check the exact resumed CLI, actor, optimizer, data and GPU2 availability before stopping the source.
3. Signal only the registered torchrun via Linux pidfd after checking PID start time, UID, exact argv/cwd and ancestry. Wait for the original launcher and both workers to exit and release their locks. GPU3 and other tasks are untouched.
4. Recheck all three GPU UUIDs and availability, acquire all three GPU locks, then launch three workers. Confirm handoff only after a newly completed epoch reports `baseline_kind=greedy_rollout`.

The old launcher may record its intentional SIGTERM as `failed`; `watcher/source_stop.json` records the planned reason. Any speculative work after checkpoint 300 is discarded. Failures are recorded rather than silently overwriting or restarting a run. If the host reboots or a prerequisite changes, inspect the recorded phase before intervening.

Output records:

- `watcher/request.json`, `status.json`, `watcher.log`, `source_stop.json`
- `stage2_config.json`, `stage2_stream_preparation.json`, `verification/`
- `runs/evrptw_rl_road_cus500_seed1234/stage2_transition.json` and `source_checkpoint_archive/`
- `launchers/local_stage2_gpu012/evrptw_rl/status.json` and training logs in the run directory

The new checkpoint's signed continuation fields permit later ordinary three-worker `--resume` with the same config, output, source and stream. Starting from a historical best checkpoint before the transition boundary is rejected as a continuation, while evaluating those historical weights remains supported.

## Validation evidence

202 deployment/distributed regression tests passed. The final 13-test continuation suite also passed after preserving migration lineage in subsequent checkpoints. The three-rank CPU Gloo test checks actual batch 24 × 30 per rank, the global gradient denominator, epoch-300 AdamW/RNG restore, contiguous samples from position 14,400, and exact interrupted/resumed state.

A read-only rehearsal with the real epoch-200 checkpoint used a private temporary 715,200-entry stream and a test-process boundary override. The actual three-worker CLI, model and AdamW checkpoint loader passed; original weights, worker 0/1 RNG and accumulated GPU hours were preserved. Production remains fixed to epoch 300 with 712,800 stream entries.

GPU2 separately passed an actual actor + greedy-baseline + backward/update using source checkpoint 200, batch24, n-traj30 and H1700. Peak process memory was 10,168 MiB (9.93 GiB), total probe time 217.58 seconds, with finite gradients and a nonzero parameter update. The H2550 validation-shaped probe used one training instance. This was a single-GPU feasibility test; simultaneous three-GPU NCCL training will first run at the planned handoff. Prior two-GPU calibration already passed the greedy transition at 10,254 MiB per rank.
