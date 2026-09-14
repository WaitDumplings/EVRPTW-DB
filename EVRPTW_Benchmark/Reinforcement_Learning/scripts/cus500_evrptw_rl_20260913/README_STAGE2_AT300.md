# EVRPTW-RL Road Cus500: enter greedy-baseline training after epoch 300

This watcher implements an explicit continuation of the existing two-GPU Road500 run. EVRPTW-RL uses hard environment constraints throughout; its phase change is **EMA baseline → greedy rollout baseline**, not DRL-TS's soft/hard constraint stages.

The original run uses EMA for 1,000 optimizer updates. The continuation changes that boundary to 300, initializes the greedy baseline from the actor saved at epoch 300, and resumes at **epoch 301**. Epochs are global optimizer updates in this deployment. Actor weights, AdamW state, per-rank RNG, stream cursor, validation history, historical best checkpoints, and the original minimum/maximum budget are preserved. Epoch 301 consumes the next global batch after cursor 14,400.

The current configuration stays on **physical GPUs 0/1**, batch **24 per rank / 48 global**, **30 trajectories per instance**, rollout caps **1700 training / 2550 validation**, and the original sum-aggregation architecture, objective, and station auxiliary. This continuation does not apply the separate mean-aggregation retraining change. Advancing the baseline schedule alone does not guarantee that cost will improve. The original calibration already exercised the greedy-baseline transition with this batch size, peaking at 10,254 MiB per GPU.

## Register on this server

Run from the isolated worktree containing these scripts. The default source is the live `cus500-evrptw-rl-after-am` experiment; the default destination is this worktree's `EVRPTW_Benchmark/results/cus500_evrptw_rl_stage2_at300_20260914`.

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus500_evrptw_rl_20260913/watch_stage2_at300.sh
```

It registers a background watcher and returns immediately. Do not register it a second time. A duplicate registration in the same output directory is rejected. Use `CUS500_STAGE2_PYTHON` to select another compatible Python, or `--source-request /absolute/path/to/launch_request.json` and `--output-root /absolute/path` for explicit paths. The checkpoint migration is restricted to this audited Road500 configuration and epoch 300.

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus500_evrptw_rl_20260913/watch_stage2_at300.sh --mode status
```

Records are under `<output-root>/watcher/`: `request.json`, `status.json`, `watcher.log`, and (after stopping) `source_stop.json`. `stage2_config.json` records the changed schedule. The resumed run is under `<output-root>/runs/evrptw_rl_road_cus500_seed1234`; its launcher is under `launchers/local_after_am_gpu01/evrptw_rl`.

## Handoff conditions

1. Wait for epoch-300 validation, `checkpoint_epoch_0300.pt`, and the matching committed state. A training or validation log line alone cannot trigger a stop.
2. Prepare an independent continuation directory, archive source checkpoint bytes with hashes, copy the epoch-300 actor into its baseline, and explicitly record the signature migration. Validate the exact resumed CLI, model, optimizer, dataset and reward contract **before stopping** the source.
3. Signal only the registered torchrun through a Linux pidfd. Verify UID, PID start time, exact command, output path, process ancestry and working directory. Wait for the source launcher and both workers to exit and release the GPU locks. Other tasks are not signalled.
4. Start the resumed two-rank trainer with the same UUID-bound GPU0/1 pair. Mark `handed_off` only after a newly completed epoch reports `baseline_kind=greedy_rollout`.

The old run and its original launch request remain available. The original launcher may record the intentional SIGTERM as `failed`; the separate `source_stop.json` explains the planned handoff. Any work begun after checkpoint 300 before the watcher stops the process is discarded; the continuation resumes exactly from checkpoint 300. The watcher records failures and does not silently restart or overwrite runs. If its host reboots or a prerequisite changes, inspect the recorded phase before intervening.

The first paired baseline comparison is at epoch 400 (`step > 300`, interval 100), while greedy-baseline training starts at 301. The inherited budget remains minimum 5,000, maximum 10,000 epochs with the original validation and early-stop settings.

## Validation

44 CPU tests passed, including watcher process guards, exact migration state, and the existing two-rank Gloo EMA/greedy/resume tests. A separate CPU-only rehearsal used the real epoch-200 checkpoint with a test-process boundary override, verified the actual resumed CLI and strict checkpoint loader, and checked exact actor, AdamW, RNG, baseline and historical-best state. It left the source checkpoint unchanged; production code remains fixed to epoch 300. No GPU training was started by that rehearsal.
