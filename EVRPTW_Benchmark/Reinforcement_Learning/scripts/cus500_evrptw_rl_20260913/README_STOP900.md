# Stop the current EVRPTW-RL Road500 run after epoch 900

This watcher implements the user's requested stop of the three-GPU continuation on physical GPUs 0/1/2. It waits for epoch **900**, including its fixed-cohort validation and checkpoint publication, then stops only the registered training process tree. It does not start another experiment or change the training budget/configuration.

From the independent `cus500-evrptw-stop900` worktree:

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus500_evrptw_rl_20260913/watch_stop900.sh
```

The command registers a background watcher and returns. Duplicate registration is rejected. Check its state with:

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus500_evrptw_rl_20260913/watch_stop900.sh --mode status
```

Default records are in the existing three-GPU experiment output root:
`EVRPTW_Benchmark/results/cus500_evrptw_rl_stage2_3gpu_20260914/stop_after_epoch900/`.

The readiness check requires:

- `checkpoint_epoch_0900.pt` and byte-identical `checkpoint_latest.pt`, fully loadable on CPU;
- matching epoch-900 validation with 500 instances;
- `data_pass_state.json` exactly matching the embedded checkpoint state, published last;
- 900 optimizer steps, **57,600** consumed instances, **28,800,000** customer exposures, nine validation checkpoints, three rank RNG states, and the expected model/topology.

Before signalling, it copies the epoch-900 checkpoint, best checkpoint aliases, validation summaries and transition provenance into `checkpoint_backup/`, checks hashes and flushes the backup files. The original run remains available.

It verifies UID, PID start time, exact command/working directory and process ancestry. Only the bound torchrun receives SIGTERM via pidfd; torchrun shuts down its three ranks, and the launcher exits afterward. The watcher waits for all five registered processes to exit. GPU3/TERRAN and unrelated processes are untouched. A failed prerequisite or changed PID causes an error without signalling replacement processes.

Terminal watcher status is `stopped_at_checkpoint`. An intentional-stop receipt is also written as `runs/evrptw_rl_road_cus500_seed1234/user_stop_after_epoch900.json`. The original launcher can record the intentional SIGTERM as `failed`; this receipt records the user's stop request. This is not completion of the configured 10,000-epoch budget, so no final training result is fabricated. Work started speculatively after checkpoint 900 can be discarded; the saved checkpoint remains at exactly 900.

`request.json`, `status.json`, `watcher.log`, and `checkpoint_backup/manifest.json` retain the process identities and checkpoint provenance. Do not edit this watcher worktree after registration; its source hash is checked before stopping.
