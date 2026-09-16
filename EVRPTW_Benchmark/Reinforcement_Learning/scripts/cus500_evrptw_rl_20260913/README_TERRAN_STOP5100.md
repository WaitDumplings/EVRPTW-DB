# Stop local TR05 TERRAN at epoch 5100

This deployment targets the existing Euclidean Cus100 TERRAN experiment TR05 on physical GPU3. It runs from the independent `/data/Maojie/ICLR/terran-stop5100` worktree, leaving both live training code and the EVRPTW-RL stop900 watcher unchanged.

Start once:

```bash
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/cus500_evrptw_rl_20260913/watch_terran_stop5100.sh
```

Add `--mode status` to inspect its background heartbeat. Default state and backups are under the existing TR05 run's `stop_after_epoch5100/` directory. Duplicate registration is rejected.

The watcher waits for epoch5100 validation over 500 instances and the final atomic `data_pass_state.json`: 1,958,400 instances, 195,840,000 customer exposures and 61,200 optimizer steps. TERRAN uses 12 optimizer updates per epoch here. Both `checkpoints/checkpoint_epoch_5100.pt` and `checkpoint_latest.pt` must load on CPU, match epoch, seed and frozen configuration, and contain identical model and optimizer state. Their serialized bytes may differ because TERRAN saves them separately.

Before signalling it backs up both checkpoint files, selected/best aliases, validation summaries/history and launch metadata, checks hashes and flushes the backup files. It then sends SIGTERM only to the registered trainer after checking UID, PID start time, command and working directory via pidfd. Other experiments and watchers are not signalled. The source launcher has no restart policy and exits when its children finish.

The terminal watcher status is `stopped_at_checkpoint`; `user_stop_after_epoch5100.json` in TR05 records the deliberate stop. The original launcher may report `failed` due to SIGTERM; the receipt distinguishes this stop from an unexplained crash. A full-budget training result is not created. Work begun after the saved epoch5100 checkpoint may be discarded.

Do not edit this worktree after registration; the source hash is checked before stopping. A failed checkpoint/identity check leaves training untouched and records `failed` with the error in watcher status.
