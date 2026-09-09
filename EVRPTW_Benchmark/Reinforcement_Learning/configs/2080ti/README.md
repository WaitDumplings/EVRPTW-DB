# RTX 2080 Ti benchmark profiles (2026-09-09)

The three 2080 Ti bundles retain the four canonical benchmark adapters and their
registered training streams. `terran_cus50.yaml` and `terran_cus100.yaml` freeze
the latest canonical TERRAN critic settings separately from the A6000 configs.
The manifest generator records both profile path and SHA-256; runtime checks the
hash before selecting the config for training, evaluation and provenance.

| Setting | Cus50 | Cus100 |
| --- | ---: | ---: |
| TERRAN physical / effective instances | 480 / 480 | 280 / 280 |
| Sampled training trajectories / instance | 50 | 50 |
| PPO minibatches / passes | 4 / 3 | 4 / 3 |
| Replay time chunk | 64 | 64 |
| Train / validation action budget | 65 / 98 | 120 / 180 |
| Actor / critic shared optimizer LR | 1e-4 | 1e-4 |
| Value loss / weight | Smooth-L1 / 0.1 | Smooth-L1 / 0.1 |
| Critic gradient multiplier into shared encoder | 0.1 | 0.1 |
| Customer progress budget / repair coefficient | 0.5 / 0.5 | 0.5 / 0.5 |
| Gamma / terminal success bonus | 1.0 / 0.0 | 1.0 / 0.0 |
| Stage-2 family cache entries | 16 | 16 |

The reference economic reward already uses separate Cus50/Cus100 divisors, so
value weights and potential budgets are not multiplied by customer count. PBRS
retains its existing cosine schedule (1.0 to 0.2 through epoch 5000). This keeps
the normalized reward contract and terminal failure charges consistent across
the benchmark comparison. Further shaping or value-weight changes need paired
validation evidence. The physical batch sizes are the complete train/validation
calibration choices in `reports/RTX2080TI_PER_JOB_MEMORY_CALIBRATION_V5.md`.
Those measurements predate the latest critic diagnostics. Fresh two-epoch
Cus50/Cus100 resource checks passed on this host, including 10-view, 100-candidate
validation; the Cus100 check exercised diagnostics at both epochs. The new
checks are smaller than a complete 500-view validation and do not demonstrate
convergence. See `reports/RTX2080TI_MACHINE_PROFILE_UPDATE_20260909.md`.

The other adapters retain their calibrated batches: AM 2304/800, EVRPTW-RL
336/96 and DRL-TS 144/40 for Cus50/Cus100. These leave little spare device memory;
increasing batch without a new measurement is not justified. A larger family
cache reduces repeated dataset reads. Wrapper defaults limit OpenMP/MKL/BLAS
threads to one unless the user has set them. Synchronous timing remains off in
formal training.

The separate A6000 `TERRAN.stable_cli` algorithm uses raw USD cost, independent
PopArt critics and a different deterministic sampler. It does not accept the
frozen stream/support-set/E queue contract and is not silently substituted into
`full.sh`. In particular, its current CLI selects the core index, whereas Cus50
on this server is under `generation_plan/compatibility_cus50`; a Cus50 stable
profile alone would therefore not be a working launcher.

Run from the repository root using the project environment:

```bash
conda activate maojie
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/2080ti_4_1/full.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/2080ti_4_1/status.sh
bash EVRPTW_Benchmark/Reinforcement_Learning/scripts/2080ti_4_1/logs.sh
```

Replace the machine identifier for the other two servers. `full.sh`, `resume.sh`,
`status.sh` and `logs.sh` in the older script location forward to `scripts/rq_v1`.
The historical `pilot.sh`, `run.sh` and `start.sh` remain legacy protocol entry
points and are not the new full launch path. Resuming a checkpoint made with a
different profile is rejected by the recorded training contract; start a fresh
run for the changed profile.

No registered stream, stream sidecar, registry hash or preparation-marker hash
was changed by these profile updates. All 2080 Ti sidecars were checked against
the registry. The local A6000 TERRAN Cus500/Cus1000 streams still have the older
640000/20000 rows versus the fetched registry's 740000/40000. This affects a
local all-server artifact rebuild and is not evidence against the selected
2080 Ti queues. Restore/rebuild those artifacts on the relevant server before
validating its queue; do not rebind a stale stream to the new checksum.
