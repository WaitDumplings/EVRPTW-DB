# AM Road Cus500 -> Cus1000, four GPUs

Place the selected **Road Cus500 stage2** `best_overall.ckpt` at
`/data/cus500_ckpt/am.ckpt`, or pass `--source-checkpoint /absolute/path`.
The launcher checks AM architecture and weights, source protocol, Cus500/G/seed1234,
training signature, and the D_time monetary objective before using the checkpoint.

```bash
conda activate maojie
./script_curriculum/cus1000_curr/am_cus500_to_1000.sh 0 1 2 3
```

The default is a detached/background job. The launcher prints its PID and log.
Use `--dry-run` for preflight only, or `--foreground` to supervise in the terminal.
Set `CURRICULUM_PYTHON=/path/to/python` when needed. The Road dataset can be supplied
with `--road-root` or `CURRICULUM_ROAD_ROOT`.

Default settings:

- Four GPUs, per-GPU batch **3**, global batch **12**, 30 training trajectories per instance.
- **2000 new logical epochs**, no early stopping; validation every100 epochs on all500
  Cus1000 validation instances, sampling30 candidates per instance.
- Inherit the selected Cus500 **actor weights**. Reset optimizer, baseline,
  sample counters, validation selection and epoch counter for the new scale.
  This follows the Cus100->500 curriculum warm-start protocol, not a same-scale resume.
- AdamW, LR1e-4, weight decay0.01, gradient norm limit1. Native AM baseline settings
  are retained: `steps_per_epoch=2500`, `baseline_warmup_epochs=1`; all2000 new updates
  therefore use EMA. This is distinct from the selected source checkpoint's epoch.
- Train/validation action caps1250/1875. Optimize monetary cost using D_time.
- Frozen training pool5000 and validation pool500, with disjoint parent families;
  seeded shuffled pool cycles and **24000 actual instance occurrences**
  (2000 updates * global batch12), corresponding to24million customer occurrences.
  Test data is not used.

Results: `/data/curriculum_stage3_cus1000/am_evrptw_G_Cus1000_stage3_seed1234_<timestamp>_<pid>/`.
Contains `request.json`, `status.json`, `training.log`, training/validation histories,
selected best aliases and latest checkpoints. Defaults can be overridden with
`--epochs`, `--batch-size`, `--validation-every`, `--validation-limit`, `--output-root`.
Batch is always instances per GPU, not trajectories.

The local source is the completed Cus500 stage2 run's best checkpoint, selected at
stage2 epoch2000 (500/500 feasible; val cost1837.8601118468303 USD). Its SHA256 and
strict source identity are recorded in `batch_profile.json` and each run's request.

The exact source passed a three-update, four-GPU batch3 smoke test, plus four
validation instances. Batch4 exceeded GPU memory during rollout and was rejected.
See `batch_profile.json` for allocation peaks and evidence. Short profiling does not
guarantee memory safety for every future instance batch. Formal training restarts
from the selected Cus500 source, not from the smoke-test weights.
