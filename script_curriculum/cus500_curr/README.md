# AM Road Cus100 → Cus500 curriculum

This stage trains **one AM policy on physical GPU 0 and 1 together**, using Road
Cus500 only. It imports the selected Road Cus100 stage-1 policy and runs **3000
additional logical epochs**, with validation every 100 epochs. Euclidean data and
the archived independently trained Cus500 checkpoint are not used.

```bash
cd /data/Maojie/ICLR/EVRPTW-DB-curriculum
./script_curriculum/cus500_curr/am.sh
```

The repository may instead be named `EVRPTW-DB`; paths are resolved from the script.
Use `--dry-run` to inspect the source, dataset, topology, and complete command
without reserving GPUs or creating a training output directory. The shell preflights and starts training in the background by default, prints the
launcher log path, and permits disconnecting SSH. For an interactive launch, use
`./script_curriculum/cus500_curr/am.sh --foreground`; this option must come first.
The shell selects the `maojie` environment unless `CURRICULUM_PYTHON` or
`CURRICULUM_CONDA_ENV` overrides it. Updating Git is explicit via `--pull`; ordinary
launches do not change a checkout that another training process may still use.

## Source weights

The default source is frozen in [source_checkpoint.json](source_checkpoint.json):

```text
$CURRICULUM_STAGE1_ROOT/am_evrptw_G_Cus100_stage1_seed1234_20260920T092713_1409604/best_overall.ckpt
```

`CURRICULUM_STAGE1_ROOT` defaults to `/data/curriculum_stage1`. This run completed
**2000** new Cus100 epochs, and its selected checkpoint is epoch **1300**, with
500/500 verified feasible validation instances and mean cost
468.9396665655092 USD. Epoch 1000 was not the final run endpoint. Its full hash is
`6ae497332b98eefb507c63d21ff5e713a4bfeb71594259872ad2cbbf77ba2ea1`.
Missing or changed default source files fail explicitly; there is no fallback to
an old checkpoint or the latest checkpoint. Source weights are never overwritten.

For an explicitly chosen alternative stage-1 AM Road Cus100 checkpoint:

```bash
./script_curriculum/cus500_curr/am.sh --source-checkpoint /absolute/path/best_overall.ckpt
```

The override must match the stage-1 protocol, Road Cus100 domain, seed, D_time
objective, and AM architecture. Its actual hash/epoch are recorded; the default
checkpoint's validation score is not attributed to an override.
An explicit source override is marked as unprofiled for GPU memory.

## Training and validation

| Setting | Default |
|---|---|
| GPUs | Physical 0, 1; two synchronous replicas of one model |
| Per-GPU / global instance batch | 12 / 24; accumulation 1 |
| New stage epochs | 1–3000; no early stopping |
| Training trajectories | 30 per instance |
| Train / validation action cap | 1700 / 2550 |
| Validation | 500 Road Cus500 instances, best-of-30, every 100 epochs |
| Training / validation seed | 1234 / 910001234 |
| Optimizer | AdamW, learning rate 1e-4, weight decay 0.01, gradient norm limit 1 |
| Model | AM, 128-dimensional embeddings, 3 encoder layers, 8 heads |
| Train corpus | 10,000 Road Cus500 views, distinct from the 500 validation views/families |
| Batch-12 sample exposure | 72,000 sampled instances / 36,000,000 customer occurrences |

One logical epoch is one synchronized optimizer update, not one pass over the
corpus. The training stream uses seeded shuffled full-pool cycles. Each rank gets
disjoint entries within a global batch. No test data is read.

This is **weights-only continuation across scales**, with explicit scale-transition
permission and strict architecture loading. The source optimizer, baseline history,
EMA value, best-selection state, and epoch/sample counters are reset. It is not a
resume of the single-GPU run. AM's default baseline schedule restarts: the first
2500 stage updates use EMA, followed by 500 greedy-baseline updates. Paired baseline
probes use the default 2500-update interval and 64 training instances.

The target is `energy_vehicle_cost`, with cost and energy both using
`running_time_path_distance_km`. Cost is
`413.6331536717643 * K + 0.39 * (100/257) * D_time_km` USD. Training switches to the
shared **Cus500** reward normalizer: objective scale 4238.927542618743, failure base
3.21013867342889, unserved coefficient 1. These are retained training-only numerical
normalizers, not a new D_time calibration. Reported cost is independently replayed
USD cost; candidate/epoch selection first maximizes complete-and-feasible rate,
then minimizes mean verified cost among feasible instances.

Gradients are globally normalized and summed before clipping and updating.
BatchNorm forward statistics are local to each rank, with rank-0 running buffers
broadcast after updates. Thus this is not claimed to be numerically equivalent to
a serial run with the same global batch. Validation shards preserve each instance's
seed and are merged before the single checkpoint-selection decision.

The current curriculum weights were measured on two 2080 Ti GPUs with per-GPU
batches 4 and 12. Batch 12 passed five EMA updates and three greedy-baseline
updates, including validation and a baseline comparison. It is the final default;
see [SMOKE_REPORT.md](SMOKE_REPORT.md) for recorded memory and test scope.
A short smoke run does not establish the peak of every future random batch.

## Data, outputs, and overrides

The complete Road release can be shared with the existing repository:

```bash
export CURRICULUM_ROAD_ROOT=/data/Maojie/ICLR/EVRPTW-DB/EVRPTW_Dataset/Instances_v2/us_11city_full_clean_v7_bbde5db_20260823
```

`CUS500_ROAD_ROOT`, `CUS100_ROAD_ROOT`, and `EVRPTW_DATASET_ROOT` are fallback variables.
Without an override the existing release locations are searched. Git transfers code,
not the Road dataset or stage-1 checkpoint.

Outputs default to `/data/curriculum_stage2_cus500`, overridable with
`CURRICULUM_CUS500_OUTPUT_ROOT` or `--output-root`. Every launch uses a new dated run
directory containing `request.json`, `status.json`, `training.log`, checkpoint files,
`logical_epoch_history.jsonl`, `validation_history.jsonl`, and
`validation_summary.csv`. The request records source identity, objective, scale,
batch, GPU identities, stream hashes, code hashes, and the executed command.

The Python interface also supports `--gpus 0,1`, `--batch-size 12`, `--epochs 3000`,
`--validation-every 100`, `--validation-limit 500`, and `--run-dir` (must be new).
Batches 4 and 12 have short-run profiles; another batch is unprofiled. Changing
the batch changes global sample exposure.
Short tests must use a separate output directory. Existing compute workloads on
either selected GPU prevent startup; the launcher does not stop them.
