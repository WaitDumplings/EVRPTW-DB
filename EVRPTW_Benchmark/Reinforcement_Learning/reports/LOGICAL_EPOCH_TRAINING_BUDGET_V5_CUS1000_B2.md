# 1,000-epoch / Cus1000 batch-two training budget v5

Date: 2026-09-03

Runtime budget ID: `drl_rq_runtime_budget_v4_cus1000_b2_val100`

Status: implemented candidate; formal launch remains subject to the existing pilot gate

## Frozen logical budget

Every method uses 1,000 logical epochs at every scale. Comparisons are made
within scale, so effective batch and the deterministic training stream are equal
across methods at a given `scale × condition × seed`; customer exposures are not
artificially equalized across different problem sizes.

| Scale | Epochs | Base instances/epoch | Effective batch | Total base instances | Customer exposures | Rollout cap |
|---|---:|---:|---:|---:|---:|---:|
| Cus50 | 1,000 | 200 | 200 | 200,000 | 10,000,000 | 80 |
| Cus100 | 1,000 | 50 | 50 | 50,000 | 5,000,000 | 140 |
| Cus500 | 1,000 | 4 | 4 | 4,000 | 2,000,000 | 600 |
| Cus1000 | 1,000 | 2 | 2 | 2,000 | 2,000,000 | 1,200 |

A base instance is one Stage-2 view. Parallel decoding trajectories are a model
detail and do not count as additional training instances. Streams are sampled
with replacement within frozen city/day-type strata and deterministically
shuffled. All methods sharing a scale, condition, and seed consume the same
stream.

## Effective and physical batches

Each cell is `effective / physical`; the quotient is the number of physical
microbatches per logical epoch.

| Method | Cus50 | Cus100 | Cus500 | Cus1000 |
|---|---:|---:|---:|---:|
| AM-EVRPTW | 200 / 200 | 50 / 50 | 4 / 4 | 2 / 1 |
| EVRPTW-RL | 200 / 100 | 50 / 25 | 4 / 2 | 2 / 1 |
| DRL-TS | 200 / 50 | 50 / 10 | 4 / 1 | 2 / 1 |
| TERRAN | 200 / 200 | 50 / 50 | 4 / 4 | 2 / 1 |

The smaller physical batches are memory controls. Exact gradient accumulation
preserves the registered effective batch for REINFORCE methods. For Cus1000,
TERRAN also collects two physical rollout buffers sequentially and accumulates
their weighted PPO gradients before each optimizer step; only one Cus1000
instance is resident on the GPU at a time.

## Validation and model selection

Validation runs at epochs `50, 100, ..., 1000`, for 20 checkpoints. Cus50,
Cus100, and Cus500 use the same fixed 500-view subset at every event; Cus1000
uses a fixed 100-view selection subset. Every event uses a full rollout horizon
and independent route verification. The lexicographic selection key is:

1. maximum complete-and-feasible verifier rate;
2. minimum mean verified directed-road distance among feasible instances.

The selected state is saved as both `best.ckpt` and
`checkpoint_selected.pt`. `checkpoint_latest.pt` remains the resume state and
must not be confused with the selected model. All results are retained in
`validation_history.jsonl`; test splits never participate in selection. After
selection, Cus1000 evaluates the selected checkpoint once on all 500 registered
val views and writes `validation_final_audit.json`. This audit cannot change the
selected checkpoint.

## Planning boundary

The linearly scaled Cus1000 estimate is approximately 48 hours per model/seed
job on one RTX A6000 and must be recalibrated by pilot evidence. Raising Cus50,
Cus100, and Cus500 to 1,000 epochs increases their previous training exposure;
old v2/v3 wall-time estimates do not apply to this budget. Validation time is
additional and must be reported separately. No SHA-256 or per-file hash scan is
part of this protocol.
