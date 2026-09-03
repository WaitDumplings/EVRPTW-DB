# DRL Experiment and GPU Script Directive

```text
Status: IMPLEMENTATION DIRECTIVE
Target branch: drl-benchmark-adapters
Canonical objective: verified directed-road total distance
Hardware: 2 x (4 x RTX 2080 Ti), 1 x (3 x RTX 2080 Ti),
          1 x (2 x RTX A6000)
```

This document tells the code agent what experiments must be run, which jobs are
currently executable, how the four servers divide work, and which launch
scripts must be delivered.  It does not authorize inventing a road-injection
model or changing a method after observing test results.

## 1. Scientific questions and experiment tracks

### Q1. How do learning solvers change with customer scale?

Train and evaluate the four frozen learning baselines:

1. AM-EVRPTW;
2. EVRPTW-RL;
3. DRL-TS; and
4. TERRAN.

The core scales are Cus100, Cus500, and Cus1000.  All methods use the same
canonical Stage-2 environment, independent verifier, and final distance
objective.  Architectures, training algorithms, and compatible auxiliary
shaping remain method-specific and are documented in each `ADAPTATION.md`.

### Q2. How do policies generalize under three different distribution shifts?

One checkpoint is trained on the released training split and then evaluated on
all applicable test splits.  Never train a separate model for a test split.

| Test | Meaning | Cities | Customer pool | Instances per scale |
|---|---|---|---|---:|
| T1 | new operational realization | ten seen cities | training communities | 500 |
| T2 | spatial holdout | ten seen cities | held-out communities | 500 |
| T3 | geographic holdout | Jacksonville | release-eligible city pool | 500 |

Cus100, Cus500, and Cus1000 use T1, T2, and T3.  Report both absolute results
and the T2-minus-T1 and T3-minus-T1 degradation.

### Q3. Are the implementations compatible with the traditional small regime?

Cus50 is an appendix compatibility track.  Train all four methods and evaluate
only the released 500-instance T1 set.  T2 and T3 Cus50 views do not exist and
must not be synthesized from larger views.

### Q4. Can a policy transfer beyond its training scale?

Use each Cus1000 checkpoint without further training on:

- the 500 released Cus2000 views; and
- their deterministic paired Cus1000 control views.

This is a zero-shot scale-transfer experiment.  Report completion and
feasibility rates, verified distance, vehicle count, inference wall time, and
peak GPU memory.  Do not train a Cus2000 policy and do not describe the
best-observed result as an exact reference.

### Q5. Does Euclidean training transfer to road-network evaluation?

This experiment is required for the road-representation analysis but is not
yet executable.  The intended comparison uses AM-EVRPTW and TERRAN:

```text
E -> R: train on a registered matched Euclidean representation; test on R
R -> R: reuse the canonical checkpoints from Q1
R -> Inject -> R: train with a separately frozen road-injection module; test on R
```

Run Cus100 in the main paper and Cus500/Cus1000 for appendix completeness.
With three seeds, E -> R adds 18 training jobs:

```text
2 methods x 3 scales x 3 seeds = 18
```

R -> Inject -> R would add another 18 jobs, but it is **blocked** until the
injection architecture, inputs, and method name are approved.  The script
generator may expose disabled track IDs for these jobs, but it must not create
or launch them by default.  A Euclidean matrix substitution or a newly written
encoder must never be silently labelled as a published method.

### Q6. Is the released corpus sufficiently varied?

Dataset coverage, city/community composition, and split diagnostics are
dataset-side analyses and require no GPU training.  T1/T2/T3 provide the
learning-side realization, spatial, and geographic tests.  Do not add a
post-hoc "easy/diverse validation subset" selected after looking at test
performance.

## 2. Baseline and scale matrix

Use the frozen training seeds:

```text
1234, 2345, 3456
```

### Immediately executable R -> R training

| Track | Methods | Scales | Seeds | Training jobs |
|---|---:|---:|---:|---:|
| Core | 4 | Cus100/Cus500/Cus1000 | 3 | 36 |
| Compatibility | 4 | Cus50 | 3 | 12 |
| Scale transfer | 4 | Cus2000 | n/a | 0 |
| **Total ready now** |  |  |  | **48** |

Edge-DIRECT is conditional and LEHD is excluded under the frozen source and
adaptation assessment.  Neither receives a GPU job.

### Evaluation protocol

Each selected checkpoint uses two registered decoding budgets:

1. `greedy`: one candidate;
2. `best_of_50`: exactly 50 seeded candidates.

Candidate count is an evaluation budget.  The same candidate IDs/seeds must be
used across methods.  Candidate generation may be chunked for memory, but the
union of candidates and the selected minimum-distance verified solution must
be identical to the unchunked protocol.  Paper-native decoding budgets may be
reported separately in the appendix and must not replace the matched main
comparison.

For Cus100/Cus500/Cus1000, each training checkpoint therefore evaluates six
cells: T1/T2/T3 times greedy/best-of-50.  Cus50 evaluates two cells.  Cus2000
evaluates the paired Cus1000 and Cus2000 cohorts under both decoding budgets.

## 3. Training-data and checkpoint protocol

Do not use the four training programs' raw CLI defaults as a cross-method
budget: they currently imply different numbers of sampled instances.

Implement a shared `data_pass` counter.  One pass means that every released
training view at the target scale is presented exactly once in a seeded
shuffle-cycle:

| Scale | Train views per pass | Customers per view | Customer exposures per pass |
|---|---:|---:|---:|
| Cus50 | 100,000 | 50 | 5,000,000 |
| Cus100 | 50,000 | 100 | 5,000,000 |
| Cus500 | 10,000 | 500 | 5,000,000 |
| Cus1000 | 5,000 | 1,000 | 5,000,000 |

The candidate full protocol is 100 data passes, or 500 million corpus customer
exposures per run.  Store this as one versioned protocol value rather than
hard-coding it in four trainers.  The pilot may motivate changing the value
once, globally, before full launch; it may not be changed by method, scale, or
seed after test results are observed.

- Validate every five data passes on the 500-view validation cohort.
- Complete the registered pass budget; do not use test performance for early
  stopping.
- Select the final checkpoint lexicographically on validation data: highest
  complete-and-feasible rate, then lowest mean verified distance.
- DRL-TS uses passes 1--50 for its soft stage and 51--100 for its canonical hard
  stage when the full budget is 100.
- Method-specific batch sizes and trajectory multiplicities may differ, but
  the data-pass count, effective batch size, optimizer steps, environment
  transitions, wall time, and GPU memory must all be reported.
- If a physical batch is reduced, use documented gradient accumulation to keep
  the registered effective batch unchanged.  Do not silently change optimizer
  semantics after an out-of-memory error.

## 4. Hardware allocation

The cluster has thirteen GPUs:

```text
Server 4T-A: 4 x RTX 2080 Ti, 11 GB each
Server 4T-B: 4 x RTX 2080 Ti, 11 GB each
Server 3T:   3 x RTX 2080 Ti, 11 GB each
Server A6:   2 x RTX A6000, 48 GB each
```

GPU memory is per device and must not be summed across devices.  Use one
independent training process per GPU.  Do not introduce DDP or model parallelism
for the first release.

### RTX 2080 Ti queue

The eleven 2080 Ti devices consume a common low/mid-memory manifest in this
priority order:

1. all Cus100 jobs;
2. Cus500 AM-EVRPTW and EVRPTW-RL jobs;
3. Cus500 DRL-TS and TERRAN only when the pilot peak is at most 9.5 GiB;
4. all Cus50 jobs;
5. greedy evaluation and verifier replay;
6. matched-representation E -> R jobs only after that track is approved.

If a job exceeds the memory gate, record `OOM_UNCHANGED_CONFIG` and transfer its
job ID to the A6000 overflow manifest.  Do not alter the job in place.

### RTX A6000 queue

The two A6000 devices consume the high-memory manifest:

1. one Cus1000 seed for each of the four methods;
2. the second Cus1000 seed for each method;
3. the third Cus1000 seed for each method;
4. Cus500 overflow from the 2080 Ti pilot;
5. best-of-50 evaluation;
6. paired Cus1000/Cus2000 zero-shot evaluation.

Round-robin by seed before completing all seeds of one method.  This produces
an early one-seed diagnostic for every method and avoids spending days on one
method before discovering that another fails at Cus1000.

Suggested fixed A6000 queues are:

```text
A6000-0: DRL-TS s1234 -> AM s1234 -> DRL-TS s2345 -> AM s2345
           -> DRL-TS s3456 -> AM s3456
A6000-1: TERRAN s1234 -> EVRPTW-RL s1234 -> TERRAN s2345
           -> EVRPTW-RL s2345 -> TERRAN s3456 -> EVRPTW-RL s3456
```

The exact order may be regenerated from the manifest, but the seed-round-robin
property must remain true.

## 5. Required three script classes

The code agent must deliver three user-facing launch wrappers.  Shared logic
belongs in one common runner; do not maintain three independent copies of the
training commands.

### 5.1 Two identical 4-GPU servers

```text
EVRPTW_Benchmark/Reinforcement_Learning/scripts/run_drl_4x2080ti.sh
```

Invocation:

```bash
SERVER_INDEX=0 bash .../run_drl_4x2080ti.sh pilot
SERVER_INDEX=1 bash .../run_drl_4x2080ti.sh pilot

SERVER_INDEX=0 bash .../run_drl_4x2080ti.sh full
SERVER_INDEX=1 bash .../run_drl_4x2080ti.sh full
```

`SERVER_INDEX` is mandatory and must be either 0 or 1.  Index 0 owns global
2080-Ti slots 0--3; index 1 owns slots 4--7.  This prevents the two identical
servers from claiming the same jobs.

### 5.2 One 3-GPU server

```text
EVRPTW_Benchmark/Reinforcement_Learning/scripts/run_drl_3x2080ti.sh
```

It owns global 2080-Ti slots 8--10 and accepts the same `pilot`, `full`,
`evaluate`, `status`, and `resume` modes.

### 5.3 One 2-GPU A6000 server

```text
EVRPTW_Benchmark/Reinforcement_Learning/scripts/run_drl_2xa6000.sh
```

It owns high-memory slots 0--1 and runs the Cus1000, overflow, best-of-50, and
scale-transfer queues.

## 6. Common runner and manifests

The wrappers must call shared implementation files such as:

```text
scripts/drl_job_runner.sh
scripts/build_drl_job_manifests.py
configs/drl_experiment_protocol_v1.yaml
manifests/drl_2080ti_jobs_v1.jsonl
manifests/drl_a6000_jobs_v1.jsonl
scripts/summarize_drl_experiments.py
```

Every job must have a stable ID containing at least:

```text
representation / method / scale / seed / stage / split / decode budget
```

Example:

```text
train__R__am_evrptw__Cus500__seed2345
eval__R__am_evrptw__Cus500__seed2345__T2__best_of_50
transfer__R__terran__Cus1000_to_Cus2000__seed1234__greedy
```

Required environment variables:

```text
EVRPTW_REPO_ROOT
EVRPTW_DATASET_ROOT
EVRPTW_OUTPUT_ROOT
```

No machine-specific absolute path may be committed.  Each wrapper must verify
the expected GPU count and model, repository branch/commit, usable Python environment,
dataset indices, writable output root, and available disk space before launch.

## 7. Resume, failure, and provenance contract

- Each GPU owns an ordered serial queue; GPUs run in parallel.
- A completed job is skipped only after its result manifest, selected
  checkpoint, and verifier summary all pass.
- A failed job writes a failure record and stops that GPU queue.  Other GPU
  queues may finish their in-flight jobs, but must not silently mutate or retry
  the failed configuration.
- `resume` continues from the first incomplete stable job ID without
  overwriting passed outputs.
- SIGINT/SIGTERM must propagate to all child process groups and leave a
  resumable state.
- Training outputs must use
  `representation/method/scale/seed/git_commit/`; evaluation outputs add
  `test_id/decode_budget/`.
- Record the executable Git commit, protocol ID, dataset release ID, command,
  seed, batch/effective-batch settings, CUDA/PyTorch versions, GPU name, peak
  memory, wall time, selected checkpoint, and verifier result.
- Use Git commit provenance during research.  Do not add SHA256 corpus scans to
  these launch scripts.

## 8. Pilot gate before `full`

`full` must refuse to start until the following pilot artifacts exist for every
method:

1. Cus100 short optimization run with finite loss and saved checkpoint;
2. Cus500 forward/backward memory measurement;
3. Cus1000 forward/backward measurement on A6000;
4. greedy route export and independent verifier PASS;
5. chunked best-of-50 equality against an unchunked small-case run;
6. data-pass counter and resume replay test;
7. estimated wall time for 100 data passes.

The pilot report must recommend one unchanged full protocol or stop for review.
It may change the global data-pass budget before launch, but it may not define
different budgets in response to method-specific test performance.

## 9. Required aggregate outputs

The summarizer must produce machine-readable CSV/JSON and paper-ready tables
with, at minimum:

- complete-and-feasible rate;
- verified distance, conditional on feasibility;
- relative gap to the named comparison reference when such a reference exists;
- vehicle count;
- inference wall time;
- training wall time and data passes;
- peak CPU and GPU memory;
- results by method, scale, seed, T1/T2/T3, and decoding budget;
- T2/T1 and T3/T1 degradation;
- paired Cus1000/Cus2000 scale-transfer change.

The aggregation must retain per-instance rows.  Means without feasibility
denominators are insufficient.

## 10. Acceptance checklist for the code agent

Before asking the user to pull and run, provide evidence that:

1. the three wrappers exist and pass `bash -n`;
2. the two 4-GPU server indices generate disjoint job IDs;
3. the union of the eleven 2080-Ti slots contains every intended low/mid job
   exactly once;
4. the A6000 manifest contains all twelve Cus1000 training jobs exactly once;
5. T1/T2/T3 never create training jobs;
6. Cus50 contains T1 only;
7. Cus2000 contains evaluation jobs only and includes paired controls;
8. E -> R and R -> Inject -> R remain disabled until their contracts pass;
9. a one-job dry run, failure test, interruption test, and resume test pass;
10. no executable code uses test metrics for checkpoint or hyperparameter
    selection; and
11. documentation gives copy-paste commands for all four servers.
