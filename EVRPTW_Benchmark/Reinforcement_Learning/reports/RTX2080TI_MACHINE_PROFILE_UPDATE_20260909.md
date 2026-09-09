# RTX 2080 Ti machine-profile integration, 2026-09-09

The three 2080 Ti full queues now select pinned Cus50/Cus100 TERRAN profiles.
The older `scripts/2080ti_*/full.sh`, `resume.sh`, `status.sh` and `logs.sh` forward
to the current RQ queues. A generated manifest records the profile path and SHA;
rebuilding the manifest retains that selection. See
[profile parameters](../configs/2080ti/README.md) for the calibration basis.

The latest critic settings are Smooth-L1, `vf_coef=0.1` and shared-encoder
critic gradient multiplier `0.1`. Small-scale reward normalization already
exists, so these weights and PBRS budgets remain common. PBRS gamma is 1,
terminal success bonus is 0, and the existing shaping schedule is retained.
Batch sizes and training trajectory counts retain the prior measured 2080 Ti
geometry. The family cache increases from 4 to 16; synchronous profiling stays
off. CPU thread defaults are one and honor explicit overrides.

## Production path checks

`run_server.sh full --dry-run --reuse-preverified-training-streams` succeeded
for all 16 selected jobs, using separate result directories. Only the other
server simulations used `--skip-gpu-preflight`.

| Bundle | Jobs | GPU preflight on this host | Result |
| --- | ---: | --- | --- |
| 2080ti_4_2 | 5 | Skipped (configuration simulation) | PASS |
| 2080ti_3_1 | 3 | Skipped (configuration simulation) | PASS |
| 2080ti_4_1 | 8 | Actual count/model: four RTX 2080 Ti | PASS |

All 16 generated commands retained the original stream contract SHA. All five
TERRAN commands selected the correct small-scale profile with a matching SHA.
The last preparation operation restored the shared marker to the eight unique
2080ti_4_1 streams. Evidence is in
`results/2080ti_update_20260909/preflight/preflight_summary.json` (under
`EVRPTW_Benchmark`). This was a dry run and launched no training.

This check found and fixed a launch blocker in 2080ti_4_2: the G and E TERRAN
jobs share an ordered stream, but the marker builder emitted duplicate artifact
entries. The runtime correctly rejected the duplicate. Marker preparation now
writes each stream once and rejects conflicting snapshots for one path before
replacing the marker. It still explicitly records no-rehash reuse and preserves
the frozen marker identity; no stream content/registry hash was relabelled.

## CPU checks

The targeted runtime, environment, machine-profile generation and marker tests
passed (94 distinct tests, with the final fixture adjustment rechecked
separately). Mock validators now accept the explicit integrity-mode keyword;
validator fixtures read committed manifests and build matching marker/registry
identities only under their temporary directory. The content-rehash test still
copies and hashes actual method-specific streams, independently of launch mode.
All edited shell files passed `bash -n`; `git diff --check` passed.

The broad all-server manifest suite is blocked by existing local A6000 TERRAN
artifacts: Cus500 has 640000 rows versus the fetched registry's 740000; Cus1000
has 20000 versus 40000. Every 2080 Ti sidecar agrees with the registry. These
A6000 files were not modified and are outside the selected production queues.

## GPU resource checks

Cus50 completed two fresh epochs at physical batch 480 and 50 trajectories on
an RTX 2080 Ti. Peak PyTorch allocated memory was 10394742784 bytes (9.68 GiB).
The ten-instance, 100-candidate validation produced ten independently verified
feasible solutions. Evidence:
`results/2080ti_update_20260909/terran_cus50_gate_v3`.

Cus100 uses a separate temporary gate config at batch 280 × 50 trajectories,
120 training / 180 validation actions, two epochs and ten validation instances
with 100 candidates. Its diagnostic interval is forced to one so both epochs
exercise critic gradient diagnostics; the formal profile remains at interval
25. Both epochs completed, including 24 optimizer updates and critic gradient
diagnostics at epochs 1 and 2. All numeric diagnostics were finite. Peak
PyTorch allocated memory was 10358958592 bytes (9.648 GiB); sampled device memory
peaked at 10716 MiB. The ten-instance, 100-candidate validation produced ten
independently verified feasible solutions. Training-reported wall time was
101.43 seconds. Evidence:
`results/2080ti_update_20260909/terran_cus100_gate/gate_summary.json`.

The device-utilization samples averaged 16.5% over the entire monitored process,
including initialization, sampling, updates and validation. The batch fills
most available device memory, but this is not a high average-utilization result.
CPU environment progression remains a relevant bottleneck. Across replayed
transitions and PPO updates, the first epoch's mean empirical approximate KL
was 2.134, while epoch 2 was 0.00379. The short resource check therefore does
not establish long-run policy stability.

The subsequent RRNCO Cus100 resource gate used batch 32, 16 leave-one-out
trajectories, stable AFT, nearest-distance sampling and temperature 5. It
completed two optimizer steps with finite losses 13.742 and 10.858. Peak
PyTorch allocated memory was 2051420672 bytes (1.911 GiB), sampled device memory
3048 MiB, and wall time 13.94 seconds. Its feasibility check did **not** pass:
only 4/10 validation instances had a verified solution among 100 candidates.
During training, 1019/1024 trajectories exhausted the 120-action budget and
only five succeeded. Evidence:
`results/2080ti_update_20260909/rrnco_cus100_b32_t16_gate/gate_summary.json`.
GPU 1 was released after both gates. The RRNCO result is a resource pass and a
short-training feasibility failure; its feasible-subset cost must not be
compared directly against the TERRAN gate's complete ten-instance cohort.

These short checks establish execution and resource feasibility. They do not
establish convergence, a cost improvement, or that RRNCO exceeds the four
benchmarks. Those conclusions require matched independent-verifier evaluation
of the trained checkpoints.
