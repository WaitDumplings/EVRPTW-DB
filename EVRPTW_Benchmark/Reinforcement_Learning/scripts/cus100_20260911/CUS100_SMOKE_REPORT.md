# Cus100: final configuration and engineering measurements, 2026-09-11

All ten configurations completed native optimizer updates with finite losses/parameters and nonzero parameter changes, followed by validation on all 500 instances with 30 candidates. E and G use the final source-specific reward contracts. This is an engineering readiness result, not convergence or evidence that any method is superior. None of these checkpoints is reused for formal training.

| TR | Server / GPU | Model / source | Physical=effective batch | Peak GiB | Probe logical / optimizer updates | Feasible val /500 | Val seconds | Estimated hours 5500–10000 incl. 15% |
|---|---|---|---:|---:|---:|---:|---:|---:|
| TR02 | 2080ti_4_1 / 0 | am_evrptw / G | 108 | 10.008 | 10 / 10 | 500 | 111.2 | 24.5–46.9 |
| TR01 | 2080ti_4_1 / 1 | am_evrptw / E | 108 | 10.008 | 10 / 10 | 485 | 197.9 | 26.2–50.9 |
| TR06 | 2080ti_4_1 / 2 | terran / G | 384 | 9.943 | 5 / 60 | 500 | 130.1 | 166.1–302.0 |
| TR05 | 2080ti_4_1 / 3 | terran / E | 384 | 9.945 | 3 / 36 | 61 | 282.2 | 191.9–348.9 |
| TR04 | 2080ti_4_2 / 0 | drl_ts / G | 24 | 10.143 | 10 / 10 | 500 | 117.7 | 15.3–26.9 |
| TR03 | 2080ti_4_2 / 1 | drl_ts / E | 24 | 10.217 | 10 / 10 | 393 | 159.3 | 21.8–40.7 |
| TR18 | 2080ti_4_2 / 2 | evrptw_rl / G | 200 | 10.025 | 2 / 2 | 500 | 232.9 | 128.9–236.1 |
| TR17 | 2080ti_4_2 / 3 | evrptw_rl / E | 200 | 10.025 | 3 / 3 | 352 | 297.4 | 136.5–249.8 |
| TR10 | 2080ti_3_1 / 0 | rrnco / G | 50 | 9.941 | 2 / 2 | 486 | 205.6 | 18.8–34.2 |
| TR09 | 2080ti_3_1 / 1 | rrnco / E | 50 | 9.941 | 10 / 10 | 495 | 184.6 | 16.7–30.3 |

E denotes TERRAN-derived synthetic Euclidean coordinates; G denotes existing Road data. Both sources have 50,000 training instances and 500 distinct validation instances. Seed 1234; train/validation trajectories 30; train/validation cap 240/360. No Cus500, Cus1000 or T1 evaluation is scheduled.

## Budget and time interpretation

The minimum is 5000 logical epochs, maximum 10000, validation every 100 and patience 5 strictly after 5000; earliest automatic stop is 5500. An epoch is one new environment batch, not one full 50k-data pass. TERRAN performs 3 PPO passes × 4 minibatches = 12 optimizer updates per logical epoch; the other methods perform one. Physical equals effective batch and differs by model, as requested for memory use; therefore the same epoch does not mean equal sample exposure. All methods within a source consume identical ordered sample-ID prefixes. Native baseline probes read additional training-only samples, recorded separately.

ETA uses actual epoch times excluding validation, then adds full 500-val every 100 epochs and 15% planning allowance. AM changes from EMA to greedy rollout baseline at 2500 updates; EVR changes at 1000; DRL soft/hard transition is 2500. Short probes accelerated these transitions only to cover both phases. Formal CLI keeps native intervals. Speed depends on route length, CPU and storage; this is not a completion guarantee.

Memory is nvidia-smi peak per process in GiB, including PyTorch cache/context. Training uses variable route lengths, so this is measured readiness rather than a bound on every future batch. AM peaks include separate greedy-baseline phase probes. DRL and EVR checkpoint decoder activations at stride 1, preserving RNG and gradients; no model parameters or architecture were removed.

## Initial policy quality and verification

The short probes have different update counts and must not be used to rank methods. Euclidean TERRAN produced 61/500 feasible validation selections after 3 epochs (36 PPO updates); EVR produced 352/500 after 3 updates; DRL 393/500, AM 485/500 and RRNCO 495/500 after 10 updates. Failures remain failures. On TERRAN E, training success fell from 3.02% to 0% and trajectories exhausted 240 steps; this is a material early-learning risk, not evidence of convergence. New reward scale 6530.963114223516 was confirmed in its logs. Independently verified 500 training-reference routes require 107–199 actions (median 124), all within 240; no mandatory unit/loader mismatch was identified. Further training must be assessed using feasible fraction together with cost, not cost on only the successful subset.

The selected feasible routes are independently verified before cost reporting. Cost identity is checked per selected route for the four common adapters and using verified-population means for TERRAN; errors are below 1e-8 USD. This does not claim replaying every one of all 15000 raw candidates. Validation is sampling 30, horizon 360, with no hidden route repair.

All methods report C_USD = 0.151750972762646 × D_km + 413.6331536717643 × K. The Road scale is 432.7147451224062 and E scale 6530.963114223516; only training normalization differs. E calibration used a deterministic 500-instance training-only cohort, never validation/test. No separate driver-hours term is added.

## Frozen data and reproducibility

The upstream TERRAN generator has physical-feasibility defects, including inverted windows. The documented canonical_candidate_feasibility_v4 guard is applied during candidate construction. This is TERRAN-derived data, not an unmodified upstream reproduction. No complete-instance filtering or learned-solver selection is used. All 50500 instances have independent feasible singleton witnesses for every customer; selected cases also pass sequential canonical mask replay and both verifiers. Train/val RNG streams are separate; raw hashes and unit conversions are frozen in corpus provenance. See TERRAN_SYNTHETIC100_DATA_CONTRACT.md.

The portable archive includes current actual code, complete E data, both sources' frozen streams and reports. It excludes existing 90.53 GiB Road parent matrices, Git metadata, Conda environment, probe checkpoints and formal outputs. Extract into an existing compatible repository with the Road release and environment installed. Source hash is checked on each host; extra or edited nonignored source files can cause a deliberate drift failure. GPU occupancy checks prevent accidental oversubscription.

Numeric evidence: EVRPTW_Benchmark/results/cus100_20260911/artifacts/final_calibration.json. Raw probe locations are recorded there. Related CPU regressions cover canonical cost/data/stream contracts, DRL/EVR checkpoint RNG-gradient equivalence and native baseline diagnostics; 146 tests passed; see artifacts/final_cpu_checks.txt for the final result.
